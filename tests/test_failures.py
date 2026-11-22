#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Eight kinds of failure of the Rabel API, each against the strategy in Pipecat's user
aggregator over two turns of a call.

Failing open, for each: no exception reaches the app and nothing is logged at ERROR; Pipecat answers the caller exactly as
it does with its own eager strategy and no Bell (the same frames out, the same replies, the same conversation); no reply is
started on a guess; and at most one line is logged for the failure. The strategy holds no frame in any of them: it is not in
the pipeline's path, and every frame reaches the aggregator and goes on before the strategy sees it.
"""

import pytest
from loguru import logger
from pipecat.tests.utils import SleepFrame

from .common import GUESS, H2, bell, one_turn, pipecat_alone, run, shape
from .standin import BlackHole, StandIn

TWO_TURNS = one_turn() + one_turn(wait=0.1)


@pytest.fixture
def logged():
    """Every line logged during a test at WARNING or above, from any part of Pipecat or the strategy."""
    lines = []
    sink = logger.add(lambda m: lines.append((m.record["level"].name, m.record["message"])), level="WARNING")
    yield lines
    logger.remove(sink)


async def fails_open(strategies, frames, logged, kinds):
    down, up, context = await run(strategies, frames)
    alone_down, alone_up, alone_context = await pipecat_alone(frames)
    assert shape(down) == shape(alone_down)
    assert shape(up) == shape(alone_up)
    assert context.messages == alone_context.messages
    assert [x for x in logged if x[0] in ("ERROR", "CRITICAL")] == []
    bells = [x for x in logged if "RabelBell" in x[1]]
    assert len(bells) <= 1, bells
    b = strategies.bell
    assert b.stats.guesses_pushed == 0
    assert set(b.stats.failures) <= set(kinds), b.stats.failures
    if kinds:
        assert len(bells) == 1 and b.stats.failures, (bells, b.stats.failures)
    return b


async def test_1_refused(logged):
    async with StandIn(refuse_http=401, guesses={H2: GUESS}) as api:
        await fails_open(bell(api.url), TWO_TURNS, logged, {"refused_401"})


async def test_2_unresolvable(logged):
    # .invalid never resolves (RFC 6761); a resolver slower than connect_timeout gives the timeout instead
    await fails_open(bell("ws://bell.invalid/v1/conversation"), TWO_TURNS, logged, {"down", "connect_timeout"})


async def test_3_a_tcp_black_hole(logged):
    async with BlackHole() as hole:
        await fails_open(bell(hole.url, connect_timeout=0.3), TWO_TURNS, logged, {"connect_timeout"})


async def test_4_a_websocket_that_never_answers(logged):
    async with StandIn(ready_delay=3600.0, guesses={H2: GUESS}) as api:
        await fails_open(bell(api.url, connect_timeout=0.3), TWO_TURNS, logged, {"connect_timeout"})


async def test_5_ready_late(logged):
    # ready comes, but after connect_timeout: the conversation is given up and retried, and the retry is late too
    async with StandIn(ready_delay=0.8, guesses={H2: GUESS}) as api:
        await fails_open(bell(api.url, connect_timeout=0.5), TWO_TURNS, logged, {"connect_timeout"})


async def test_6_ready_then_mute(logged):
    # ready, then nothing: no guess and no answer to anything, the connection open
    async with StandIn(mute=True, guesses={H2: GUESS}) as api:
        b = await fails_open(bell(api.url), TWO_TURNS, logged, set())
        assert b.stats.turns_sent == 2 and len(api.received[0]) >= 6  # the words went to Bell; nothing came back


async def test_7_guesses_three_seconds_late(logged):
    async with StandIn(guesses={H2: GUESS}, guess_delay=3.0) as api:
        b = await fails_open(bell(api.url), one_turn() + [SleepFrame(3.3)], logged, set())
    assert b.stats.guesses_received == 1 and b.stats.guesses_dropped == 1  # it came after its turn had ended


async def test_8_the_api_stalled_mid_call(logged):
    # after the first words the API stops reading and answering, without closing
    async with StandIn(stall_after_heard=1, guesses={H2: GUESS}) as api:
        b = await fails_open(bell(api.url), TWO_TURNS, logged, set())
    assert b.stats.guesses_received == 0
