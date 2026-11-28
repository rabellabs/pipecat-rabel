#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RabelBellUserTurnStopStrategy in Pipecat's user aggregator, against a local stand-in for Bell's API: no key and no
network.

The fail-open tests all ask the same thing: with the API failing in one way, does Pipecat answer the caller exactly as it
does with its own eager strategy and no Bell, and is exactly one line logged for that kind of failure?
"""

import asyncio
import importlib.metadata
import sys

import pytest
from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, LLMContextFrame, UserTurnResponseCancelFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

import pipecat_rabel
from pipecat_rabel import RabelBellUserTurnStopStrategy, RabelBellUserTurnStrategies
from pipecat_rabel import strategy as bell_module

from .common import (
    FINAL,
    GUESS,
    H1,
    H2,
    bell,
    eager,
    final,
    interim,
    one_turn,
    pipecat_alone,
    replies,
    run,
    shape,
    start,
    stop,
)
from .standin import BlackHole, SilentProxy, StandIn, unused_url


@pytest.fixture
def warnings():
    """The WARNING lines the strategy logs during a test."""
    lines = []
    sink = logger.add(lambda m: lines.append(m.record["message"]), level="WARNING",
                      filter=lambda r: "RabelBell" in r["message"])
    yield lines
    logger.remove(sink)


async def assert_fails_open(strategies, frames, warnings, kind, wait_after=0.0):
    """Pipecat answers as it does on its own; one line for this kind; no reply started on a guess."""
    if wait_after:
        frames = frames + [SleepFrame(wait_after)]
    down, up, context = await run(strategies, frames)
    alone_down, alone_up, alone_context = await pipecat_alone(frames)
    assert shape(down) == shape(alone_down)
    assert shape(up) == shape(alone_up)
    assert context.messages == alone_context.messages
    assert len(warnings) == 1, warnings
    assert strategies.bell.stats.guesses_pushed == 0
    assert kind in strategies.bell.stats.failures, strategies.bell.stats.failures
    return down


# ---------------------------------------------------------------------------------------------------- the working path


async def test_the_bundle_is_pipecats_external_strategies_with_bells_stop_strategy():
    s = RabelBellUserTurnStrategies(api_key="test-key", agent="front-desk")
    assert isinstance(s, ExternalUserTurnStrategies)
    assert [type(x).__name__ for x in s.start] == ["ExternalUserTurnStartStrategy"]
    assert len(s.stop) == 1 and s.bell is s.stop[0] and isinstance(s.bell, RabelBellUserTurnStopStrategy)
    assert s.bell.agent == "front-desk"
    assert "test-key" not in repr(s)


async def test_a_guess_starts_an_early_reply_that_is_kept(warnings):
    async with StandIn(guesses={H2: GUESS}) as api:
        s = bell(api.url, agent="front-desk", min_chance=0.7, vendor="deepgram/flux-general-en")
        down, up, context = await run(s, one_turn())
    # one reply, started on Bell's guess before the turn ended, and kept: no second reply, no withdrawal
    assert replies(down) == [(True, GUESS)]
    assert not any(isinstance(f, UserTurnResponseCancelFrame) for f in down + up)
    # the conversation holds the speech-to-text's final words, never the guess
    assert context.messages == [{"role": "user", "content": FINAL}]
    # the report: the guess kept, with its head start (the 0.2 s between Bell's guess and the final)
    head_start_ms = api.received[0][-1].pop("head_start_ms")
    assert 150 <= head_start_ms <= 600, head_start_ms
    assert api.received[0] == [
        {"type": "heard", "text": H1},
        {"type": "heard", "text": H2},
        {"type": "end_of_turn", "text": FINAL, "guess": "kept"},
    ]
    st = s.bell.stats
    assert st.guesses_pushed == 1 and st.guesses_right == 1 and st.turns_sent == 1
    assert warnings == []


async def test_hello_takes_the_new_bell_and_the_key_goes_in_the_header():
    async with StandIn() as api:
        s = bell(api.url, agent="front-desk", learn=False, vendor="deepgram/flux-general-en")
        await run(s, one_turn())
    # version 3, and no min_chance: the new Bell's balanced setting
    assert api.hellos == [{
        "type": "hello", "v": 3, "agent": "front-desk", "learn": False,
        "host": "pipecat", "host_version": importlib.metadata.version("pipecat-ai"),
        "hookup": pipecat_rabel.__version__, "vendor": "deepgram/flux-general-en",
    }]
    assert api.headers[0]["Authorization"] == "Bearer test-key"
    assert "key" not in api.hellos[0]
    assert s.bell.min_chance is None


async def test_hello_carries_min_chance_only_when_set():
    async with StandIn() as api:
        s = bell(api.url, min_chance=0.8)
        await run(s, one_turn())
    assert api.hellos[0]["v"] == 3 and api.hellos[0]["min_chance"] == 0.8


async def test_with_no_min_chance_the_new_bell_adds_no_bar_and_a_version_2_api_keeps_0_7():
    # A version 3 API sends only guesses as sure as its balanced setting allows; the strategy adds no bar of its own. A
    # version 2 API with no min_chance from either side keeps 0.3.0's 0.7.
    got = []
    for version in (None, 2):
        async with StandIn(guesses={H2: GUESS}, chance=0.5, version=version, ready_extra={"min_chance": None}) as api:
            s = bell(api.url)
            down, _, _ = await run(s, one_turn())
        got.append(replies(down))
    assert got == [[(True, GUESS)], [(False, FINAL)]]


async def test_a_version_3_ready_is_the_new_bell_and_a_version_2_ready_works_as_before():
    seen = []
    for version in (None, 2):
        async with StandIn(guesses={H2: GUESS}, version=version) as api:
            s = bell(api.url)

            @s.bell.event_handler("on_ready")
            async def on_ready(strategy, ready):
                seen.append((ready["v"], strategy.v3))

            down, _, _ = await run(s, one_turn())
        assert replies(down) == [(True, GUESS)]
        assert api.received[0][-1]["guess"] == "kept"
    assert seen == [(3, True), (2, False)]


async def test_ready_hold_line_is_kept_from_the_api():
    async with StandIn(hold_line=0.85) as api:
        s = bell(api.url)
        seen = []

        @s.bell.event_handler("on_ready")
        async def on_ready(strategy, ready):
            seen.append((ready, strategy.hold_line))

        await run(s, [SleepFrame(0.3)])
    assert seen == [({"v": 3, "hold_line": 0.85, "calls_heard": 143}, 0.85)]


@pytest.mark.parametrize(
    "extra, always, updates",
    [
        ({"hold_always": True, "hold_updates": 2}, True, 2),
        ({"hold_always": False, "hold_updates": 1}, False, 1),
        ({"hold_always": "yes", "hold_updates": 3}, None, None),
        ({"hold_updates": True}, None, None),
    ],
    ids=["set", "today", "bad", "true-is-not-a-count"],
)
async def test_ready_hold_settings_are_read_and_change_nothing(extra, always, updates, warnings):
    """``hold_always`` and ``hold_updates`` are LiveKit's rule: read as ``hold_line`` is, never
    applied. Pipecat answers the same as with a ``ready`` that has neither."""

    async def frames_out(ready_extra):
        async with StandIn(guesses={H2: GUESS}, ready_extra=ready_extra) as api:
            s = bell(api.url)
            seen = []

            @s.bell.event_handler("on_ready")
            async def on_ready(strategy, ready):
                seen.append((strategy.hold_line, strategy.hold_always, strategy.hold_updates))

            down, up, _ = await run(s, one_turn())
        return shape(down), shape(up), seen, s.bell.stats

    base = await frames_out({})
    got = await frames_out(extra)
    assert base[2] == [(0.9, None, None)]
    assert got[2] == [(0.9, always, updates)]
    assert got[0] == base[0] and got[1] == base[1]
    assert (got[3].guesses_pushed, got[3].guesses_right) == (base[3].guesses_pushed, base[3].guesses_right) == (1, 1)
    assert warnings == []


async def test_a_version_2_ready_has_no_hold_settings():
    """``hold_always`` and ``hold_updates`` are the new Bell's: a version 2 ``ready`` that carries them is not read for
    them, as the LiveKit hookup does not read them."""
    async with StandIn(version=2, ready_extra={"hold_always": True, "hold_updates": 2}) as api:
        s = bell(api.url)
        seen = []

        @s.bell.event_handler("on_ready")
        async def on_ready(strategy, ready):
            seen.append((strategy.hold_line, strategy.hold_always, strategy.hold_updates))

        await run(s, [SleepFrame(0.3)])
    assert seen == [(0.9, None, None)]


async def test_a_wrong_guess_is_withdrawn_and_the_turn_end_reports_it():
    async with StandIn(guesses={H2: GUESS}) as api:
        s = bell(api.url)
        ended = []

        @s.bell.event_handler("on_turn_end")
        async def on_turn_end(strategy, final_text, guess, right):
            ended.append((final_text, guess, right))

        down, _, context = await run(s, one_turn("I would like to reset my pin."))
    assert replies(down) == [(True, GUESS), (False, "I would like to reset my pin.")]
    early, cancel, answer = (
        next(i for i, f in enumerate(down) if isinstance(f, LLMContextFrame)),
        next(i for i, f in enumerate(down) if isinstance(f, UserTurnResponseCancelFrame)),
        [i for i, f in enumerate(down) if isinstance(f, LLMContextFrame)][-1],
    )
    assert early < cancel < answer
    assert context.messages == [{"role": "user", "content": "I would like to reset my pin."}]
    assert ended == [("I would like to reset my pin.", GUESS, False)]
    assert s.bell.stats.guesses_right == 0


async def test_min_chance_is_clamped_and_agent_name_checked():
    assert RabelBellUserTurnStopStrategy(api_key="k", min_chance=0.2).min_chance == 0.67
    assert RabelBellUserTurnStopStrategy(api_key="k", min_chance=0.6).min_chance == 0.67
    assert bell("ws://x", min_chance=1.5).bell.min_chance == 0.99
    with pytest.raises(ValueError):
        bell("ws://x", agent="Front Desk")


async def test_base_address_becomes_the_conversation_endpoint():
    assert bell_module._ws_url("https://api.rabellabs.com") == "wss://api.rabellabs.com/v1/conversation"
    assert bell_module._ws_url("http://localhost:8000/") == "ws://localhost:8000/v1/conversation"
    assert bell_module._ws_url("wss://api.rabellabs.com/v1/conversation") == "wss://api.rabellabs.com/v1/conversation"


async def test_one_guess_in_each_of_two_turns():
    async with StandIn(guesses={H1: "i would like to pay", H2: GUESS}) as api:
        s = bell(api.url)
        frames = [SleepFrame(0.3), start(), interim(H1), SleepFrame(0.1), final("I would like to pay."), stop(),
                  SleepFrame(0.1), start(), interim(H2), SleepFrame(0.1), final(FINAL), stop(), SleepFrame(0.1)]
        down, _, context = await run(s, frames)
    # two turns, one early reply each, both kept: no withdrawal needed across turns
    assert replies(down) == [(True, "i would like to pay"), (True, GUESS)]
    assert not any(isinstance(f, UserTurnResponseCancelFrame) for f in down)
    assert [m["content"] for m in context.messages] == ["I would like to pay.", FINAL]
    assert s.bell.stats.guesses_right == 2


async def test_two_guesses_in_one_turn_withdraw_the_first():
    # The new Bell can send a second guess in a turn; the first is withdrawn before the second starts.
    first = '{"type": "guess", "turn": 1, "id": 1, "text": "i would like to pay", "heard": "i would like", "chance": 0.8}'
    async with StandIn(raw_guesses={H1: first}, guesses={H2: GUESS}) as api:
        s = bell(api.url)
        frames = [SleepFrame(0.3), start(), interim(H1), SleepFrame(0.1), interim(H2), SleepFrame(0.1), final(FINAL),
                  stop(), SleepFrame(0.1)]
        down, _, context = await run(s, frames)
    kinds = [k for k in shape(down) if k[0] in ("LLMContextFrame", "UserTurnResponseCancelFrame")]
    assert kinds == [("LLMContextFrame", True, "i would like to pay"), ("UserTurnResponseCancelFrame", None),
                     ("LLMContextFrame", True, GUESS)]
    assert context.messages == [{"role": "user", "content": FINAL}]


async def test_the_speech_to_texts_early_reply_on_the_words_stays_until_the_words_reach_bells_guess():
    # Flux's own eager end of turn on the words so far stays; Bell's guess is started when the words reach it.
    async with StandIn(guesses={H2: GUESS}, guess_delay=0.05) as api:
        s = bell(api.url)
        frames = [SleepFrame(0.3), start(), interim(H2), eager(H2), SleepFrame(0.2), interim(GUESS), SleepFrame(0.05),
                  final(FINAL), stop(), SleepFrame(0.1)]
        down, _, context = await run(s, frames)
    kinds = [k for k in shape(down) if k[0] in ("LLMContextFrame", "UserTurnResponseCancelFrame")]
    assert kinds == [("LLMContextFrame", True, H2), ("UserTurnResponseCancelFrame", None),
                     ("LLMContextFrame", True, GUESS)]
    assert context.messages == [{"role": "user", "content": FINAL}]
    assert s.bell.stats.eager_held_back == 0


async def test_bell_then_the_speech_to_texts_early_end_of_turn_on_the_same_words_is_not_started():
    # Bell's early reply is under way; Flux's own on the same words (cased and punctuated) would make Pipecat start a
    # second reply and drop the one under way on Bell's guess, so it is not started.
    async with StandIn(guesses={H2: GUESS}) as api:
        s = bell(api.url)
        frames = one_turn()
        frames.insert(5, eager("I would like to reset my password."))  # after Bell's guess, before the final
        down, up, context = await run(s, frames)
    assert replies(down) == [(True, GUESS)]
    assert not any(isinstance(f, UserTurnResponseCancelFrame) for f in down + up)
    assert context.messages == [{"role": "user", "content": FINAL}]
    assert s.bell.stats.eager_held_back == 1 and s.bell.stats.guesses_right == 1


async def test_pipecats_eager_strategy_starts_its_early_reply_through_this_method(monkeypatch):
    # The held-back check above lives in an override of start_user_turn_speculation, which Pipecat's eager strategy
    # calls for the speech-to-text's own early end of turn. If Pipecat stops calling it, this test fails.
    seen = []
    original = RabelBellUserTurnStopStrategy.start_user_turn_speculation

    async def spy(self, speculation):
        seen.append(speculation.text)
        await original(self, speculation)

    monkeypatch.setattr(RabelBellUserTurnStopStrategy, "start_user_turn_speculation", spy)
    async with StandIn(guesses={}) as api:
        s = bell(api.url)
        frames = one_turn()
        frames.insert(5, eager("I would like to reset my password."))
        down, _, _ = await run(s, frames)
    assert seen == ["I would like to reset my password."]
    assert replies(down) == [(True, "I would like to reset my password.")]


async def test_bell_then_the_speech_to_texts_early_end_of_turn_on_other_words_replaces_bells():
    async with StandIn(guesses={H2: GUESS}) as api:
        s = bell(api.url)
        frames = one_turn("I would like to reset my pin.")
        frames.insert(5, eager("I would like to reset my pin."))
        down, _, context = await run(s, frames)
    assert replies(down) == [(True, GUESS), (True, "I would like to reset my pin.")]
    assert context.messages == [{"role": "user", "content": "I would like to reset my pin."}]
    assert s.bell.stats.eager_held_back == 0


async def test_bell_withdrawn_then_the_speech_to_texts_early_end_of_turn_starts():
    # Flux withdrew (the caller resumed): its UserTurnResponseCancelFrame voids Bell's early reply too, so Flux's next
    # early end of turn on the same words is the only one and starts.
    async with StandIn(guesses={H2: GUESS}) as api:
        s = bell(api.url)
        frames = one_turn()
        frames[5:5] = [UserTurnResponseCancelFrame(), eager("I would like to reset my password.")]
        down, _, context = await run(s, frames)
    assert replies(down) == [(True, GUESS), (True, "I would like to reset my password.")]
    assert context.messages == [{"role": "user", "content": FINAL}]
    assert s.bell.stats.eager_held_back == 0


async def test_a_conversation_opened_mid_turn_waits_for_the_next_turn():
    async with StandIn(ready_delay=0.3, guesses={H2: GUESS}) as api:
        s = bell(api.url, connect_timeout=2.0)
        frames = [start(), interim(H1), SleepFrame(0.5), interim(H2), final(FINAL), stop()] + one_turn(wait=0.05)
        down, _, _ = await run(s, frames)
    # The first turn began before Bell was ready: none of it went to Bell. The second went whole.
    assert api.received[0][-1].pop("head_start_ms") >= 0  # the report: the second turn's guess kept
    assert api.received[0] == [{"type": "heard", "text": H1}, {"type": "heard", "text": H2},
                               {"type": "end_of_turn", "text": FINAL, "guess": "kept"}]
    assert [r[0] for r in replies(down)] == [False, True]


async def test_a_clean_close_after_a_long_conversation_reopens_at_once(monkeypatch, warnings):
    monkeypatch.setattr(bell_module, "_LONG_CONVERSATION_S", 0.0)
    async with StandIn(guesses={H2: GUESS}, close_clean_after_turns=1) as api:
        s = bell(api.url)
        down, _, _ = await run(s, one_turn() + one_turn(wait=0.2))
    # Each conversation closes after one turn and is reopened at once: the third waits, unused, for a third turn.
    assert api.connections == 3 and s.bell.stats.conversations == 3
    assert len(api.received[1]) == 3  # the second turn went whole to the second conversation
    assert replies(down) == [(True, GUESS), (True, GUESS)]
    assert warnings == []


async def test_a_late_guess_after_the_turn_ended_is_dropped():
    async with StandIn(guesses={H2: GUESS}, guess_delay=0.3) as api:
        s = bell(api.url, answer_timeout=5.0)
        frames = [SleepFrame(0.3), start(), interim(H1), interim(H2), final(FINAL), stop(), SleepFrame(0.5)]
        down, _, _ = await run(s, frames)
    alone, _, _ = await pipecat_alone(frames)
    assert shape(down) == shape(alone)
    assert s.bell.stats.guesses_dropped == 1


# ------------------------------------------------------------------------------------------------------ failing open


async def test_no_key(warnings, monkeypatch):
    monkeypatch.delenv("RABEL_API_KEY", raising=False)
    async with StandIn() as api:
        s = RabelBellUserTurnStrategies(url=api.url)
        await assert_fails_open(s, one_turn(), warnings, "no_key")
    assert api.connections == 0


async def test_api_down(warnings):
    # A closed local port refuses at once on Linux: "down". On Windows the refusal takes about two seconds, longer than
    # the connect timeout here, so the same failure is counted as "connect_timeout". Either way Bell fails open.
    kind = "connect_timeout" if sys.platform == "win32" else "down"
    await assert_fails_open(bell(unused_url()), one_turn(), warnings, kind)


async def test_api_slow_to_connect(warnings):
    async with BlackHole() as hole:
        await assert_fails_open(bell(hole.url, connect_timeout=0.3), one_turn(wait=0.5), warnings, "connect_timeout")


async def test_api_slow_to_answer_ready(warnings):
    async with StandIn(ready_delay=5.0, guesses={H2: GUESS}) as api:
        await assert_fails_open(bell(api.url, connect_timeout=0.3), one_turn(wait=0.5), warnings, "connect_timeout")


async def test_api_slow_to_answer_a_guess(warnings):
    async with StandIn(guesses={H2: GUESS}, guess_delay=0.4) as api:
        frames = [SleepFrame(0.3), start(), interim(H1), interim(H2), SleepFrame(0.6), final(FINAL), stop(),
                  SleepFrame(0.1)]
        await assert_fails_open(bell(api.url, answer_timeout=0.2), frames, warnings, "slow_answer")


@pytest.mark.parametrize("status,stays_off", [(401, True), (402, True), (429, False), (529, False)])
async def test_api_refuses_at_the_door(warnings, status, stays_off):
    async with StandIn(refuse_http=status) as api:
        # Long enough for the first retry (1 s) of a refusal that is retried.
        await assert_fails_open(bell(api.url), one_turn(), warnings, f"refused_{status}", wait_after=1.0)
    assert (api.connections == 1) is stays_off, api.connections
    assert str(status) in warnings[0]


@pytest.mark.parametrize("status,stays_off", [(401, True), (402, True), (429, False), (529, False)])
async def test_api_refuses_with_an_error_message(warnings, status, stays_off):
    async with StandIn(refuse_message=status) as api:
        await assert_fails_open(bell(api.url), one_turn(), warnings, f"refused_{status}", wait_after=1.0)
    assert (api.connections == 1) is stays_off, api.connections


async def test_api_refusal_retry_after_is_honoured(warnings):
    async with StandIn(refuse_http=529, retry_after_s=30) as api:
        await assert_fails_open(bell(api.url), one_turn(), warnings, "refused_529", wait_after=1.2)
    assert api.connections == 1  # waiting the API's 30 s, not the 1 s back-off


async def test_api_closes_mid_call_and_is_reopened(warnings):
    async with StandIn(close_after_heard=1, guesses={H2: GUESS}) as api:
        await assert_fails_open(bell(api.url), one_turn(), warnings, "closed", wait_after=1.2)
        assert api.connections == 2  # reopened after the 1 s back-off


async def test_an_api_that_closes_right_after_ready_is_reopened_on_the_growing_waits(warnings):
    # ready, then the line drops at once (a worker that accepts and dies). The back-off starts again only after a
    # conversation that stayed up, so it is reopened 1, 2, 4 s later and so on, not every second for the whole call.
    async with StandIn(close_after_ready=0.05) as api:
        s = bell(api.url)
        await run(s, [SleepFrame(4.5)])
    assert api.connections <= 3, api.connections  # at 0, 1 and 3 s; every second would be 5
    assert s.bell.stats.failures.get("closed", 0) >= 2
    assert len(warnings) == 1, warnings


async def test_a_connection_that_goes_silent_after_ready_is_noticed_and_reopened(warnings, monkeypatch):
    # The socket stays open but nothing comes back, not even an answer to a ping (a half-open socket). The strategy
    # pings, counts the connection dead, and opens another; the next turn's guess is started as usual. With the
    # library's own timing (a ping every 20 s, 20 s to answer) the dead line would hold the words for 40 s.
    monkeypatch.setattr(bell_module, "_PING_INTERVAL_S", 0.2)
    monkeypatch.setattr(bell_module, "_LIVENESS_TIMEOUT_S", 0.3)
    monkeypatch.setattr(bell_module, "_CLOSE_TIMEOUT_S", 0.2)
    async with StandIn(guesses={H2: GUESS}) as api, SilentProxy(api.url) as proxy:
        s = bell(proxy.url)
        readies = []

        @s.bell.event_handler("on_ready")
        async def on_ready(strategy, ready):
            readies.append(ready)
            if len(readies) == 1:
                proxy.silence()

        down, _, _ = await run(s, [SleepFrame(0.3), SleepFrame(2.2)] + one_turn())
    assert proxy.connections == 2 and len(readies) == 2
    assert s.bell.stats.conversations == 2 and "closed" in s.bell.stats.failures
    assert len(warnings) == 1, warnings
    assert replies(down) == [(True, GUESS)]


@pytest.mark.parametrize(
    "raw",
    [
        "this is not json",
        '{"type": "guess", "turn": 1, "id": 1, "heard": "i would like to reset", "chance": 0.9}',
        '{"type": "guess", "turn": 1, "id": 1, "text": "", "heard": "i would like to reset", "chance": 0.9}',
        '{"type": "guess", "turn": 1, "id": 1, "text": "x y", "heard": "i would like to reset", "chance": 7}',
        '{"type": "guess", "turn": 1, "id": 1, "text": "x y", "heard": "i would like to reset", "chance": "high"}',
        '{"type": "guess", "turn": 1, "id": 1, "text": "x y", "heard": "i would like to reset", "chance": 0.3}',
        '{"type": "guess", "turn": 1, "id": 1, "text": "x y", "heard": "i would like to reset"}',
        '{"type": "guess", "turn": 1, "id": 1, "text": "x y", "heard": "words never sent", "chance": 0.9}',
        '["a", "list"]',
    ],
    ids=["not-json", "no-text", "empty-text", "chance-over-1", "chance-not-a-number", "chance-under-min",
         "no-chance", "unknown-heard", "not-an-object"],
)
async def test_api_sends_wrong_data(warnings, raw):
    # min_chance 0.8 set: the API's ready says it, and a guess under it is not used
    async with StandIn(raw_guesses={H2: raw}) as api:
        await assert_fails_open(bell(api.url, min_chance=0.8), one_turn(), warnings, "wrong_data")
        # the conversation went on; no guess was started, so the report says none
        assert api.received[0][-1] == {"type": "end_of_turn", "text": FINAL, "guess": "none"}


async def test_api_sends_wrong_data_instead_of_ready(warnings):
    async with StandIn(not_ready='{"type": "welcome"}') as api:
        await assert_fails_open(bell(api.url), one_turn(), warnings, "wrong_data")


async def test_wrong_data_many_times_is_still_one_line(warnings):
    bad = "this is not json"
    async with StandIn(raw_guesses={H1: bad, H2: bad}) as api:
        s = bell(api.url)
        await assert_fails_open(s, one_turn() + one_turn(wait=0.05), warnings, "wrong_data")
    assert s.bell.stats.failures["wrong_data"] == 4


async def test_unknown_message_types_are_ignored(warnings):
    async with StandIn(raw_guesses={H1: '{"type": "status", "calls_heard": 144}'}, guesses={H2: GUESS}) as api:
        down, _, _ = await run(bell(api.url), one_turn())
    assert replies(down) == [(True, GUESS)]
    assert warnings == []


async def test_end_frame_closes_the_conversation_after_the_last_words():
    async with StandIn() as api:
        s = bell(api.url)
        await run(s, [SleepFrame(0.3), start(), interim(H1), final("I would like."), stop()])
        await asyncio.sleep(0.1)
    assert api.received[0] == [{"type": "heard", "text": H1},
                               {"type": "end_of_turn", "text": "I would like.", "guess": "none"}]
    assert not s.bell.ready.is_set()


# ---------------------------------------------------------------------------------------------------- the end of a call


async def test_cleanup_stops_bell_when_no_end_frame_reached_the_strategy():
    # Pipecat always cleans the strategies up at teardown, while a call cancelled during start-up never shows the
    # strategy its CancelFrame. cleanup() alone must stop Bell's tasks and close the conversation, so nothing keeps
    # reopening it.
    class NoEndHandling(RabelBellUserTurnStopStrategy):
        async def process_frame(self, frame):
            if isinstance(frame, (EndFrame, CancelFrame)):
                return
            await super().process_frame(frame)

    async with StandIn() as api:
        s = RabelBellUserTurnStrategies()
        s.stop = [NoEndHandling(url=api.url, api_key="test-key", connect_timeout=0.5)]
        await run(s, [SleepFrame(0.3)])
        b = s.stop[0]
        assert b._connection_task is None and b._sender_task is None and b._conv is None
        assert not b.ready.is_set()
        await asyncio.sleep(1.2)  # longer than the first wait before a reopening
    assert api.connections == 1


async def test_setup_again_after_cleanup_opens_bell_again():
    # Pipecat cleans a strategy up and sets it up again when a service's recommended strategies are merged in
    # (UserTurnController.update_strategies). The strategy closes its conversation and opens a new one.
    async with StandIn(guesses={H2: GUESS}) as api:
        s = bell(api.url)
        controller_setups = []
        original = s.bell.setup

        async def setup_twice(setup):
            controller_setups.append(setup)
            await original(setup)
            if len(controller_setups) == 1:
                await asyncio.sleep(0.2)  # the first conversation opens
                await s.bell.cleanup()
                await original(setup)

        s.bell.setup = setup_twice
        down, _, _ = await run(s, one_turn())
    assert api.connections == 2 and s.bell.stats.conversations == 2
    assert replies(down) == [(True, GUESS)]


async def test_cancel_frame_passes_on_before_bell_closes():
    # A cancelled call ends at once: the aggregator sends the CancelFrame on before the strategy closes Bell.
    order = []
    async with StandIn() as api:
        s = bell(api.url)
        user = LLMUserAggregator(LLMContext(), params=LLMUserAggregatorParams(user_turn_strategies=s))
        stop_bell, push = s.bell._stop, user.push_frame

        async def recorded_stop(polite, final):
            order.append(("stop", polite, final))
            await stop_bell(polite, final)

        async def recorded_push(frame, direction=FrameDirection.DOWNSTREAM):
            if isinstance(frame, CancelFrame):
                order.append("cancel passed on")
            await push(frame, direction)

        s.bell._stop, user.push_frame = recorded_stop, recorded_push
        await run_test(Pipeline([user]), frames_to_send=[SleepFrame(0.3), CancelFrame()], send_end_frame=False)
    assert order[:2] == ["cancel passed on", ("stop", False, True)], order
    assert not s.bell.ready.is_set()


async def test_the_address_comes_from_rabel_api_url(monkeypatch):
    async with StandIn() as api:
        monkeypatch.setenv("RABEL_API_URL", api.url.replace("ws://", "http://").replace("/v1/conversation", ""))
        s = RabelBellUserTurnStrategies(api_key="test-key", connect_timeout=0.5)
        assert s.bell._url == api.url
        await run(s, one_turn())
    assert len(api.hellos) == 1
    monkeypatch.setenv("RABEL_API_URL", "ws://127.0.0.1:9/v1/conversation")
    assert RabelBellUserTurnStopStrategy(api_key="k", url=api.url)._url == api.url  # an explicit url wins
    monkeypatch.delenv("RABEL_API_URL")
    assert RabelBellUserTurnStopStrategy(api_key="k")._url == bell_module.RABEL_API_URL
