#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bell's early reply through Pipecat's own user aggregator and turn controller, against the local stand-in.

These show the claim the README makes: Pipecat answers Bell's guess early, keeps that answer when the turn's final words
equal the guess, and withdraws it and answers the final words when they differ. Nothing made from a wrong guess reaches
the conversation. They also show the door the strategy uses: its early reply is Pipecat's speculation, started and
withdrawn by the strategy, and the strategy sends no frame of its own.
"""

from pipecat.frames.frames import (
    LLMContextFrame,
    UserStoppedSpeakingFrame,
    UserTurnResponseCancelFrame,
)
from pipecat.tests.utils import SleepFrame

from pipecat_rabel import RabelBellUserTurnStrategies

from .common import FINAL, GUESS, H1, H2, eager, final, interim, replies, run, start, stop
from .standin import StandIn


def turn(final_text):
    return [SleepFrame(0.3), start(), interim(H1), interim(H2), SleepFrame(0.2), final(final_text), SleepFrame(0.1),
            stop(), SleepFrame(1.0)]


async def run_frames(frames, **kwargs):
    async with StandIn(guesses={H2: GUESS}) as api:
        s = RabelBellUserTurnStrategies(api_key="test-key", url=api.url, connect_timeout=0.5, **kwargs)
        down, up, context = await run(s, frames)
    return s.bell, context, down, up


async def test_a_right_guess_is_answered_early_and_kept():
    _, context, down, _ = await run_frames(turn(FINAL))
    contexts = [f for f in down if isinstance(f, LLMContextFrame)]
    # One answer, started on Bell's guess before the turn ended, and kept: no second answer, no withdrawal.
    assert len(contexts) == 1 and contexts[0].speculation
    assert contexts[0].context.messages[-1] == {"role": "user", "content": GUESS}
    assert not any(isinstance(f, UserTurnResponseCancelFrame) for f in down)
    assert sum(isinstance(f, UserStoppedSpeakingFrame) for f in down) == 1
    # The conversation holds the speech-to-text's final words, never the guess.
    assert context.messages == [{"role": "user", "content": FINAL}]


async def test_a_wrong_guess_is_withdrawn_and_the_final_words_answered():
    _, context, down, _ = await run_frames(turn("I would like to reset my pin."))
    contexts = [f for f in down if isinstance(f, LLMContextFrame)]
    cancel = next(i for i, f in enumerate(down) if isinstance(f, UserTurnResponseCancelFrame))
    # The early answer on the guess, its withdrawal, then a fresh answer on the final words.
    assert [c.speculation for c in contexts] == [True, False]
    assert down.index(contexts[0]) < cancel < down.index(contexts[1])
    assert contexts[1].context.messages[-1] == {"role": "user", "content": "I would like to reset my pin."}
    assert context.messages == [{"role": "user", "content": "I would like to reset my pin."}]


async def test_the_strategy_sends_no_frame_of_its_own():
    # A stop strategy can push or broadcast frames through the aggregator (its on_push_frame and on_broadcast_frame
    # events). Bell's never does: its early reply and the withdrawal of a wrong one are Pipecat's speculation, which the
    # turn controller and the aggregator send.
    for final_text in (FINAL, "I would like to reset my pin."):
        sent = []
        async with StandIn(guesses={H2: GUESS}) as api:
            s = RabelBellUserTurnStrategies(api_key="test-key", url=api.url, connect_timeout=0.5)
            s.bell.add_event_handler("on_push_frame", lambda strategy, frame, direction: sent.append(frame))
            s.bell.add_event_handler("on_broadcast_frame", lambda strategy, frame_cls, **kw: sent.append(frame_cls))
            down, _, _ = await run(s, turn(final_text))
        assert sent == [], sent
        assert replies(down)[0] == (True, GUESS)


async def test_bells_reply_is_kept_when_the_speech_to_texts_early_end_of_turn_on_the_same_words_follows():
    # Without the rule, Pipecat starts a second reply on Flux's early end of turn and drops the one under way on Bell's
    # guess.
    frames = turn(FINAL)
    frames.insert(5, eager("I would like to reset my password."))  # after Bell's guess, before the final
    bell, context, down, _ = await run_frames(frames)
    contexts = [f for f in down if isinstance(f, LLMContextFrame)]
    assert len(contexts) == 1 and contexts[0].speculation
    assert contexts[0].context.messages[-1] == {"role": "user", "content": GUESS}
    assert not any(isinstance(f, UserTurnResponseCancelFrame) for f in down)
    assert bell.stats.eager_held_back == 1
    assert context.messages == [{"role": "user", "content": FINAL}]


async def test_after_bells_reply_times_out_the_speech_to_texts_early_end_of_turn_starts():
    # Bell's early reply goes unresolved past speculation_timeout and is withdrawn; Flux's early end of turn on the same
    # words then starts the reply.
    frames = turn(FINAL)
    frames[5:5] = [SleepFrame(0.8), eager("I would like to reset my password.")]
    bell, context, down, _ = await run_frames(frames, speculation_timeout=0.5)
    assert replies(down) == [(True, GUESS), (True, "I would like to reset my password.")]
    contexts = [f for f in down if isinstance(f, LLMContextFrame)]
    cancel = next(i for i, f in enumerate(down) if isinstance(f, UserTurnResponseCancelFrame))
    assert down.index(contexts[0]) < cancel < down.index(contexts[1])
    assert bell.stats.eager_held_back == 0
    assert context.messages == [{"role": "user", "content": FINAL}]
