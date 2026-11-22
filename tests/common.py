#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""What the tests share: the words of a turn, the frames a speech-to-text sends, and Pipecat's own user aggregator to
run them through.

The strategy is run where a developer puts it: in the user aggregator's turn strategies. A test reads Pipecat's own
decisions from the frames the aggregator sends on: an early reply is an ``LLMContextFrame`` with ``speculation`` true, its
withdrawal a ``UserTurnResponseCancelFrame``, the reply to the final words an ``LLMContextFrame`` without it, and the
conversation is the context the aggregator writes.
"""

from pipecat.frames.frames import (
    EagerTranscriptionFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMUserAggregatorParams
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_turn_strategies import EagerUserTurnStrategies

from pipecat_rabel import RabelBellUserTurnStrategies

TS = "2026-11-19T00:00:00Z"
H1 = "i would like"
H2 = "i would like to reset"
GUESS = "i would like to reset my password"
FINAL = "I would like to reset my password."


def interim(text):
    return InterimTranscriptionFrame(text, "caller", TS)


def final(text):
    # Flux's EndOfTurn: the turn's one committed transcript, finalized
    return TranscriptionFrame(text, "caller", TS, finalized=True)


def eager(text):
    # the speech-to-text's own eager end of turn (Flux's EagerEndOfTurn), sent by the test as Flux would send it
    return EagerTranscriptionFrame(text, "caller", TS)


def start():
    return ProposedUserStartedSpeakingFrame()


def stop():
    return ProposedUserStoppedSpeakingFrame()


def one_turn(final_text=FINAL, wait=0.3):
    """A turn as Flux sends it, with time for the connection to open and for Bell to answer: StartOfTurn, two Updates,
    EndOfTurn (the final transcript, then the proposed stop)."""
    return [SleepFrame(wait), start(), interim(H1), interim(H2), SleepFrame(0.2), final(final_text), stop(),
            SleepFrame(0.1)]


def bell(url, **kwargs):
    """The bundle against the stand-in, with a short connect timeout."""
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("connect_timeout", 0.5)
    return RabelBellUserTurnStrategies(url=url, **kwargs)


async def run(strategies, frames, **kwargs):
    """The frames through Pipecat's user aggregator on these strategies: (frames out, frames up, the context)."""
    context = LLMContext()
    user = LLMUserAggregator(context, params=LLMUserAggregatorParams(user_turn_strategies=strategies))
    down, up = await run_test(Pipeline([user]), frames_to_send=frames, **kwargs)
    return down, up, context


def shape(frames):
    """The frames out as Pipecat's decisions read them: each frame's kind, and for a reply whether it is early and the
    words it answers. Two runs with the same shape gave the caller the same thing."""
    out = []
    for f in frames:
        if isinstance(f, LLMContextFrame):
            out.append(("LLMContextFrame", bool(f.speculation), f.context.messages[-1].get("content")))
        else:
            out.append((type(f).__name__, getattr(f, "text", None)))
    return out


def replies(frames):
    """The replies Pipecat started: (early, the words answered)."""
    return [(bool(f.speculation), f.context.messages[-1]["content"]) for f in frames if isinstance(f, LLMContextFrame)]


async def pipecat_alone(frames, **kwargs):
    """The same frames through Pipecat's own eager strategies, without Bell: what failing open must give."""
    return await run(EagerUserTurnStrategies(), frames, **kwargs)


def ends(api):
    """The end_of_turn messages the stand-in received, in order."""
    return [m for conv in api.received for m in conv if m.get("type") == "end_of_turn"]


def without_sleeps(frames):
    return [f for f in frames if not isinstance(f, SleepFrame)]
