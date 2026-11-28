#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""P1 and P2 of the code audit of 22 November 2026, and P3 to P6 for the new Bell (28 November): Bell beside Flux's eager
end of turn, with a real LLM service.

The harness runs Pipecat's own user aggregator with the strategies under test, then a real ``LLMService`` subclass, then
a recorder. The service is a local stand-in that runs one completion at a time inline, as Pipecat's OpenAI and Anthropic
services do: on each ``LLMContextFrame`` it pushes ``LLMFullResponseStartFrame``, waits ``TTFT`` (its time to first
token), pushes ``PARTS`` text frames ``FRAME_DELAY`` apart, then ``LLMFullResponseEndFrame``. No network. Pipecat 1.16.0
hosts its ``SpeculationGate`` inside every ``LLMService``, so what the recorder sees is what passed the gate: what the
caller would hear. Each text frame names the completion it belongs to, and the service records the words each completion
answers. Bell's API is the stand-in of ``standin.py``.

Each test runs the same frames twice: with Bell (``RabelBellUserTurnStrategies``) and with Pipecat's own
``EagerUserTurnStrategies`` alone. Its measure is the time from the turn's end (``UserStoppedSpeakingFrame`` past the
gate) to the first text frame past the gate of a reply to the caller's real words. With Bell it must be no later than
with Pipecat alone, within ``TOLERANCE`` (the stand-in's frame delay).
"""

import asyncio
import time

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_stop import NormalizedMatch
from pipecat.turns.user_turn_strategies import EagerUserTurnStrategies

from .common import FINAL, GUESS, H1, H2, bell, eager, final, interim, start, stop
from .standin import StandIn

TTFT = 0.4  # the stand-in LLM's time to first token, in seconds
FRAME_DELAY = 0.1  # between its text frames
PARTS = 3  # text frames a completion sends: a completion lasts TTFT + (PARTS - 1) * FRAME_DELAY = 0.6 s
TOLERANCE = FRAME_DELAY

_MATCH = NormalizedMatch()


class SlowLLM(LLMService):
    """A local stand-in LLM service: one completion at a time, inline, a few slow text frames, no network."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.runs: list[tuple[float, str, bool]] = []  # (start, the words answered, early) for each completion

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            n = len(self.runs)
            words = frame.context.messages[-1]["content"]
            self.runs.append((time.monotonic(), words, bool(frame.speculation)))
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await asyncio.sleep(TTFT)
                for k in range(PARTS):
                    if k:
                        await asyncio.sleep(FRAME_DELAY)
                    await self._push_llm_text(f"{n}#{k}")
            finally:
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class Recorder(FrameProcessor):
    """What passed the LLM service's gate, downstream, with when."""

    def __init__(self):
        super().__init__()
        self.seen: list[tuple[float, Frame]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.seen.append((time.monotonic(), frame))
        await self.push_frame(frame, direction)


class Run:
    """One run's record: the completions the LLM ran and the frames past its gate."""

    def __init__(self, strategies, llm: SlowLLM, recorder: Recorder):
        self.strategies = strategies
        self.runs = llm.runs
        self.seen = recorder.seen

    def heard(self) -> list[tuple[float, int, str]]:
        """The text frames past the gate: (when, the completion's number, the words it answers)."""
        out = []
        for t, f in self.seen:
            if isinstance(f, LLMTextFrame):
                n = int(f.text.split("#")[0])
                out.append((t, n, self.runs[n][1]))
        return out

    def turn_end(self) -> float:
        return next(t for t, f in self.seen if isinstance(f, UserStoppedSpeakingFrame))

    def first_reply_to(self, words: str) -> float | None:
        """Seconds from the turn's end to the first text frame past the gate of a reply to these words."""
        t = next((t for t, _, w in self.heard() if _MATCH.matches(w, words)), None)
        return None if t is None else t - self.turn_end()

    def describe(self) -> str:
        t0 = self.runs[0][0] if self.runs else 0.0
        runs = ", ".join(f"#{n} {w!r} early={e} at {t - t0:+.2f}s" for n, (t, w, e) in enumerate(self.runs))
        heard = ", ".join(f"#{n} at {t - self.turn_end():+.2f}s" for t, n, _ in self.heard())
        held = ""
        if hasattr(self.strategies, "bell"):
            held = f"; Flux's early ends of turn held back by Bell: {self.strategies.bell.stats.eager_held_back}"
        return f"completions: [{runs}]; text past the gate, from the turn's end: [{heard}]{held}"


async def through_llm(strategies, frames) -> Run:
    """The frames through Pipecat's user aggregator on these strategies, then the stand-in LLM, then the recorder."""
    context = LLMContext()
    user = LLMUserAggregator(context, params=LLMUserAggregatorParams(user_turn_strategies=strategies))
    llm = SlowLLM()
    recorder = Recorder()
    await run_test(Pipeline([user, llm, recorder]), frames_to_send=frames)
    return Run(strategies, llm, recorder)


async def bell_and_alone(guesses, frames, **standin) -> tuple[Run, Run]:
    async with StandIn(guesses=guesses, **standin) as api:
        with_bell = await through_llm(bell(api.url), frames)
    alone = await through_llm(EagerUserTurnStrategies(), frames)
    return with_bell, alone


# P1: Flux eager on. Bell guesses wrong on the interim "i need to"; Flux's eager end of turn then comes on other words.
P1_HEARD = "i need to"
P1_GUESS = "i need to reset my password"
P1_EAGER = "i need to check my balance"
P1_FINAL = "I need to check my balance."


def p1_only_flux_eager_reply_heard(with_bell: Run) -> str | None:
    """(c) The completions whose text passes the gate are exactly one: Flux's early reply. A reply Pipecat's eager
    strategy forgot would run again on the final words and pass the gate too."""
    passed = sorted({n for _, n, _ in with_bell.heard()})
    if len(passed) == 1:
        _, words, early = with_bell.runs[passed[0]]
        if early and _MATCH.matches(words, P1_EAGER):
            return None
    return f"(c) the completions whose text passed the gate were {passed}, not only Flux's early reply"


def p1_frames(eager_to_final: float):
    """Bell's reply B starts on the guess about 0.1 s before Flux's eager frame; the final words and the turn's end come
    ``eager_to_final`` seconds after the eager frame."""
    return [SleepFrame(0.3), start(), interim(P1_HEARD), SleepFrame(0.1), eager(P1_EAGER), SleepFrame(eager_to_final),
            final(P1_FINAL), stop(), SleepFrame(2.0)]


async def test_p1_bells_wrong_reply_is_not_heard_and_flux_eager_reply_is_not_late():
    # The turn ends while B is still generating (0.3 s after B starts; B lasts 0.6 s).
    with_bell, alone = await bell_and_alone({P1_HEARD: P1_GUESS}, p1_frames(eager_to_final=0.2))
    faults = []
    b_heard = [n for _, n, w in with_bell.heard() if _MATCH.matches(w, P1_GUESS)]
    if b_heard:
        faults.append(f"(a) the reply started on Bell's wrong guess passed the gate: {len(b_heard)} text frames")
    late_bell, late_alone = with_bell.first_reply_to(P1_FINAL), alone.first_reply_to(P1_FINAL)
    if late_bell is None or late_bell > late_alone + TOLERANCE:
        faults.append(
            f"(b) the reply to the real words started {late_bell} s after the turn's end with Bell, "
            f"{late_alone:.2f} s with Pipecat's eager strategy alone (tolerance {TOLERANCE} s)"
        )
    if fault := p1_only_flux_eager_reply_heard(with_bell):
        faults.append(fault)
    assert not faults, "; ".join(faults) + f". With Bell: {with_bell.describe()}. Alone: {alone.describe()}"


async def test_p1_flux_eager_reply_is_not_late_when_bells_reply_ended_before_the_turn():
    # The finding's second shape: B ends before the turn does (0.7 s after B starts), so E supersedes it at the gate,
    # but E waits in the LLM behind B.
    with_bell, alone = await bell_and_alone({P1_HEARD: P1_GUESS}, p1_frames(eager_to_final=0.6))
    faults = []
    b_heard = [n for _, n, w in with_bell.heard() if _MATCH.matches(w, P1_GUESS)]
    if b_heard:
        faults.append(f"(a) the reply started on Bell's wrong guess passed the gate: {len(b_heard)} text frames")
    late_bell, late_alone = with_bell.first_reply_to(P1_FINAL), alone.first_reply_to(P1_FINAL)
    if late_bell is None or late_bell > late_alone + TOLERANCE:
        faults.append(
            f"(b) the reply to the real words started {late_bell} s after the turn's end with Bell, "
            f"{late_alone:.2f} s with Pipecat's eager strategy alone (tolerance {TOLERANCE} s)"
        )
    if fault := p1_only_flux_eager_reply_heard(with_bell):
        faults.append(fault)
    assert not faults, "; ".join(faults) + f". With Bell: {with_bell.describe()}. Alone: {alone.describe()}"


# P2: Flux eager on and a transport VAD. Bell guesses right; the caller pauses past VAD's stop time and goes on.
def p2_frames():
    return [SleepFrame(0.3), start(), VADUserStartedSpeakingFrame(), interim(H1), interim(H2), SleepFrame(0.1),
            VADUserStoppedSpeakingFrame(stop_secs=0.2), SleepFrame(0.1), VADUserStartedSpeakingFrame(),
            interim(GUESS), eager(GUESS), SleepFrame(0.6), final(FINAL), stop(), SleepFrame(2.0)]


async def test_p2_after_a_vad_resume_flux_eager_reply_is_not_held_back():
    with_bell, alone = await bell_and_alone({H2: GUESS}, p2_frames())
    late_bell, late_alone = with_bell.first_reply_to(FINAL), alone.first_reply_to(FINAL)
    assert late_bell is not None and late_bell <= late_alone + TOLERANCE, (
        f"the reply to the final words started {late_bell} s after the turn's end with Bell, {late_alone:.2f} s with "
        f"Pipecat's eager strategy alone (tolerance {TOLERANCE} s). With Bell: {with_bell.describe()}. "
        f"Alone: {alone.describe()}"
    )


# P3 to P6, 28 November 2026 (ENG-62): the new Bell (version 3) can send a second guess in a turn when the words leave its
# first, and a guess equal to the words heard (its bet that the turn ends there). The same measure: with Bell, the reply
# to the caller's real words starts no later than with Pipecat's eager strategies alone, and no reply on a wrong guess
# passes the gate.
P3_H1, P3_G1 = "i need to", "i need to reset my password"
P3_H2, P3_G2 = "i need to check", "i need to check my balance"
P3_EAGER = "i need to check my balance"
P3_FINAL = "I need to check my balance."


def no_wrong_guess_heard(with_bell: Run, *wrong: str) -> list[str]:
    return [f"a reply started on Bell's wrong guess {w!r} passed the gate" for w in wrong
            if any(_MATCH.matches(words, w) for _, _, words in with_bell.heard())]


def not_later(with_bell: Run, alone: Run, final_words: str) -> list[str]:
    late_bell, late_alone = with_bell.first_reply_to(final_words), alone.first_reply_to(final_words)
    if late_bell is None or late_alone is None or late_bell > late_alone + TOLERANCE:
        return [f"the reply to the real words started {late_bell} s after the turn's end with Bell, {late_alone} s with "
                f"Pipecat's eager strategy alone (tolerance {TOLERANCE} s)"]
    return []


async def test_p3_a_second_guess_that_is_right_is_kept_and_no_reply_is_later():
    # Flux eager on. Bell's first guess is wrong; the words leave it and its second, right, starts before Flux's eager
    # end of turn on the same words, which is held back.
    frames = [SleepFrame(0.3), start(), interim(P3_H1), SleepFrame(0.1), interim(P3_H2), SleepFrame(0.1),
              interim(P3_EAGER), eager(P3_EAGER), SleepFrame(0.3), final(P3_FINAL), stop(), SleepFrame(2.0)]
    with_bell, alone = await bell_and_alone({P3_H1: P3_G1, P3_H2: P3_G2}, frames, guesses_a_turn=2)
    faults = no_wrong_guess_heard(with_bell, P3_G1) + not_later(with_bell, alone, P3_FINAL)
    assert with_bell.strategies.bell.stats.guesses_pushed == 2
    assert not faults, "; ".join(faults) + f". With Bell: {with_bell.describe()}. Alone: {alone.describe()}"


async def test_p4_two_wrong_guesses_then_flux_eager_reply_on_other_words_is_not_late():
    # Flux eager on. Both of Bell's guesses are wrong; Flux's eager end of turn comes on other words while the second
    # runs, and the turn ends while Flux's reply is under way.
    final_words = "I need to check my card."
    frames = [SleepFrame(0.3), start(), interim(P3_H1), SleepFrame(0.1), interim(P3_H2), SleepFrame(0.1),
              eager("i need to check my card"), SleepFrame(0.2), final(final_words), stop(), SleepFrame(2.0)]
    with_bell, alone = await bell_and_alone({P3_H1: P3_G1, P3_H2: P3_G2}, frames, guesses_a_turn=2)
    faults = no_wrong_guess_heard(with_bell, P3_G1, P3_G2) + not_later(with_bell, alone, final_words)
    assert with_bell.strategies.bell.stats.guesses_pushed == 2
    assert not faults, "; ".join(faults) + f". With Bell: {with_bell.describe()}. Alone: {alone.describe()}"


async def test_p5_bell_alone_two_wrong_guesses_the_last_under_way_at_the_turns_end_is_not_late():
    # The README's measured setup: Flux's eager end of turn off. Bell's second wrong guess is still generating when the
    # turn ends; its withdrawal must stop it at once, so the reply to the final words starts as it does without Bell.
    final_words = "I need to check my card."
    frames = [SleepFrame(0.3), start(), interim(P3_H1), SleepFrame(0.1), interim(P3_H2), SleepFrame(0.15),
              final(final_words), stop(), SleepFrame(2.0)]
    with_bell, alone = await bell_and_alone({P3_H1: P3_G1, P3_H2: P3_G2}, frames, guesses_a_turn=2)
    faults = no_wrong_guess_heard(with_bell, P3_G1, P3_G2) + not_later(with_bell, alone, final_words)
    assert with_bell.strategies.bell.stats.guesses_pushed == 2
    assert not faults, "; ".join(faults) + f". With Bell: {with_bell.describe()}. Alone: {alone.describe()}"


async def test_p6_bells_bet_that_the_turn_ends_on_the_words_heard_is_kept_and_not_late():
    # The new Bell's guess equal to the words heard, before Flux's eager end of turn on the same words.
    frames = [SleepFrame(0.3), start(), interim(P3_H1), SleepFrame(0.05), interim(P3_EAGER), SleepFrame(0.1),
              eager(P3_EAGER), SleepFrame(0.3), final(P3_FINAL), stop(), SleepFrame(2.0)]
    with_bell, alone = await bell_and_alone({P3_EAGER: P3_EAGER}, frames)
    faults = not_later(with_bell, alone, P3_FINAL)
    st = with_bell.strategies.bell.stats
    assert st.guesses_pushed == 1 and st.eager_held_back == 1
    assert not faults, "; ".join(faults) + f". With Bell: {with_bell.describe()}. Alone: {alone.describe()}"
