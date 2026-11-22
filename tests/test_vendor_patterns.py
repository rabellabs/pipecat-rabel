#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The strategy on three ways a speech-to-text sends words, against the local stand-in: no key and no network.

Each test sends, with made-up words, the frames a Pipecat speech-to-text service pushes in that pattern:

- Deepgram Flux (``DeepgramFluxSTTService``, eager end of turn on): StartOfTurn a ``ProposedUserStartedSpeakingFrame``;
  each Update an ``InterimTranscriptionFrame`` of the turn so far, word by word; EagerEndOfTurn an
  ``EagerTranscriptionFrame``; TurnResumed a ``UserTurnResponseCancelFrame``; EndOfTurn a ``TranscriptionFrame`` and a
  ``ProposedUserStoppedSpeakingFrame``.
- Part words: some speech-to-text services show a word while it is still being said ("wal" before "wallet"), each
  ``InterimTranscriptionFrame`` carrying the words so far; no early end of turn of their own.
- The last words in one burst: some services hold a turn's last words and show them in quick succession, faster than
  Bell's answer; with their own eager end of turn on, an ``EagerTranscriptionFrame``, a ``UserTurnResponseCancelFrame``
  when the turn resumes, and the final at the end.
"""

from pipecat.frames.frames import UserTurnResponseCancelFrame
from pipecat.tests.utils import SleepFrame

from .common import bell, eager, final, interim, pipecat_alone, replies, run, shape, start, stop
from .standin import StandIn

# ------------------------------------------------------------------------------------------------------------ Flux


async def test_flux_steady_updates_and_its_eager_end_of_turn_on_bells_words():
    # Bell's guess is under way. Flux's updates reach it word by word: the guess is not started again when they do.
    # Flux's EagerEndOfTurn on the same words is not started, so Pipecat keeps the reply already under way on Bell's.
    guess = "please send me a new card"
    async with StandIn(guesses={"please send": guess}) as api:
        s = bell(api.url)
        frames = [SleepFrame(0.3), start(), interim("please send"), SleepFrame(0.1), interim("please send me"),
                  interim("please send me a"), interim("please send me a new card"), eager("Please send me a new card."),
                  SleepFrame(0.05), final("Please send me a new card."), stop(), SleepFrame(0.1)]
        down, up, context = await run(s, frames)
    assert replies(down) == [(True, guess)]
    assert not any(isinstance(f, UserTurnResponseCancelFrame) for f in down + up)
    assert context.messages == [{"role": "user", "content": "Please send me a new card."}]
    st = s.bell.stats
    assert st.guesses_pushed == 1 and st.eager_held_back == 1 and st.guesses_right == 1


async def test_flux_turn_resumed_then_the_words_reach_bells_guess():
    # Flux's EagerEndOfTurn on fewer words starts and takes Pipecat's one early reply; TurnResumed withdraws it; when the
    # caller's words then reach Bell's guess, the guess is started again.
    guess = "can you freeze my card"
    async with StandIn(guesses={"can you freeze": guess}) as api:
        s = bell(api.url)
        frames = [SleepFrame(0.3), start(), interim("can you"), interim("can you freeze"), SleepFrame(0.1),
                  eager("Can you freeze?"), SleepFrame(0.05), UserTurnResponseCancelFrame(), interim("can you freeze my"),
                  interim("can you freeze my card"), SleepFrame(0.05), final("Can you freeze my card?"), stop(),
                  SleepFrame(0.1)]
        down, _, context = await run(s, frames)
    assert replies(down) == [(True, guess), (True, "Can you freeze?"), (True, guess)]
    assert context.messages == [{"role": "user", "content": "Can you freeze my card?"}]
    assert s.bell.stats.guesses_pushed == 2 and s.bell.stats.guesses_right == 1


# -------------------------------------------------------------------------------------------------------- part words


async def test_part_words_pass_and_bells_guess_stays():
    # The service shows a word while it is said ("wal" before "wallet"). Every interim goes to Bell as heard; none
    # withdraws Bell's reply under way, and the words reaching it do not start it again.
    guess = "i lost my wallet"
    async with StandIn(guesses={"I lost": guess}) as api:
        s = bell(api.url)
        frames = [SleepFrame(0.3), start(), interim("I lost"), SleepFrame(0.1), interim("I lost my"),
                  interim("I lost my wal"), interim("I lost my wallet"), SleepFrame(0.05), final("I lost my wallet."),
                  stop(), SleepFrame(0.1)]
        down, _, context = await run(s, frames)
    assert replies(down) == [(True, guess)]
    assert context.messages == [{"role": "user", "content": "I lost my wallet."}]
    assert [m["text"] for m in api.received[0] if m["type"] == "heard"] == [
        "I lost", "I lost my", "I lost my wal", "I lost my wallet"]
    assert s.bell.stats.guesses_pushed == 1 and s.bell.stats.guesses_right == 1


# -------------------------------------------------------------------------------------------- the last words in a burst

BURST_GUESS = {"I'd like": "i'd like to close my account"}


def last_words_burst():
    # the turn's last words held and shown in one burst, faster than Bell's answer
    return [SleepFrame(0.3), start(), interim("I'd like"), SleepFrame(0.02), interim("I'd like to close"),
            SleepFrame(0.02), interim("I'd like to close my account.")]


async def test_last_words_burst_reaches_bells_guess_before_it_arrives():
    # Bell alone (the service's eager end of turn off): the burst shows the guessed words before Bell's answer arrives.
    # The reply is started on the guess, not dropped as equal to the words so far.
    async with StandIn(guesses=BURST_GUESS, guess_delay=0.1) as api:
        s = bell(api.url)
        frames = last_words_burst() + [SleepFrame(0.2), final("I'd like to close my account."), stop(), SleepFrame(0.1)]
        down, _, _ = await run(s, frames)
    assert replies(down) == [(True, "i'd like to close my account")]
    assert s.bell.stats.guesses_pushed == 1 and s.bell.stats.guesses_right == 1


async def test_burst_with_its_own_early_end_of_turn_then_the_turn_resumes():
    # The service's eager end of turn on: its early reply comes with the burst. Bell's guess, arriving after, has the
    # words so far and that early reply is on them: it is not started. The resume and the words after are Pipecat's own.
    async with StandIn(guesses=BURST_GUESS, guess_delay=0.1) as api:
        s = bell(api.url)
        frames = last_words_burst() + [
            eager("I'd like to close my account."), SleepFrame(0.2), UserTurnResponseCancelFrame(),
            interim("I'd like to close my account today."), eager("I'd like to close my account today."),
            SleepFrame(0.05), final("I'd like to close my account today."), stop(), SleepFrame(0.1)]
        down, up, _ = await run(s, frames)
    alone_down, alone_up, _ = await pipecat_alone(frames)
    assert shape(down) == shape(alone_down) and shape(up) == shape(alone_up)
    assert s.bell.stats.guesses_received == 1 and s.bell.stats.guesses_pushed == 0


async def test_the_turn_resuming_withdraws_bells_reply():
    # Bell's reply is under way before the burst; the service's early end of turn on its words is not started. The turn
    # resuming withdraws the early reply, Bell's with it, so the service's next early end of turn on the same words is
    # the only one and starts.
    guess = "can you tell me my balance"
    async with StandIn(guesses={"Can you": guess}) as api:
        s = bell(api.url)
        frames = [SleepFrame(0.3), start(), interim("Can you"), SleepFrame(0.1), interim("Can you tell me"),
                  interim("Can you tell me my balance"), eager("Can you tell me my balance?"), SleepFrame(0.05),
                  UserTurnResponseCancelFrame(), eager("Can you tell me my balance?"), SleepFrame(0.05),
                  final("Can you tell me my balance?"), stop(), SleepFrame(0.1)]
        down, _, context = await run(s, frames)
    assert replies(down) == [(True, guess), (True, "Can you tell me my balance?")]
    assert context.messages == [{"role": "user", "content": "Can you tell me my balance?"}]
    assert s.bell.stats.eager_held_back == 1 and s.bell.stats.guesses_pushed == 1
