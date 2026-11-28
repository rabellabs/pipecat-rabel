#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A guess written as the call's speech-to-text writes its words, so Pipecat's match policy can keep it.

Bell's guesses come in the API's words (case-folded, punctuation dropped, a mark inside a word dropped too). Pipecat's
``NormalizedMatch`` splits a word at a mark inside it, so before 0.4.0 a guess with such a word never matched the final.
The tests run Pipecat's own user aggregator against the local stand-in, in a language other than English.
"""

from pipecat.frames.frames import UserTurnResponseCancelFrame
from pipecat.tests.utils import SleepFrame
from pipecat.turns.user_stop import NormalizedMatch

from pipecat_rabel._form import CallForm, words_of, written

from .common import bell, ends, final, interim, replies, run, start, stop
from .standin import StandIn

DE_TEACH_HEARD = "ich habe eine neue"
DE_TEACH_FINAL = "Ich habe eine neue E-Mail-Adresse."
DE_HEARD = "ich möchte meine"
DE_GUESS = "ich möchte meine emailadresse ändern"  # the API's words
DE_FINAL = "Ich möchte meine E-Mail-Adresse ändern."


def test_words_of_counts_words_as_the_api_does():
    assert words_of("I'd like the E-Mail, please.") == ("id", "like", "the", "email", "please")
    assert words_of(DE_FINAL) == tuple(DE_GUESS.split())
    assert not NormalizedMatch().matches(DE_GUESS, DE_FINAL)  # the API's words do not match the final as they are


def test_a_word_is_written_as_the_calls_finals_last_wrote_it():
    form = CallForm()
    assert written(DE_HEARD, ("emailadresse", "ändern"), form) == DE_GUESS  # nothing shown yet: as Bell sent it
    form.final(DE_TEACH_FINAL)
    assert written(DE_HEARD, ("emailadresse", "ändern"), form) == "ich möchte meine E-Mail-Adresse ändern"
    # the opening word of a final is not read: its capital may be the sentence's
    form.final("Konto gesperrt.")
    assert written("mein", ("konto",), form) == "mein konto"


async def test_a_guess_with_a_word_the_call_wrote_with_a_mark_is_kept():
    frames = [SleepFrame(0.3), start(), interim(DE_TEACH_HEARD), SleepFrame(0.1), final(DE_TEACH_FINAL), stop(),
              SleepFrame(0.1), start(), interim(DE_HEARD), SleepFrame(0.2), final(DE_FINAL), stop(), SleepFrame(0.1)]
    async with StandIn(guesses={DE_HEARD: DE_GUESS}) as api:
        s = bell(api.url)
        down, _, context = await run(s, frames)
    # the reply started on the guess as the call writes it, and kept: no withdrawal, no second reply (the first turn's
    # reply carries Pipecat's own context, which by the end holds both turns, so only its kind is read)
    got = replies(down)
    assert [early for early, _ in got] == [False, True]
    assert got[1] == (True, "ich möchte meine E-Mail-Adresse ändern")
    assert not any(isinstance(f, UserTurnResponseCancelFrame) for f in down)
    assert [m["content"] for m in context.messages] == [DE_TEACH_FINAL, DE_FINAL]
    assert [e["guess"] for e in ends(api)] == ["none", "kept"]
    assert s.bell.stats.guesses_right == 1
