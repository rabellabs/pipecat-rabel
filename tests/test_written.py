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


# ---------------------------------------------------------------------------------- scripts written without spaces

JA_HEARD = "パスワードを"
JA_GUESS = "パ ス ワ ー ド を リ セ ッ ト し た い"  # the API's words: one character at a time
JA_FINAL = "パスワードをリセットしたい。"


def test_no_space_is_added_in_a_script_written_without_spaces():
    form = CallForm()
    assert words_of(JA_FINAL) == tuple(JA_GUESS.split())
    assert written(JA_HEARD, tuple("リセットしたい"), form) == "パスワードをリセットしたい"
    # a word in a script with spaces beside one without: no space on either side of the unspaced script
    assert written("暗証番号を", ("pin", "に", "変", "更"), form) == "暗証番号をpinに変更"
    assert written("my", ("pin", "is"), form) == "my pin is"
    assert written("รหัส", tuple("ผ่าน"), form) == "รหัสผ่าน"


async def test_a_right_guess_in_japanese_is_kept():
    frames = [SleepFrame(0.3), start(), interim(JA_HEARD), SleepFrame(0.2), final(JA_FINAL), stop(), SleepFrame(0.1)]
    async with StandIn(guesses={JA_HEARD: JA_GUESS}) as api:
        s = bell(api.url)
        down, _, context = await run(s, frames)
    assert replies(down) == [(True, "パスワードをリセットしたい")]
    assert not any(isinstance(f, UserTurnResponseCancelFrame) for f in down)
    assert context.messages == [{"role": "user", "content": JA_FINAL}]
    assert ends(api)[0]["guess"] == "kept"


async def test_finals_that_carry_their_own_spaces_are_joined_as_pipecat_joins_them():
    # A speech-to-text that marks its finals as carrying their own spaces: Pipecat's eager strategy joins them with none
    # added, and Bell's check joins them the same way, so the reply on the guess is kept as Pipecat would keep it.
    parts = []
    for text in ("パスワードを", "リセットしたい。"):
        f = final(text)
        f.includes_inter_frame_spaces = True
        parts.append(f)
    frames = [SleepFrame(0.3), start(), interim(JA_HEARD), SleepFrame(0.2), parts[0], parts[1], stop(),
              SleepFrame(0.1)]
    async with StandIn(guesses={JA_HEARD: JA_GUESS}) as api:
        s = bell(api.url)
        down, _, _ = await run(s, frames)
    assert replies(down) == [(True, "パスワードをリセットしたい")]
    assert not any(isinstance(f, UserTurnResponseCancelFrame) for f in down)
