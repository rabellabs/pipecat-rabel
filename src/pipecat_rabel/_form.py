#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A guess written as this call's speech-to-text writes its words.

Bell's guesses come in the API's words: case-folded, with punctuation dropped, a mark inside a word dropped too
("e-mail" is "email", "3.5" is "35"), and in a script written without spaces (kana, ideographs, Thai) one character at a
time with a space between each. Pipecat's match policy (``NormalizedMatch`` by default) folds case and punctuation the
same way, but it splits a word at a mark inside it ("e-mail" is "e mail") and keeps the spaces. So a guess in the API's
words never matched a final with such a word or in such a script, and its early reply was always withdrawn.

The guess is therefore written from the speech-to-text's own text of the words heard, as it showed them, followed by the
words Bell adds, each as the speech-to-text last wrote it in a final transcript of this call. A word the call has not
shown in a final stays as Bell sent it. A word goes after a space unless the text before it or the word itself is in a
script written without spaces. Nothing about a language is assumed: the call's own transcripts are copied.
"""

import re
import unicodedata

# The API's word split (``bell_memory.norm_host``), which is LiveKit Agents' ``split_words(text,
# ignore_punctuation=True, split_character=True)``: words split on spaces, a character of a script written without
# spaces is a word on its own, punctuation is dropped, and each word is case-folded.
_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~±—‘’“”…")
_UNSPACED = re.compile(r"[぀-ヿ㐀-䶿一-鿿฀-๿]")


def _is_punctuation(c: str) -> bool:
    return c in _PUNCTUATION or unicodedata.category(c).startswith("P")


def words_of(text: str) -> tuple[str, ...]:
    """The words of ``text`` as the API counts them."""
    words: list[str] = []

    def add(a: int, b: int):
        w = "".join(c for c in text[a:b] if not _is_punctuation(c))
        if w:
            words.append(w.casefold())

    start = 0
    for pos, ch in enumerate(text):
        if ch.isspace():
            add(start, pos)
            start = pos + 1
        elif _UNSPACED.match(ch):
            if start < pos:
                add(start, pos)
            add(pos, pos + 1)
            start = pos + 1
    add(start, len(text))
    return tuple(words)


def _is_mark(c: str) -> bool:
    return c.isspace() or unicodedata.category(c).startswith("P")


class CallForm:
    """How this call's speech-to-text has written each word in its final transcripts."""

    def __init__(self):
        self._surface: dict[str, str] = {}

    def final(self, text: str):
        """A final transcript, as the speech-to-text wrote it. A word that opens it is not read: its capital may be
        the sentence's, not the word's."""
        for k, tok in enumerate(text.split()):
            w = words_of(tok)
            if len(w) != 1 or k == 0:
                continue
            a, b = 0, len(tok)
            while a < b and _is_mark(tok[a]):
                a += 1
            while b > a and _is_mark(tok[b - 1]):
                b -= 1
            self._surface[w[0]] = tok[a:b]

    def word(self, w: str) -> str:
        """The word as the call last wrote it, else as Bell sent it."""
        return self._surface.get(w, w)


def _unspaced(c: str) -> bool:
    """A character of a script written without spaces between words."""
    return bool(_UNSPACED.match(c))


def written(shown: str, added: tuple[str, ...], form: CallForm) -> str:
    """The guess: ``shown`` (the speech-to-text's text of the words heard, as it showed them), then the words Bell
    adds after them, each as this call's finals have written it, after a space unless the text before it or the word
    is in a script written without spaces."""
    out = shown.strip()
    for w in (form.word(w) for w in added):
        if out and not (_unspaced(out[-1]) or _unspaced(w[:1])):
            out += " "
        out += w
    return out
