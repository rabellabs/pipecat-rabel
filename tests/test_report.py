#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The report on each turn's end, the kept line, the call's summary and ``user``, against the local stand-in.

"Kept" follows Pipecat's own decision on whether it uses the reply started on Bell's guess: Bell's early reply is still the
one under way when the final words arrive, and the match policy (``NormalizedMatch`` by default) finds the final words
equal to it. Every test runs Pipecat's real user aggregator, and the last ones read its decision from the frames it
sends, so the report is checked against Pipecat itself, not against this file's idea of it.
"""

import pytest
from loguru import logger
from pipecat.frames.frames import LLMContextFrame, UserTurnResponseCancelFrame
from pipecat.tests.utils import SleepFrame
from pipecat.turns.user_stop import NormalizedMatch

from pipecat_rabel import RabelBellUserTurnStopStrategy, RabelBellUserTurnStrategies

from .common import FINAL, GUESS, H1, H2, bell, eager, ends, final, interim, one_turn, replies, run, start, stop
from .standin import StandIn

PAY_HEARD = "i would like to pay"
PAY_GUESS = "i would like to pay my bill"
PAY_FINAL = "I would like to pay my card."


def pay_turn():
    return [start(), interim(PAY_HEARD), SleepFrame(0.2), final(PAY_FINAL), stop(), SleepFrame(0.1)]


async def report_run(frames, guesses=None, **kwargs):
    async with StandIn(guesses={H2: GUESS} if guesses is None else guesses) as api:
        s = bell(api.url, **kwargs)
        down, up, context = await run(s, frames)
    return s.bell, api, down, up


@pytest.fixture
def lines():
    """Every line the strategy logs during a test, with its level and its bound fields."""
    got = []
    sink = logger.add(
        lambda m: got.append((m.record["level"].name, m.record["message"], dict(m.record["extra"]))),
        level="DEBUG",
        filter=lambda r: "RabelBell" in r["message"],
    )
    yield got
    logger.remove(sink)


# ------------------------------------------------------------------------------------------------- kept, dropped, none


async def test_report_kept_with_its_head_start():
    b, api, down, _ = await report_run(one_turn())
    assert replies(down) == [(True, GUESS)]
    (end,) = ends(api)
    assert end["guess"] == "kept" and end["text"] == FINAL
    # from starting the reply on Bell's guess to the final words reaching the strategy: the 0.2 s of the turn between
    assert 150 <= end["head_start_ms"] <= 600, end
    assert b.summary == {"turns": 1, "guesses": 1, "kept": 1, "dropped": 0, "head_start_ms": end["head_start_ms"]}


@pytest.mark.parametrize(
    "final_text",
    [
        "I would like to reset my password.",  # capitals and a full stop: kept
        "i would like to reset my password",
        "I WOULD LIKE TO RESET MY PASSWORD!",
        "I would like to reset my password?",
        "I would like, to reset my password.",
        "I would like to reset my pin.",  # other words: withdrawn
        "I would like to reset my password now.",
        "I would like to reset password.",
    ],
)
async def test_report_follows_pipecats_own_check(final_text):
    # The reply is kept when Pipecat's match policy (NormalizedMatch by default) finds the final words equal to the
    # guess; the report says kept exactly then, by Pipecat's own class
    _, api, down, _ = await report_run(one_turn(final_text))
    (end,) = ends(api)
    want = "kept" if NormalizedMatch().matches(GUESS, final_text) else "dropped"
    assert end["guess"] == want, (final_text, end)
    assert ("head_start_ms" in end) is (want == "kept")
    assert any(isinstance(f, UserTurnResponseCancelFrame) for f in down) is (want == "dropped")


async def test_report_dropped_when_the_speech_to_text_withdraws_bells_early_reply():
    # Flux's TurnResumed withdraws whatever early reply is under way, Bell's with it; the final then equals the guess, but
    # no early reply is left for Pipecat to keep
    frames = one_turn()
    frames.insert(5, UserTurnResponseCancelFrame())
    b, api, down, _ = await report_run(frames)
    assert [e["guess"] for e in ends(api)] == ["dropped"]
    assert b.summary["dropped"] == 1 and b.summary["kept"] == 0 and b.summary["head_start_ms"] == 0
    assert [r[0] for r in replies(down)] == [True, False]


async def test_report_dropped_when_the_speech_to_texts_early_reply_replaces_bells():
    # Bell's withdrawn, then Flux's own on the same words starts and is the one Pipecat keeps: not Bell's
    frames = one_turn()
    frames[5:5] = [UserTurnResponseCancelFrame(), eager("I would like to reset my password.")]
    _, api, down, _ = await report_run(frames)
    assert replies(down) == [(True, GUESS), (True, "I would like to reset my password.")]
    assert [e["guess"] for e in ends(api)] == ["dropped"]


async def test_report_dropped_when_the_speech_to_texts_early_end_of_turn_on_other_words_follows():
    # Flux's early end of turn on other words starts and replaces Bell's; Pipecat checks the final against it
    frames = one_turn()
    frames.insert(5, eager("I would like to reset my pin."))
    _, api, _, _ = await report_run(frames)
    assert [e["guess"] for e in ends(api)] == ["dropped"]


async def test_report_kept_when_the_speech_to_texts_same_early_end_of_turn_is_not_started():
    # Flux's early end of turn on Bell's words is not started: Bell's stays and is kept
    frames = one_turn()
    frames.insert(5, eager("I would like to reset my password."))
    b, api, _, _ = await report_run(frames)
    assert [e["guess"] for e in ends(api)] == ["kept"]
    assert b.stats.eager_held_back == 1


@pytest.mark.parametrize(
    "case,kwargs,api_kw",
    [
        ("no guess", {}, {"guesses": {}}),
        ("a late guess, never started", {"answer_timeout": 0.05}, {"guesses": {H2: GUESS}, "guess_delay": 0.15}),
        ("a guess under min_chance, never started", {"min_chance": 0.9}, {"guesses": {H2: GUESS}}),
    ],
)
async def test_report_none(case, kwargs, api_kw):
    async with StandIn(**api_kw) as api:
        s = bell(api.url, **kwargs)
        await run(s, one_turn())
    assert ends(api) == [{"type": "end_of_turn", "text": FINAL, "guess": "none"}], case
    assert s.bell.summary == {"turns": 1, "guesses": 0, "kept": 0, "dropped": 0, "head_start_ms": 0}


async def test_report_counts_each_turn_on_its_own():
    _, api, _, _ = await report_run(one_turn() + pay_turn(), guesses={H2: GUESS, PAY_HEARD: PAY_GUESS})
    assert [e["guess"] for e in ends(api)] == ["kept", "dropped"]
    assert "head_start_ms" not in ends(api)[1]


# ------------------------------------------------------------------------------------------------- a guess never started


async def test_a_guess_before_pipecats_turn_opens_that_the_words_never_reach_is_dropped():
    # Flux sends worded Updates before its StartOfTurn, so Bell's guess can come while Pipecat's turn is closed. Nothing
    # is started then; when the words never reach the guess inside the turn, it is a guess not used: dropped, in the
    # stats and in the turn's report
    frames = [SleepFrame(0.3), interim(H1), interim(H2), SleepFrame(0.2), start(), final(FINAL), stop(), SleepFrame(0.1)]
    b, api, down, _ = await report_run(frames)
    assert replies(down) == [(False, FINAL)]
    assert not any(isinstance(f, UserTurnResponseCancelFrame) for f in down)
    assert (b.stats.guesses_received, b.stats.guesses_pushed, b.stats.guesses_dropped) == (1, 0, 1)
    assert [e["guess"] for e in ends(api)] == ["dropped"]
    assert b.summary == {"turns": 1, "guesses": 1, "kept": 0, "dropped": 1, "head_start_ms": 0}


async def test_a_guess_before_pipecats_turn_opens_that_the_words_reach_is_started_and_not_dropped():
    frames = [SleepFrame(0.3), interim(H1), interim(H2), SleepFrame(0.2), start(), interim(GUESS), SleepFrame(0.2),
              final(FINAL), stop(), SleepFrame(0.1)]
    b, api, down, _ = await report_run(frames)
    assert replies(down) == [(True, GUESS)]
    assert (b.stats.guesses_received, b.stats.guesses_pushed, b.stats.guesses_dropped) == (1, 1, 0)
    assert [e["guess"] for e in ends(api)] == ["kept"]


async def test_a_guess_not_started_beside_the_speech_to_texts_reply_on_the_same_words_is_dropped():
    # The speech-to-text's early reply on the words heard so far stays, so Bell's guess on those words is not started;
    # the turn ends before the words reach it: a guess not used
    frames = [SleepFrame(0.3), start(), interim(H2), eager(H2), SleepFrame(0.2), final(FINAL), stop(), SleepFrame(0.1)]
    b, api, down, _ = await report_run(frames)
    assert (b.stats.guesses_received, b.stats.guesses_pushed, b.stats.guesses_dropped) == (1, 0, 1)
    assert [e["guess"] for e in ends(api)] == ["dropped"]
    assert [r[0] for r in replies(down)] == [True, False]


# ------------------------------------------------------------------------------------------------- the flag and the lines


async def test_report_on_a_turn_with_two_guesses_started_counts_the_first_dropped_before():
    # The new Bell guesses on "i would like", then again on "i would like to reset" when the words leave its first guess.
    guesses = {H1: PAY_HEARD, H2: GUESS}
    frames = [SleepFrame(0.3), start(), interim(H1), SleepFrame(0.1), interim(H2), SleepFrame(0.2), final(FINAL),
              stop(), SleepFrame(0.1)]
    for version, expected in ((None, {"dropped_before": 1}), (2, {})):
        async with StandIn(guesses=guesses, guesses_a_turn=2, version=version) as api:
            s = bell(api.url)
            down, _, _ = await run(s, frames)
        assert replies(down) == [(True, PAY_HEARD), (True, GUESS)]
        (end,) = ends(api)
        end.pop("head_start_ms")
        # dropped_before is the new Bell's field; a version 2 API is sent the report as before
        assert end == {"type": "end_of_turn", "text": FINAL, "guess": "kept", **expected}
        assert s.bell.summary == {**s.bell.summary, "turns": 1, "guesses": 2, "kept": 1, "dropped": 1}


async def test_report_off_leaves_the_field_out():
    b_off, api_off, down_off, _ = await report_run(one_turn(), report=False)
    assert ends(api_off) == [{"type": "end_of_turn", "text": FINAL}]
    # the turn itself is the same with the report on or off: the same replies, the same counts
    b_on, _, down_on, _ = await report_run(one_turn())
    assert replies(down_off) == replies(down_on)
    assert b_off.stats == b_on.stats
    for bad in ("no", None, 0, 1):
        with pytest.raises(ValueError):
            RabelBellUserTurnStopStrategy(api_key="k", report=bad)


async def test_log_lines_kept_and_summary_carry_no_words(lines):
    b, api, _, _ = await report_run(one_turn() + pay_turn(), guesses={H2: GUESS, PAY_HEARD: PAY_GUESS})
    hs = ends(api)[0]["head_start_ms"]
    # one line for the kept guess, at DEBUG as Pipecat's own "eager end of turn held, keeping the speculative response"
    kept = [x for x in lines if "Bell's guess held, keeping the speculative response" in x[1]]
    assert len(kept) == 1 and kept[0][0] == "DEBUG" and kept[0][2] == {"head_start_ms": hs}, kept
    assert kept[0][1].endswith(f"({hs} ms head start)")
    # one summary when the call ends, at INFO
    summary = [x for x in lines if "head start in all" in x[1]]
    assert len(summary) == 1 and summary[0][0] == "INFO", summary
    assert summary[0][1].endswith(f": Bell: 2 turns, 2 guesses, 1 kept, 1 dropped, {hs} ms head start in all")
    assert summary[0][2] == {"turns": 2, "guesses": 2, "kept": 1, "dropped": 1, "head_start_ms_total": hs}
    # no word of the call in any line of the strategy's: not the words heard, the finals, nor Bell's guesses
    said = set()
    for text in ("i would like", H2, FINAL, GUESS, PAY_HEARD, PAY_GUESS, PAY_FINAL):
        said |= set(text.lower().strip(".").split())
    for level, message, extra in lines:
        words = set(message.lower().replace(":", " ").replace(",", " ").replace("(", " ").replace(")", " ").split())
        assert not words & said, message
        assert all(not isinstance(v, str) for v in extra.values()), extra


async def test_no_key_logs_no_summary(lines, monkeypatch):
    monkeypatch.delenv("RABEL_API_KEY", raising=False)
    s = RabelBellUserTurnStrategies(url="ws://127.0.0.1:9/v1/conversation")
    await run(s, one_turn())
    assert not any("head start in all" in x[1] for x in lines)


async def test_an_api_that_refuses_the_report_fails_open(lines):
    # no API refuses it (one that does not know the field ignores it); if one did, it is a 400, and the strategy goes on
    # without Bell for the rest of the call, as after any 400: the later turns are Pipecat's own
    async with StandIn(guesses={H2: GUESS, PAY_HEARD: PAY_GUESS}, refuse_report=True) as api:
        s = bell(api.url)
        down, _, _ = await run(s, one_turn() + [SleepFrame(0.2)] + pay_turn())
    assert replies(down) == [(True, GUESS), (False, PAY_FINAL)]  # the first turn's, before the refusal
    assert api.connections == 1  # a 400 opens no other conversation
    warnings = [x for x in lines if x[0] == "WARNING"]
    assert len(warnings) == 1 and "400 bad_message" in warnings[0][1], warnings


# ------------------------------------------------------------------------------------------------- user


async def test_user_sent_in_the_hello_when_set():
    async with StandIn() as api:
        await run(bell(api.url, user="caller-0042:eu.1"), one_turn())
        await run(bell(api.url), one_turn())
    assert api.hellos[0]["user"] == "caller-0042:eu.1"
    assert "user" not in api.hellos[1]


@pytest.mark.parametrize("bad", ["Mara Quill", "mara.quill@example.com", "", "x" * 129, "caller/1", 12345, b"caller"])
async def test_user_form_checked_and_not_repeated(bad):
    with pytest.raises(ValueError) as e:
        RabelBellUserTurnStopStrategy(api_key="k", user=bad)
    assert repr(bad) not in str(e.value) and (not isinstance(bad, str) or not bad or bad not in str(e.value))


async def test_user_not_in_the_bundles_repr():
    s = RabelBellUserTurnStrategies(api_key="k", user="caller-0042")
    assert "caller-0042" not in repr(s)


# ------------------------------------------------------------------------------------------------- against Pipecat itself


def turn(final_text):
    return [SleepFrame(0.3), start(), interim("i would like"), interim(H2), SleepFrame(0.2), final(final_text),
            SleepFrame(0.1), stop(), SleepFrame(1.0)]


async def run_with_pipecat(frames, guesses=None, **kwargs):
    """Pipecat's decision read from its frames: kept when its last answer was the early one on Bell's guess."""
    async with StandIn(guesses={H2: GUESS} if guesses is None else guesses) as api:
        s = RabelBellUserTurnStrategies(api_key="test-key", url=api.url, connect_timeout=0.5, **kwargs)
        down, _, _ = await run(s, frames)
    contexts = [f for f in down if isinstance(f, LLMContextFrame)]
    pipecat_kept_bells = (
        bool(contexts)
        and contexts[-1].speculation
        and contexts[-1].context.messages[-1] == {"role": "user", "content": GUESS}
    )
    return s.bell, api, down, pipecat_kept_bells


@pytest.mark.parametrize(
    "final_text",
    [FINAL, "i would like to reset my password", "I would like to reset my password?", "I would like to reset my pin."],
)
async def test_report_agrees_with_pipecats_own_decision(final_text):
    _, api, down, pipecat_kept_bells = await run_with_pipecat(turn(final_text))
    (end,) = ends(api)
    assert end["guess"] == ("kept" if pipecat_kept_bells else "dropped"), (final_text, end)
    assert pipecat_kept_bells is NormalizedMatch().matches(GUESS, final_text)
    assert any(isinstance(f, UserTurnResponseCancelFrame) for f in down) is (not pipecat_kept_bells)


async def test_report_agrees_with_pipecat_when_bells_reply_times_out():
    # Bell's early reply is withdrawn unresolved (speculation_timeout); Flux's early end of turn on the same words then
    # starts the reply Pipecat keeps: Bell's guess is dropped although the final equals it
    frames = turn(FINAL)
    frames[5:5] = [SleepFrame(0.8), eager("I would like to reset my password.")]
    _, api, down, pipecat_kept_bells = await run_with_pipecat(frames, speculation_timeout=0.5)
    assert [c[0] for c in replies(down)] == [True, True] and not pipecat_kept_bells
    assert [e["guess"] for e in ends(api)] == ["dropped"]
