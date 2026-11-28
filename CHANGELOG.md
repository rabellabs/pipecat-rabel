# Changelog

## 0.4.0 - 2026-11-28

- The new Bell. The hello says `"v": 3` and carries `min_chance` only when you set it. Its default is now None, the
  new Bell's balanced setting. An API without the new Bell answers with a version 2 `ready`, and the strategy works as
  0.3.0 did, with `min_chance` 0.7.
- `hold_always` and `hold_updates` are read from a version 3 `ready` only. `strategy.v3` says whether the open
  conversation has the new Bell, and `on_ready`'s dict carries the version (`v`). A version 3 guess with no `chance` is
  not used.
- The new Bell can send more than one guess in a turn. Each new guess withdraws the reply under way before its own
  starts, as before. The turn's report is on the guess started last, and `dropped_before` counts the guesses started
  before it. The summary counts each of them as dropped.
- A guess is written as the call's speech-to-text writes its words: the words heard as it showed them, then each word
  Bell adds as this call's finals last wrote it, with no space in a script written without spaces (Japanese, Chinese,
  Thai). Bell's guesses come in the API's words, and Pipecat's `NormalizedMatch` splits "E-Mail" into two words and
  keeps the spaces between characters, so a right guess with such a word, or in such a script, was always withdrawn.
  Where a script without spaces meets a spaced one (Chinese and a Latin word, or a Chinese name in an English turn), a
  space is written unless the call's finals write none there.
- A guess that comes while the speech-to-text's own early reply is under way no longer withdraws it. It waits, and
  starts if the caller's words reach it. Before, a guess made on earlier words that arrived just after Flux's eager end
  of turn on other words replaced Flux's reply. When the turn then ended on Flux's words, the reply started at the turn's
  end, later than without Bell (0.41 s against 0.09 s in the new test). 0.3.0 has the same fault.
- The speech-to-text's early end of turn on the words of Bell's reply under way is held back only when the match
  policy is sure to judge every final the same against both: always with Pipecat's own policies, and only for the same
  string with a developer's own policy, which need not be transitive. Before, with such a policy the reply could start
  at the turn's end, later than without Bell.
- The check at the turn's end joins the committed transcripts as Pipecat's eager strategy joins them.
- A turn of finals only goes to Bell, so Bell learns from a speech-to-text that sends no interims. It still guesses
  only on interims.
- A connection to the API that goes silent after `ready` is noticed within about 20 seconds and reopened: a ping every
  5 seconds, 15 seconds for its answer, 1 second for the close.
- An API that sends `ready` and then closes is reopened on the growing waits (1, 2, 4 seconds and on, up to 30), not
  every second for the whole call. The waits start again only after a conversation that stayed up 30 seconds.
- The later-reply check run again, with six new cases: a second guess right, two wrong beside Flux's eager end of
  turn, two wrong with Bell alone, a guess that the turn ends on the words heard, and a guess that comes after Flux's
  eager end of turn, and a developer's own match policy that is not transitive. With Bell, no reply to the caller's
  words starts later than with Pipecat's eager strategies alone. 137 tests, on Pipecat 1.16.0 and 1.17.0.
- Not taken from the LiveKit hookup's 1.9.0.1 and 1.9.0.2, because they are LiveKit's own: the count of LiveKit's early
  starts, no guess while the agent's reply is under way (Pipecat starts no early reply outside the caller's turn, and a
  guess never counts as the caller's words here), a final after the end of speech (Bell's turn here ends at the final,
  not at the end of speech), the wrapped stream's flush need, and `turn_detection="stt"`'s early start.

## 0.3.0 - 2026-11-23

- Bell is now a user turn stop strategy, on the door Pipecat's docs give a custom stop strategy for answering early:
  `RabelBellUserTurnStopStrategy`, installed with the bundle `RabelBellUserTurnStrategies` as Pipecat's
  `EagerUserTurnStopStrategy` is with `EagerUserTurnStrategies`. Pass the bundle as the user aggregator's
  `user_turn_strategies`; nothing goes in the pipeline.
- The early reply on Bell's guess is started with `start_user_turn_speculation()`, kept at the turn's end only when the
  final transcript matches it by the match policy, and withdrawn with `cancel_user_turn_speculation()` otherwise. The
  strategy pushes no frame and holds none back.
- `RabelBellProcessor` is removed (never uploaded). Its settings, events, stats, report and summary line are the
  strategy's, with the same names; the events' first argument is the strategy. New settings passed to Pipecat's eager
  strategy: `match_policy` and `speculation_timeout`.
- Flux's own eager end of turn keeps working beside Bell. Its early end of turn on the words of Bell's reply under way
  is not started (`stats.eager_held_back`).
- Flux's early end of turn on other words withdraws Bell's reply under way before its own starts. Bell's reply no longer
  holds the LLM ahead of it or reaches the caller when the turn ends.
- When a local VAD hears the caller go on after a pause, Pipecat voids the early reply under way, and Bell's no longer
  counts as running: Flux's early end of turn on the same words starts, and words that reach Bell's guess start it again.
- Each turn's end tells the API whether Bell's guess was kept or dropped (`report`, on by default), with no words. One
  DEBUG line for each guess kept and one INFO line with the call's totals. `user` is sent when set.
- A guess never started in its turn (it came before Pipecat's turn opened, or beside an early reply on the same words,
  and the words never reached it) counts in `stats.guesses_dropped`, and the turn's report says dropped.
- The conversation opens when the strategy is set up and is closed at the pipeline's teardown, also when no end frame
  arrived. The API's address can be set with `RABEL_API_URL`. `websockets>=15`.
- Tests rewritten for the strategy in Pipecat's user aggregator, with eight kinds of API failure, each failing open.
  Three more run Bell beside Flux's eager end of turn through a local stand-in LLM service and Pipecat's speculation
  gate, and check what passes the gate against Pipecat's eager strategies alone.

## 0.2.0 - 2026-11-19

- An installable package, `pipecat-rabel`, for Pipecat 1.16.0: `from pipecat_rabel import RabelBellProcessor`.
- Version 2 of Rabel's API: the hello carries `agent`, `min_chance` and `learn`, the key goes in the `Authorization`
  header, and each guess carries its `chance`. Bell now learns each agent on its own.
- Fails open: with the API down, slow to connect, slow to answer, refusing, closing mid-call or sending what the
  processor cannot read, the frames out are exactly the speech-to-text's, with one logged line for each kind of failure.
  A connect timeout, an answer timeout, and reopening by the API's rules.
- The words go out from a queue, so the pipeline never waits on the network.
- Tracks the speech-to-text's own early start (Flux's eager end of turn), so Bell's guess and Flux's prediction share
  Pipecat's one early reply.
- Event handlers `on_ready`, `on_guess` and `on_turn_end`, and `processor.stats`.
- `examples/bank_call.py`, a bank's phone agent on `demo-bank`, replaces `compare.py`.
- Tests against a local stand-in for the API, and a nightly run against Pipecat's main branch.
- While Bell's early start is live, the speech-to-text's own early start on the same words is held back
  (`stats.eager_held_back`): Pipecat would otherwise start a second reply and drop the one under way on Bell's guess.
- `min_chance` is 0.67 to 0.99, the API's range; a value outside it is clamped.

## 0.1.0 - 2026-11-14

- `RabelBellProcessor` and `compare.py`, tested with Pipecat 1.15.0.
