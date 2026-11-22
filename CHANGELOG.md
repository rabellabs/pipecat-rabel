# Changelog

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
