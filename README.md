# Bell for Pipecat

Starts your [Pipecat](https://github.com/pipecat-ai/pipecat) voice agent's reply on Bell's guess at the caller's words,
before the caller finishes.

Bell is a hosted model from Rabel Labs that learns each of your agents' callers from that agent's own calls. It is in
beta. It works best on agents whose callers come for the same few things, like a bank's phone line or a booking desk.
Where callers talk freely, Bell has little to guess and mostly stays quiet, and the call runs as it would without it.

As your speech-to-text hears a turn, Bell may guess the whole turn.
`RabelBellUserTurnStopStrategy`, a user turn stop strategy, starts Pipecat's early reply on that guess. At the end of the
turn Pipecat keeps the reply only if the final words match the guess; otherwise it discards the reply unheard and answers
the final words. The package adds nothing to your pipeline, sends no frame and replaces no service.

Bell runs on Rabel's servers: there is nothing to host and no GPU. The API's docs:
[rabellabs.com/docs](https://rabellabs.com/docs). Tested with Pipecat 1.16.0. The numbers here come from recorded,
scripted calls to one bank, in English, on Deepgram Flux's words; the rest is under [Limits](#limits).

**What Rabel keeps.** Bell never receives audio. It takes nothing but the words from your
speech-to-text and holds a call's words only while that call is open. When the call closes, Bell learns from them for
your agent and drops them. What it learns stays with your agent and is never filed under a call or under a caller's name
or number. Part of it is kept as codes that Rabel holds the key to, and the rest is short phrases kept in words. A phrase
can be a name or a number your callers say. You can list your agent's phrases at any time and delete any of them.
Deleting an agent erases what Bell learned for it. Rabel's notice says it in full:
[rabellabs.com/api-privacy](https://rabellabs.com/api-privacy).

This is a community integration, built and maintained by Rabel Labs, which runs Bell. Rabel Labs is not affiliated with,
endorsed by or sponsored by LiveKit, Pipecat (Daily) or Deepgram; they are named only to say what the package works with.

## Installation

```bash
uv add pipecat-rabel
```

It needs Pipecat 1.16.0 or later (below 2) and Python 3.11 or later, and a Rabel API key in `RABEL_API_KEY`
([sign up](https://rabellabs.com/signup); the first 10,000 minutes are free).

## Usage with a Pipecat pipeline

Give Bell's turn strategies to your user aggregator. Nothing goes in the pipeline.

```python
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

from pipecat_rabel import RabelBellUserTurnStrategies

stt = DeepgramFluxSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), enable_eager_end_of_turn=True)

context = LLMContext()
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        user_turn_strategies=RabelBellUserTurnStrategies(agent="demo-bank"),
    ),
)

# transport, llm and tts are your own; the pipeline is Pipecat's usual one
pipeline = Pipeline(
    [transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator]
)
```

`RabelBellUserTurnStrategies` is built on Pipecat's `ExternalUserTurnStrategies`, as `EagerUserTurnStrategies` is,
with Bell's stop strategy, a subclass of `EagerUserTurnStopStrategy`. It uses Pipecat's documented door for answering
early: a custom stop strategy that calls `start_user_turn_speculation()`, `cancel_user_turn_speculation()` and
`report_user_turn_stopped()`. With Flux's eager end of turn on, as above, Pipecat answers early on Flux's ends of turn
and on Bell's guesses; with it off, on Bell's guesses alone.

The bundle replaces the turn strategies Flux recommends. If you set Flux's `should_interrupt=False`, pass
`RabelBellUserTurnStrategies(enable_interruptions=False)`.

`demo-bank` is a demonstration agent already trained on a bank's callers, so it guesses from the first call. Use your
own agent's name once it takes calls. A new name sends no guess until Bell has learned enough of its calls. The
conversation always holds the speech-to-text's final words, never the guess.

## Running the example

[`examples/bank_call.py`](https://github.com/rabellabs/pipecat-rabel/blob/main/examples/bank_call.py) is a bank's phone agent: Deepgram Flux, Bell, Claude Haiku 5.5 and
Deepgram Aura-2, talking to you in the browser.

```bash
uv add "pipecat-rabel[example]"
export RABEL_API_KEY=... DEEPGRAM_API_KEY=... ANTHROPIC_API_KEY=...
python examples/bank_call.py      # then open http://localhost:7860 and talk
```

Say "I need to reset my password". Each turn logs Bell's guess, whether it was right, and how long before the final
words the reply on it started. Its calls are from the Gridspace-Stanford Harper Valley dataset of simulated
bank calls (CC BY 4.0).

## Settings

The bundle and the strategy take the same settings. The strategy is `strategies.bell`.

| Setting | Default | What it does |
|---|---|---|
| `api_key` | `RABEL_API_KEY` | Without a key Bell stays off, the strategy is Pipecat's eager one, and one line is logged. |
| `agent` | `"default"` | The agent Bell learns for: 1 to 64 lower-case letters, digits, `-` and `_`. Calls that share an agent name share what Bell learns. Give each voice agent you run its own name, such as `front-desk` or `billing`, and keep all of that agent's calls under it. |
| `min_chance` | `0.7` | How sure Bell must be before it guesses, 0.67 to 0.99. Higher means fewer guesses and fewer wrong ones. |
| `learn` | `True` | `False` means this call teaches Bell nothing. |
| `report` | `True` | Tells the API at each turn's end whether the guess was kept or dropped, with no words, so the agent's status can show it. |
| `user` | `None` | An id you choose for the person on the call. Bell does not use it yet. Never a name, a number or an address. |
| `vendor` | `None` | Your speech-to-text, for example `"deepgram/flux-general-en"`, for usage reports. |
| `connect_timeout` | `2.0` | Seconds allowed to open a conversation with the API. |
| `answer_timeout` | `0.5` | Seconds after the words it answers within which a guess is still used. |
| `match_policy` | `NormalizedMatch()` | Pipecat's check of an early reply's words against the final words. |
| `speculation_timeout` | `5.0` | Seconds an early reply may go unresolved before it is withdrawn. |
| `url` | `RABEL_API_URL`, then Rabel's API | The API's address. |

Events on the strategy: `on_ready(strategy, ready)`, `on_guess(strategy, text, chance)` and
`on_turn_end(strategy, final, guess, right)`. `strategy.stats` counts turns, guesses received, started, dropped and right,
and failures by kind.

## What was measured

On recorded, scripted calls to one bank (Harper Valley), in English, on Deepgram Flux's words. Bell learned from 1,070
calls by 27 callers and was measured on 376 calls by 28 callers it had never heard (95% intervals, by whole calls).

- On Pipecat 1.16.0 with Flux deciding the end of the turn, Bell started the reply 170 ms a turn sooner [159, 182] than
  with no early start, at 0.07 more LLM calls a turn. That is when the LLM starts. Pipecat holds an early reply until the
  turn ends, so the caller hears less of it: in one reading with Claude Haiku 5.5 and Deepgram Aura-2, the first audio came
  133 ms a turn sooner [125, 142]. The package ran as `RabelBellUserTurnStrategies` on the user aggregator, with Flux's
  eager end of turn off and no local VAD.
- Under LiveKit's default (its own turn detector with preemptive generation, on LiveKit Agents 1.8.13 on our own
  machines), Bell started the reply 118 ms a turn sooner [108, 128], at 0.11 more LLM calls a turn. In one reading the
  first audio came 115 ms a turn sooner [105, 125].

The method, the other rows, the script and its data are in [Rabel's benchmark](https://github.com/rabellabs/bell-benchmark).

## Limits

- **The numbers come from one kind of call:** scripted calls to one bank, in English, on Flux's words. On freer speech we
  have only looked informally, and Bell mostly stayed quiet. No live deployment's callers and no other language have been
  measured.
- **A wrong guess costs one LLM call** and nothing the caller hears, when these turn strategies end the turn. If your app,
  or a turn processor ahead of the aggregator such as `UserTurnProcessor`, pushes `UserStoppedSpeakingFrame` itself,
  Pipecat releases the early reply before the words are checked and the caller can hear both replies.
- **It takes the place of Smart Turn.** Like Pipecat's eager strategies, it leaves the end of the turn to the
  speech-to-text, so the speech-to-text must end turns and send interim transcriptions, as Flux does.
- **A local VAD can withdraw an early start** when the caller pauses and speaks again. That start costs one LLM call and
  gains nothing; the reply starts early again when the words reach Bell's guess or Flux's eager end of turn comes.
- **A reply started on a guess never runs a tool call.** Pipecat withdraws it and the reply to the final words runs it.
- **It fails open.** While the API is unreachable, slow, refusing or sending what the strategy cannot read, Bell starts no
  early reply, the turn goes exactly as with Pipecat's eager strategy, and one line is logged for each kind of failure.
  The pipeline never waits for the API. After a busy, slow or unreachable API it tries again on its own; after a refused
  key, no minutes left or an unknown agent, Bell stays off for the rest of the call.
- **The stats and the turn's report count a refused guess differently.** A guess the strategy refuses as unreadable,
  under `min_chance`, late or for other words counts as dropped in `strategy.stats` and shows as "none" in the turn's
  report.

## Tests

```bash
uv run --extra test pytest
```

112 tests, with no key and no network, against a local stand-in for the API, with the strategy in Pipecat's own user
aggregator. A [workflow](https://github.com/rabellabs/pipecat-rabel/blob/main/.github/workflows/nightly.yml) also runs them every night against Pipecat's main branch.

Tested with Pipecat 1.16.0 on Python 3.11, 3.12 and 3.13.

## License

BSD 2-Clause. See [LICENSE](https://github.com/rabellabs/pipecat-rabel/blob/main/LICENSE).
