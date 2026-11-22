#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A bank's phone agent with Bell: the reply starts on Bell's guess at the caller's words.

    uv add "pipecat-rabel[example]"
    python bank_call.py            # then open http://localhost:7860 and talk

Keys, from the environment or a .env file: RABEL_API_KEY, DEEPGRAM_API_KEY, ANTHROPIC_API_KEY.

The agent is Deepgram Flux (speech-to-text, which also decides where the turn ends) -> Claude Haiku 5.5 -> Deepgram
Aura-2, with Bell in the user aggregator's turn strategies. Flux's eager end of turn is on, so Pipecat answers Flux's
early ends of turn; Bell adds its guesses to Flux's own. Bell uses ``demo-bank``, a demonstration agent every
account can use, already trained on a bank's callers, so it guesses from the first call. Try "I need to reset my
password".

Each turn logs Bell's guess, if it made one, whether it was right, and how long before the turn's final words the reply
started on it. When the call ends, the strategy's own line gives the call's kept and dropped guesses.
"""

import os
import time

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from pipecat_rabel import RabelBellUserTurnStrategies

load_dotenv(override=True)

SYSTEM = (
    "You answer the phone for a retail bank. Your words are spoken aloud: answer in one short sentence, "
    "with no lists or formatting, and ask one question at a time."
)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Builds and runs the agent on one connected caller."""
    # Flux's eager end of turn on: Pipecat then answers early, on Flux's ends of turn and on Bell's guesses, and keeps an
    # answer only when the turn's final words match the words it was started on.
    stt = DeepgramFluxSTTService(api_key=os.environ["DEEPGRAM_API_KEY"], enable_eager_end_of_turn=True)

    # Pipecat's eager turn strategies with Bell's added, in place of the ones Flux recommends.
    strategies = RabelBellUserTurnStrategies(
        api_key=os.environ["RABEL_API_KEY"],
        agent="demo-bank",
        vendor="deepgram/flux-general-en",
    )
    bell = strategies.bell

    llm = AnthropicLLMService(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        settings=AnthropicLLMService.Settings(
            model="claude-haiku-5-5", max_tokens=80, system_instruction=SYSTEM
        ),
    )
    tts = DeepgramTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramTTSService.Settings(voice="aura-2-thalia-en"),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(user_turn_strategies=strategies)
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @bell.event_handler("on_ready")
    async def on_ready(strategy, ready):
        logger.info(f"Bell is ready on {strategy.agent}")

    guessed_at = {}  # when the reply was last started on a guess

    @bell.event_handler("on_guess")
    async def on_guess(strategy, text, chance):
        guessed_at["t"] = time.monotonic()
        logger.info(f"Bell's guess, the reply started on it: {text!r} (chance {chance})")

    @bell.event_handler("on_turn_end")
    async def on_turn_end(strategy, final, guess, right):
        if guess is None:
            logger.info(f"Turn ended: {final!r}; Bell made no guess")
        elif right:
            ahead_ms = (time.monotonic() - guessed_at.get("t", time.monotonic())) * 1000
            logger.info(
                f"Turn ended: {final!r}; Bell's guess was right, "
                f"the reply on it started {ahead_ms:.0f} ms before the final words"
            )
        else:
            logger.info(f"Turn ended: {final!r}; Bell's guess {guess!r} differed, Pipecat answered the final words")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        context.add_message({"role": "user", "content": "(The caller is on the line. Greet them briefly.)"})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point for Pipecat's runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
