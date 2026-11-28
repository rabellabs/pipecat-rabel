#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bell, as a Pipecat user turn stop strategy: an early reply started on Bell's guess at the caller's words.

The strategy sits in the user aggregator's turn strategies, where Pipecat's own eager end of turn sits. It reads the words
as the speech-to-text sends them, sends them to Rabel's API (version 3 of its contract: the new Bell), and when Bell guesses
the rest of the turn it starts Pipecat's early reply on the guess with ``start_user_turn_speculation()``. At the turn's end it keeps
that reply only when the committed transcript matches the guess, by the strategy's match policy (``NormalizedMatch`` by
default), and withdraws it with ``cancel_user_turn_speculation()`` otherwise, so a wrong guess costs one model call and
nothing the caller hears. It pushes no frame and changes none: the speech-to-text's frames reach the aggregator as they
were sent. With the API down, slow, refusing, closing or sending something the strategy cannot read, Bell starts no early
reply, the turn ends as Pipecat's own eager strategy ends it, and one line is logged for each kind of failure.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import Any

import websockets
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserTurnResponseCancelFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.turns.types import UserTurnSpeculation
from pipecat.turns.user_stop import EagerMatchPolicy, EagerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text
from websockets.exceptions import InvalidStatus

from pipecat_rabel import __version__
from pipecat_rabel._form import CallForm, words_of, written

RABEL_API_URL = "wss://api.rabellabs.com/v1/conversation"
"""The address of Bell's conversation endpoint."""

DEFAULT_AGENT = "default"
DEFAULT_MIN_CHANCE = None  # the new Bell's balanced setting: the hello sends no min_chance
# The bar a guess must reach when neither the developer nor the API's ready gives one and the API is not version 3, as in
# 0.3.0. A version 3 API with no min_chance set serves its balanced setting, and no bar is added here.
_V2_MIN_CHANCE = 0.7
MIN_CHANCE_RANGE = (0.67, 0.99)  # the API's range; a value outside it is clamped here, as the API would bring it in

_AGENT_NAME = re.compile(r"^[a-z0-9_-]{1,64}$")
# the hello's `user`: 1 to 128 of A-Z, a-z, 0-9, '.', '_', ':' and '-'
_USER = re.compile(r"[A-Za-z0-9._:-]{1,128}")
# the report's head start is a number of ms from 0 to the conversation's limit (an hour)
_MAX_HEAD_START_MS = 3_600_000

# The API's refusals. A refusal in STOP ends Bell for the rest of the call: retrying cannot change
# the answer. One in RETRY is retried with the back-off below, or after the API's retry_after_s when that is longer.
_STOP_STATUSES = {400, 401, 402, 404, 409}
_RETRY_STATUSES = {408, 429, 503, 529}

# Waits between attempts to open a conversation after a failure, in seconds; the last one repeats.
_BACKOFF = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
# A conversation that stayed open this long and then closed cleanly (the API closes one after an hour) is reopened at once.
_LONG_CONVERSATION_S = 60.0


def _pipecat_version() -> str:
    try:
        return _package_version("pipecat-ai")
    except PackageNotFoundError:  # pragma: no cover - only when Pipecat runs from a source tree
        return "unknown"


def _ws_url(url: str) -> str:
    """Accepts the endpoint itself, or the API's base address in http(s) or ws(s) form."""
    url = url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://") :]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://") :]
    if not url.endswith("/v1/conversation"):
        url += "/v1/conversation"
    return url


@dataclass
class _Conversation:
    """One open conversation with the API."""

    ws: Any
    opened_at: float
    v: int | None = None  # the version its ready said: 3 for the new Bell, 2 for an API without it
    min_chance: float | None = None  # the ready's min_chance in force; None for the new Bell's balanced setting
    hold_line: float | None = None
    hold_always: bool | None = None
    hold_updates: int | None = None
    calls_heard: int | None = None
    refusal: dict | None = None  # an error message the API sent before closing


@dataclass
class BellStats:
    """Counts kept for one call, for logs and tests.

    Parameters:
        conversations: Conversations opened with the API.
        turns_sent: Turns whose words were sent to Bell.
        guesses_received: Guesses Bell sent.
        guesses_pushed: Early replies started on Bell's guesses.
        guesses_dropped: Guesses not used: late, outside the turn, unreadable, under ``min_chance``, or never started
            in their turn.
        guesses_right: Turns whose guess equalled the final words, as the match policy compares them.
        eager_held_back: The speech-to-text's own early ends of turn not started because Bell's early reply on the same
            words was already under way.
        failures: Failures by kind, each kind logged once.
    """

    conversations: int = 0
    turns_sent: int = 0
    guesses_received: int = 0
    guesses_pushed: int = 0
    guesses_dropped: int = 0
    guesses_right: int = 0
    eager_held_back: int = 0
    failures: dict[str, int] = field(default_factory=dict)


class RabelBellUserTurnStopStrategy(EagerUserTurnStopStrategy):
    """Starts Pipecat's early reply on Bell's guess at the caller's words, and keeps it only when the turn matches.

    A user turn stop strategy, installed with :class:`RabelBellUserTurnStrategies` as Pipecat's
    ``EagerUserTurnStopStrategy`` is installed with ``EagerUserTurnStrategies``. It is that eager strategy with Bell added,
    so a speech-to-text's own eager end of turn (Deepgram Flux's, when switched on) keeps working beside Bell, and decides
    the turn's end as it does. What it adds:

    - The words: each :class:`~pipecat.frames.frames.InterimTranscriptionFrame` with words is sent to Bell as ``heard``,
      each :class:`~pipecat.frames.frames.TranscriptionFrame` with words as ``end_of_turn``.
    - Bell's guess at the whole turn, written as the speech-to-text writes the call's words: the words heard as it
      showed them, then each word Bell adds as it last wrote that word in this call. An early reply is started on it
      with ``start_user_turn_speculation()``, the call
      Pipecat's docs give a stop strategy for answering ahead of the turn ending. Pipecat holds the reply until the turn
      is confirmed and never writes the guess to the conversation.
    - The turn's end: the reply is kept when the transcript committed since the guess matches it by the match policy
      (``NormalizedMatch`` unless another is given), and withdrawn with ``cancel_user_turn_speculation()`` when it does
      not, when the caller resumes, or when the turn stays unresolved for ``speculation_timeout``. So a wrong guess is
      never heard when the turn ends through these strategies. When the app or an upstream turn processor such as
      Pipecat's ``UserTurnProcessor`` ends the turn itself, Pipecat releases the held reply before the check, and the
      caller can hear both the reply to the guess and the reply to the final words.

    One early reply runs at a time. A newer guess withdraws the reply under way before it starts its own. The
    speech-to-text's early end of turn on the words of Bell's reply under way is not started (it would replace that reply
    with an identical one started later; counted in ``stats.eager_held_back``); one on other words withdraws Bell's
    reply, then starts. When a VAD hears the caller go on after a pause, Pipecat voids the reply under way, and Bell's
    no longer counts as running.

    Failing open: the connection to the API is opened in the background when the pipeline is set up, and nothing waits
    for it. Until the API answers ``ready``, while it is down, when it refuses, when it closes the conversation mid-call,
    and when it sends a message this strategy cannot read, Bell starts no early reply and the turn ends exactly as
    Pipecat's eager strategy ends it. Each kind of failure is logged once a call at WARNING, and again only at DEBUG.

    Event handlers available:

    - on_ready(strategy, ready): the API answered the hello; ``ready`` is a dict with the version the API answered
      with (``v``: 3 for the new Bell), its ``hold_line`` and ``calls_heard`` (how many of this agent's calls Bell has
      learned from).
    - on_guess(strategy, text, chance): an early reply was started on a guess. ``chance`` is None when an earlier guess
      is started again because the words heard have reached it.
    - on_turn_end(strategy, final, guess, right): a turn sent to Bell ended; ``guess`` is the guess started in that
      turn or None, ``right`` whether it equals the final words as the match policy compares them.

    The report: at each turn's end the strategy tells the API whether Bell's guess was kept or dropped, with the head
    start of a kept one, so the agent's status can show them. On a turn where the new Bell sent more than one guess and
    more than one was started, the report is on the last one started, and ``dropped_before`` counts those before it. Kept means Bell's early reply is still the one under way
    when the final words arrive and the match policy finds the final words equal to it. One DEBUG line is logged for
    each kept guess, and one INFO line with the call's totals when it ends; neither carries a word of the call.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent: str = DEFAULT_AGENT,
        min_chance: float | None = DEFAULT_MIN_CHANCE,
        learn: bool = True,
        url: str | None = None,
        vendor: str | None = None,
        connect_timeout: float = 2.0,
        answer_timeout: float = 0.5,
        user: str | None = None,
        report: bool = True,
        match_policy: EagerMatchPolicy | None = None,
        speculation_timeout: float = 5.0,
        **kwargs,
    ):
        """Initialize the strategy.

        Args:
            api_key: The Rabel API key. Defaults to the ``RABEL_API_KEY`` environment variable. Without one Bell stays
                off and the strategy is Pipecat's eager strategy.
            agent: Your agent's name, which Bell learns for: lower-case letters, digits, ``-`` and ``_``, 1 to
                64 characters. ``demo-bank`` is a demonstration agent every account can use.
            min_chance: How sure Bell must be before it guesses, 0.67 to 0.99, or None (the default) for the new Bell's
                balanced setting. Out of range is clamped. A value set here is sent in the hello, and every guess must
                then be at least that sure: a higher value sends fewer guesses, fewer wrong ones and fewer right ones.
            learn: False means this call teaches Bell nothing.
            url: The conversation endpoint, or the API's base address. Defaults to the ``RABEL_API_URL`` environment
                variable, and without it to Rabel's API.
            vendor: The speech-to-text in front of the aggregator (for example ``deepgram/flux-general-en``). Sent to
                the API for its usage reports; never required.
            connect_timeout: Seconds allowed to open a conversation, from the connection to Bell's ``ready``.
            answer_timeout: Seconds after the words it answers within which a guess is still used. A later guess is
                dropped.
            user: An id you choose for the person on the call, sent in the hello: 1 to 128 letters, digits, ``.``,
                ``_``, ``:`` or ``-``. It must never be a name, a number or an address. Bell does not use it yet.
                Defaults to None (not sent).
            report: True (the default) tells the Rabel API at each turn's end whether Bell's guess was kept or
                dropped, with the head start of a kept one, so your agent's status can show them. No words are
                added. False leaves it out.
            match_policy: Decides whether the committed transcript matches an early reply's words, Bell's and the
                speech-to-text's alike. Defaults to Pipecat's ``NormalizedMatch``.
            speculation_timeout: Seconds an early reply may go unresolved before it is withdrawn, Bell's and the
                speech-to-text's alike.
            **kwargs: Passed to Pipecat's ``EagerUserTurnStopStrategy``.

        Raises:
            ValueError: If ``agent`` is not a valid agent name, ``user`` is not in its form, or ``report`` is not
                True or False.
        """
        super().__init__(match_policy=match_policy, speculation_timeout=speculation_timeout, **kwargs)
        if not _AGENT_NAME.match(agent):
            raise ValueError(
                f"agent must be 1 to 64 lower-case letters, digits, '-' or '_', not {agent!r}"
            )
        if user is not None and (not isinstance(user, str) or not _USER.fullmatch(user)):
            # the error does not repeat the value: it may be a name or an address
            raise ValueError(
                "user must be 1 to 128 letters, digits, '.', '_', ':' or '-' (an id you choose, never a name, a number "
                "or an address)"
            )
        if not isinstance(report, bool):
            raise ValueError(f"report must be True or False (got {report!r})")
        lo, hi = MIN_CHANCE_RANGE
        if min_chance is not None and not lo <= min_chance <= hi:
            logger.warning(f"{self}: min_chance {min_chance} is outside {lo} to {hi}; clamped")
            min_chance = min(max(min_chance, lo), hi)

        self._api_key = api_key if api_key is not None else os.environ.get("RABEL_API_KEY", "")
        self._agent = agent
        self._min_chance = min_chance
        self._learn = learn
        self._url = _ws_url(url if url is not None else os.environ.get("RABEL_API_URL") or RABEL_API_URL)
        self._vendor = vendor
        self._connect_timeout = connect_timeout
        self._answer_timeout = answer_timeout
        self._user = user
        self._report = report
        self._bell_speculation_timeout = speculation_timeout

        self._conv: _Conversation | None = None
        self._connection_task: asyncio.Task | None = None
        self._sender_task: asyncio.Task | None = None
        self._outbox: asyncio.Queue | None = None
        self._stopped = False

        # The turn as Bell's API counts it: from its first words to its final words.
        self._in_turn = False
        self._turn_conv: _Conversation | None = None  # the conversation this turn's words go to, if any
        self._heard = ""
        self._heard_at: dict[str, float] = {}  # words sent as heard -> when
        self._guesses: list[str] = []
        self._unused: list[str] = []  # guesses taken in this turn and not yet started, counted dropped if never started
        self._pushed_guess: str | None = None  # the guess started last in this turn
        self._form = CallForm()  # how this call's speech-to-text writes its words, for writing a guess the same way
        self._started: list[str] = []  # the guesses started in this turn, each once, the last started last

        # Pipecat's turn and its one early reply.
        self._pipecat_turn_open = False
        self._live_text: str | None = None  # the words of the early reply under way, Bell's or the speech-to-text's
        self._bell: UserTurnSpeculation | None = None  # Bell's early reply, while it is the one under way
        self._bell_at: float | None = None  # when it was started (the report's head start)
        self._bell_timer: asyncio.Task | None = None
        # transcripts committed since it was started, which it is checked against, joined as Pipecat's eager strategy
        # joins them (no space added where the speech-to-text says its text carries its own)
        self._bell_parts: list[TextPartForConcatenation] = []
        self._bell_confirmed: UserTurnSpeculation | None = None  # the one this turn's end kept
        self._withdrawing_for_eager = False  # while Bell's reply is withdrawn for the speech-to-text's early reply

        self.stats = BellStats()
        # what this call's guesses did for the host, for the summary logged when the call ends (numbers only)
        self.summary: dict[str, int] = {"turns": 0, "guesses": 0, "kept": 0, "dropped": 0, "head_start_ms": 0}
        self._summary_logged = False
        self.ready = asyncio.Event()
        """Set while a conversation with Bell is open and ready for words."""

        self._register_event_handler("on_ready")
        self._register_event_handler("on_guess")
        self._register_event_handler("on_turn_end")

    @property
    def agent(self) -> str:
        """The name of the agent this strategy talks to Bell for."""
        return self._agent

    @property
    def min_chance(self) -> float | None:
        """How sure Bell must be before it guesses, as set; None for the new Bell's balanced setting."""
        return self._min_chance

    @property
    def v3(self) -> bool:
        """Whether the open conversation serves the new Bell (its ``ready`` said ``"v": 3``). False before a ``ready``
        and with an API that answered with a version 2 ``ready``."""
        return self._conv is not None and self._conv.v == 3

    @property
    def hold_line(self) -> float | None:
        """The API's ``hold_line`` from the open conversation's ``ready``, or None.

        For an app to log what the API sent; this strategy does not apply it.
        """
        return self._conv.hold_line if self._conv else None

    @property
    def hold_always(self) -> bool | None:
        """The API's ``hold_always`` from the open conversation's version 3 ``ready`` (true or false), or None.

        For an app to log what the API sent; this strategy does not apply it, like ``hold_line``.
        """
        return self._conv.hold_always if self._conv else None

    @property
    def hold_updates(self) -> int | None:
        """The API's ``hold_updates`` from the open conversation's version 3 ``ready`` (1 or 2), or None.

        For an app to log what the API sent; this strategy does not apply it, like ``hold_line``.
        """
        return self._conv.hold_updates if self._conv else None

    # ------------------------------------------------------------------ the strategy's life

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the strategy and open Bell's conversation in the background.

        Pipecat sets a strategy up again after cleaning it up when a service's recommended strategies are merged in;
        the conversation is then opened again.
        """
        await super().setup(setup)
        self._start()

    async def cleanup(self):
        """Stop Bell's tasks and close its conversation at the pipeline's teardown."""
        await super().cleanup()
        await self._stop(polite=False, final=False)
        await self._take_bell()

    # ------------------------------------------------------------------ frames

    async def process_frame(self, frame: Frame):
        """Send the words to Bell, then let Pipecat's eager strategy see the frame.

        Args:
            frame: The frame to be analyzed.
        """
        if isinstance(frame, (EndFrame, CancelFrame)):
            await super().process_frame(frame)
            # An ended call sends its last turn's words to Bell before the conversation closes: Bell learns from a call
            # when it closes. The frame is already on its way; this waits at most about 2 seconds, and only when the
            # API is slow.
            await self._stop(polite=isinstance(frame, EndFrame), final=True)
            return

        if isinstance(frame, UserTurnResponseCancelFrame):
            # The early reply under way was withdrawn (the speech-to-text's resumed turn, a timeout, or a withdrawal of
            # this strategy's own, which Pipecat feeds back to every strategy): none is under way.
            await self._take_bell()
            self._live_text = None
            if self._withdrawing_for_eager:
                # This strategy's own withdrawal of Bell's reply, made in start_user_turn_speculation just before the
                # speech-to-text's early reply starts. Pipecat's eager strategy has already recorded that reply and
                # would forget it on this frame, so the frame goes no further.
                return
        elif isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame, ProposedUserStartedSpeakingFrame)):
            # Speech voids the turn's answer, and with it the confirmation of the guess that gave it.
            self._bell_confirmed = None
            if isinstance(frame, VADUserStartedSpeakingFrame) and self._bell is not None:
                # The caller went on after the VAD heard them stop. Pipecat resumes the turn and voids the early reply
                # under way (the user aggregator interrupts it or withdraws it downstream, and tells no strategy), so
                # Bell's no longer runs: the speech-to-text's early end of turn on its words is not held back, and words
                # that reach the guess start Bell's again.
                await self._take_bell()
                self._live_text = None
        elif isinstance(frame, TranscriptionFrame) and frame.text:
            # Committed before Pipecat's strategy sees it, which may end the turn on it.
            self._bell_parts.append(
                TextPartForConcatenation(
                    frame.text, includes_inter_part_spaces=getattr(frame, "includes_inter_frame_spaces", False)
                )
            )
            try:
                await self._on_final(frame)
            except Exception as e:  # noqa: BLE001 - Bell must never stop the turn
                self._failure("internal", f"Bell's strategy hit an error ({e!r}); the turn goes on")

        await super().process_frame(frame)

        if isinstance(frame, InterimTranscriptionFrame) and frame.text:
            reached = None
            try:
                reached = self._on_heard(frame)
            except Exception as e:  # noqa: BLE001
                self._failure("internal", f"Bell's strategy hit an error ({e!r}); the turn goes on")
            if reached is not None:
                # The words have reached a guess Bell made earlier in the turn: start the reply on it.
                await self._start_bell(reached)

    async def handle_user_turn_started(self):
        """Ready the strategy for the turn now starting."""
        await super().handle_user_turn_started()
        unresolved = await self._take_bell()
        self._bell_confirmed = None
        self._bell_parts = []
        self._pipecat_turn_open = True
        if unresolved is not None:
            await self.cancel_user_turn_speculation()

    async def handle_user_turn_stopped(self):
        """Clear the turn's state; withdraw Bell's early reply if the turn ended without resolving it."""
        unresolved = await self._take_bell()
        self._bell_confirmed = None
        self._bell_parts = []
        self._live_text = None
        self._pipecat_turn_open = False
        await super().handle_user_turn_stopped()
        if unresolved is not None:
            # The turn ended without the check (the turn controller's stop timeout, an interruption, the call's end).
            logger.debug(f"{self}: turn ended unresolved, discarding the reply started on Bell's guess")
            await self.cancel_user_turn_speculation()

    async def report_user_turn_stopped(
        self,
        *,
        enable_user_speaking_frames: bool | None = None,
        confirms_speculation: bool = False,
    ):
        """Report that the user is done, keeping the reply started on Bell's guess only if the turn matches it.

        Without a reply of Bell's under way, Pipecat's eager strategy decides, on its own early reply if it has one.

        Args:
            enable_user_speaking_frames: Whether to emit the ``UserStoppedSpeakingFrame`` for this turn.
            confirms_speculation: Whether a reply generated ahead of this turn ending already answers it. Decided here
                for Bell's own reply.
        """
        bell = await self._take_bell() or self._bell_confirmed
        if bell is None:
            await super().report_user_turn_stopped(
                enable_user_speaking_frames=enable_user_speaking_frames,
                confirms_speculation=confirms_speculation,
            )
            return

        if self.match_policy.matches(bell.text, self._committed()):
            self._bell_confirmed = bell
            # The reply already runs on Bell's guess. The turn's end releases it, and the flag stops a second inference
            # answering the same turn.
            await super().report_user_turn_stopped(
                enable_user_speaking_frames=enable_user_speaking_frames,
                confirms_speculation=True,
            )
            return

        self._bell_confirmed = None
        await self.cancel_user_turn_speculation()
        # The committed transcript gets an inference of its own, which the turn controller runs for this decision.
        await super().report_user_turn_stopped(enable_user_speaking_frames=enable_user_speaking_frames)

    async def start_user_turn_speculation(self, speculation: UserTurnSpeculation):
        """Start the speech-to-text's own early reply, unless Bell's on the same words is under way.

        Pipecat's eager strategy calls this on the speech-to-text's eager end of turn. Bell's own replies are started
        through the base class and do not pass here.

        This override relies on a detail Pipecat's docs do not promise: ``EagerUserTurnStopStrategy`` starts its
        own early reply by calling ``self.start_user_turn_speculation()`` (Pipecat 1.16.0,
        ``eager_user_turn_stop_strategy.py`` line 216, in ``_speculate``), so the call reaches this method. If a later
        Pipecat started it another way, the speech-to-text's early end of turn on the words of Bell's reply under way
        would start instead of being held back, Bell's reply would not be withdrawn first, and ``stats.eager_held_back``
        would stay at 0. ``test_pipecats_eager_strategy_starts_its_early_reply_through_this_method`` fails if that
        happens, and the nightly workflow runs it against Pipecat's main branch.

        Withdrawing Bell's reply here relies on another detail: Pipecat feeds the withdrawal back to every strategy as a
        ``UserTurnResponseCancelFrame`` before this call returns, and its eager strategy would forget the early reply it
        has just recorded. :meth:`process_frame` keeps that frame from it while this method withdraws Bell's reply.

        Args:
            speculation: The speech-to-text's early end of turn.
        """
        if self._bell is not None and self._same(self._live_text, speculation.text):
            # Starting it would replace the reply under way on Bell's guess with an identical one started later.
            self.stats.eager_held_back += 1
            logger.debug(f"{self}: the speech-to-text's early end of turn has the words of Bell's, under way; not started")
            return
        if await self._take_bell() is not None:
            # Bell's reply on other words is under way. Pipecat withdraws nothing when an early reply starts, so it is
            # withdrawn here first: left running, it would hold the LLM and pass the gate at the turn's end.
            self._withdrawing_for_eager = True
            try:
                await self.cancel_user_turn_speculation()
            finally:
                self._withdrawing_for_eager = False
        self._live_text = speculation.text
        await super().start_user_turn_speculation(speculation)

    # ------------------------------------------------------------------ Bell's early reply

    async def _start_bell(self, text: str, chance: float | None = None):
        """Withdraw the early reply under way, if any, and start one on this guess."""
        if not self._pipecat_turn_open:
            return  # Pipecat starts no early reply outside its turn; the guess waits for the words to reach it
        self._unused = [g for g in self._unused if g != text]
        previous = await self._take_bell()
        if previous is not None or self._live_text is not None:
            await self.cancel_user_turn_speculation()
        speculation = UserTurnSpeculation(text=text)
        self._bell = speculation
        self._bell_at = time.monotonic()
        self._bell_parts = []
        self._bell_confirmed = None
        self._live_text = text
        self._pushed_guess = text
        self._started = [g for g in self._started if g != text] + [text]
        self.stats.guesses_pushed += 1
        self._bell_timer = self.create_task(self._bell_timeout(speculation), "bell_speculation_timeout")
        logger.debug(f"{self}: starting an early reply on Bell's guess")
        # The base class's call, not this class's override, which handles the speech-to-text's early end of turn.
        await super().start_user_turn_speculation(speculation)
        await self._call_event_handler("on_guess", text, chance)

    async def _bell_timeout(self, speculation: UserTurnSpeculation):
        """Withdraw a reply on Bell's guess that the turn never resolved."""
        await asyncio.sleep(self._bell_speculation_timeout)
        self._bell_timer = None
        if self._bell is not speculation:
            return
        logger.debug(f"{self}: Bell's guess unresolved after {self._bell_speculation_timeout}s, discarding the reply")
        self._bell = None
        await self.cancel_user_turn_speculation()

    async def _take_bell(self) -> UserTurnSpeculation | None:
        """Take Bell's early reply under way, if any, stopping its clock."""
        speculation, self._bell = self._bell, None
        if self._bell_timer is not None:
            task, self._bell_timer = self._bell_timer, None
            if task is not asyncio.current_task():
                await self.cancel_task(task)
        return speculation

    def _committed(self) -> str:
        """The transcript committed since Bell's reply started, as Pipecat's eager strategy joins it."""
        return concatenate_aggregated_text(self._bell_parts)

    def _same(self, a: str | None, b: str | None) -> bool:
        """Whether two transcripts are the same words, as the match policy compares them."""
        return bool(a) and bool(b) and self.match_policy.matches(a, b)

    # ------------------------------------------------------------------ Bell's turn

    def _on_heard(self, frame: InterimTranscriptionFrame) -> str | None:
        """Sends the words to Bell; returns an earlier guess the words have reached, to start the reply on."""
        if not self._in_turn:
            self._begin_turn()
        self._heard = frame.text
        if self._turn_conv is not None:
            self._heard_at.setdefault(frame.text, time.monotonic())
            self._send(self._turn_conv, {"type": "heard", "text": frame.text})
        earlier = next((g for g in self._guesses if self._same(g, self._heard)), None)
        if earlier and not self._same(earlier, self._live_text):
            return earlier
        return None

    async def _on_final(self, frame: TranscriptionFrame):
        final_at = time.monotonic()
        try:
            self._form.final(frame.text)
        except Exception:  # noqa: BLE001 - the form is only for writing guesses
            pass
        conv, guess = self._turn_conv, self._pushed_guess
        report, head_start_ms = self._report_of(frame.text, final_at)
        # The new Bell can send more than one guess a turn. The report is on the guess started last. Each guess started
        # before it was withdrawn when a newer one started, and counts as dropped before it.
        dropped_before = max(len(self._started) - 1, 0) if self._in_turn and report != "none" else 0
        self._count(report, head_start_ms, dropped_before)
        self._drop_unused()
        if conv is not None:
            message: dict[str, Any] = {"type": "end_of_turn", "text": frame.text}
            if self._report:
                message["guess"] = report
                if report == "kept" and head_start_ms is not None:
                    message["head_start_ms"] = head_start_ms
                if dropped_before > 0 and conv.v == 3:
                    message["dropped_before"] = dropped_before
            self._send(conv, message)
            right = self._same(guess, frame.text)
            self.stats.guesses_right += int(right)
            await self._call_event_handler("on_turn_end", frame.text, guess, right)
        self._in_turn = False
        self._turn_conv = None
        self._live_text = None  # the turn's end settles the early reply, through the check

    def _report_of(self, final: str, final_at: float) -> tuple[str, int | None]:
        """What happened to Bell's guess in the turn: "kept" with its head start when the reply started on it is the
        one under way and the transcript committed since it matches it, "dropped" when one was started and not kept or
        when a guess taken in the turn was never started (it came before Pipecat's turn opened and the words never
        reached it), "none" otherwise. This is the check :meth:`report_user_turn_stopped` makes when the turn ends on
        these final words.
        """
        if not self._in_turn:
            return "none", None
        if self._pushed_guess is None:
            return ("dropped" if self._unused else "none"), None
        if self._bell is not None and self.match_policy.matches(self._bell.text, self._committed()):
            return "kept", None if self._bell_at is None else _ms(final_at - self._bell_at)
        return "dropped", None

    def _drop_unused(self):
        """Count as dropped the guesses taken in the turn that ending it leaves unused (never started)."""
        if self._in_turn:
            self.stats.guesses_dropped += len(self._unused)
        self._unused = []

    def _count(self, report: str, head_start_ms: int | None, dropped_before: int = 0):
        """The call's totals for the summary, and the line for a kept guess (numbers only, no word of the call)."""
        s = self.summary
        s["turns"] += 1
        if report == "none":
            return
        s["guesses"] += 1 + dropped_before
        s[report] += 1
        s["dropped"] += dropped_before
        if report == "kept":
            s["head_start_ms"] += head_start_ms or 0
            # in the form and at the level of Pipecat's own line for an early reply kept (EagerUserTurnStopStrategy:
            # "eager end of turn held, keeping the speculative response", DEBUG)
            logger.bind(head_start_ms=head_start_ms).debug(
                f"{self}: Bell's guess held, keeping the speculative response ({head_start_ms} ms head start)"
            )

    def _log_summary(self):
        """One line when the call ends: numbers only, no word of the call."""
        if self._summary_logged:
            return
        self._summary_logged = True
        s = self.summary
        try:
            logger.bind(
                turns=s["turns"], guesses=s["guesses"], kept=s["kept"], dropped=s["dropped"],
                head_start_ms_total=s["head_start_ms"],
            ).info(
                f"{self}: Bell: {s['turns']} turns, {s['guesses']} guesses, {s['kept']} kept, {s['dropped']} dropped, "
                f"{s['head_start_ms']} ms head start in all"
            )
        except Exception:  # noqa: BLE001 - a fault here never reaches the pipeline
            pass

    def _begin_turn(self):
        self._in_turn = True
        self._heard = ""
        self._heard_at = {}
        self._guesses = []
        self._unused = []
        self._pushed_guess = None
        self._started = []
        # A turn goes to Bell only if its first words do: a conversation opened mid-turn waits for the next turn.
        conv = self._conv
        self._turn_conv = conv if conv is not None and self.ready.is_set() else None
        if self._turn_conv is not None:
            self.stats.turns_sent += 1

    async def _on_guess(self, conv: _Conversation, m: dict):
        """A guess from Bell: check it, then start the early reply on it or drop it."""
        self.stats.guesses_received += 1
        text, heard, chance = m.get("text"), m.get("heard"), m.get("chance")
        if (
            not isinstance(text, str)
            or not text.strip()
            or (heard is not None and not isinstance(heard, str))
            or (chance is not None and (isinstance(chance, bool) or not isinstance(chance, (int, float))))
            or (chance is not None and not 0.0 <= chance <= 1.0)
        ):
            self.stats.guesses_dropped += 1
            self._failure("wrong_data", f"Bell sent a guess this strategy cannot read ({_short(m)}); not used")
            return
        # The bar in force: the API's (its ready says the value it took), else the developer's. With an API that is not
        # version 3 and none set, 0.3.0's 0.7. A version 3 API with none set serves its balanced setting: no bar here.
        line = conv.min_chance if conv.min_chance is not None else self._min_chance
        if line is None and conv.v != 3:
            line = _V2_MIN_CHANCE
        if chance is None and (conv.min_chance is not None or conv.v == 3):
            self.stats.guesses_dropped += 1
            self._failure("wrong_data", "Bell sent a guess with no chance; not used")
            return
        if chance is not None and line is not None and chance < line:
            self.stats.guesses_dropped += 1
            self._failure("wrong_data", f"Bell sent a guess at chance {chance}, under min_chance {line}; not used")
            return
        if not self._in_turn or self._turn_conv is not conv:
            self.stats.guesses_dropped += 1  # the turn it answers is over
            return
        sent_at = self._heard_at.get(heard) if heard is not None else None
        if heard is not None and sent_at is None:
            sent_at = next((t for h, t in self._heard_at.items() if self._same(h, heard)), None)
            if sent_at is None:
                self.stats.guesses_dropped += 1
                self._failure("wrong_data", "Bell sent a guess made from words this call did not send; not used")
                return
        if sent_at is not None:
            late = time.monotonic() - sent_at
            if late > self._answer_timeout:
                self.stats.guesses_dropped += 1
                self._failure(
                    "slow_answer",
                    f"Bell's guess came {late * 1000:.0f} ms after its words, over answer_timeout "
                    f"{self._answer_timeout * 1000:.0f} ms; not used",
                )
                return

        text = self._written(text, heard)
        self._guesses.append(text)
        self._unused.append(text)
        # An early reply under way on the words heard so far stays; otherwise the newest guess replaces it.
        if self._live_text is not None and self._same(self._live_text, self._heard):
            return
        await self._start_bell(text, chance)

    def _written(self, text: str, heard: str | None) -> str:
        """Bell's guess written as this call's speech-to-text writes its words (see ``_form``): the words heard as the
        speech-to-text showed them (the newest, else the ones the guess was made at), then the words Bell adds."""
        g = words_of(text)
        if not g:
            return text
        for shown in (self._heard, heard):
            h = words_of(shown) if shown else ()
            if h and g[: len(h)] == h:
                return written(shown, g[len(h) :], self._form) or text
        return written("", g, self._form) or text

    # ------------------------------------------------------------- connection

    def _start(self):
        if self._connection_task is not None:
            return
        self._stopped = False
        if not self._api_key:
            self._failure("no_key", "no Rabel API key (RABEL_API_KEY); Bell is off for this call")
            return
        self._outbox = asyncio.Queue()
        self._sender_task = self.create_task(self._sender(), "bell_sender")
        self._connection_task = self.create_task(self._run_connection(), "bell_connection")

    async def _stop(self, polite: bool, final: bool):
        """Close Bell's conversation. ``final`` when the call itself has ended (its end or cancel frame)."""
        bell_was_on = self._connection_task is not None
        if final:
            self._drop_unused()  # the call ended inside a turn: its guesses never started are not used
        if bell_was_on and (final or self.summary["turns"] > 0):
            self._log_summary()
        self._stopped = True
        connection, self._connection_task = self._connection_task, None
        if connection is not None:
            await self.cancel_task(connection)
        sender, self._sender_task = self._sender_task, None
        if sender is not None:
            if polite and self._outbox is not None and self._conv is not None:
                # The last turn's words, then a clean close: Bell learns from a call when it closes.
                try:
                    await asyncio.wait_for(self._outbox.join(), 1.0)
                except (asyncio.TimeoutError, TimeoutError):
                    pass
            await self.cancel_task(sender)
        conv, self._conv = self._conv, None
        self.ready.clear()
        if conv is not None:
            try:
                await asyncio.wait_for(conv.ws.close(), 1.0)
            except Exception:  # noqa: BLE001 - already gone, or slow to close; the call goes on either way
                pass

    def _send(self, conv: _Conversation, message: dict):
        if self._outbox is not None:
            self._outbox.put_nowait((conv, message))

    async def _sender(self):
        """Sends queued messages, so the pipeline never waits on the network."""
        assert self._outbox is not None
        outbox = self._outbox
        while True:
            conv, message = await outbox.get()
            try:
                if conv is self._conv:
                    await asyncio.wait_for(conv.ws.send(json.dumps(message)), self._connect_timeout)
            except Exception:  # noqa: BLE001 - the receiver sees the close and reopens
                if conv is self._conv:
                    self._conv = None
                    self.ready.clear()
                    self._failure("closed", "the Rabel API connection failed while sending; going on without Bell")
                    try:
                        await asyncio.wait_for(conv.ws.close(), 0.5)
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                outbox.task_done()

    def _hello(self) -> dict:
        # "v": 3 takes the new Bell. An API without it reads the same hello as a version 2 hello and answers with a
        # version 2 ready (min_chance 0.7 when the hello has none). The strategy then works as 0.3.0 did.
        hello: dict[str, Any] = {
            "type": "hello",
            "v": 3,
            "agent": self._agent,
            "learn": self._learn,
            "host": "pipecat",
            "host_version": _pipecat_version(),
            "hookup": __version__,
        }
        if self._min_chance is not None:
            hello["min_chance"] = self._min_chance
        if self._vendor:
            hello["vendor"] = self._vendor
        if self._user is not None:
            hello["user"] = self._user
        return hello

    async def _run_connection(self):
        """Opens conversations with the API for as long as the call lasts, by the contract's rules."""
        attempt = 0
        while not self._stopped:
            delay: float | None
            try:
                conv = await asyncio.wait_for(self._open(), self._connect_timeout)
            except _Refused as r:
                delay = self._on_refusal(r.message, attempt)
            except (asyncio.TimeoutError, TimeoutError):
                self._failure(
                    "connect_timeout",
                    f"the Rabel API did not answer within connect_timeout {self._connect_timeout} s; "
                    "going on without Bell and retrying",
                )
                delay = _backoff(attempt)
            except _WrongData as w:
                self._failure("wrong_data", f"the Rabel API's first answer was not ready ({w}); retrying")
                delay = _backoff(attempt)
            except Exception as e:  # noqa: BLE001 - down, refused at the socket, a bad address
                self._failure("down", f"could not reach the Rabel API ({type(e).__name__}); going on without Bell")
                delay = _backoff(attempt)
            else:
                attempt = 0
                self.stats.conversations += 1
                self._conv = conv
                self.ready.set()
                await self._call_event_handler(
                    "on_ready", {"v": conv.v, "hold_line": conv.hold_line, "calls_heard": conv.calls_heard}
                )
                await self._receive(conv)
                if self._conv is conv:
                    self._conv = None
                    self.ready.clear()
                if self._stopped:
                    return
                if conv.refusal is not None:
                    delay = self._on_refusal(conv.refusal, attempt)
                else:
                    lasted = time.monotonic() - conv.opened_at
                    code = getattr(conv.ws, "close_code", None)
                    if lasted >= _LONG_CONVERSATION_S and code == 1000:
                        logger.debug(f"{self}: the Rabel API closed the conversation after {lasted:.0f} s; reopening")
                        delay = 0.0
                    else:
                        self._failure(
                            "closed",
                            f"the Rabel API closed the conversation mid-call (code {code}); going on without Bell "
                            "and reopening",
                        )
                        delay = _backoff(attempt)
            if delay is None:
                return
            attempt += 1
            await asyncio.sleep(delay)

    async def _open(self) -> _Conversation:
        try:
            ws = await websockets.connect(
                self._url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                open_timeout=None,
                max_size=2**20,
                user_agent_header=f"pipecat-rabel/{__version__}",
            )
        except InvalidStatus as e:
            status = e.response.status_code
            body = _parse(e.response.body) if e.response.body else None
            message = body if isinstance(body, dict) else {}
            raise _Refused({"status": status, **{k: v for k, v in message.items() if k != "status"}}) from None
        try:
            await ws.send(json.dumps(self._hello()))
            m = _parse(await ws.recv())
            if isinstance(m, dict) and m.get("type") == "error":
                raise _Refused(m)
            if not isinstance(m, dict) or m.get("type") != "ready":
                raise _WrongData(_short(m))
            v = m.get("v")
            v = v if isinstance(v, int) and not isinstance(v, bool) else None
            min_chance = m.get("min_chance")
            hold_line = m.get("hold_line")
            # hold_always and hold_updates are read from a version 3 ready only, as the LiveKit hookup reads them
            hold_always = m.get("hold_always") if v == 3 else None
            hold_updates = m.get("hold_updates") if v == 3 else None
            calls_heard = m.get("calls_heard")
            return _Conversation(
                ws=ws,
                opened_at=time.monotonic(),
                v=v,
                min_chance=float(min_chance)
                if isinstance(min_chance, (int, float)) and not isinstance(min_chance, bool)
                else None,
                hold_line=float(hold_line) if isinstance(hold_line, (int, float)) else None,
                hold_always=hold_always if isinstance(hold_always, bool) else None,
                hold_updates=int(hold_updates)
                if isinstance(hold_updates, (int, float)) and not isinstance(hold_updates, bool) and hold_updates in (1, 2)
                else None,
                calls_heard=calls_heard if isinstance(calls_heard, int) else None,
            )
        except BaseException:
            try:
                await asyncio.wait_for(ws.close(), 0.5)
            except Exception:  # noqa: BLE001
                pass
            raise

    async def _receive(self, conv: _Conversation):
        try:
            async for raw in conv.ws:
                if self._conv is not conv:
                    return
                m = _parse(raw)
                kind = m.get("type") if isinstance(m, dict) else None
                if kind == "guess":
                    try:
                        await self._on_guess(conv, m)
                    except Exception as e:  # noqa: BLE001
                        self._failure("internal", f"Bell's strategy hit an error on a guess ({e!r}); not used")
                elif kind == "end_of_turn":
                    pass  # Bell's own record of the turn; the match policy already decided
                elif kind == "error":
                    conv.refusal = m
                elif kind is None or kind == "ready":
                    self._failure("wrong_data", f"the Rabel API sent a message this strategy cannot read ({_short(m)})")
                # Message types this version does not know are ignored, as the contract asks.
        except websockets.ConnectionClosed:
            pass
        except Exception as e:  # noqa: BLE001
            self._failure("closed", f"the Rabel API connection failed ({type(e).__name__}); going on without Bell")

    def _on_refusal(self, m: dict, attempt: int) -> float | None:
        """Logs a refusal; returns the wait before the next attempt, or None to stay off for the call."""
        status = m.get("status")
        name = m.get("error") or ""
        if status in _STOP_STATUSES:
            self._failure(
                f"refused_{status}",
                f"the Rabel API refused the conversation ({status} {name}); Bell is off for this call",
            )
            return None
        self._failure(
            f"refused_{status}",
            f"the Rabel API refused the conversation ({status} {name}); going on without Bell and retrying",
        )
        retry_after = m.get("retry_after_s")
        wait = _backoff(attempt)
        if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool) and retry_after > wait:
            wait = min(float(retry_after), 300.0)
        return wait

    def _failure(self, kind: str, line: str):
        """One logged line per kind of failure a call; repeats go to DEBUG."""
        n = self.stats.failures.get(kind, 0)
        self.stats.failures[kind] = n + 1
        if n == 0:
            logger.warning(f"{self}: {line}")
        else:
            logger.debug(f"{self}: {line} (again, {n + 1} times)")


@dataclass
class RabelBellUserTurnStrategies(ExternalUserTurnStrategies):
    """Turn strategies for a speech-to-text with its own turn detection, with Bell's early reply added.

    Built on Pipecat's ``ExternalUserTurnStrategies``, as ``EagerUserTurnStrategies`` is, with
    :class:`RabelBellUserTurnStopStrategy` (a subclass of ``EagerUserTurnStopStrategy``) as its stop strategy: the
    service (Deepgram Flux, for one) still starts and ends the turn, its own eager end of turn still answers early when
    it is switched on, and Bell's guess at the caller's words starts the reply earlier still. Pass it as the user
    aggregator's ``user_turn_strategies``. The strategy is :attr:`bell`, for its events and its counts.

    Parameters:
        api_key: The Rabel API key; defaults to ``RABEL_API_KEY``. Not shown in this object's repr.
        agent: Your agent's name, which Bell learns for.
        min_chance: How sure Bell must be before it guesses, 0.67 to 0.99, or None for the new Bell's balanced setting.
        learn: False means this call teaches Bell nothing.
        url: The conversation endpoint or the API's base address; defaults to ``RABEL_API_URL``, then Rabel's API.
        vendor: The speech-to-text in front of the aggregator, for the API's usage reports.
        connect_timeout: Seconds allowed to open a conversation.
        answer_timeout: Seconds after its words within which a guess is still used.
        user: An id you choose for the person on the call; never a name, a number or an address.
        report: Whether each turn's end tells the API if Bell's guess was kept or dropped.
        match_policy: Decides whether the committed transcript matches an early reply. Defaults to ``NormalizedMatch``.
        speculation_timeout: Seconds an early reply may go unresolved before it is withdrawn.

    Example::

        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=RabelBellUserTurnStrategies(agent="front-desk"),
            ),
        )
    """

    api_key: str | None = field(default=None, repr=False)
    agent: str = DEFAULT_AGENT
    min_chance: float | None = DEFAULT_MIN_CHANCE
    learn: bool = True
    url: str | None = None
    vendor: str | None = None
    connect_timeout: float = 2.0
    answer_timeout: float = 0.5
    user: str | None = field(default=None, repr=False)
    report: bool = True
    match_policy: EagerMatchPolicy | None = None
    speculation_timeout: float = 5.0

    def __post_init__(self):
        super().__post_init__()
        self.stop = [
            RabelBellUserTurnStopStrategy(
                api_key=self.api_key,
                agent=self.agent,
                min_chance=self.min_chance,
                learn=self.learn,
                url=self.url,
                vendor=self.vendor,
                connect_timeout=self.connect_timeout,
                answer_timeout=self.answer_timeout,
                user=self.user,
                report=self.report,
                match_policy=self.match_policy,
                speculation_timeout=self.speculation_timeout,
            )
        ]

    @property
    def bell(self) -> RabelBellUserTurnStopStrategy:
        """Bell's stop strategy in this bundle."""
        return self.stop[0]


class _Refused(Exception):
    def __init__(self, message: dict):
        super().__init__(message.get("status"))
        self.message = message


class _WrongData(Exception):
    pass


def _backoff(attempt: int) -> float:
    return _BACKOFF[min(attempt, len(_BACKOFF) - 1)]


def _parse(raw: Any) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _short(m: Any) -> str:
    s = repr(m)
    return s if len(s) <= 80 else s[:77] + "..."


def _ms(seconds: float) -> int:
    """A head start in whole ms, from 0 to the contract's limit."""
    return max(0, min(round(seconds * 1000), _MAX_HEAD_START_MS))
