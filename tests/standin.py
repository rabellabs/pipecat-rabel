#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A local stand-in for Bell's API (versions 2 and 3 of the contract), so the tests need no key and no network.

It speaks the conversation endpoint's messages (hello, ready, heard, guess, end_of_turn, error) and can be told to
misbehave in each way the strategy must survive: refuse with an HTTP status, refuse with an error message, be slow to
answer ``ready``, close mid-call, stall mid-call, send unreadable data, or answer late.
"""

import asyncio
import json
import socket
from dataclasses import dataclass, field

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

_REFUSALS = {
    400: "bad_message",
    401: "bad_key",
    402: "no_seconds",
    404: "unknown_agent",
    408: "no_message",
    429: "key_at_once",
    503: "waking",
    529: "out_of_capacity",
}


@dataclass
class StandIn:
    """Bell's conversation endpoint on localhost.

    Parameters:
        guesses: Words heard -> the guess Bell sends for them (at most ``guesses_a_turn`` a turn).
        guesses_a_turn: Guesses sent at most in one turn: 1, as version 2 of the contract says; the new Bell (version
            3) may send more.
        version: The version ``ready`` says: None answers the hello's (3 to a version 3 hello, else 2), 2 is an API
            without the new Bell.
        chance: The chance sent with each guess.
        refuse_http: Refuse the WebSocket upgrade with this HTTP status.
        refuse_message: Accept, read the hello, then send an error with this status and close.
        retry_after_s: The ``retry_after_s`` sent with a refusal.
        ready_delay: Seconds to wait before answering the hello with ``ready``.
        not_ready: Answer the hello with this message instead of ``ready``.
        guess_delay: Seconds to wait before sending a guess.
        raw_guesses: Words heard -> a raw message (any string) sent in place of a guess.
        close_after_heard: Close the conversation (code 1011) after this many ``heard`` messages.
        close_clean_after_turns: Close the conversation cleanly (code 1000) after this many turns.
        hold_line: The ``hold_line`` sent in ``ready``.
        refuse_report: Answer an ``end_of_turn`` that carries the report (``guess``) with an error 400 and close, as an
            API that refused the field would (none does: one that does not know it ignores it).
        ready_extra: More fields sent in ``ready`` (``hold_always``, ``hold_updates``).
        mute: After ``ready``, read every message and send nothing more: no guess, no ``end_of_turn``.
        stall_after_heard: Stop reading and answering, without closing, after this many ``heard`` messages: the API
            stalled mid-call.
    """

    guesses: dict[str, str] = field(default_factory=dict)
    guesses_a_turn: int = 1
    version: int | None = None
    chance: float = 0.86
    refuse_http: int | None = None
    refuse_message: int | None = None
    retry_after_s: float = 0
    ready_delay: float = 0.0
    not_ready: str | None = None
    guess_delay: float = 0.0
    raw_guesses: dict[str, str] = field(default_factory=dict)
    close_after_heard: int | None = None
    close_clean_after_turns: int | None = None
    hold_line: float = 0.9
    refuse_report: bool = False
    ready_extra: dict = field(default_factory=dict)
    stall_after_heard: int | None = None
    mute: bool = False

    connections: int = 0
    hellos: list[dict] = field(default_factory=list)
    headers: list[Headers] = field(default_factory=list)
    received: list[list[dict]] = field(default_factory=list)  # one list a conversation

    async def __aenter__(self) -> "StandIn":
        self._server = await serve(self._handler, "127.0.0.1", 0, process_request=self._process_request)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/v1/conversation"
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    def _process_request(self, connection, request):
        self.connections += 1
        self.headers.append(request.headers)
        if self.refuse_http is not None:
            body = json.dumps(
                {"error": _REFUSALS.get(self.refuse_http, "refused"), "message": "stand-in refusal",
                 "retry_after_s": self.retry_after_s}
            )
            return Response(self.refuse_http, "Refused", Headers({"Content-Type": "application/json"}), body.encode())
        return None

    async def _handler(self, ws):
        messages: list[dict] = []
        self.received.append(messages)
        hello = json.loads(await ws.recv())
        self.hellos.append(hello)
        if self.refuse_message is not None:
            await ws.send(json.dumps({"type": "error", "status": self.refuse_message,
                                      "error": _REFUSALS.get(self.refuse_message, "refused"),
                                      "message": "stand-in refusal", "retry_after_s": self.retry_after_s}))
            await ws.close(1008)
            return
        if self.ready_delay and await _closed_within(ws, self.ready_delay):
            return
        if self.not_ready is not None:
            await ws.send(self.not_ready)
        else:
            v = self.version if self.version is not None else (3 if hello.get("v") == 3 else 2)
            ready = {"type": "ready", "v": v, "agent": hello.get("agent", "default"),
                     "min_chance": hello.get("min_chance", None if v == 3 else 0.7), "hold_line": self.hold_line,
                     "learning": hello.get("learn", True), "calls_heard": 143}
            await ws.send(json.dumps({**ready, **self.ready_extra}))
        turn, guess_id, heard_count, turns, guessed, sent = 1, 0, 0, 0, None, 0
        async for raw in ws:
            m = json.loads(raw)
            messages.append(m)
            if self.mute:
                continue
            if m.get("type") == "heard":
                heard_count += 1
                if self.close_after_heard is not None and heard_count >= self.close_after_heard:
                    await ws.close(1011)
                    return
                if self.stall_after_heard is not None and heard_count >= self.stall_after_heard:
                    await ws.wait_closed()  # read nothing more, answer nothing, until the client or the test closes it
                    return
                text = m["text"]
                if text in self.raw_guesses:
                    await ws.send(self.raw_guesses[text])
                elif text in self.guesses and sent < self.guesses_a_turn and (
                    guessed is None or guessed[1] != self.guesses[text]
                ):
                    guess_id += 1
                    sent += 1
                    guessed = (guess_id, self.guesses[text])
                    asyncio.ensure_future(self._send_guess(ws, turn, guess_id, self.guesses[text], text))
            elif m.get("type") == "end_of_turn":
                if self.refuse_report and "guess" in m:
                    await ws.send(json.dumps({"type": "error", "status": 400, "error": "bad_message",
                                              "message": "stand-in refusal of the report"}))
                    await ws.close(1008)
                    return
                right = None
                if guessed is not None:
                    right = _norm(guessed[1]) == _norm(m["text"])
                await ws.send(json.dumps({"type": "end_of_turn", "turn": turn, "text": m["text"],
                                          "guess_id": guessed[0] if guessed else None, "right": right}))
                turn, guessed, turns, sent = turn + 1, None, turns + 1, 0
                if self.close_clean_after_turns is not None and turns >= self.close_clean_after_turns:
                    await ws.close(1000)
                    return

    async def _send_guess(self, ws, turn, guess_id, text, heard):
        if self.guess_delay:
            await asyncio.sleep(self.guess_delay)
        try:
            await ws.send(json.dumps({"type": "guess", "turn": turn, "id": guess_id, "text": text,
                                      "heard": heard, "chance": self.chance}))
        except Exception:  # noqa: BLE001 - the conversation closed first
            pass


async def _closed_within(ws, seconds: float) -> bool:
    """Waits up to ``seconds``; True if the connection closed first (the test's end closes it)."""
    try:
        await asyncio.wait_for(ws.wait_closed(), seconds)
        return True
    except (asyncio.TimeoutError, TimeoutError):
        return False


class BlackHole:
    """A TCP server that accepts and never answers: the API slow to connect."""

    async def __aenter__(self) -> "BlackHole":
        self._writers = []
        self._done = asyncio.Event()
        self._server = await asyncio.start_server(self._hold, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/v1/conversation"
        return self

    async def _hold(self, reader, writer):
        self._writers.append(writer)
        await self._done.wait()

    async def __aexit__(self, *exc):
        self._done.set()
        for w in self._writers:
            w.close()
        self._server.close()


class SilentProxy:
    """A TCP proxy in front of a stand-in that can go silent: after :meth:`silence`, the connections open through it pass
    nothing either way and stay open, as a half-open socket does. Connections opened after it pass as before."""

    def __init__(self, target_url: str):
        self._port = int(target_url.split(":")[2].split("/")[0])
        self._conns: list[dict] = []
        self._writers: list[asyncio.StreamWriter] = []
        self.connections = 0

    async def __aenter__(self) -> "SilentProxy":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/v1/conversation"
        return self

    def silence(self):
        for conn in self._conns:
            conn["silent"] = True

    async def _handle(self, reader, writer):
        self.connections += 1
        conn = {"silent": False}
        self._conns.append(conn)
        up_reader, up_writer = await asyncio.open_connection("127.0.0.1", self._port)
        self._writers += [writer, up_writer]

        async def pipe(r, w):
            try:
                while data := await r.read(65536):
                    if not conn["silent"]:
                        w.write(data)
                        await w.drain()
            except Exception:  # noqa: BLE001 - either side closed
                pass
            finally:
                w.close()

        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))

    async def __aexit__(self, *exc):
        for w in self._writers:
            w.close()
        self._server.close()


def unused_url() -> str:
    """An address nothing listens on: the API down."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"ws://127.0.0.1:{port}/v1/conversation"


def _norm(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())
