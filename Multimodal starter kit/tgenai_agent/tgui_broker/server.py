"""Redis-backed TGUI broker service."""

from __future__ import annotations

# =================================== IMPORTS ==================================

# Standard Library
import asyncio
import json
import logging
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Deque, Optional, Set

# Third Party
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis

# Local
from tgenai_agent.config import (
    BACKEND_EVENT_STREAM,
    GUI_EVENT_BUFFER_SIZE,
    GUI_TOOL_DETAIL_BUFFER_SIZE,
    REDIS_URL,
    TGUI_REPLAY_COUNT,
)
from tgenai_agent.eventing.schema import BackendEvent, EventChannel


# =================================== CONSTANTS ==================================

LOGGER = logging.getLogger(__name__)
GUI_STATIC_DIR = Path(__file__).resolve().parents[2] / "gui"


# ==================================== CLASSES ====================================

class BrowserEventBroker:
    """Consume Redis backend events and fan them out to browser clients."""

    def __init__(
        self,
        redis_url: str,
        stream_name: str,
        replay_count: int,
        timeline_buffer_size: int,
        detail_buffer_size: int,
    ) -> None:
        self._redis_url = redis_url
        self._stream_name = stream_name
        self._replay_count = replay_count
        self._timeline_buffer: Deque[dict[str, Any]] = deque(maxlen=timeline_buffer_size)
        self._detail_buffer: Deque[dict[str, Any]] = deque(maxlen=detail_buffer_size)
        self._timeline_queues: Set[asyncio.Queue[Optional[dict[str, Any]]]] = set()
        self._detail_queues: Set[asyncio.Queue[Optional[dict[str, Any]]]] = set()
        self._client: Optional[Redis] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._redis_connected = False

    async def start(self) -> None:
        """Start the Redis consumer task."""
        if self._task is None:
            self._task = asyncio.create_task(self._consume_forever())

    async def stop(self) -> None:
        """Stop the Redis consumer task and close Redis."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ping(self) -> bool:
        """Return whether Redis responds to a ping."""
        try:
            self._redis_connected = bool(await self._get_client().ping())
        except Exception:
            self._redis_connected = False
        return self._redis_connected

    def health(self) -> dict[str, Any]:
        """Return broker health details."""
        return {
            "status": "ok" if self._task is not None else "starting",
            "service": "tgui-broker",
            "redis_connected": self._redis_connected,
            "stream": self._stream_name,
        }

    def timeline_replay(self) -> list[dict[str, Any]]:
        """Return buffered timeline events."""
        return list(self._timeline_buffer)

    def detail_replay(self) -> list[dict[str, Any]]:
        """Return buffered detail messages."""
        return list(self._detail_buffer)

    def subscribe_timeline(self) -> "BrowserEventSubscription":
        """Return a timeline event subscription."""
        return BrowserEventSubscription(self._timeline_queues)

    def subscribe_details(self) -> "BrowserEventSubscription":
        """Return a tool detail event subscription."""
        return BrowserEventSubscription(self._detail_queues)

    async def _consume_forever(self) -> None:
        """Continuously read backend events from Redis Streams."""
        last_id = await self._load_replay()
        while True:
            try:
                response = await self._get_client().xread(
                    {self._stream_name: last_id},
                    count=100,
                    block=5000,
                )
                self._redis_connected = True
            except asyncio.CancelledError:
                raise
            except Exception:
                self._redis_connected = False
                LOGGER.exception("Failed to read Redis stream %s", self._stream_name)
                await asyncio.sleep(2)
                continue

            for _, entries in response:
                for entry_id, fields in entries:
                    last_id = _decode(entry_id)
                    await self._dispatch(BackendEvent.from_redis_fields(fields), broadcast=True)

    async def _load_replay(self) -> str:
        """Load recent stream entries into in-memory replay buffers."""
        try:
            entries = await self._get_client().xrevrange(
                self._stream_name,
                count=self._replay_count,
            )
            self._redis_connected = True
        except Exception:
            self._redis_connected = False
            LOGGER.exception("Failed to load Redis replay from %s", self._stream_name)
            return "$"

        last_id = "$"
        for entry_id, fields in reversed(entries):
            last_id = _decode(entry_id)
            await self._dispatch(BackendEvent.from_redis_fields(fields), broadcast=False)
        return last_id

    async def _dispatch(self, event: BackendEvent, *, broadcast: bool) -> None:
        """Store and optionally broadcast one backend event."""
        if str(event.channel) == EventChannel.TOOL_DETAIL:
            await self._publish(event.payload, self._detail_buffer, self._detail_queues, broadcast)
            return
        await self._publish(event.payload, self._timeline_buffer, self._timeline_queues, broadcast)

    async def _publish(
        self,
        message: dict[str, Any],
        buffer: Deque[dict[str, Any]],
        queues: Set[asyncio.Queue[Optional[dict[str, Any]]]],
        broadcast: bool,
    ) -> None:
        """Store and optionally broadcast one browser message."""
        buffer.append(message)
        if not broadcast:
            return
        dead: Set[asyncio.Queue[Optional[dict[str, Any]]]] = set()
        for queue in queues:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                dead.add(queue)
        queues -= dead

    def _get_client(self) -> Redis:
        """Return a lazy Redis client."""
        if self._client is None:
            self._client = Redis.from_url(
                self._redis_url,
                decode_responses=False,
                socket_timeout=30,
                socket_connect_timeout=10,
                socket_keepalive=True,
            )
        return self._client


class BrowserEventSubscription:
    """Async context manager for browser event subscriptions."""

    def __init__(self, queues: Set[asyncio.Queue[Optional[dict[str, Any]]]], maxsize: int = 256) -> None:
        self._queues = queues
        self._queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue(maxsize=maxsize)

    async def __aenter__(self) -> "BrowserEventSubscription":
        self._queues.add(self._queue)
        return self

    async def __aexit__(self, *_: Any) -> None:
        self._queues.discard(self._queue)
        while not self._queue.empty():
            self._queue.get_nowait()

    def __aiter__(self) -> "BrowserEventSubscription":
        return self

    async def __anext__(self) -> dict[str, Any]:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item


# ==================================== GLOBALS ====================================

broker = BrowserEventBroker(
    redis_url=REDIS_URL,
    stream_name=BACKEND_EVENT_STREAM,
    replay_count=TGUI_REPLAY_COUNT,
    timeline_buffer_size=GUI_EVENT_BUFFER_SIZE,
    detail_buffer_size=GUI_TOOL_DETAIL_BUFFER_SIZE,
)
router = APIRouter(prefix="/gui", tags=["gui"])


# ================================== LIFESPAN ==================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop the Redis consumer with the FastAPI app."""
    await broker.start()
    yield
    await broker.stop()


# ==================================== ROUTES ====================================

@router.get("", include_in_schema=False)
async def gui_root() -> RedirectResponse:
    """Redirect the GUI root to the static index page."""
    return RedirectResponse(url="/gui/index.html")


@router.get("/events", summary="Replay buffered GUI events as JSON")
async def get_events() -> list[dict[str, Any]]:
    """Return buffered timeline events."""
    return broker.timeline_replay()


@router.get("/tool-details/events", summary="Replay buffered tool detail events as JSON")
async def get_tool_detail_events() -> list[dict[str, Any]]:
    """Return buffered tool detail messages."""
    return broker.detail_replay()


@router.websocket("/ws")
async def gui_websocket(websocket: WebSocket) -> None:
    """Stream replayed and live lightweight timeline events to the browser."""
    await websocket.accept()
    for event in broker.timeline_replay():
        if not await _send_json_message(websocket, event):
            return

    async with broker.subscribe_timeline() as subscription:
        receive_task = asyncio.create_task(_receive_loop(websocket))
        try:
            async for event in subscription:
                if not await _send_json_message(websocket, event):
                    break
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass


@router.websocket("/tool-details/ws")
async def tool_detail_websocket(websocket: WebSocket) -> None:
    """Stream replayed and live custom tool detail messages to the browser."""
    await websocket.accept()
    for message in broker.detail_replay():
        if not await _send_json_message(websocket, message):
            return

    async with broker.subscribe_details() as subscription:
        receive_task = asyncio.create_task(_receive_loop(websocket))
        try:
            async for message in subscription:
                if not await _send_json_message(websocket, message):
                    break
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass


async def health() -> dict[str, Any]:
    """Return broker service health."""
    await broker.ping()
    return broker.health()


# ================================ HELPER FUNCTIONS ================================

async def _receive_loop(websocket: WebSocket) -> None:
    """Consume browser messages and respond to websocket keep-alive pings."""
    try:
        while True:
            text = await websocket.receive_text()
            data = _parse_json_object(text)
            if data.get("type") == "ping":
                if not await _send_text(websocket, '{"type":"pong"}'):
                    return
    except (WebSocketDisconnect, Exception):
        pass


async def _send_json_message(websocket: WebSocket, message: dict[str, Any]) -> bool:
    """Safely send a JSON message to a websocket client."""
    return await _send_text(websocket, json.dumps(message))


async def _send_text(websocket: WebSocket, text: str) -> bool:
    """Send websocket text and return whether the client is still connected."""
    try:
        await websocket.send_text(text)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse websocket text into a JSON object."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"payload": {"value": text}}
    if isinstance(data, dict):
        return data
    return {"payload": {"value": data}}


def _decode(value: Any) -> str:
    """Decode Redis byte values to strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# ====================================== APP ======================================

app = FastAPI(title="FAIR Assistant TGUI Broker", lifespan=lifespan)
app.include_router(router)
app.add_api_route("/health", health, methods=["GET"], tags=["health"])

if GUI_STATIC_DIR.is_dir():
    app.mount("/gui", StaticFiles(directory=str(GUI_STATIC_DIR), html=True), name="gui_static")
