"""GUI event contracts and async publish/subscribe buses.

The main ``GuiEvent`` stream is intentionally lightweight.  It is used by
the browser timeline and mirrors coarse Agno run/tool activity.  Rich data
from custom tools travels through ``ToolDetailMessage`` on a separate bus so
MCP tool results can stay simple and agent-facing.
"""

from __future__ import annotations

# =================================== IMPORTS ==================================

# Standard Library
import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Deque, Optional, Set, TypedDict

# Third Party
# None

# Local
# None


# ===================================== TYPES =====================================


class GuiEventType(StrEnum):
    """Timeline event types sent on ``/gui/ws``."""

    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_ERROR = "run.error"
    RUN_CONTENT = "run.content"
    TOOL_CALL_STARTED = "tool.call_started"
    TOOL_CALL_COMPLETED = "tool.call_completed"
    TOOL_CALL_ERROR = "tool.call_error"
    MODEL_REQUEST_STARTED = "model.request_started"
    MODEL_REQUEST_COMPLETED = "model.request_completed"


class ToolDetailMessageType(StrEnum):
    """Tool detail message types sent on ``/gui/tool-details/ws``."""

    STATUS = "status"
    STEP = "step"
    IMAGE = "image"
    DETECTIONS = "detections"
    RESULT = "result"
    ERROR = "error"


class RunStartedPayload(TypedDict):
    """Payload for a run start event."""

    message: str


class RunCompletedPayload(TypedDict):
    """Payload for a completed run event."""

    content: str


class RunErrorPayload(TypedDict):
    """Payload for a failed run event."""

    error: str


class RunContentPayload(TypedDict):
    """Payload for a streamed content chunk."""

    token: str


class ToolCallStartedPayload(TypedDict, total=False):
    """Payload for a tool call start event."""

    tool: str
    tool_call_id: str
    args: Any


class ToolCallCompletedPayload(TypedDict, total=False):
    """Payload for a completed tool call event."""

    tool: str
    tool_call_id: str
    result_preview: str
    duration_ms: int


class ToolCallErrorPayload(TypedDict, total=False):
    """Payload for a failed tool call event."""

    tool: str
    tool_call_id: str
    error: str
    duration_ms: int


@dataclass
class GuiEvent:
    """Stable event envelope sent to every connected browser client."""

    event_type: str | GuiEventType
    payload: dict[str, Any]
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Return a JSON string that is safe to send over a websocket."""
        return json.dumps(
            {
                "id": self.id,
                "timestamp": self.timestamp,
                "event_type": str(self.event_type),
                "run_id": self.run_id,
                "session_id": self.session_id,
                "payload": _json_safe(self.payload),
            }
        )


@dataclass
class ToolDetailMessage:
    """Rich, optional detail message emitted by a custom tool server."""

    tool_name: str
    message_type: str | ToolDetailMessageType
    payload: dict[str, Any]
    title: Optional[str] = None
    source: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "message_type": str(self.message_type),
            "title": self.title,
            "source": self.source,
            "payload": _json_safe(self.payload),
        }

    def to_json(self) -> str:
        """Return a JSON string that is safe to send over a websocket."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, tool_name: str, data: dict[str, Any]) -> "ToolDetailMessage":
        """Create a detail message from producer websocket data."""
        message_type = (
            data.get("message_type")
            or data.get("type")
            or ToolDetailMessageType.STATUS
        )
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            payload = {"value": payload}
        return cls(
            tool_name=tool_name,
            message_type=str(message_type),
            payload=payload,
            title=data.get("title"),
            source=data.get("source"),
        )


# ==================================== CLASSES ====================================


class EventBus:
    """
    Async publish/subscribe bus with a bounded replay buffer.

    Usage::

        bus = EventBus(buffer_size=200)
        await bus.publish(GuiEvent("run.started", payload={...}))

        async for event in bus.subscribe():
            ...  # yields GuiEvent; exit the loop to unsubscribe
    """

    def __init__(self, buffer_size: int = 200) -> None:
        self._buffer: Deque[GuiEvent] = deque(maxlen=buffer_size)
        self._queues: Set[asyncio.Queue[Optional[GuiEvent]]] = set()

    async def publish(self, event: GuiEvent) -> None:
        """Publish an event to all current subscribers and the replay buffer."""
        self._buffer.append(event)
        dead: Set[asyncio.Queue[Optional[GuiEvent]]] = set()
        for queue in self._queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead.add(queue)
        self._queues -= dead

    def replay(self) -> list[GuiEvent]:
        """Return buffered events for replaying to a newly connected client."""
        return list(self._buffer)

    def subscribe(self) -> "_Subscription":
        """Return an async context manager / async iterator for receiving events."""
        return _Subscription(self)


class _Subscription:
    """Async context manager that wraps a per-subscriber asyncio.Queue."""

    def __init__(self, bus: EventBus, maxsize: int = 256) -> None:
        self._bus = bus
        self._queue: asyncio.Queue[Optional[GuiEvent]] = asyncio.Queue(maxsize=maxsize)

    async def __aenter__(self) -> "_Subscription":
        self._bus._queues.add(self._queue)
        return self

    async def __aexit__(self, *_: Any) -> None:
        self._bus._queues.discard(self._queue)
        # Drain so the producer is never blocked
        while not self._queue.empty():
            self._queue.get_nowait()

    def __aiter__(self) -> "_Subscription":
        return self

    async def __anext__(self) -> GuiEvent:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item


# =============================== HELPER FUNCTIONS ================================


def _json_safe(value: Any) -> Any:
    """Convert arbitrary values to JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return _json_safe(value.dict())
    return str(value)


# ==================================== GLOBALS ====================================

_default_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Return the module-level EventBus, creating it on first call."""
    global _default_bus
    if _default_bus is None:
        from tgenai_agent.config import GUI_EVENT_BUFFER_SIZE  # imported lazily to avoid circular deps
        _default_bus = EventBus(buffer_size=GUI_EVENT_BUFFER_SIZE)
    return _default_bus
