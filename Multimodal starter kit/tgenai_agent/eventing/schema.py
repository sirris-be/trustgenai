"""Typed event contracts shared by backend services and the TGUI broker."""

from __future__ import annotations

# =================================== IMPORTS ==================================

# Standard Library
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Optional, TypedDict

# Third Party
# None

# Local
# None


# ===================================== TYPES =====================================

class EventChannel(StrEnum):
    """Logical frontend channels carried over the backend Redis stream."""

    TIMELINE = "timeline"
    TOOL_DETAIL = "tool_detail"


class EventSeverity(StrEnum):
    """Severity values for backend events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


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
    INFO = "info"
    STEP = "step"
    IMAGE = "image"
    DETECTIONS = "detections"
    PLOTLY = "plotly"
    RESULT = "result"
    ERROR = "error"


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
    images: list[dict[str, Any]]


class ToolCallErrorPayload(TypedDict, total=False):
    """Payload for a failed tool call event."""

    tool: str
    tool_call_id: str
    error: str
    duration_ms: int



# ==================================== CLASSES ====================================

@dataclass
class GuiEvent:
    """Browser-compatible timeline event."""

    event_type: str | GuiEventType
    payload: dict[str, Any]
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": str(self.event_type),
            "run_id": self.run_id,
            "session_id": self.session_id,
            "payload": json_safe(self.payload),
        }

    def to_json(self) -> str:
        """Return a JSON string that is safe to send over a websocket."""
        return json.dumps(self.to_dict())


@dataclass
class ToolDetailMessage:
    """Browser-compatible rich detail message emitted by a custom tool server."""

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
            "payload": json_safe(self.payload),
        }

    def to_json(self) -> str:
        """Return a JSON string that is safe to send over a websocket."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, tool_name: str, data: dict[str, Any]) -> "ToolDetailMessage":
        """Create a detail message from producer data."""
        message_type = data.get("message_type") or data.get("type") or ToolDetailMessageType.STATUS
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


@dataclass
class BackendEvent:
    """Redis-backed event envelope shared by backend services."""

    source: str
    type: str
    payload: dict[str, Any]
    channel: str | EventChannel
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    severity: str | EventSeverity = EventSeverity.INFO
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_redis_fields(self) -> dict[str, str]:
        """Return Redis Stream field values as strings."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "type": self.type,
            "channel": str(self.channel),
            "run_id": self.run_id or "",
            "session_id": self.session_id or "",
            "severity": str(self.severity),
            "payload": json.dumps(json_safe(self.payload)),
        }

    @classmethod
    def from_redis_fields(cls, fields: Mapping[Any, Any]) -> "BackendEvent":
        """Create a backend event from Redis Stream fields."""
        values = {_decode(key): _decode(value) for key, value in fields.items()}
        payload = _parse_payload(values.get("payload", "{}"))
        return cls(
            event_id=values.get("event_id") or str(uuid.uuid4()),
            timestamp=values.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            source=values.get("source") or "unknown",
            type=values.get("type") or "unknown",
            channel=values.get("channel") or EventChannel.TIMELINE,
            run_id=values.get("run_id") or None,
            session_id=values.get("session_id") or None,
            severity=values.get("severity") or EventSeverity.INFO,
            payload=payload,
        )

    @classmethod
    def from_gui_event(cls, event: GuiEvent, source: str = "agent-system") -> "BackendEvent":
        """Create a backend event for a browser timeline event."""
        return cls(
            source=source,
            type=str(event.event_type),
            channel=EventChannel.TIMELINE,
            run_id=event.run_id,
            session_id=event.session_id,
            payload=event.to_dict(),
        )

    @classmethod
    def from_tool_detail(
        cls,
        message: ToolDetailMessage,
        source: str = "vlm-mcp-service",
    ) -> "BackendEvent":
        """Create a backend event for a browser tool-detail message."""
        return cls(
            source=source,
            type=f"tool_detail.{message.message_type}",
            channel=EventChannel.TOOL_DETAIL,
            payload=message.to_dict(),
        )


# ================================ HELPER FUNCTIONS ================================

def json_safe(value: Any) -> Any:
    """Convert arbitrary values to JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return json_safe(value.dict())
    return str(value)


def _decode(value: Any) -> str:
    """Decode Redis byte values to strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _parse_payload(raw: str) -> dict[str, Any]:
    """Parse a JSON payload field into a dictionary."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"value": raw}
    if isinstance(payload, dict):
        return payload
    return {"value": payload}
