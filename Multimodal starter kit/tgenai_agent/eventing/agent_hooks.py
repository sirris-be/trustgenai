"""Bridge Agno run/tool activity into Redis-backed timeline events."""

from __future__ import annotations

# =================================== IMPORTS ==================================

# Standard Library
import base64
import inspect
import re
import time
import uuid
from typing import Any, Dict, Optional

# Third Party
# None

# Local
from .publisher import publish_event, publish_event_background
from .schema import BackendEvent, GuiEvent, GuiEventType


# ================================ HOOK FUNCTIONS ================================

def agent_pre_hook(run_input: Any, run_context: Any, agent: Any) -> None:
    """Agno pre-hook that publishes ``run.started`` to Redis."""
    try:
        input_str = (
            run_input.input_content_string()
            if hasattr(run_input, "input_content_string")
            else str(run_input)
        )
    except Exception:
        input_str = ""

    publish_event_background(
        BackendEvent.from_gui_event(
            GuiEvent(
                event_type=GuiEventType.RUN_STARTED,
                run_id=getattr(run_context, "run_id", None),
                session_id=getattr(run_context, "session_id", None),
                payload={"message": input_str},
            )
        )
    )


def agent_post_hook(run_output: Any, run_context: Any, agent: Any) -> None:
    """Agno post-hook that publishes ``run.completed`` to Redis."""
    try:
        content = (
            run_output.get_content_as_string()
            if hasattr(run_output, "get_content_as_string")
            else str(run_output)
        )
    except Exception:
        content = ""

    publish_event_background(
        BackendEvent.from_gui_event(
            GuiEvent(
                event_type=GuiEventType.RUN_COMPLETED,
                run_id=getattr(run_context, "run_id", None),
                session_id=getattr(run_context, "session_id", None),
                payload={"content": content},
            )
        )
    )


async def agent_tool_hook(name: str, func: Any, args: Dict[str, Any], run_context: Any) -> Any:
    """Agno tool-hook that publishes tool lifecycle events to Redis."""
    run_id = getattr(run_context, "run_id", None)
    session_id = getattr(run_context, "session_id", None)
    tool_call_id = str(uuid.uuid4())

    await publish_event(
        BackendEvent.from_gui_event(
            GuiEvent(
                event_type=GuiEventType.TOOL_CALL_STARTED,
                run_id=run_id,
                session_id=session_id,
                payload={"tool": name, "tool_call_id": tool_call_id, "args": _safe_repr(args)},
            )
        )
    )

    start = time.perf_counter()
    try:
        result = func(**args)
        if inspect.iscoroutine(result):
            result = await result
    except Exception as exc:
        await publish_event(
            BackendEvent.from_gui_event(
                GuiEvent(
                    event_type=GuiEventType.TOOL_CALL_ERROR,
                    run_id=run_id,
                    session_id=session_id,
                    payload={
                        "tool": name,
                        "tool_call_id": tool_call_id,
                        "error": str(exc),
                        "duration_ms": round((time.perf_counter() - start) * 1000),
                    },
                )
            )
        )
        raise

    await publish_event(
        BackendEvent.from_gui_event(
            GuiEvent(
                event_type=GuiEventType.TOOL_CALL_COMPLETED,
                run_id=run_id,
                session_id=session_id,
                payload={
                    "tool": name,
                    "tool_call_id": tool_call_id,
                    "result_preview": _safe_preview(result),
                    "images": _extract_images(result),
                    "duration_ms": round((time.perf_counter() - start) * 1000),
                },
            )
        )
    )
    return result


async def stream_run_to_events(
    agent: Any,
    message: str,
    *,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """Run the agent with full streaming and publish all events to Redis."""
    from agno.run.agent import RunEvent  # type: ignore[import-untyped]

    full_content: list[str] = []
    kwargs: Dict[str, Any] = {"stream": True, "stream_events": True}
    if session_id:
        kwargs["session_id"] = session_id
    if run_id:
        kwargs["run_id"] = run_id

    async for event in await agent.arun(message, **kwargs):  # type: ignore[arg-type]
        event_name = getattr(event, "event", "")
        event_run_id = getattr(event, "run_id", None)
        event_session_id = getattr(event, "session_id", None)

        if event_name == RunEvent.run_started.value:
            await _publish_gui_event(GuiEvent(
                GuiEventType.RUN_STARTED,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={"message": message},
            ))
        elif event_name == RunEvent.run_content.value:
            token = getattr(event, "content", "") or ""
            full_content.append(token)
            await _publish_gui_event(GuiEvent(
                GuiEventType.RUN_CONTENT,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={"token": token},
            ))
        elif event_name == RunEvent.run_completed.value:
            await _publish_gui_event(GuiEvent(
                GuiEventType.RUN_COMPLETED,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={"content": "".join(full_content)},
            ))
        elif event_name == RunEvent.run_error.value:
            await _publish_gui_event(GuiEvent(
                GuiEventType.RUN_ERROR,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={"error": str(getattr(event, "error", ""))},
            ))

    return "".join(full_content)


# ================================ HELPER FUNCTIONS ================================

async def _publish_gui_event(event: GuiEvent) -> None:
    """Publish one browser-compatible timeline event."""
    await publish_event(BackendEvent.from_gui_event(event))


def _safe_repr(value: Any, max_len: int = 500) -> Any:
    """Return a JSON-safe, truncated representation of a value."""
    if isinstance(value, dict):
        return {key: _safe_repr(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_repr(item) for item in value]
    text = str(value)
    return text if len(text) <= max_len else text[:max_len] + "..."


def _safe_preview(value: Any, max_len: int = 131000) -> str:
    """Return a human-readable text preview of a tool result."""
    try:
        text = str(value)
    except Exception:
        return "<unprintable>"
    text = re.sub(r"content=b'(?:\\'|[^'])*'", "content=<base64 image>", text)
    text = re.sub(r"content=<base64 image>(?:\\x[0-9a-fA-F]{2})+", "content=<base64 image>", text)
    return text if len(text) <= max_len else text[:max_len] + "..."


def _extract_images(result: Any) -> list[dict[str, str]]:
    """Extract base64-encoded images from an Agno result if present."""
    raw_images = getattr(result, "images", None)
    if not raw_images:
        return []
    images: list[dict[str, str]] = []
    for image in raw_images:
        data = getattr(image, "content", None)
        if isinstance(data, bytes):
            images.append({"data": base64.b64encode(data).decode(), "mime_type": "image/png"})
    return images
