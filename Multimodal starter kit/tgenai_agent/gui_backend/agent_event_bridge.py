"""Bridge Agno run/tool activity into lightweight GUI timeline events."""

from __future__ import annotations

# =================================== IMPORTS ==================================

# Standard Library
import asyncio
import base64
import time
import uuid
from typing import Any, Dict, Optional

# Third Party
# None

# Local
from .events import GuiEvent, GuiEventType, get_event_bus


# ================================ HELPER FUNCTIONS ================================

def _publish_sync(event: GuiEvent) -> None:
    """Fire-and-forget: schedule publish on the running event loop if present."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(get_event_bus().publish(event))
        else:
            loop.run_until_complete(get_event_bus().publish(event))
    except RuntimeError:
        pass


# ================================ HOOK FUNCTIONS ================================

def gui_pre_hook(run_input: Any, run_context: Any, agent: Any) -> None:
    """
    Agno pre_hook: publishes ``run.started`` to the GUI event bus.

    Parameters are injected by Agno based on parameter names.
    """
    try:
        input_str = (
            run_input.input_content_string()
            if hasattr(run_input, "input_content_string")
            else str(run_input)
        )
    except Exception:
        input_str = ""

    event = GuiEvent(
        event_type=GuiEventType.RUN_STARTED,
        run_id=getattr(run_context, "run_id", None),
        session_id=getattr(run_context, "session_id", None),
        payload={"message": input_str},
    )
    _publish_sync(event)


def gui_post_hook(run_output: Any, run_context: Any, agent: Any) -> None:
    """
    Agno post_hook: publishes ``run.completed`` to the GUI event bus.
    """
    try:
        content = (
            run_output.get_content_as_string()
            if hasattr(run_output, "get_content_as_string")
            else str(run_output)
        )
    except Exception:
        content = ""

    event = GuiEvent(
        event_type=GuiEventType.RUN_COMPLETED,
        run_id=getattr(run_context, "run_id", None),
        session_id=getattr(run_context, "session_id", None),
        payload={"content": content},
    )
    _publish_sync(event)


async def gui_tool_hook(name: str, func: Any, args: Dict[str, Any], run_context: Any) -> Any:
    """
    Agno tool_hook: wraps tool execution with minimal timeline events.

    Defined as async so it works correctly for async tools (which is the common
    case for MCP-based tools). Agno's async execution chain will await this hook;
    the sync chain skips async hooks with a warning (acceptable — sync tools are
    rare in this project).

    ``func`` here is ``next_func`` — calling (and awaiting) it continues the hook
    chain and ultimately invokes the actual tool.
    """
    import inspect

    run_id = getattr(run_context, "run_id", None)
    session_id = getattr(run_context, "session_id", None)
    tool_call_id = str(uuid.uuid4())

    _publish_sync(
        GuiEvent(
            event_type=GuiEventType.TOOL_CALL_STARTED,
            run_id=run_id,
            session_id=session_id,
            payload={"tool": name, "tool_call_id": tool_call_id, "args": _safe_repr(args)},
        )
    )

    start = time.perf_counter()
    result: Any = None

    try:
        result = func(**args)
        if inspect.iscoroutine(result):
            result = await result
    except Exception as exc:
        _publish_sync(
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
        raise

    duration_ms = round((time.perf_counter() - start) * 1000)
    _publish_sync(
        GuiEvent(
            event_type=GuiEventType.TOOL_CALL_COMPLETED,
            run_id=run_id,
            session_id=session_id,
            payload={
                "tool": name,
                "tool_call_id": tool_call_id,
                "result_preview": _safe_preview(result),
                "images": _extract_images(result),
                "duration_ms": duration_ms,
            },
        )
    )

    return result


# =============================== STREAMING BRIDGE ================================

async def stream_run_to_gui(
    agent: Any,
    message: str,
    *,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """
    Run the agent with full streaming and publish all events to the GUI bus.

    Returns the final response text.
    """
    from agno.run.agent import RunEvent  # type: ignore[import-untyped]

    bus = get_event_bus()
    full_content = []

    kwargs: Dict[str, Any] = {"stream": True, "stream_events": True}
    if session_id:
        kwargs["session_id"] = session_id
    if run_id:
        kwargs["run_id"] = run_id

    async for event in await agent.arun(message, **kwargs):  # type: ignore[arg-type]
        event_name: str = getattr(event, "event", "")
        event_run_id = getattr(event, "run_id", None)
        event_session_id = getattr(event, "session_id", None)

        if event_name == RunEvent.run_started.value:
            await bus.publish(GuiEvent(
                GuiEventType.RUN_STARTED,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={"message": message},
            ))

        elif event_name == RunEvent.run_content.value:
            token = getattr(event, "content", "") or ""
            full_content.append(token)
            await bus.publish(GuiEvent(
                GuiEventType.RUN_CONTENT,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={"token": token},
            ))

        elif event_name == RunEvent.run_completed.value:
            await bus.publish(GuiEvent(
                GuiEventType.RUN_COMPLETED,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={"content": "".join(full_content)},
            ))

        elif event_name == RunEvent.run_error.value:
            await bus.publish(GuiEvent(
                GuiEventType.RUN_ERROR,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={"error": str(getattr(event, "error", ""))},
            ))

        elif event_name == RunEvent.tool_call_started.value:
            tool_name = getattr(event, "tool_name", "") or getattr(event, "function_name", "") or ""
            tool_call_id = _extract_tool_call_id(event)
            await bus.publish(GuiEvent(
                GuiEventType.TOOL_CALL_STARTED,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={"tool": tool_name, "tool_call_id": tool_call_id},
            ))

        elif event_name == RunEvent.tool_call_completed.value:
            tool_name = getattr(event, "tool_name", "") or getattr(event, "function_name", "") or ""
            tool_call_id = _extract_tool_call_id(event)
            await bus.publish(GuiEvent(
                GuiEventType.TOOL_CALL_COMPLETED,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={
                    "tool": tool_name,
                    "tool_call_id": tool_call_id,
                    "result_preview": _safe_preview(getattr(event, "tool_call_content", "")),
                },
            ))

        elif event_name == RunEvent.tool_call_error.value:
            tool_name = getattr(event, "tool_name", "") or getattr(event, "function_name", "") or ""
            tool_call_id = _extract_tool_call_id(event)
            await bus.publish(GuiEvent(
                GuiEventType.TOOL_CALL_ERROR,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={
                    "tool": tool_name,
                    "tool_call_id": tool_call_id,
                    "error": str(getattr(event, "error", "")),
                },
            ))

        elif event_name == RunEvent.model_request_started.value:
            await bus.publish(GuiEvent(
                GuiEventType.MODEL_REQUEST_STARTED,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={},
            ))

        elif event_name == RunEvent.model_request_completed.value:
            await bus.publish(GuiEvent(
                GuiEventType.MODEL_REQUEST_COMPLETED,
                run_id=event_run_id,
                session_id=event_session_id,
                payload={},
            ))

    return "".join(full_content)


# ================================ PRIVATE HELPERS ================================

def _safe_repr(value: Any, max_len: int = 500) -> Any:
    """Return a JSON-safe, truncated representation of a value."""
    if isinstance(value, dict):
        return {k: _safe_repr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_repr(v) for v in value]
    text = str(value)
    return text if len(text) <= max_len else text[:max_len] + "..."


def _extract_tool_call_id(event: Any) -> str:
    """Return a stable tool-call identifier when the runtime provides one."""
    for attr in ("tool_call_id", "function_call_id", "call_id", "id"):
        value = getattr(event, attr, None)
        if value:
            return str(value)
    return str(uuid.uuid4())


def _safe_preview(value: Any, max_len: int = 131000) -> str: # limit to 128Kb Messages
    """Return a short human-readable text preview of a tool result."""
    import re
    try:
        text = str(value)
    except Exception:
        return "<unprintable>"
    # Replace raw base64/binary image blobs with a short placeholder so the
    # rest of the message stays readable.
    text = re.sub(r"content=b'(?:\\'|[^'])*'", "content=<base64 image>", text)
    text = re.sub(r"content=<base64 image>(?:\\x[0-9a-fA-F]{2})+", "content=<base64 image>", text)
    return text if len(text) <= max_len else text[:max_len] + "..."


def _extract_images(result: Any) -> list[dict]:
    """Extract base64-encoded images from an Agno result if present."""
    raw_images = getattr(result, "images", None)
    if not raw_images:
        return []
    images = []
    for img in raw_images:
        data = getattr(img, "content", None)
        if isinstance(data, bytes):
            images.append({"data": base64.b64encode(data).decode(), "mime_type": "image/png"})
    return images
