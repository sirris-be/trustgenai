"""FastAPI routes for the GUI timeline and optional tool detail streams."""

from __future__ import annotations

# =================================== IMPORTS ==================================

# Standard Library
import asyncio
import json
from pathlib import Path
from typing import Any

# Third Party
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

# Local
from .events import ToolDetailMessage, get_event_bus
from .tool_detail_broker import get_tool_detail_broker


# ==================================== ROUTER ====================================

router = APIRouter(prefix="/gui", tags=["gui"])

_GUI_STATIC_DIR = Path(__file__).resolve().parent.parent / "gui"


@router.get("", include_in_schema=False)
async def gui_root() -> RedirectResponse:
    """Redirect the GUI root to the static index page."""
    return RedirectResponse(url="/gui/index.html")


@router.get("/events", summary="Replay buffered GUI events as JSON")
async def get_events() -> list[dict[str, Any]]:
    """Return buffered timeline events as JSON-compatible dictionaries."""
    bus = get_event_bus()
    return [json.loads(event.to_json()) for event in bus.replay()]


@router.get("/tool-details/events", summary="Replay buffered tool detail events as JSON")
async def get_tool_detail_events() -> list[dict[str, Any]]:
    """Return buffered tool detail messages for debugging."""
    broker = get_tool_detail_broker()
    return [message.to_dict() for message in broker.replay()]


@router.websocket("/ws")
async def gui_websocket(websocket: WebSocket) -> None:
    """Stream replayed and live lightweight timeline events to the browser."""
    await websocket.accept()
    bus = get_event_bus()

    for event in bus.replay():
        await websocket.send_text(event.to_json())

    async with bus.subscribe() as subscription:
        receive_task = asyncio.create_task(_receive_loop(websocket))
        try:
            async for event in subscription:
                try:
                    await websocket.send_text(event.to_json())
                except Exception:
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
    broker = get_tool_detail_broker()
    print("Tool detail websocket listener connected")

    for message in broker.replay():
        await websocket.send_text(message.to_json())

    async with broker.subscribe() as subscription:
        receive_task = asyncio.create_task(_receive_loop(websocket))
        try:
            async for message in subscription:
                try:
                    await websocket.send_text(message.to_json())
                except Exception:
                    break
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass


@router.websocket("/tool-details/{tool_name}/publish")
async def publish_tool_detail_websocket(websocket: WebSocket, tool_name: str) -> None:
    """Receive optional GUI detail messages from a custom MCP tool server."""
    await websocket.accept()
    broker = get_tool_detail_broker()
    print(f"Tool detail websocket connected for tool: {tool_name}")
    try:
        while True:
            text = await websocket.receive_text()
            data = _parse_json_object(text)
            if data.get("type") == "ping":
                await websocket.send_text('{"type":"pong"}')
                continue
            message = ToolDetailMessage.from_dict(tool_name=tool_name, data=data)
            await broker.publish(message)
    except (WebSocketDisconnect, Exception):
        pass


# ================================ HELPER FUNCTIONS ================================

async def _receive_loop(websocket: WebSocket) -> None:
    """Consume browser messages and respond to websocket keep-alive pings."""
    try:
        while True:
            text = await websocket.receive_text()
            data = _parse_json_object(text)
            if data.get("type") == "ping":
                await websocket.send_text('{"type":"pong"}')
    except (WebSocketDisconnect, Exception):
        pass


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse websocket text into a JSON object."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"payload": {"value": text}}
    if isinstance(data, dict):
        return data
    return {"payload": {"value": data}}


# ================================ MAIN FUNCTIONS ================================

def mount_static(app: Any) -> None:
    """Mount the ``./gui`` directory under ``/gui`` on the given FastAPI app."""
    if _GUI_STATIC_DIR.is_dir():
        app.mount("/gui", StaticFiles(directory=str(_GUI_STATIC_DIR), html=True), name="gui_static")
