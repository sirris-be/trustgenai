#!/usr/bin/env python3
"""Camera MCP Server.

A FastMCP HTTP server exposing camera capture tools to the Agentic LLM.
The server captures frames from attached cameras, returns the image through
MCP, and publishes optional tool-detail events for the backend visualizer.

Run as an HTTP MCP service via Docker Compose, or locally with:

    python -m tgenai_agent.tools.camera_tool_mcp_server
"""


# =================================== IMPORTS ==================================

# Standard Library
import asyncio
import base64
import importlib
import io
import logging
import os
import sys
import time
from typing import Any, Optional

# Third Party
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

# Local
from tgenai_agent.config import (
    CAMERA_EVENT_SOURCE,
    CAMERA_MCP_PORT
)
from tgenai_agent.eventing.publisher import publish_event
from tgenai_agent.eventing.schema import BackendEvent, ToolDetailMessage


# ==================================== CONFIG ====================================

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

_CAMERA_SCAN_LIMIT = int(os.getenv("CAMERA_SCAN_LIMIT", "8"))


def _load_cv2() -> Any | None:
    """Import OpenCV lazily so startup still works when it is unavailable."""
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError:
        logger.exception("OpenCV is required for camera capture but is not installed")
        return None

# ==================================== CLASSES ====================================

class ToolDetailEventPublisher:
    """Publish optional GUI-only tool details through Redis events."""

    def __init__(self, tool_name: str, source: str) -> None:
        self._tool_name = tool_name
        self._source = source

    async def __aenter__(self) -> "ToolDetailEventPublisher":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def publish(
        self,
        message_type: str,
        payload: dict[str, Any],
        title: str | None = None,
    ) -> None:
        """Publish one detail message to the backend event stream."""
        message = ToolDetailMessage(
            tool_name=self._tool_name,
            message_type=message_type,
            payload=payload,
            title=title,
            source=self._source,
        )
        await publish_event(BackendEvent.from_tool_detail(message, source=self._source))


def _candidate_device_indices(camera_index: int | None) -> list[int]:
    """Return the device indices that should be probed for a camera frame."""
    if camera_index is not None and camera_index >= 0:
        return [camera_index]
    return list(range(_CAMERA_SCAN_LIMIT))


def _capture_png_from_device(camera_index: int) -> tuple[bytes, dict[str, Any]] | None:
    """Capture one PNG frame from a specific camera device index."""
    cv2 = _load_cv2()
    if cv2 is None:
        return None

    capture = cv2.VideoCapture(camera_index)
    try:
        if not capture.isOpened():
            return None

        frame = None
        for _ in range(3):
            success, candidate = capture.read()
            if success and candidate is not None:
                frame = candidate
                break

        if frame is None:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = PILImage.fromarray(rgb_frame)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        metadata = {
            "camera_index": camera_index,
            "width": int(rgb_frame.shape[1]),
            "height": int(rgb_frame.shape[0]),
        }
        return buffer.getvalue(), metadata
    finally:
        capture.release()


def _capture_from_available_camera(camera_index: int | None = None) -> tuple[bytes, dict[str, Any]] | None:
    """Capture one frame from the first available camera device."""
    for index in _candidate_device_indices(camera_index):
        captured = _capture_png_from_device(index)
        if captured is not None:
            return captured
    return None


# ==================================== MCP SERVER ====================================

mcp = FastMCP(
    "basic-camera-server",
    host="0.0.0.0",
    port=CAMERA_MCP_PORT,
)


# ==================================== MCP TOOLS ====================================

@mcp.tool()
async def get_rgb_image(camera_index: int | None = None) -> Image:
    """Get the latest RGB image from an attached camera.

    If ``camera_index`` is omitted, the server scans the configured device range
    and returns the first camera that produces a frame.
    """
    async with ToolDetailEventPublisher("get_rgb_image", CAMERA_EVENT_SOURCE) as gui:
        candidates = _candidate_device_indices(camera_index)
        await gui.publish(
            "status",
            {"message": "get_rgb_image started", "camera_index": camera_index},
            title="Camera capture",
        )
        await gui.publish(
            "info",
            {"message": "Probing camera devices", "candidates": candidates},
            title="Camera probe",
        )

        t0 = time.perf_counter()
        captured = await asyncio.to_thread(_capture_from_available_camera, camera_index)
        step = {
            "name": "capture_frame",
            "duration_ms": round((time.perf_counter() - t0) * 1000),
        }
        await gui.publish("step", step, title="Capture frame")

        if captured is None:
            message = "No attached camera could be opened or no frame was captured"
            logger.error(message)
            await gui.publish("error", {"message": message, "candidates": candidates}, title="Camera error")
            return None

        png_bytes, metadata = captured
        await gui.publish(
            "image",
            {
                "data": base64.b64encode(png_bytes).decode(),
                "mime_type": "image/png",
                "camera_index": metadata["camera_index"],
            },
            title="RGB frame",
        )
        await gui.publish(
            "result",
            {
                "camera_index": metadata["camera_index"],
                "width": metadata["width"],
                "height": metadata["height"],
            },
            title="Capture result",
        )

        return Image(data=png_bytes, format="png")


# ================================ ENTRY POINT ================================
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
