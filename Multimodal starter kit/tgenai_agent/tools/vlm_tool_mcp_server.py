#!/usr/bin/env python3
"""
VLM Robotics MCP Server
========================

A FastMCP stdio server exposing VLM-powered robotic perception tools.
Connects to the camera MCP server for image data and uses Gemini Robotics-ER
for visual reasoning.

Run as an HTTP MCP service via Docker Compose.
"""

# =================================== IMPORTS ==================================

# Standard Library
import base64
import io
import json
import logging
import os
import sys
import time
from typing import Any

# Third Party
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from PIL import Image

# Local
from tgenai_agent.eventing.publisher import publish_event
from tgenai_agent.eventing.schema import BackendEvent, ToolDetailMessage
from tgenai_agent.vlm import (
    GeminiRoboticsER,
    LabeledPoint2D,
    LabeledPoint3D,
    ObjectFindingResult,
    Vec3,
    annotate_points_on_image,
)


# ==================================== CONFIG ====================================

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
MCP_SERVER_CAMERA_URL: str = os.getenv("MCP_SERVER_CAMERA_URL", "http://127.0.0.1:5000/mcp")

VLM_MODEL_ID: str = os.getenv("VLM_MODEL_ID", "gemini-robotics-er-1.6-preview")
VLM_EVENT_SOURCE: str = "vlm-mcp-service"
DETECT_TOOL_NAME: str = "detect_object_coordinates"


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


# ================================ HELPER FUNCTIONS ================================


async def _get_rgb_image(session: ClientSession) -> bytes | None:
    """Fetch the latest RGB image from the camera MCP server."""
    result = await session.call_tool("get_rgb_image")
    for content in result.content:
        if content.type == "image":
            return base64.b64decode(content.data)
    return None


async def _pixel_to_3d(session: ClientSession, u: float, v: float) -> Vec3 | None:
    """Convert a normalized 2D pixel [u, v] to a 3D coordinate via the camera MCP server."""
    result = await session.call_tool("pixel_to_3d_point", {"u": u, "v": v})
    for content in result.content:
        if content.type == "text":
            return Vec3.model_validate_json(content.text)
    return None


def _make_step(name: str, start_time: float) -> dict[str, int | str]:
    """Build a timing step payload."""
    return {"name": name, "duration_ms": round((time.perf_counter() - start_time) * 1000)}


def _points_2d_payload(points: list[LabeledPoint2D]) -> list[dict[str, float | str]]:
    """Convert validated 2D points into GUI payload dictionaries."""
    return [
        {"label": point.label, "y": point.point[0], "x": point.point[1]}
        for point in points
    ]


def _points_3d_payload(points: list[LabeledPoint3D]) -> list[dict[str, float | str]]:
    """Convert validated 3D points into agent and GUI payload dictionaries."""
    return [
        {"label": point.label, "x": point.point.x, "y": point.point.y, "z": point.point.z}
        for point in points
    ]


def _encode_png_image(image: Image.Image) -> str:
    """Encode a PIL image into a base64 PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ==================================== MCP SERVER ==================================

mcp = FastMCP(
    "vlm-tools-server",
    host="0.0.0.0",
    port=int(os.getenv("VLM_MCP_PORT", "5002")),
)

_vlm_client = GeminiRoboticsER(api_key=GOOGLE_API_KEY, model_id=VLM_MODEL_ID)


@mcp.tool()
async def detect_object_coordinates(description: str) -> str:
    """Detect the 2D location of a specific object visible in the camera image.

    Fetches the current RGB frame from the camera, queries the VLM to locate
    the object.

    Args:
        description: Natural-language description of the object(s) to locate
                     (e.g. "the red block", "the screwdriver handle").

    Returns:
        JSON object with 'points_2d' (list of {label, x, y}) for agent
        reasoning. GUI-only image, 2D detections, and timing details are
        published through the optional GUI detail websocket.
    """
    steps: list[dict] = []

    async with ToolDetailEventPublisher(DETECT_TOOL_NAME, VLM_EVENT_SOURCE) as gui:
        await gui.publish("status", {"message": "detect_object_coordinates started"})
        await gui.publish("info", {"message": f"Finding objects with the following description: '{description}'"})

        async with streamable_http_client(MCP_SERVER_CAMERA_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                t0 = time.perf_counter()
                image_bytes = await _get_rgb_image(session)
                step = _make_step("fetch_image", t0)
                steps.append(step)
                await gui.publish("step", step, title="Fetch image")

                if image_bytes is None:
                    message = "Failed to fetch RGB image from camera"
                    await gui.publish("error", {"message": message})
                    return json.dumps({"error": message})

                await gui.publish(
                    "image",
                    {
                        "data": base64.b64encode(image_bytes).decode(),
                        "mime_type": "image/png",
                    },
                    title="RGB frame",
                )

                t0 = time.perf_counter()
                try:
                    vlm_result: ObjectFindingResult = await _vlm_client.detect_object_coordinates(
                        image_bytes=image_bytes,
                        description=description,
                    )
                except Exception as exc:
                    message = f"VLM detect_object_coordinates failed: {exc}"
                    logger.exception(message)
                    await gui.publish("error", {"message": message})
                    return json.dumps({"error": message})

                step = _make_step("vlm_inference", t0)
                steps.append(step)
                await gui.publish("step", step, title="VLM inference")

                await gui.publish("info", {"message": f"Description of the VLM result: '{vlm_result.description}'"})

                points_2d_payload = _points_2d_payload(vlm_result.objects)
                await gui.publish(
                    "detections",
                    {"points_2d": points_2d_payload, "points_3d": []},
                    title="2D detections",
                )

                # annotated = annotate_points_on_image(
                #     Image.open(io.BytesIO(image_bytes)),
                #     vlm_result.objects,
                # )
                # await gui.publish(
                #     "image",
                #     {
                #         "data": _encode_png_image(annotated),
                #         "mime_type": "image/png",
                #     },
                #     title="Annotated detections",
                # )

                await gui.publish("result", {"points_2d": points_2d_payload, "steps": steps})

                return json.dumps({"points_2d": points_2d_payload, "description": vlm_result.description})


# ================================ ENTRY POINT ================================
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
