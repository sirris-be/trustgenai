"""
FAIR Assistant Agent
================

This agent serves as a demonstration of how to integrate the FAIR Assistant with a Camera MCP (Model Control Plane) using the Agno framework. 
The agent utilizes Google's Gemini Pro model and is designed to interact with a Camera MCP server via streaming HTTP.

"""

# =================================== IMPORTS ==================================

# Standard Library
import logging
import socket
from typing import Any
from urllib.parse import urlparse

# Third Party
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAILike
from agno.tools.mcp import MCPTools
from agno.tools.websearch import WebSearchTools

# Local
from .config import (
    AGENT_DATABASE_FILE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    MCP_SERVER_CAMERA_URL,
    MCP_SERVER_KNOWLEDGE_URL,
    MCP_SERVER_ROBOT_URL,
    MCP_SERVER_VLM_URL,
)
from .eventing.agent_hooks import agent_post_hook, agent_pre_hook, agent_tool_hook


# =================================== CONSTANTS ==================================

LOGGER = logging.getLogger(__name__)
MCP_CONNECT_TIMEOUT_SECONDS = 1.5

MCP_ENDPOINTS: list[tuple[str, str]] = [
    ("camera", MCP_SERVER_CAMERA_URL),
    ("robot", MCP_SERVER_ROBOT_URL),
    ("vlm", MCP_SERVER_VLM_URL),
    ("knowledge", MCP_SERVER_KNOWLEDGE_URL),
]

SYSTEM_PROMPT = """You are a digital assistant in an an industrial assembly environment. You are currently part of a demo setup.

If during the demo, the user asks to point to one or multiple items on the desk, you should do the following.
1) Use the VLM to get the 3D location of an item(s).
2) To point to an item, move the gripper of the Robot to the position 10 cm above the item (+ 0.10m in the z-axis), to make sure you don't hit the object. For the orientation, you should keep using 'Orientation(x=0.94, y=0.32, z=-0.06, w=-0.0096)'
"""


# ================================ HELPER FUNCTIONS ================================

def _check_tcp_endpoint(url: str) -> bool:
    """Return whether the TCP endpoint for the URL is reachable."""
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    socket.create_connection((host, port), timeout=MCP_CONNECT_TIMEOUT_SECONDS).close()
    return True


def _get_mcp_preflight_statuses() -> dict[str, bool]:
    """Run MCP endpoint preflight checks and return per-tool reachability."""
    statuses: dict[str, bool] = {}
    for tool_name, url in MCP_ENDPOINTS:
        try:
            statuses[tool_name] = _check_tcp_endpoint(url)
        except OSError as exc:
            parsed = urlparse(url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            LOGGER.error(
                "MCP preflight failed tool=%s url=%s host=%s port=%s error=%s",
                tool_name,
                url,
                host,
                port,
                exc,
            )
            statuses[tool_name] = False

    status_parts = [f"{name}={'ok' if ok else 'fail'}" for name, ok in statuses.items()]
    LOGGER.warning("MCP preflight summary: %s", ", ".join(status_parts))
    return statuses


def _build_agent_tools(mcp_preflight: dict[str, bool]) -> list[Any]:
    """Build the tool list while skipping unreachable MCP endpoints."""
    tools: list[Any] = [WebSearchTools()]

    if mcp_preflight.get("camera", False):
        tools.append(MCPTools(transport="streamable-http", url=MCP_SERVER_CAMERA_URL))
    else:
        LOGGER.warning("Skipping camera MCP toolkit because preflight failed")

    if mcp_preflight.get("robot", False):
        tools.append(MCPTools(transport="streamable-http", url=MCP_SERVER_ROBOT_URL))
    else:
        LOGGER.warning("Skipping robot MCP toolkit because preflight failed")

    if mcp_preflight.get("vlm", False):
        tools.append(MCPTools(transport="streamable-http", url=MCP_SERVER_VLM_URL))
    else:
        LOGGER.warning("Skipping VLM MCP toolkit because preflight failed")

    if mcp_preflight.get("knowledge", False):
        tools.append(MCPTools(transport="streamable-http", url=MCP_SERVER_KNOWLEDGE_URL))
    else:
        LOGGER.warning("Skipping knowledge MCP toolkit because preflight failed")

    return tools


# ===================================== AGENT =====================================

# Setting up the model
model = OpenAILike(
    id="gpt-5.4-nano",
    api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_ENDPOINT,
)

# NOTE: AgnoOS sometimes failed as soon as one MCP tool was unreachable,
# so we build the tool list dynamically with preflight checks and allow
# the agent to start with a partial toolset.
# TODO: In a production system, you might want to implement more robust
# retry logic or fallback behaviors for unreachable tools.
MCP_PREFLIGHT = _get_mcp_preflight_statuses()
AGENT_TOOLS = _build_agent_tools(MCP_PREFLIGHT)

# Create the FAIR Assistant Agent
fair_agent = Agent(
    name="Fair Assistant",
    description=SYSTEM_PROMPT,
    debug_mode=True,
    model=model,
    db=SqliteDb(db_file=AGENT_DATABASE_FILE),
    tools=AGENT_TOOLS,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=3,
    markdown=True,
    pre_hooks=[agent_pre_hook],
    post_hooks=[agent_post_hook],
    tool_hooks=[agent_tool_hook],
)
