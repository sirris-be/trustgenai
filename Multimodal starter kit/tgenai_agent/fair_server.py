"""
FAIR Assistant Runtime Server
================

This server serves as the runtime environment for the FAIR Assistant Agent, which is designed to interact with a Camera MCP (Model Control Plane) using the Agno framework. 
The server utilizes Google's Gemini Pro model and is set up to handle streaming HTTP requests from the Camera MCP.

"""

# ======================== IMPORTS ========================

# Standard Library
from typing import Any

# Third Party
from agno.os import AgentOS
from agno.db.sqlite import SqliteDb

# Local
from .agent import fair_agent
from .config import SERVER_DATABASE_FILE


# ===================== RUNTIME SERVER ======================

# Setting up the model
agent_os = AgentOS(
    agents=[fair_agent],
    tracing=True,
    db=SqliteDb(db_file=SERVER_DATABASE_FILE),  # separate DB for this agent
)

# Get the FastAPI app from AgentOS to serve the agent
app = agent_os.get_app()


@app.get("/health", tags=["health"])
async def health() -> dict[str, Any]:
    """Return agent service health."""
    return {"status": "ok", "service": "agent-system"}
