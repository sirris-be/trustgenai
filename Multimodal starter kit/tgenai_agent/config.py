"""
FAIR Assistant Configuration
================

This module loads all environment variables used across the FAIR Assistant application.

"""

# ======================== IMPORTS ========================

# Standard Library
import os

# ========================= CONFIG =========================

def _get_bool(name: str, default: bool) -> bool:
	"""Read a boolean environment variable."""
	default_value = "true" if default else "false"
	return os.getenv(name, default_value).lower() not in ("false", "0", "no")


# Cloud API Keys and Endpoints
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "...")  # Set your Google API key here or via environment variable
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "...")  # Set your Azure API key here or via environment variable
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "...")  # Set your Azure endpoint here or via environment variable

# Database file paths for the agent and server
AGENT_DATABASE_FILE: str = os.getenv("AGENT_DATABASE_FILE", "fair_agent.db")  # SQLite database file for session storage
SERVER_DATABASE_FILE: str = os.getenv("SERVER_DATABASE_FILE", "fair_server.db")  # SQLite database file for server session storage

# Port configuration
AGENT_PORT: int = int(os.getenv("AGENT_PORT", "6000"))
CAMERA_MCP_PORT: int = int(os.getenv("CAMERA_MCP_PORT", "5000"))
ROBOT_MCP_PORT: int = int(os.getenv("ROBOT_MCP_PORT", "5001"))
VLM_MCP_PORT: int = int(os.getenv("VLM_MCP_PORT", "5002"))
KNOWLEDGE_MCP_PORT: int = int(os.getenv("KNOWLEDGE_MCP_PORT", "5003"))
TGUI_PORT: int = int(os.getenv("TGUI_PORT", "3001"))

# Redis event backbone configuration
REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")
BACKEND_EVENT_STREAM: str = os.getenv("BACKEND_EVENT_STREAM", "stream:backend-events")
REDIS_STREAM_MAXLEN: int = int(os.getenv("REDIS_STREAM_MAXLEN", "1000"))
TGUI_REPLAY_COUNT: int = int(os.getenv("TGUI_REPLAY_COUNT", "200"))
BACKEND_EVENTS_ENABLED: bool = _get_bool("BACKEND_EVENTS_ENABLED", True)

# GUI configuration
GUI_ENABLED: bool = _get_bool("GUI_ENABLED", True)
GUI_EVENT_BUFFER_SIZE: int = int(os.getenv("GUI_EVENT_BUFFER_SIZE", "200"))
GUI_TOOL_DETAILS_ENABLED: bool = _get_bool("GUI_TOOL_DETAILS_ENABLED", True)
GUI_TOOL_DETAIL_BUFFER_SIZE: int = int(os.getenv("GUI_TOOL_DETAIL_BUFFER_SIZE", "200"))
GUI_TOOL_DETAILS_BASE_WS_URL: str = os.getenv("GUI_TOOL_DETAILS_BASE_WS_URL", f"ws://127.0.0.1:{AGENT_PORT}/gui/tool-details")

# URL for the MCP servers (Model Context Protocol) that the FAIR Assistant will interact with
MCP_SERVER_CAMERA_URL: str = os.getenv("MCP_SERVER_CAMERA_URL", f"http://127.0.0.1:{CAMERA_MCP_PORT}/mcp")
MCP_SERVER_ROBOT_URL: str = os.getenv("MCP_SERVER_ROBOT_URL", f"http://127.0.0.1:{ROBOT_MCP_PORT}/mcp")
MCP_SERVER_VLM_URL: str = os.getenv("MCP_SERVER_VLM_URL", f"http://127.0.0.1:{VLM_MCP_PORT}/mcp")
MCP_SERVER_KNOWLEDGE_URL: str = os.getenv("MCP_SERVER_KNOWLEDGE_URL", f"http://127.0.0.1:{KNOWLEDGE_MCP_PORT}/mcp")
VLM_MODEL_ID: str = os.getenv("VLM_MODEL_ID", "gemini-robotics-er-1.6-preview")

# Camera tool configuration
CAMERA_EVENT_SOURCE: str = os.getenv("CAMERA_EVENT_SOURCE", "camera-mcp-service")

# Knowledge retrieval configuration
KNOWLEDGE_EMBEDDING_PROVIDER: str = os.getenv("KNOWLEDGE_EMBEDDING_PROVIDER", "azure_foundry")
KNOWLEDGE_EMBEDDING_MODEL: str = os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "text-embedding-3-large")
KNOWLEDGE_VECTOR_DB_URI: str = os.getenv("KNOWLEDGE_VECTOR_DB_URI", "knowledge.lancedb")
KNOWLEDGE_TABLE_NAME: str = os.getenv("KNOWLEDGE_TABLE_NAME", "knowledge_pieces")
KNOWLEDGE_EVENT_SOURCE: str = os.getenv("KNOWLEDGE_EVENT_SOURCE", "knowledge-mcp-service")
KNOWLEDGE_RETRIEVE_TOOL_NAME: str = os.getenv("KNOWLEDGE_RETRIEVE_TOOL_NAME", "retrieve_relevant_knowledge")
KNOWLEDGE_TEXT_SEARCH_TOOL_NAME: str = os.getenv("KNOWLEDGE_TEXT_SEARCH_TOOL_NAME", "search_knowledge_text")
