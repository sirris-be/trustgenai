#!/usr/bin/env python3
"""Knowledge Retrieval MCP Server.

A FastMCP HTTP server exposing knowledge search tools to the Agentic LLM.
Provides semantic (embedding-based) search and simple text search over a
LanceDB knowledge store.

Run as an HTTP MCP service via Docker Compose, or locally with:

  python -m fair_agent.tools.knowledge_tool_mcp_server
"""


# =================================== IMPORTS ==================================

# Standard Library
import asyncio
import json
import logging
import sys
import time
from typing import Any, Optional

# Third Party
from mcp.server.fastmcp import FastMCP

# Local
from tgenai_agent.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    KNOWLEDGE_EMBEDDING_MODEL,
    KNOWLEDGE_EMBEDDING_PROVIDER,
    KNOWLEDGE_EVENT_SOURCE,
    KNOWLEDGE_MCP_PORT,
    KNOWLEDGE_RETRIEVE_TOOL_NAME,
    KNOWLEDGE_TABLE_NAME,
    KNOWLEDGE_TEXT_SEARCH_TOOL_NAME,
    KNOWLEDGE_VECTOR_DB_URI,
)
from tgenai_agent.eventing.publisher import publish_event
from tgenai_agent.eventing.schema import BackendEvent, ToolDetailMessage


# ==================================== CONFIG ====================================

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

RETRIEVE_TOOL_NAME: str = KNOWLEDGE_RETRIEVE_TOOL_NAME
TEXT_SEARCH_TOOL_NAME: str = KNOWLEDGE_TEXT_SEARCH_TOOL_NAME


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


# ==================================== MCP SERVER ====================================

mcp = FastMCP(
    "knowledge-tools-server",
    host="0.0.0.0",
    port=KNOWLEDGE_MCP_PORT,
)

# Service instances are created lazily on first tool call so that a missing
# or misconfigured DB / API key does not crash the server at startup.
_retrieval_service = None

# Lazily fitted visualizer — None until the first successful fit attempt.
_visualizer: Optional[Any] = None
_visualizer_fitted: bool = False


def _get_retrieval_service():
    """Return the lazily initialised :class:`KnowledgeRetrievalService`."""
    global _retrieval_service
    if _retrieval_service is None:
        # Import here to avoid circular imports and to keep startup cheap.
        from tgenai_agent.knowledge.embeddings import create_embedding_provider
        from tgenai_agent.knowledge.service import KnowledgeRetrievalService
        from tgenai_agent.knowledge.vector_store import LanceDBKnowledgeStore

        embedding_provider = create_embedding_provider(
            KNOWLEDGE_EMBEDDING_PROVIDER,
            api_key=AZURE_OPENAI_API_KEY,
            endpoint=AZURE_OPENAI_ENDPOINT,
            model=KNOWLEDGE_EMBEDDING_MODEL,
        )
        store = LanceDBKnowledgeStore(
            db_uri=KNOWLEDGE_VECTOR_DB_URI,
            table_name=KNOWLEDGE_TABLE_NAME,
        )
        _retrieval_service = KnowledgeRetrievalService(
            embedding_provider=embedding_provider,
            store=store,
        )
        logger.info(
            "Initialised KnowledgeRetrievalService provider=%s model=%s db=%s table=%s",
            KNOWLEDGE_EMBEDDING_PROVIDER,
            KNOWLEDGE_EMBEDDING_MODEL,
            KNOWLEDGE_VECTOR_DB_URI,
            KNOWLEDGE_TABLE_NAME,
        )
    return _retrieval_service


async def _get_visualizer() -> Optional[Any]:
    """Return a lazily fitted EmbeddingVisualizer, or None on failure or empty store.

    The visualizer is fitted once (on first call) using all pieces currently in
    the knowledge store and then cached for the lifetime of the server process.
    Fitting is offloaded to a thread pool via ``asyncio.to_thread`` so it does
    not block the async event loop during the potentially multi-second t-SNE fit.
    """
    global _visualizer, _visualizer_fitted
    if _visualizer_fitted:
        return _visualizer

    # Lazy import keeps startup cheap and avoids circular-import issues.
    from tgenai_agent.knowledge.visualization import EmbeddingVisualizer

    service = _get_retrieval_service()
    try:
        pieces = await service.get_all_pieces()
    except Exception as exc:
        logger.warning("Failed to load knowledge pieces for visualizer: %s", exc)
        _visualizer_fitted = True
        return None

    if not pieces:
        logger.warning(
            "No knowledge pieces found; embedding visualizer will be disabled."
        )
        _visualizer_fitted = True
        return None

    try:
        viz = EmbeddingVisualizer()
        await asyncio.to_thread(viz.fit, pieces)
        _visualizer = viz
        logger.info("EmbeddingVisualizer fitted on %d pieces.", len(pieces))
    except Exception as exc:
        logger.warning("EmbeddingVisualizer fit failed: %s", exc)

    _visualizer_fitted = True
    return _visualizer


# ================================ HELPER FUNCTIONS ================================

def _make_step(name: str, start_time: float) -> dict[str, int | str]:
    """Build a timing step payload."""
    return {"name": name, "duration_ms": round((time.perf_counter() - start_time) * 1000)}


def _results_payload(results: list) -> list[dict[str, Any]]:
    """Convert search results to a JSON-safe list of dicts."""
    return [r.to_dict() for r in results]


# ==================================== MCP TOOLS ====================================

@mcp.tool()
async def retrieve_relevant_knowledge(description: str, limit: int = 5) -> str:
    """Retrieve the most relevant knowledge pieces for a given question or topic.

    Uses semantic (embedding-based) similarity search over the knowledge base.
    The results are ranked by relevance score (cosine similarity, 0–1; higher is better).

    Args:
        description: The question or topic to search for (e.g. "How do I pick up the red cube?").
        limit: Maximum number of results to return (1–50, default 5).

    Returns:
        JSON object with 'results' — a list of {id, text, source, metadata, score} entries.
    """
    steps: list[dict] = []

    async with ToolDetailEventPublisher(RETRIEVE_TOOL_NAME, KNOWLEDGE_EVENT_SOURCE) as gui:
        await gui.publish("status", {"message": "retrieve_relevant_knowledge started"})
        await gui.publish("info", {"message": f"Searching for: '{description}' (limit={limit})"})

        service = _get_retrieval_service()

        # Step 1: create embedding
        t0 = time.perf_counter()
        try:
            embedding = await service._embedding_provider.embed_text(description)
        except Exception as exc:
            message = f"Embedding creation failed: {exc}"
            logger.exception(message)
            await gui.publish("error", {"message": message})
            return json.dumps({"error": message})

        step = _make_step("create_embedding", t0)
        steps.append(step)
        await gui.publish("step", step, title="Create embedding")

        # Step 2: vector search
        t0 = time.perf_counter()
        try:
            results = await service._store.semantic_search(embedding, limit=limit)
        except Exception as exc:
            message = f"Vector search failed: {exc}"
            logger.exception(message)
            await gui.publish("error", {"message": message})
            return json.dumps({"error": message})

        step = _make_step("vector_search", t0)
        steps.append(step)
        await gui.publish("step", step, title="Vector search")

        # Step 3: build embedding-space visualization
        t0 = time.perf_counter()
        visualizer = await _get_visualizer()
        if visualizer is not None:
            try:
                fig = visualizer.plot_query_similarity(description, embedding)
                await gui.publish(
                    "plotly",
                    {"figure": json.loads(fig.to_json())},
                    title="Embedding space",
                )
            except Exception as exc:
                logger.warning("Failed to publish embedding visualization: %s", exc)

        step = _make_step("embedding_visualization", t0)
        steps.append(step)
        await gui.publish("step", step, title="Embedding visualization")

        results_payload = _results_payload(results)
        await gui.publish(
            "result",
            {"results": results_payload, "count": len(results), "steps": steps},
            title=f"{len(results)} knowledge pieces found",
        )

        logger.info("retrieve_relevant_knowledge: %d results for '%s'", len(results), description)
        return json.dumps({"results": results_payload})


@mcp.tool()
async def search_knowledge_text(query: str, limit: int = 10) -> str:
    """Search the knowledge base using simple text matching.

    Ranks stored knowledge pieces by the fraction of query tokens found in the
    text (case-insensitive substring matching). Useful for keyword-based look-up
    when semantic search is not required.

    Args:
        query: One or more search terms (e.g. "robot arm gripper").
        limit: Maximum number of results to return (1–50, default 10).

    Returns:
        JSON object with 'results' — a list of {id, text, source, metadata, score} entries.
    """
    steps: list[dict] = []

    async with ToolDetailEventPublisher(TEXT_SEARCH_TOOL_NAME, KNOWLEDGE_EVENT_SOURCE) as gui:
        await gui.publish("status", {"message": "search_knowledge_text started"})
        await gui.publish("info", {"message": f"Text search for: '{query}' (limit={limit})"})

        service = _get_retrieval_service()

        t0 = time.perf_counter()
        try:
            results = await service._store.text_search(query, limit=limit)
        except Exception as exc:
            message = f"Text search failed: {exc}"
            logger.exception(message)
            await gui.publish("error", {"message": message})
            return json.dumps({"error": message})

        step = _make_step("text_search", t0)
        steps.append(step)
        await gui.publish("step", step, title="Text search")

        results_payload = _results_payload(results)
        await gui.publish(
            "result",
            {"results": results_payload, "count": len(results), "steps": steps},
            title=f"{len(results)} knowledge pieces found",
        )

        logger.info("search_knowledge_text: %d results for '%s'", len(results), query)
        return json.dumps({"results": results_payload})


# ================================ ENTRY POINT ================================
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
