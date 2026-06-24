"""High-level knowledge retrieval and indexing service.

Typical usage (retrieval):

  service = KnowledgeRetrievalService(embedding_provider, store)
  results = await service.retrieve_relevant("How do I pick up the cube?", limit=5)

Typical usage (indexing — internal, not exposed via MCP):

  indexer = KnowledgeIndexer(embedding_provider, store)
  await indexer.index([KnowledgeUpsertItem(id="1", text="The robot arm...")])
"""


# =================================== IMPORTS ==================================

# Standard Library
import logging

# Third Party
# None

# Local
from .embeddings import EmbeddingProvider
from .models import KnowledgePiece, KnowledgeSearchResult, KnowledgeUpsertItem
from .vector_store import KnowledgeStore


# =================================== CONSTANTS ==================================

LOGGER = logging.getLogger(__name__)

_MAX_LIMIT = 50


# ==================================== CLASSES ====================================

class KnowledgeRetrievalService:
    """Orchestrates embedding creation and vector/text search queries.

    This class is designed to be used by the MCP server for search-only
    operations exposed to the LLM. Indexing is handled separately by
    :class:`KnowledgeIndexer`.

    Args:
        embedding_provider: Provider used to embed query text.
        store: Backing knowledge store.
    """

    def __init__(self, embedding_provider: EmbeddingProvider, store: KnowledgeStore) -> None:
        self._embedding_provider = embedding_provider
        self._store = store

    # ----------------------- Query Operations -----------------------

    async def retrieve_relevant(
        self,
        question: str,
        limit: int = 5,
    ) -> list[KnowledgeSearchResult]:
        """Return the most semantically relevant knowledge pieces for ``question``.

        1. Embeds ``question`` with the configured :class:`EmbeddingProvider`.
        2. Queries the vector store with the resulting vector.

        Args:
            question: The natural-language question to search for.
            limit: Maximum number of results. Clamped to ``_MAX_LIMIT``.

        Returns:
            Results sorted by descending cosine similarity.

        Raises:
            ValueError: If ``question`` is empty.
        """
        if not question or not question.strip():
            raise ValueError("question must be non-empty")

        limit = _clamp_limit(limit)
        embedding = await self._embedding_provider.embed_text(question)
        return await self._store.semantic_search(embedding, limit=limit)

    async def search_text(
        self,
        query: str,
        limit: int = 10,
    ) -> list[KnowledgeSearchResult]:
        """Return knowledge pieces that match ``query`` by text search.

        Uses simple case-insensitive substring/token matching over stored text.

        Args:
            query: Free-text search string.
            limit: Maximum number of results. Clamped to ``_MAX_LIMIT``.

        Returns:
            Results sorted by descending match score.

        Raises:
            ValueError: If ``query`` is empty.
        """
        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        limit = _clamp_limit(limit)
        return await self._store.text_search(query, limit=limit)

    async def get_all_pieces(self) -> list[KnowledgePiece]:
        """Return all stored knowledge pieces, including their embeddings.

        Delegates directly to the underlying :class:`KnowledgeStore`.  The
        returned pieces include pre-computed embedding vectors, making them
        suitable for offline visualization (e.g. t-SNE projections).

        Returns:
            All :class:`KnowledgePiece` objects in the store; empty list if the
            store has not been populated yet.
        """
        return await self._store.get_all_pieces()


class KnowledgeIndexer:
    """Embeds and upserts knowledge pieces into the store.

    This class is intentionally *not* exposed via the MCP server; it is
    meant to be used from scripts or background jobs to maintain the index.

    Args:
        embedding_provider: Provider used to embed item texts.
        store: Backing knowledge store.
    """

    def __init__(self, embedding_provider: EmbeddingProvider, store: KnowledgeStore) -> None:
        self._embedding_provider = embedding_provider
        self._store = store

    # ----------------------- Indexing Operations -----------------------

    async def index(self, items: list[KnowledgeUpsertItem]) -> None:
        """Embed and upsert ``items`` into the knowledge store.

        Items with a pre-computed ``embedding`` are upserted directly.
        Items without an embedding are embedded first using the configured
        provider, batched in a single call where possible.

        Args:
            items: Items to index; at least one must be provided.

        Raises:
            ValueError: If ``items`` is empty.
        """
        if not items:
            raise ValueError("items must be non-empty")

        needs_embedding = [item for item in items if item.embedding is None]
        if needs_embedding:
            texts = [item.text for item in needs_embedding]
            LOGGER.info("Embedding %d items via provider batch call.", len(texts))
            embeddings = await self._embedding_provider.embed_texts(texts)
            for item, vector in zip(needs_embedding, embeddings):
                item.embedding = vector

        await self._store.upsert(items)
        LOGGER.info("Indexed %d items into the knowledge store.", len(items))


# ================================ HELPER FUNCTIONS ================================

def _clamp_limit(limit: int) -> int:
    """Return ``limit`` clamped to the allowed range [1, _MAX_LIMIT]."""
    return max(1, min(limit, _MAX_LIMIT))
