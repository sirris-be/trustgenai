"""Knowledge store interface and LanceDB implementation.

Typical usage:

  store = LanceDBKnowledgeStore(db_uri="/app/data/knowledge.lancedb")
  await store.upsert([KnowledgeUpsertItem(id="1", text="...", embedding=[...])])
  results = await store.semantic_search(query_embedding, limit=5)
"""


# =================================== IMPORTS ==================================

# Standard Library
import json
import logging
from typing import Any, Optional, Protocol, runtime_checkable

# Third Party
import lancedb
import numpy as np
import pyarrow as pa

# Local
from .models import KnowledgePiece, KnowledgeSearchResult, KnowledgeUpsertItem


# =================================== CONSTANTS ==================================

LOGGER = logging.getLogger(__name__)

# LanceDB column names
_COL_ID = "id"
_COL_TEXT = "text"
_COL_SOURCE = "source"
_COL_METADATA = "metadata"
_COL_EMBEDDING = "embedding"


# ==================================== PROTOCOLS ====================================

@runtime_checkable
class KnowledgeStore(Protocol):
    """Generic interface for storing and retrieving knowledge pieces."""

    # ----------------------- Retrieval -----------------------

    async def semantic_search(
        self,
        embedding: list[float],
        limit: int,
    ) -> list[KnowledgeSearchResult]:
        """Return the ``limit`` most similar knowledge pieces to ``embedding``.

        Args:
            embedding: Query embedding vector.
            limit: Maximum number of results to return.

        Returns:
            Results sorted by descending relevance.
        """
        ...

    async def text_search(
        self,
        query: str,
        limit: int,
    ) -> list[KnowledgeSearchResult]:
        """Return up to ``limit`` knowledge pieces that contain terms from ``query``.

        Uses case-insensitive substring/token matching and ranks by the fraction
        of query terms found in the stored text.

        Args:
            query: Free-text search query.
            limit: Maximum number of results to return.

        Returns:
            Results sorted by descending match score.
        """
        ...

    async def get_all_pieces(self) -> list[KnowledgePiece]:
        """Return all knowledge pieces stored in the database, including their embeddings.

        Returns:
            A list of all :class:`KnowledgePiece` objects (may be empty if the
            table does not exist yet).
        """
        ...

    # ----------------------- Indexing -----------------------

    async def upsert(self, items: list[KnowledgeUpsertItem]) -> None:
        """Insert or overwrite knowledge pieces in the store.

        Items are matched by ``id``; existing rows are replaced, new rows are
        inserted. Each item must already have an ``embedding`` set.

        Args:
            items: Knowledge items to upsert; each must have ``embedding`` set.

        Raises:
            ValueError: If any item is missing its ``embedding``.
        """
        ...


# ==================================== CLASSES ====================================

class LanceDBKnowledgeStore:
    """Knowledge store backed by a LanceDB vector database.

    Args:
        db_uri: Path or URI to the LanceDB database directory.
        table_name: Name of the LanceDB table to use (created on first write).
        embedding_dim: Dimensionality of the embedding vectors. Must match the
            vectors produced by the configured :class:`EmbeddingProvider`.
    """

    def __init__(
        self,
        db_uri: str,
        table_name: str = "knowledge_pieces",
        embedding_dim: int = 3072,
    ) -> None:
        self._db_uri = db_uri
        self._table_name = table_name
        self._embedding_dim = embedding_dim
        # Connection and table are opened lazily to avoid failures at import time.
        self._db: Optional[lancedb.AsyncConnection] = None
        self._table: Optional[Any] = None

    # ----------------------- Retrieval -----------------------

    async def semantic_search(
        self,
        embedding: list[float],
        limit: int,
    ) -> list[KnowledgeSearchResult]:
        """Return the ``limit`` most similar knowledge pieces using vector search.

        Cosine similarity is used to rank results.

        Args:
            embedding: Query embedding vector.
            limit: Maximum number of results.

        Returns:
            Results sorted by descending cosine similarity.
        """
        table = await self._get_table()
        if table is None:
            return []

        try:
            query_vec = np.array(embedding, dtype=np.float32)
            rows = (
                await (await table.search(query_vec))
                .distance_type("cosine")
                .limit(limit)
                .to_list()
            )
        except Exception as exc:
            LOGGER.warning("LanceDB semantic_search failed: %s", exc)
            return []

        results: list[KnowledgeSearchResult] = []
        for row in rows:
            piece = _row_to_piece(row)
            # LanceDB returns _distance as 1 - cosine_similarity for cosine metric.
            distance = float(row.get("_distance", 1.0))
            similarity = max(0.0, 1.0 - distance)
            results.append(KnowledgeSearchResult(piece=piece, score=similarity))

        return results

    async def text_search(
        self,
        query: str,
        limit: int,
    ) -> list[KnowledgeSearchResult]:
        """Return up to ``limit`` pieces matching ``query`` by simple token ranking.

        Ranks rows by the fraction of lowercase query tokens found in the stored
        text.  Only rows with at least one matching token are returned.

        Args:
            query: Free-text search query.
            limit: Maximum number of results.

        Returns:
            Results sorted by descending match score.
        """
        LOGGER.info("Performing text search for query '%s'", query)
        table = await self._get_table()
        LOGGER.info("LanceDB table '%s' opened for text search: %s", self._table_name, "found" if table else "not found")
        if table is None:
            return []

        tokens = [t.lower() for t in query.split() if t]
        LOGGER.info("Extracted %d tokens from query: %s", len(tokens), tokens)
        if not tokens:
            return []


        try:
            rows = await table.query().to_list()
        except Exception as exc:
            LOGGER.warning("LanceDB text_search scan failed: %s", exc)
            return []

        scored: list[tuple[float, KnowledgePiece]] = []
        for row in rows:
            LOGGER.info("Scoring row id=%s for text search query '%s'", row.get(_COL_ID), query)
            text_lower = str(row.get(_COL_TEXT, "")).lower()
            hits = sum(1 for t in tokens if t in text_lower)
            if hits:
                score = hits / len(tokens)
                scored.append((score, _row_to_piece(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            KnowledgeSearchResult(piece=piece, score=score)
            for score, piece in scored[:limit]
        ]

    async def get_all_pieces(self) -> list[KnowledgePiece]:
        """Return all knowledge pieces stored in the database, including their embeddings.

        Returns:
            A list of all stored :class:`KnowledgePiece` objects; empty if the
            table does not exist yet or the store is empty.
        """
        table = await self._get_table()
        if table is None:
            return []

        try:
            rows = await table.query().to_list()
        except Exception as exc:
            LOGGER.warning("LanceDB get_all_pieces scan failed: %s", exc)
            return []

        return [_row_to_piece(row) for row in rows]

    # ----------------------- Indexing -----------------------

    async def upsert(self, items: list[KnowledgeUpsertItem]) -> None:
        """Insert or overwrite knowledge pieces in the LanceDB table.

        Args:
            items: Items to upsert; each must have ``embedding`` set.

        Raises:
            ValueError: If any item is missing its ``embedding``.
        """
        for item in items:
            if item.embedding is None:
                raise ValueError(f"Item '{item.id}' is missing its embedding.")

        rows = [_upsert_item_to_row(item) for item in items]
        db = await self._get_db()
        existing_table_names = await db.list_tables()

        if self._table_name not in existing_table_names.tables:
            # Create the table from the first batch.
            schema = _build_schema(self._embedding_dim)
            self._table = await db.create_table(
                self._table_name,
                data=rows,
                schema=schema,
                mode="overwrite",
            )
            LOGGER.info(
                "Created LanceDB table '%s' with %d rows.",
                self._table_name,
                len(rows),
            )
        else:
            table = await self._get_table()
            # Overwrite matching rows using merge-insert on id.
            await table.merge_insert(_COL_ID).when_matched_update_all().when_not_matched_insert_all().execute(rows)
            LOGGER.info("Upserted %d rows into '%s'.", len(rows), self._table_name)

    # ----------------------- Internal Helpers -----------------------

    async def _get_db(self) -> lancedb.AsyncConnection:
        """Return a lazily opened LanceDB connection."""
        if self._db is None:
            LOGGER.info("Opening LanceDB connection to '%s'", self._db_uri)
            self._db = await lancedb.connect_async(self._db_uri)
        return self._db

    async def _get_table(self) -> Optional[Any]:
        """Return the LanceDB table, or ``None`` if it does not exist yet."""
        db = await self._get_db()
        existing = await db.list_tables()
        if self._table_name not in existing.tables:
            LOGGER.debug(
                "LanceDB table '%s' does not exist yet — returning empty results.",
                self._table_name,
            )
            return None
        if self._table is None:
            self._table = await db.open_table(self._table_name)
        return self._table


# ================================ HELPER FUNCTIONS ================================

def _build_schema(embedding_dim: int) -> pa.Schema:
    """Return the PyArrow schema for the knowledge table."""
    return pa.schema([
        pa.field(_COL_ID, pa.string()),
        pa.field(_COL_TEXT, pa.string()),
        pa.field(_COL_SOURCE, pa.string()),
        pa.field(_COL_METADATA, pa.string()),   # JSON-encoded dict
        pa.field(_COL_EMBEDDING, pa.list_(pa.float32(), embedding_dim)),
    ])


def _row_to_piece(row: dict[str, Any]) -> KnowledgePiece:
    """Convert a LanceDB row dict to a :class:`KnowledgePiece`."""
    raw_meta = row.get(_COL_METADATA, "{}")
    try:
        metadata = json.loads(raw_meta) if raw_meta else {}
    except (json.JSONDecodeError, TypeError):
        metadata = {}

    raw_embedding = row.get(_COL_EMBEDDING)
    embedding = list(raw_embedding) if raw_embedding is not None else None

    return KnowledgePiece(
        id=str(row.get(_COL_ID, "")),
        text=str(row.get(_COL_TEXT, "")),
        source=row.get(_COL_SOURCE) or None,
        metadata=metadata,
        embedding=embedding,
    )


def _upsert_item_to_row(item: KnowledgeUpsertItem) -> dict[str, Any]:
    """Convert a :class:`KnowledgeUpsertItem` to a LanceDB-ready row dict."""
    return {
        _COL_ID: item.id,
        _COL_TEXT: item.text,
        _COL_SOURCE: item.source or "",
        _COL_METADATA: json.dumps(item.metadata),
        _COL_EMBEDDING: np.array(item.embedding, dtype=np.float32),
    }
