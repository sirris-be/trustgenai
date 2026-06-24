"""Typed domain models for the knowledge retrieval package."""


# =================================== IMPORTS ==================================

# Standard Library
from dataclasses import dataclass, field
from typing import Any, Optional


# ================================== CLASSES ===================================

@dataclass
class KnowledgePiece:
    """A single piece of stored knowledge with its associated embedding.

    This is the on-disk/database representation of a knowledge item.
    """

    # Unique identifier within the knowledge table.
    id: str

    # The raw text content.
    text: str

    # Optional human-readable source label (e.g. document name, URL).
    source: Optional[str] = None

    # Arbitrary key/value metadata stored alongside the text.
    metadata: dict[str, Any] = field(default_factory=dict)

    # Pre-computed embedding vector; may be absent for newly created items.
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation (without the embedding)."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeUpsertItem:
    """A single item to upsert into the knowledge store.

    Supply ``embedding`` if you have already computed it; otherwise the
    :class:`KnowledgeIndexer` will embed the ``text`` before writing.
    """

    # Unique identifier; used to deduplicate / overwrite existing rows.
    id: str

    # The raw text content to store and embed.
    text: str

    # Optional human-readable source label.
    source: Optional[str] = None

    # Arbitrary key/value metadata.
    metadata: dict[str, Any] = field(default_factory=dict)

    # Pre-computed embedding; if ``None`` the indexer will embed ``text``.
    embedding: Optional[list[float]] = None


@dataclass
class KnowledgeSearchResult:
    """A knowledge piece returned by a search query, decorated with a relevance score."""

    # The matched knowledge piece.
    piece: KnowledgePiece

    # Relevance score.
    # Semantic search: cosine similarity [0, 1] — higher is more relevant.
    # Text search: simple match score [0, 1] — higher is more relevant.
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation."""
        return {
            **self.piece.to_dict(),
            "score": round(self.score, 6),
        }
