"""Knowledge retrieval package for the FAIR Assistant Agent."""

from .embeddings import AzureFoundryEmbeddingProvider, EmbeddingProvider, create_embedding_provider
from .models import KnowledgePiece, KnowledgeSearchResult, KnowledgeUpsertItem
from .service import KnowledgeIndexer, KnowledgeRetrievalService
from .vector_store import KnowledgeStore, LanceDBKnowledgeStore
from .visualization import EmbeddingVisualizer

__all__ = [
    "AzureFoundryEmbeddingProvider",
    "EmbeddingProvider",
    "EmbeddingVisualizer",
    "KnowledgeIndexer",
    "KnowledgePiece",
    "KnowledgeRetrievalService",
    "KnowledgeSearchResult",
    "KnowledgeStore",
    "KnowledgeUpsertItem",
    "LanceDBKnowledgeStore",
    "create_embedding_provider",
]
