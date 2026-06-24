"""RAG embedding visualization using t-SNE and Plotly.

Provides a 2D projection of knowledge-piece embeddings stored in the vector
database, and supports overlaying cosine-similarity heat maps for arbitrary
queries.

Typical usage:

  visualizer = EmbeddingVisualizer()
  visualizer.fit(pieces)          # pieces loaded via KnowledgeRetrievalService

  fig = visualizer.plot_documents()
  fig.show()

  query_embedding = await provider.embed_text("How high can cats jump?")
  fig = visualizer.plot_query_similarity("How high can cats jump?", query_embedding)
  fig.show()
"""


# =================================== IMPORTS ==================================

# Standard Library
from typing import Optional

# Third Party
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openTSNE import TSNE

# Local
from .models import KnowledgePiece


# =================================== CONSTANTS ==================================

_PLOT_WIDTH: int = 900
_PLOT_HEIGHT: int = 600
_WRAP_WIDTH: int = 70


# ==================================== CLASSES ====================================

class EmbeddingVisualizer:
    """Builds 2D t-SNE projections of knowledge embeddings and overlays query similarity.

    Workflow:
    1. Call :meth:`fit` with a list of :class:`KnowledgePiece` objects that
       have pre-computed embeddings (e.g. loaded from :class:`LanceDBKnowledgeStore`).
    2. Call :meth:`plot_documents` to obtain a base scatter plot of the
       projected embedding space.
    3. Call :meth:`plot_query_similarity` one or more times — each call colours
       the document points by their cosine similarity to the supplied query
       embedding.

    Args:
        perplexity: t-SNE perplexity parameter. Controls the effective number
            of neighbours considered during projection. A value between 5 and
            50 is usually appropriate.
        random_state: Seed for reproducible t-SNE results.
    """

    def __init__(self, perplexity: int = 30, random_state: int = 42) -> None:
        self._perplexity = perplexity
        self._random_state = random_state

        # Set after fit() is called.
        self._df: Optional[pd.DataFrame] = None
        self._embeddings: Optional[np.ndarray] = None

    # ----------------------- Fitting -----------------------

    def fit(self, pieces: list[KnowledgePiece]) -> None:
        """Fit t-SNE on the embeddings of the provided knowledge pieces.

        Args:
            pieces: Knowledge pieces to project. Each piece must have a
                non-``None`` ``embedding`` attribute.

        Raises:
            ValueError: If ``pieces`` is empty or any piece is missing its
                embedding.
        """
        if not pieces:
            raise ValueError("pieces must be non-empty")

        missing = [p.id for p in pieces if p.embedding is None]
        if missing:
            raise ValueError(
                f"{len(missing)} piece(s) are missing their embedding: {missing[:5]}"
            )

        # Stack embeddings into a float32 matrix for t-SNE.
        self._embeddings = np.array(
            [p.embedding for p in pieces], dtype=np.float32
        )

        # Clamp perplexity so it never exceeds n_samples - 1 (openTSNE requirement).
        effective_perplexity = min(self._perplexity, len(pieces) - 1)

        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            random_state=self._random_state,
        )
        components = np.asarray(tsne.fit(self._embeddings))

        self._df = pd.DataFrame(components, columns=["Component 1", "Component 2"])
        self._df["text"] = [p.text for p in pieces]
        self._df["source"] = [p.source or "" for p in pieces]
        self._df["hover_text"] = (
            self._df["text"]
            .str.wrap(_WRAP_WIDTH)
            .str.replace("\n", "<br>", regex=False)
        )

    # ----------------------- Plotting -----------------------

    def plot_documents(self) -> go.Figure:
        """Return a scatter plot of all document embeddings in t-SNE space.

        Each point represents one knowledge piece.  Hover over a point to see
        its full text.

        Returns:
            A Plotly figure ready to be shown or saved.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        self._require_fit()

        fig = px.scatter(
            self._df,
            x="Component 1",
            y="Component 2",
            color="source",
            custom_data=["hover_text"],
            title="t-SNE Visualization of Document Embeddings",
        )
        fig.update_traces(
            marker={"size": 8, "opacity": 0.85},
            hovertemplate="<b>Text</b><br>%{customdata[0]}<extra></extra>",
        )
        fig.update_layout(width=_PLOT_WIDTH, height=_PLOT_HEIGHT)
        return fig

    def plot_query_similarity(
        self,
        query_text: str,
        query_embedding: list[float],
    ) -> go.Figure:
        """Return a scatter plot coloured by cosine similarity to ``query_text``.

        Args:
            query_text: Human-readable query string used as the plot subtitle.
            query_embedding: Dense embedding vector of ``query_text``.  Must
                have the same dimensionality as the document embeddings supplied
                to :meth:`fit`.

        Returns:
            A Plotly figure with a ``bupu`` colour scale where darker points
            indicate higher similarity to the query.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
            ValueError: If ``query_embedding`` is empty.
        """
        self._require_fit()

        if not query_embedding:
            raise ValueError("query_embedding must be non-empty")

        similarity = _cosine_similarity_batch(
            np.array(query_embedding, dtype=np.float32),
            self._embeddings,  # type: ignore[arg-type]
        )

        df = self._df.copy()  # type: ignore[union-attr]
        df["similarity"] = similarity

        fig = px.scatter(
            df,
            x="Component 1",
            y="Component 2",
            color="similarity",
            color_continuous_scale="bupu",
            custom_data=["hover_text"],
            title="t-SNE of Document Embeddings (Colored by Query Similarity)",
            subtitle=f'Query: "{query_text}"',
        )
        fig.update_traces(
            marker={"size": 10, "opacity": 0.85},
            hovertemplate="<b>Text</b><br>%{customdata[0]}<extra></extra>",
        )
        fig.update_layout(width=_PLOT_WIDTH, height=_PLOT_HEIGHT)
        return fig

    # ----------------------- Internal Helpers -----------------------

    def _require_fit(self) -> None:
        """Raise if :meth:`fit` has not been called yet."""
        if self._df is None or self._embeddings is None:
            raise RuntimeError(
                "EmbeddingVisualizer.fit() must be called before plotting."
            )


# ================================ HELPER FUNCTIONS ================================

def _cosine_similarity_batch(
    query: np.ndarray,
    documents: np.ndarray,
) -> np.ndarray:
    """Return cosine similarities between ``query`` and each row in ``documents``.

    Args:
        query: 1-D embedding vector of shape ``(D,)``.
        documents: 2-D matrix of shape ``(N, D)`` containing document embeddings.

    Returns:
        1-D array of shape ``(N,)`` with similarities in ``[-1, 1]``.
    """
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    doc_norms = np.linalg.norm(documents, axis=1, keepdims=True) + 1e-10
    normalized_docs = documents / doc_norms
    return (normalized_docs @ query_norm).astype(np.float64)
