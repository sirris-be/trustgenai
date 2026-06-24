"""Embedding provider interface and concrete implementations.

Typical usage:

  provider = create_embedding_provider("azure_foundry")
  vector = await provider.embed_text("The robot arm grasps the red cube.")
"""


# =================================== IMPORTS ==================================

# Standard Library
import logging
from typing import Protocol, runtime_checkable

# Third Party
from openai import AsyncOpenAI

# Local
# None


# =================================== CONSTANTS ==================================

LOGGER = logging.getLogger(__name__)


# ==================================== PROTOCOLS ====================================

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Generic interface for embedding text into a dense float vector.

    Implementations must be safe to call concurrently from async code.
    """

    # ----------------------- Core Operations -----------------------

    async def embed_text(self, text: str) -> list[float]:
        """Return a dense embedding vector for ``text``.

        Args:
            text: The text to embed. Must be non-empty.

        Returns:
            A list of floats representing the embedding.

        Raises:
            ValueError: If ``text`` is empty.
            RuntimeError: If the underlying provider returns an error.
        """
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for each entry in ``texts``.

        A default batch implementation is provided that calls :meth:`embed_text`
        sequentially; concrete classes may override this for true batch requests.

        Args:
            texts: A non-empty list of strings to embed.

        Returns:
            A list of embedding vectors with the same length as ``texts``.
        """
        ...


# ==================================== CLASSES ====================================

class AzureFoundryEmbeddingProvider:
    """Embedding provider backed by Azure OpenAI (Azure AI Foundry).

    Uses the ``openai`` async client against the Azure-compatible endpoint
    already configured in ``fair_agent.config``.

    Args:
        api_key: Azure OpenAI API key.
        endpoint: Azure OpenAI endpoint URL (e.g. ``https://<resource>.openai.azure.com``).
        model: Deployment/model name to use for embeddings.
    """

    def __init__(self, api_key: str, endpoint: str, model: str) -> None:
        self._model = model
        # Use AsyncOpenAI with base_url so the SDK appends /embeddings directly
        # to the supplied endpoint. This is consistent with how the rest of the
        # project talks to Azure AI Foundry (via OpenAILike), and avoids the
        # path-doubling that occurs when using AsyncAzureOpenAI with an endpoint
        # that already contains the /openai/v1 path segment.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint,
        )

    # ----------------------- Core Operations -----------------------

    async def embed_text(self, text: str) -> list[float]:
        """Return a dense embedding vector for ``text`` using Azure OpenAI.

        Args:
            text: The text to embed. Must be non-empty.

        Returns:
            Embedding as a list of floats.

        Raises:
            ValueError: If ``text`` is empty.
            RuntimeError: If the Azure API returns an error.
        """
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            raise RuntimeError(f"Azure embedding request failed: {exc}") from exc

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for each entry in ``texts`` in a single batch request.

        Args:
            texts: A non-empty list of strings to embed.

        Returns:
            A list of embedding vectors.

        Raises:
            ValueError: If ``texts`` is empty.
            RuntimeError: If the Azure API returns an error.
        """
        if not texts:
            raise ValueError("texts must be non-empty")

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            # The API guarantees results in the same order as the input.
            return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        except Exception as exc:
            raise RuntimeError(f"Azure batch embedding request failed: {exc}") from exc


# ================================ MAIN FUNCTIONS ================================

def create_embedding_provider(
    provider_name: str,
    *,
    api_key: str,
    endpoint: str,
    model: str,
) -> EmbeddingProvider:
    """Factory that returns a concrete :class:`EmbeddingProvider` by name.

    Args:
        provider_name: Identifier for the desired provider. Currently supports
            ``"azure_foundry"``.
        api_key: API key passed to the provider.
        endpoint: Endpoint URL passed to the provider.
        model: Model/deployment name passed to the provider.

    Returns:
        A configured :class:`EmbeddingProvider` instance.

    Raises:
        ValueError: If ``provider_name`` is not a supported provider.
    """
    match provider_name:
        case "azure_foundry":
            LOGGER.info(
                "Creating AzureFoundryEmbeddingProvider model=%s endpoint=%s",
                model,
                endpoint,
            )
            return AzureFoundryEmbeddingProvider(
                api_key=api_key,
                endpoint=endpoint,
                model=model,
            )
        case _:
            raise ValueError(
                f"Unknown embedding provider '{provider_name}'. "
                "Supported values: 'azure_foundry'."
            )
