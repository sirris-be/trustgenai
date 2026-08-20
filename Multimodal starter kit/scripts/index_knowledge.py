#!/usr/bin/env python3
"""Index knowledge facts from a text file into the LanceDB knowledge store.

Each non-empty line in the input file is treated as one knowledge fact and
embedded using the configured embedding provider (default: Azure Foundry
``text-embedding-3-large``).

Usage:
    python scripts/index_knowledge.py <facts_file>
    python scripts/index_knowledge.py <facts_file> --source my-doc.txt
    python scripts/index_knowledge.py <facts_file> --db /custom/path/knowledge.lancedb

Examples:
    python scripts/index_knowledge.py data/robot_facts.txt
    python scripts/index_knowledge.py data/assembly_manual.txt --source assembly_manual.txt

Environment variables (all optional, fall back to config.py defaults):
    AZURE_OPENAI_API_KEY        Azure API key for the embedding provider.
    AZURE_OPENAI_ENDPOINT       Azure endpoint URL.
    KNOWLEDGE_EMBEDDING_PROVIDER  Provider name (default: azure_foundry).
    KNOWLEDGE_EMBEDDING_MODEL     Model/deployment name (default: text-embedding-3-large).
    KNOWLEDGE_VECTOR_DB_URI       Path to the LanceDB database directory.
    KNOWLEDGE_TABLE_NAME          Name of the LanceDB table.
"""


# =================================== IMPORTS ==================================

# Standard Library
import argparse
import asyncio
import hashlib
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
# Third Party
# None

# Local
from tgenai_agent.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    KNOWLEDGE_EMBEDDING_MODEL,
    KNOWLEDGE_EMBEDDING_PROVIDER,
    KNOWLEDGE_TABLE_NAME,
    KNOWLEDGE_VECTOR_DB_URI,
)
from tgenai_agent.knowledge.embeddings import create_embedding_provider
from tgenai_agent.knowledge.models import KnowledgeUpsertItem
from tgenai_agent.knowledge.service import KnowledgeIndexer
from tgenai_agent.knowledge.vector_store import LanceDBKnowledgeStore


# =================================== CONSTANTS ==================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Number of facts to embed in one batch request to the provider.
_BATCH_SIZE = 20


# ================================ HELPER FUNCTIONS ================================

def _load_facts(file_path: Path) -> list[str]:
    """Return non-empty, stripped lines from ``file_path``.

    Args:
        file_path: Path to the newline-separated facts file.

    Returns:
        A list of non-empty text strings.

    Raises:
        SystemExit: If the file cannot be read.
    """
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.error("Cannot read '%s': %s", file_path, exc)
        sys.exit(1)

    facts = [line.strip() for line in lines if line.strip()]
    return facts


def _fact_id(text: str, source: str) -> str:
    """Return a deterministic ID for a fact based on its content and source.

    Using a content hash means re-running the script with the same file is
    idempotent — existing rows are overwritten, not duplicated.
    """
    digest = hashlib.sha256(f"{source}:{text}".encode()).hexdigest()[:16]
    return digest


def _build_upsert_items(facts: list[str], source: str) -> list[KnowledgeUpsertItem]:
    """Convert a list of fact strings into :class:`KnowledgeUpsertItem` objects."""
    return [
        KnowledgeUpsertItem(
            id=_fact_id(text, source),
            text=text,
            source=source,
        )
        for text in facts
    ]


# ================================ MAIN FUNCTIONS ================================

async def _run(args: argparse.Namespace) -> None:
    """Execute the indexing pipeline."""
    file_path = Path(args.file)
    source = args.source or file_path.name
    db_uri = args.db or KNOWLEDGE_VECTOR_DB_URI
    table_name = args.table or KNOWLEDGE_TABLE_NAME

    # Load facts from file.
    facts = _load_facts(file_path)
    if not facts:
        logger.error("No facts found in '%s'. Exiting.", file_path)
        sys.exit(1)

    logger.info("Loaded %d facts from '%s'.", len(facts), file_path)

    # Build upsert items.
    items = _build_upsert_items(facts, source)

    # Initialise provider and store.
    logger.info(
        "Embedding provider: %s | model: %s",
        KNOWLEDGE_EMBEDDING_PROVIDER,
        KNOWLEDGE_EMBEDDING_MODEL,
    )
    logger.info("Target DB: %s | table: %s", db_uri, table_name)

    provider = create_embedding_provider(
        KNOWLEDGE_EMBEDDING_PROVIDER,
        api_key=AZURE_OPENAI_API_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        model=KNOWLEDGE_EMBEDDING_MODEL,
    )
    store = LanceDBKnowledgeStore(db_uri=db_uri, table_name=table_name)
    indexer = KnowledgeIndexer(provider, store)

    # Index in batches so progress is visible for large files.
    total = len(items)
    indexed = 0
    for batch_start in range(0, total, _BATCH_SIZE):
        batch = items[batch_start : batch_start + _BATCH_SIZE]
        logger.info(
            "Indexing facts %d–%d of %d ...",
            batch_start + 1,
            batch_start + len(batch),
            total,
        )
        await indexer.index(batch)
        indexed += len(batch)

    logger.info("Done. Indexed %d / %d facts into '%s'.", indexed, total, table_name)


# ================================ ENTRY POINT ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index newline-separated facts into the knowledge LanceDB store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1].strip() if "Examples:" in __doc__ else "",
    )
    parser.add_argument(
        "file",
        help="Path to a text file where each non-empty line is one knowledge fact.",
    )
    parser.add_argument(
        "--source",
        default=None,
        metavar="LABEL",
        help="Optional source label stored alongside the facts (defaults to the file name).",
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help=f"Path to the LanceDB database directory (default: {KNOWLEDGE_VECTOR_DB_URI!r}).",
    )
    parser.add_argument(
        "--table",
        default=None,
        metavar="NAME",
        help=f"LanceDB table name (default: {KNOWLEDGE_TABLE_NAME!r}).",
    )

    parsed = parser.parse_args()
    asyncio.run(_run(parsed))
