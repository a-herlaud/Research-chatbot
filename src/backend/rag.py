"""Shared RAG core.

This single module is the source of truth for the chunking parameters, the
embedding model, and the ChromaDB layout. Both the root ``ingest_papers.py``
script and the FastAPI backend import it, which guarantees the index is built
and queried with the *exact same* embedding method.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100  # 20% of CHUNK_SIZE
TOP_K = 3
COLLECTION_NAME = "research_papers"
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma embedding function backed by sentence-transformers.

    Used for both adding documents and querying, so ingest and retrieval can
    never drift onto different vector spaces.
    """

    def __call__(self, input: Documents) -> Embeddings:
        vectors = _model().encode(
            list(input),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    @staticmethod
    def name() -> str:
        return "sentence_transformers"

    def get_config(self) -> dict[str, Any]:
        return {"model": EMBEDDING_MODEL}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> SentenceTransformerEmbeddingFunction:
        return SentenceTransformerEmbeddingFunction()

    def embed_text(self, input: Documents) -> Embeddings:
        return self(input)

    def default_space(self) -> str:
        return "cosine"


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction()


def build_splitter() -> CharacterTextSplitter:
    return CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separator="",
    )


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(create: bool = False):
    """Return the papers collection, or None when it does not exist yet."""
    client = get_client()
    if not create:
        try:
            return client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=get_embedding_function(),
            )
        except Exception:
            return None
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def retrieve(query: str, k: int = TOP_K) -> list[dict[str, Any]]:
    """Embed ``query`` with the same model and return the top-k chunks."""
    collection = get_collection(create=False)
    if collection is None or collection.count() == 0:
        return []

    result = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    chunks: list[dict[str, Any]] = []
    for text, meta, distance in zip(documents, metadatas, distances):
        meta = meta or {}
        chunks.append(
            {
                "text": text,
                "paper_id": meta.get("paper_id", ""),
                "title": meta.get("title", ""),
                # cosine distance -> similarity in [-1, 1]
                "score": round(1.0 - float(distance), 4),
            }
        )
    return chunks
