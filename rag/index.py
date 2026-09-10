"""Task 3. Embed every chunk and index each strategy in its own collection.

normalize_embeddings=True together with {"hnsw:space": "cosine"} is what makes
similarity = 1 - distance correct in rag/retrieve.py. Changing one without the
other silently breaks every measured number downstream, so both live here.
"""

import functools
import os

from dataclasses import dataclass

import config
from rag import chunking, kb

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    title: str
    topic: str
    required: bool
    chunk_index: int
    strategy: str

    def metadata(self) -> dict:
        """Chroma accepts scalars only, which is all the mapping to a parent needs."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "topic": self.topic,
            "required": self.required,
            "chunk_index": self.chunk_index,
            "strategy": self.strategy,
        }


def build_chunks(strategy: str, documents=None) -> list[Chunk]:
    """Every chunk of every document under one strategy, in document order."""
    documents = kb.load_documents() if documents is None else documents
    chunks: list[Chunk] = []
    for document in documents:
        pieces = chunking.chunk(document.body, strategy)
        if len(pieces) < config.MIN_CHUNKS_PER_DOCUMENT:
            raise ValueError(
                f"{document.doc_id} produced {len(pieces)} chunk(s) under {strategy!r}, "
                f"needs at least {config.MIN_CHUNKS_PER_DOCUMENT}. A document with one "
                f"chunk can never satisfy the same-parent support rule, so it would be "
                f"permanently unanswerable. Lengthen the document or retune the chunker."
            )
        for position, text in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{strategy}::{position:03d}",
                    text=text,
                    doc_id=document.doc_id,
                    title=document.title,
                    topic=document.topic,
                    required=document.required,
                    chunk_index=position,
                    strategy=strategy,
                )
            )
    return chunks


@functools.lru_cache(maxsize=1)
def get_embedder():
    """The local SentenceTransformers model. Cached: loading costs about 0.3s."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Unit-normalised embeddings, so cosine distance is 1 minus cosine similarity."""
    vectors = get_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return [vector.tolist() for vector in vectors]


@functools.lru_cache(maxsize=1)
def get_client():
    import chromadb
    from chromadb.config import Settings

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def collection_name(strategy: str) -> str:
    try:
        return config.COLLECTION_FOR_STRATEGY[strategy]
    except KeyError:
        raise ValueError(
            f"unknown strategy {strategy!r}, expected one of "
            f"{sorted(config.COLLECTION_FOR_STRATEGY)}"
        ) from None


def get_collection(strategy: str):
    """The built collection. Raises if build_index has never run."""
    return get_client().get_collection(name=collection_name(strategy))


def build_index(rebuild: bool = False) -> dict[str, int]:
    """Embed and index both strategies. Returns collection name to chunk count."""
    client = get_client()
    counts: dict[str, int] = {}

    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        name = collection_name(strategy)
        if rebuild:
            try:
                client.delete_collection(name=name)
            except Exception:
                pass
        collection = client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

        chunks = build_chunks(strategy)
        if collection.count() != len(chunks):
            if collection.count():
                client.delete_collection(name=name)
                collection = client.get_or_create_collection(
                    name=name, metadata={"hnsw:space": "cosine"}
                )
            collection.add(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                embeddings=embed([c.text for c in chunks]),
                metadatas=[c.metadata() for c in chunks],
            )
        counts[name] = collection.count()

    return counts


def chunk_counts_by_document(strategy: str) -> dict[str, int]:
    """Chunks per parent document, for the Task 3 transcript and open item 2."""
    counts: dict[str, int] = {}
    for chunk in build_chunks(strategy):
        counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
    return dict(sorted(counts.items()))
