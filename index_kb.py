"""Embed both chunk sets and index each in its own ChromaDB collection (Task 3).

Two collections, never one with a filter: the brief asks for separate indexes so
Task 5 can compare them on equal terms.

Embeddings come from a free local SentenceTransformers model. Nothing here needs
a network connection after the model has been cached once, and nothing needs an
API key.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import Chunk, build_chunks, load_documents

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = Path(__file__).parent / "chroma"
COLLECTIONS = {"fixed": "kb_fixed_size", "sentence": "kb_sentence"}

_model: SentenceTransformer | None = None


def model() -> SentenceTransformer:
    """Load once, reuse. Loading costs a few seconds; embedding does not."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    return model().encode(texts, normalize_embeddings=True).tolist()


def client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(strategy: str):
    """Fetch an existing collection. Cosine space, matching how it was built."""
    return client().get_collection(name=COLLECTIONS[strategy])


def index_strategy(strategy: str, chunks: list[Chunk]) -> int:
    """Build one collection from scratch."""
    c = client()
    name = COLLECTIONS[strategy]
    try:
        c.delete_collection(name)
    except Exception:
        pass  # first run: nothing to delete

    # Cosine, not Chroma's default squared-L2, so that similarity is 1 - distance
    # and the calibrated threshold in rag.py means what it says.
    collection = c.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[ch.chunk_id for ch in chunks],
        documents=[ch.text for ch in chunks],
        embeddings=embed([ch.text for ch in chunks]),
        metadatas=[{"doc_id": ch.doc_id, "strategy": ch.strategy} for ch in chunks],
    )
    return collection.count()


def main() -> None:
    docs = load_documents()
    print(f"Loaded {len(docs)} knowledge-base documents.")
    print(f"Embedding model: {MODEL_NAME} (local, no API key)\n")

    for strategy, name in COLLECTIONS.items():
        chunks = build_chunks(docs, strategy)
        n = index_strategy(strategy, chunks)
        print(f"  {name:<16} {n:>3} chunks indexed "
              f"({strategy}, from {len(docs)} documents)")

    print("\nSanity check: the same query against both collections.")
    probe = "What is the minimum balance on a savings account?"
    q_emb = embed([probe])
    for strategy, name in COLLECTIONS.items():
        res = get_collection(strategy).query(query_embeddings=q_emb, n_results=3)
        print(f"\n  [{name}] {probe!r}")
        for doc_id, dist, text in zip(
            [m["doc_id"] for m in res["metadatas"][0]],
            res["distances"][0],
            res["documents"][0],
        ):
            print(f"    {doc_id}  cos={1 - dist:.3f}  {text[:72]}...")


if __name__ == "__main__":
    main()
