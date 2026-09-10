"""Task 4, retrieval half. Top-k chunks, cosine similarity, and the support rule.

Callers never see ChromaDB. That is deliberate: the V2 upgrade to hybrid BM25
plus dense retrieval with reciprocal rank fusion replaces this file alone.
"""

from dataclasses import dataclass

import config
from rag import index


@dataclass(frozen=True)
class Hit:
    text: str
    doc_id: str
    title: str
    chunk_index: int
    similarity: float


def retrieve(query: str, strategy: str, k: int | None = None) -> list[Hit]:
    """The top k chunks for a query, highest cosine similarity first."""
    k = config.TOP_K if k is None else k
    collection = index.get_collection(strategy)  # raises on an unknown strategy
    result = collection.query(
        query_embeddings=index.embed([query]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        # Embeddings are unit-normalised and the space is cosine, so this is
        # an identity, not an approximation. See rag/index.py.
        hits.append(
            Hit(
                text=text,
                doc_id=metadata["doc_id"],
                title=metadata["title"],
                chunk_index=int(metadata["chunk_index"]),
                similarity=round(1.0 - float(distance), 4),
            )
        )
    return sorted(hits, key=lambda h: h.similarity, reverse=True)


def parent_documents(hits: list[Hit]) -> list[str]:
    """Distinct parent doc_ids in rank order. This is the dedup Task 5 requires."""
    seen: list[str] = []
    for h in hits:
        if h.doc_id not in seen:
            seen.append(h.doc_id)
    return seen


def top1_similarity(hits: list[Hit]) -> float:
    return hits[0].similarity if hits else 0.0


def has_same_parent_support(hits: list[Hit], minimum: int | None = None) -> bool:
    """At least `minimum` of the top-k chunks agree on one parent document."""
    minimum = config.SUPPORT_MIN_SHARED if minimum is None else minimum
    counts: dict[str, int] = {}
    for h in hits:
        counts[h.doc_id] = counts.get(h.doc_id, 0) + 1
    return bool(counts) and max(counts.values()) >= minimum


def is_supported(hits: list[Hit], threshold: float) -> bool:
    """The answer decision: a calibrated threshold AND agreement on one parent.

    Similarity alone lost because a single confident chunk can be confidently
    wrong; the second condition asks the knowledge base to agree with itself
    before the system speaks.
    """
    return top1_similarity(hits) >= threshold and has_same_parent_support(hits)
