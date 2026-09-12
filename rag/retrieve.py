"""Task 4, retrieval half. Top-k chunks, cosine similarity, and the support rule.

Callers never see ChromaDB. That is deliberate: the V2 upgrade to hybrid BM25
plus dense retrieval with reciprocal rank fusion replaces this file alone.
"""

from dataclasses import dataclass

import config
from rag import index, kb


@dataclass(frozen=True)
class Hit:
    text: str
    doc_id: str
    title: str
    chunk_index: int
    similarity: float


def retrieve(
    query: str, strategy: str, k: int | None = None, product: str | None = None
) -> list[Hit]:
    """The top k chunks for a query, highest cosine similarity first.

    `product` narrows the search to the documents catalogue.json tags with that
    product, per D-53. The narrowing is conditional on purpose: 9 of the 12
    in-scope calibration probes name no product at all, so an unconditional
    filter would search an empty subset for three quarters of real questions.
    With no product the query is issued exactly as it was before the gate
    existed, with no `where` clause at all.

    The clause is a `doc_id` `$in` built from the catalogue, never a product
    field on the chunk. Measured against ChromaDB 1.5.9 on 2026-09-12, a
    `$contains` clause on a delimited string metadata field returns an empty
    result rather than raising, so a per-chunk product field would have made
    every product-named query retrieve nothing, silently. Nothing here touches
    the index, so neither collection is rebuilt for the filter.
    """
    k = config.TOP_K if k is None else k
    collection = index.get_collection(strategy)  # raises on an unknown strategy
    narrowing = {}
    if product:
        narrowing["where"] = {"doc_id": {"$in": kb.documents_for_product(product)}}
    result = collection.query(
        query_embeddings=index.embed([query]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
        **narrowing,
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
    # An explicit total order. Similarity is rounded to 4 dp on the way in, so
    # sorting on it alone manufactures ties the raw distances never had, and a
    # stable sort then silently inherits whatever order ChromaDB returned.
    # (doc_id, chunk_index) identifies a chunk uniquely, so this key can never
    # tie and the rank order is the same bytes on every machine and every
    # ChromaDB version.
    return sorted(hits, key=lambda h: (-h.similarity, h.doc_id, h.chunk_index))


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
