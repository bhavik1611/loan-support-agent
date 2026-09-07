"""Precision@3 and Recall@3 for both collections (Part 1, Task 5).

Scoring is at DOCUMENT level, not chunk level: retrieved chunks are mapped back
to their parent documents and deduplicated before anything is counted. A
strategy that puts three chunks of the same document in the top three has
retrieved one document, not three.

To reach three distinct documents the retriever is given a deeper chunk pool
(CHUNK_POOL) and the first three distinct parents are taken. Both strategies get
the same pool, so neither is advantaged by emitting more chunks per document.

Labels come from queries.py and were committed before any retrieval was run.
"""

from __future__ import annotations

from dataclasses import dataclass

from index_kb import embed, get_collection
from queries import IN_SCOPE, Query

CHUNK_POOL = 12   # chunks fetched per query, to yield 3 distinct parents
AT_K = 3


@dataclass(frozen=True)
class QueryResult:
    query: Query
    strategy: str
    retrieved_docs: tuple[str, ...]     # top-3 distinct parents, in rank order
    hit_flags: tuple[bool, ...]
    precision: float
    recall: float


def retrieve_documents(query: str, strategy: str, k: int = AT_K) -> list[str]:
    """Top-k distinct parent documents for a query."""
    res = get_collection(strategy).query(
        query_embeddings=embed([query]), n_results=CHUNK_POOL
    )
    ordered = [m["doc_id"] for m in res["metadatas"][0]]
    return list(dict.fromkeys(ordered))[:k]


def score(q: Query, strategy: str) -> QueryResult:
    retrieved = retrieve_documents(q.text, strategy)
    relevant = set(q.relevant_docs)
    flags = tuple(d in relevant for d in retrieved)
    n_hits = sum(flags)
    return QueryResult(
        query=q,
        strategy=strategy,
        retrieved_docs=tuple(retrieved),
        hit_flags=flags,
        precision=n_hits / AT_K,
        recall=n_hits / len(relevant) if relevant else 0.0,
    )


def report_strategy(strategy: str) -> list[QueryResult]:
    print(f"\n{'=' * 78}\nCOLLECTION: {strategy}\n{'=' * 78}")
    results = []
    for q in IN_SCOPE:
        r = score(q, strategy)
        results.append(r)
        marks = " ".join(
            f"{d}{'(hit)' if h else '(miss)'}" for d, h in zip(r.retrieved_docs, r.hit_flags)
        )
        n_hits = sum(r.hit_flags)
        print(f"\n{q.query_id}  {q.text}")
        print(f"    relevant  : {', '.join(q.relevant_docs)}")
        print(f"    retrieved : {marks}")
        print(f"    P@3 = {n_hits}/{AT_K} = {r.precision:.3f}    "
              f"R@3 = {n_hits}/{len(q.relevant_docs)} = {r.recall:.3f}")

    mean_p = sum(r.precision for r in results) / len(results)
    mean_r = sum(r.recall for r in results) / len(results)
    print(f"\n  MEAN over {len(results)} queries:  "
          f"Precision@3 = {mean_p:.3f}   Recall@3 = {mean_r:.3f}")
    return results


def main() -> None:
    ceiling = sum(min(len(q.relevant_docs), AT_K) / AT_K for q in IN_SCOPE) / len(IN_SCOPE)
    print(f"Document-level Precision@3 and Recall@3, {len(IN_SCOPE)} queries, "
          f"2 relevant documents each.")
    print(f"Ceiling: Precision@3 cannot exceed {ceiling:.3f} with 2 relevant "
          f"documents; Recall@3 can reach 1.000.")

    all_results = {s: report_strategy(s) for s in ("fixed", "sentence")}

    print(f"\n{'=' * 78}\nSIDE BY SIDE\n{'=' * 78}")
    print(f"{'query':<7} {'fixed P@3':>10} {'sent P@3':>10} "
          f"{'fixed R@3':>10} {'sent R@3':>10}   winner")
    for i, q in enumerate(IN_SCOPE):
        f, s = all_results["fixed"][i], all_results["sentence"][i]
        if f.recall > s.recall:
            winner = "fixed"
        elif s.recall > f.recall:
            winner = "sentence"
        else:
            winner = "tie"
        print(f"{q.query_id:<7} {f.precision:>10.3f} {s.precision:>10.3f} "
              f"{f.recall:>10.3f} {s.recall:>10.3f}   {winner}")

    print()
    for s in ("fixed", "sentence"):
        rs = all_results[s]
        print(f"{s:<9} mean Precision@3 = "
              f"{sum(r.precision for r in rs) / len(rs):.3f}   "
              f"mean Recall@3 = {sum(r.recall for r in rs) / len(rs):.3f}   "
              f"queries with both relevant docs found: "
              f"{sum(1 for r in rs if r.recall == 1.0)}/{len(rs)}")


if __name__ == "__main__":
    main()
