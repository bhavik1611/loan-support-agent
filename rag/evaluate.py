"""Task 5. Document-level Precision@3 and Recall@3 for both collections.

The denominator is |R|, the number of distinct parent documents among the top
three chunks, not a flat 3. That avoids penalising a collection for agreeing
with itself, and it does favour the collection that concentrates. Rather than
hide that, every row prints |R| and the recommendation has to argue with it.
"""

from dataclasses import dataclass

import config
from eval.queries import EVAL_QUERIES, EvalQuery
from rag import retrieve


@dataclass(frozen=True)
class QueryScore:
    query_id: str
    query: str
    strategy: str
    retrieved: tuple[str, ...]
    gold: tuple[str, ...]
    overlap: tuple[str, ...]
    precision: float
    recall: float


def score_from_sets(
    query_id: str, query: str, strategy: str, retrieved, gold: tuple[str, ...]
) -> QueryScore:
    """The arithmetic on its own, so it can be tested without a vector store."""
    retrieved = tuple(retrieved)
    gold_set = set(gold)
    overlap = tuple(d for d in retrieved if d in gold_set)
    return QueryScore(
        query_id=query_id,
        query=query,
        strategy=strategy,
        retrieved=retrieved,
        gold=tuple(gold),
        overlap=overlap,
        precision=len(overlap) / len(retrieved) if retrieved else 0.0,
        recall=len(overlap) / len(gold) if gold else 0.0,
    )


def score_query(query: EvalQuery, strategy: str) -> QueryScore:
    hits = retrieve.retrieve(query.text, strategy, k=config.TOP_K)
    return score_from_sets(
        query_id=query.query_id,
        query=query.text,
        strategy=strategy,
        retrieved=retrieve.parent_documents(hits),  # the dedup the brief requires
        gold=query.gold_doc_ids,
    )


def evaluate(strategy: str) -> tuple[list[QueryScore], float, float]:
    """Every query scored against one collection, plus the two macro averages."""
    scores = [score_query(q, strategy) for q in EVAL_QUERIES]
    n = len(scores)
    return (
        scores,
        sum(s.precision for s in scores) / n,
        sum(s.recall for s in scores) / n,
    )


def _table(scores: list[QueryScore], strategy: str, precision: float, recall: float) -> list[str]:
    lines = [
        f"--- collection: {config.COLLECTION_FOR_STRATEGY[strategy]} "
        f"(strategy: {strategy}) ---",
        "",
        f"{'query':<7} {'|R|':>3} {'|G|':>3} {'|R and G|':>9}  "
        f"{'Precision@3':>13}  {'Recall@3':>11}",
    ]
    for s in scores:
        p = f"{len(s.overlap)}/{len(s.retrieved) or 1}"
        r = f"{len(s.overlap)}/{len(s.gold)}"
        lines.append(
            f"{s.query_id:<7} {len(s.retrieved):>3} {len(s.gold):>3} {len(s.overlap):>9}  "
            f"{p:>13}  {r:>11}"
        )
    lines += [
        "",
        f"{'mean':<7} {'':>3} {'':>3} {'':>9}  {precision:>13.4f}  {recall:>11.4f}",
        "",
        "per-query detail",
    ]
    for s in scores:
        lines += [
            f"  {s.query_id}  {s.query}",
            f"    retrieved parents R : {list(s.retrieved)}",
            f"    gold documents    G : {list(s.gold)}",
            f"    intersection        : {list(s.overlap)}",
            f"    Precision@3 = |R and G| / |R| = {len(s.overlap)}/{len(s.retrieved) or 1} "
            f"= {s.precision:.4f}",
            f"    Recall@3    = |R and G| / |G| = {len(s.overlap)}/{len(s.gold)} "
            f"= {s.recall:.4f}",
            "",
        ]
    return lines


def format_comparison() -> str:
    """Per-query arithmetic for both collections, then the four averages."""
    lines = [
        "Chunking strategy comparison - document-level Precision@3 and Recall@3",
        "",
        "Chunks are mapped back to their parent doc_id and deduplicated before",
        "scoring. R is the set of distinct parents among the top 3 chunks, G is",
        "the hand-authored gold set, committed before any retrieval was run.",
        "",
        "  Precision@3 = |R and G| / |R|",
        "  Recall@3    = |R and G| / |G|",
        "",
        "Dividing by |R| rather than by 3 avoids penalising a collection for",
        "agreeing with itself, and it does favour the collection that",
        "concentrates. |R| is printed on every row for exactly that reason.",
        "",
    ]

    summary = {}
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        scores, precision, recall = evaluate(strategy)
        summary[strategy] = (precision, recall)
        lines += _table(scores, strategy, precision, recall)

    lines += ["--- both collections side by side ---", ""]
    lines.append(f"{'collection':<20} {'Precision@3':>13} {'Recall@3':>11}")
    for strategy, (precision, recall) in sorted(summary.items()):
        lines.append(
            f"{config.COLLECTION_FOR_STRATEGY[strategy]:<20} {precision:>13.4f} {recall:>11.4f}"
        )
    return "\n".join(lines) + "\n"
