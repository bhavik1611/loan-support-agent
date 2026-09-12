"""Task 5. Document-level Precision@3 and Recall@3 for both collections.

The denominator is |R|, the number of distinct parent documents among the top
three chunks, not a flat 3. That avoids penalising a collection for agreeing
with itself, and it does favour the collection that concentrates. Rather than
hide that, every row prints |R| and the recommendation has to argue with it.

Task 19 adds a second, separate pass at the bottom of this file. The metrics
above are scored over the 12 answerable items only, because only they carry
gold documents; the decision table is scored over all 29 and measures a
different thing, per spec section 9.4.
"""

from dataclasses import dataclass

import config
from eval.queries import (
    EVAL_QUERIES,
    GOLDEN_DATASET,
    KIND_INSIDE_UNCOVERED,
    KIND_OUTSIDE_BOUNDARY,
    KINDS,
    GoldenItem,
)
from rag import generate, retrieve


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


def score_query(query: GoldenItem, strategy: str) -> QueryScore:
    hits = retrieve.retrieve(query.text, strategy, k=config.TOP_K)
    return score_from_sets(
        query_id=query.item_id,
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


# --- Task 19. Decision-level evaluation, spec section 9.4 --------------------
#
# Precision@3 and Recall@3 above answer "did retrieval find the right
# document". They cannot answer "should the system have spoken at all", which
# after D-51 is a separate decision made by a separate mechanism. So this
# second pass runs all 29 golden items through rag/generate.answer and records
# which of its three outcomes each produced.
#
# Per D-56 only two of the four classes are ever asserted by a test. The
# numbers below are reported, and the reporting is the point: the
# inside_uncovered readings and the near-domain false-answer rate stay visible
# without a test pinning a number that a legitimate retune moves.

OUTCOMES = (
    generate.OUTCOME_ANSWERED,
    generate.OUTCOME_REFUSED_GATE,
    generate.OUTCOME_REFUSED_THRESHOLD,
)

# The two classes that sound like Meridian Bank business and are not. They are
# the tier D-48 kept precisely so this rate could be measured.
NEAR_DOMAIN_KINDS = (KIND_OUTSIDE_BOUNDARY, KIND_INSIDE_UNCOVERED)

# The before figure is computed, never quoted. It used to be two literals, 14
# of 30, carried over from a 15-item near-domain tier that no longer exists;
# today's tier is 12 items and 24 readings, so the printed comparison had a
# different numerator population and a different denominator on the two sides
# of the word "before". Both halves now come from decisions(gate=...) over the
# same items on the same day, which is a comparison that cannot go stale.


@dataclass(frozen=True)
class DecisionRow:
    item_id: str
    query: str
    kind: str
    strategy: str
    outcome: str
    top1_similarity: float
    product: str
    citations: tuple[str, ...]


def decide(item: GoldenItem, strategy: str, gate: bool = True) -> DecisionRow:
    """One golden item's decision, read off GroundedAnswer.outcome directly.

    The outcome is not re-derived from `supported` and `top1_similarity` here.
    rag/generate.py owns that rule, and a second copy of it in the scorer would
    let the two disagree about what the system did.

    `gate=False` is the counterfactual the before figure needs: the same query,
    the same collection, the same threshold and support rule, with rag/scope.py
    switched off.
    """
    result = generate.answer(item.text, strategy, gate=gate)
    return DecisionRow(
        item_id=item.item_id,
        query=item.text,
        kind=item.kind,
        strategy=strategy,
        outcome=result.outcome,
        top1_similarity=result.top1_similarity,
        product=result.product,
        citations=result.citations,
    )


def decisions(strategy: str, gate: bool = True) -> list[DecisionRow]:
    """Every golden item scored against one collection, in dataset order."""
    return [decide(item, strategy, gate=gate) for item in GOLDEN_DATASET]


def counts_by_kind(rows: list[DecisionRow]) -> dict[str, dict[str, int]]:
    """class -> outcome -> count, in the declared KINDS and OUTCOMES order.

    Both levels are seeded from tuples rather than accumulated from the rows,
    so the iteration order of the result is the declared order and never the
    order the data happened to arrive in.
    """
    tally = {kind: {outcome: 0 for outcome in OUTCOMES} for kind in KINDS}
    for row in rows:
        tally[row.kind][row.outcome] += 1
    return tally


def near_domain_false_answers(rows: list[DecisionRow]) -> tuple[int, int]:
    """(answered, readings) over the outside_boundary and inside_uncovered rows.

    An answer to either class is a false answer: outside_boundary names a
    product Meridian Bank does not sell, and inside_uncovered is inside the
    boundary with no document behind it, so there is nothing to ground on.
    """
    near = [row for row in rows if row.kind in NEAR_DOMAIN_KINDS]
    answered = sum(1 for row in near if row.outcome == generate.OUTCOME_ANSWERED)
    return answered, len(near)


def _decision_table(rows: list[DecisionRow], strategy: str) -> list[str]:
    lines = [
        f"--- collection: {config.COLLECTION_FOR_STRATEGY[strategy]} "
        f"(strategy: {strategy}) ---",
        "",
        f"{'item':<6} {'class':<18} {'outcome':<18} {'top-1':>7}  {'product':<20} citations",
    ]
    for row in rows:
        lines.append(
            f"{row.item_id:<6} {row.kind:<18} {row.outcome:<18} "
            f"{row.top1_similarity:>7.4f}  {row.product or '-':<20} "
            f"{list(row.citations)}"
        )

    tally = counts_by_kind(rows)
    lines += [
        "",
        "per-class summary",
        "",
        f"  {'class':<18} {'n':>3} {'answered':>9} {'refused_gate':>13} "
        f"{'refused_threshold':>18}",
    ]
    for kind in KINDS:
        row = tally[kind]
        lines.append(
            f"  {kind:<18} {sum(row.values()):>3} "
            f"{row[generate.OUTCOME_ANSWERED]:>9} "
            f"{row[generate.OUTCOME_REFUSED_GATE]:>13} "
            f"{row[generate.OUTCOME_REFUSED_THRESHOLD]:>18}"
        )

    answered, readings = near_domain_false_answers(rows)
    lines += [
        "",
        f"  near-domain false answers on this collection: {answered} of {readings}",
        "",
    ]
    return lines


def format_decision_report() -> str:
    """The decision table for both collections, plus the near-domain rate."""
    lines = [
        f"Decision-level evaluation - {len(GOLDEN_DATASET)} golden items, both collections",
        "",
        "Precision@3 and Recall@3 measure whether retrieval found the right",
        "document. They cannot measure whether the system should have spoken at",
        "all, which after D-51 is decided by a separate mechanism before any",
        "vector search runs. This table records that decision instead.",
        "",
        "  answered           the gate passed, T was cleared, and the support rule agreed",
        "  refused_gate       the product gate of rag/scope.py refused before retrieval",
        "  refused_threshold  retrieval ran, and T or the support rule refused",
        "",
        f"threshold T = {config.SIMILARITY_THRESHOLD}, top-k = {config.TOP_K}, support rule:",
        f"at least {config.SUPPORT_MIN_SHARED} of the top {config.TOP_K} chunks share one "
        "parent document.",
        "",
        "A refused_gate row reports top-1 similarity 0.0000 because no retrieval",
        "ran, not because a search scored zero.",
        "",
        "Per D-56 a test asserts two of these four classes and no more:",
        "outside_boundary must refuse at the gate, and far_out_of_scope must",
        "refuse by either mechanism. The answerable and inside_uncovered numbers",
        "are reported here and pinned nowhere, for the reason D-13 gives.",
        "",
    ]

    combined: list[DecisionRow] = []
    ungated: list[DecisionRow] = []
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        rows = decisions(strategy)
        combined += rows
        ungated += decisions(strategy, gate=False)
        lines += _decision_table(rows, strategy)

    answered, readings = near_domain_false_answers(combined)
    before_answered, before_readings = near_domain_false_answers(ungated)
    lines += [
        "--- the near-domain false-answer rate ---",
        "",
        "The near-domain tier is outside_boundary plus inside_uncovered: the",
        "questions that sound like Meridian Bank business and are not. Each item",
        "is counted once per collection, so both figures below run over exactly",
        "the same readings.",
        "",
        f"  before the product gate : {before_answered} of {before_readings} "
        "readings answered outright",
        f"  after the product gate  : {answered} of {readings} readings answered outright",
        "",
        "Both figures are computed on this run, over the same items, by the same",
        "function. The before figure runs every golden item through the same",
        "answer decision with rag/scope.py switched off: no refusal before",
        "retrieval and no D-53 product filter on the search. Nothing here is",
        "quoted from an earlier measurement, so the two sides cannot drift onto",
        "different populations.",
        "",
    ]
    still = [row for row in combined if row.kind in NEAR_DOMAIN_KINDS
             and row.outcome == generate.OUTCOME_ANSWERED]
    if still:
        lines.append("Still answered, named rather than hidden:")
        lines.append("")
        for row in still:
            lines.append(
                f"  {row.item_id} ({row.kind}) on "
                f"{config.COLLECTION_FOR_STRATEGY[row.strategy]} "
                f"at {row.top1_similarity:.4f}  {row.query}"
            )
        lines += [
            "",
            "These are coverage gaps rather than scope failures: the question is",
            "inside the product boundary of D-47 and no document answers it, so",
            "the support rule agrees with itself about a document that does not",
            "answer the question. The fix belongs to Part 2 or to V2's two-signal",
            "fallback, never to rewording the probe until it passes.",
            "",
        ]
    else:
        lines += ["No near-domain reading is answered on either collection.", ""]
    return "\n".join(lines) + "\n"
