"""Probe queries for the empirical threshold calibration in Task 4.

The brief's floor is 3 in-scope and 2 out-of-scope probes. Twelve and five
oversample it deliberately: the gap between the two clusters is the entire
justification for the chosen threshold, and three points do not make a cluster.

The in-scope probes are one per required topic and are deliberately worded
differently from the twelve evaluation queries, so that calibration and
scoring are not measuring the same twelve strings twice.
"""

from dataclasses import dataclass

import config
from rag import retrieve

IN_SCOPE_PROBES: list[str] = [
    "Who is eligible to apply for a business loan?",
    "Show me the formula used to work out a monthly instalment.",
    "What annual fee does the credit card carry?",
    "Which identity proof is accepted when opening an account?",
    "Someone used my card without permission, what happens next?",
    "How long does it take to shut a bank account?",
    "What rate of interest applies to an education loan?",
    "Will I be charged for repaying a fixed-rate loan ahead of schedule?",
    "What happens if my average balance falls below the requirement?",
    "How much does a missed payment hurt my credit rating?",
    "Can two people hold one account together?",
    "Which account types can a person living abroad hold?",
]

OUT_OF_SCOPE_PROBES: list[str] = [
    "What is the best recipe for a chocolate sponge cake?",
    "Which team won the football World Cup in 2018?",
    "How do I replace the timing belt on a diesel engine?",
    "What is the boiling point of liquid nitrogen at sea level?",
    "Recommend a three-day hiking route in the Western Ghats.",
]


@dataclass(frozen=True)
class ProbeResult:
    """The result of a probe against one collection."""
    probe: str
    in_scope: bool
    strategy: str
    top1: float
    top_doc_id: str


def measure(strategy: str) -> list[ProbeResult]:
    """Top-1 cosine similarity for all seventeen probes against one collection."""
    results = []
    for probes, in_scope in ((IN_SCOPE_PROBES, True), (OUT_OF_SCOPE_PROBES, False)):
        for probe in probes:
            hits = retrieve.retrieve(probe, strategy, k=1)
            results.append(
                ProbeResult(
                    probe=probe,
                    in_scope=in_scope,
                    strategy=strategy,
                    top1=retrieve.top1_similarity(hits),
                    top_doc_id=hits[0].doc_id if hits else "",
                )
            )
    return results


def calibrate() -> dict:
    """Pool both collections and set the threshold midway between the clusters.

    One threshold, not one per collection: the answer decision has to be a
    single rule that Part 2 imports, and Part 2's collection is not chosen
    until Task 5's numbers exist.
    """
    by_strategy = {s: measure(s) for s in sorted(config.COLLECTION_FOR_STRATEGY)}
    results = [r for rs in by_strategy.values() for r in rs]

    in_scope = [r.top1 for r in results if r.in_scope]
    out_of_scope = [r.top1 for r in results if not r.in_scope]

    min_in_scope = min(in_scope)
    max_out_of_scope = max(out_of_scope)

    return {
        "by_strategy": by_strategy,
        "results": results,
        "min_in_scope": round(min_in_scope, 4),
        "max_out_of_scope": round(max_out_of_scope, 4),
        "gap": round(min_in_scope - max_out_of_scope, 4),
        "threshold": round((min_in_scope + max_out_of_scope) / 2.0, 4),
    }


def format_report(summary: dict) -> str:
    """The transcript body: every measured value, clustered, then the arithmetic."""
    lines = [
        "Threshold calibration - measured, not preset",
        "",
        "The brief forbids an untested preset. 0.5, 0.6 and 0.7 are tutorial",
        "defaults that do not reliably separate short policy-sentence embeddings",
        "from unrelated queries, so every value below was measured on this",
        "knowledge base with all-MiniLM-L6-v2 and cosine similarity.",
        "",
        f"Probes: {len(IN_SCOPE_PROBES)} in-scope, {len(OUT_OF_SCOPE_PROBES)} out-of-scope, "
        f"measured against both collections.",
        "",
    ]
    for strategy, results in summary["by_strategy"].items():
        lines.append(f"--- collection: {config.COLLECTION_FOR_STRATEGY[strategy]} ---")
        for label, wanted in (("IN SCOPE", True), ("OUT OF SCOPE", False)):
            lines.append(f"  {label}")
            for r in sorted(
                (r for r in results if r.in_scope is wanted),
                key=lambda r: r.top1,
                reverse=True,
            ):
                lines.append(f"    {r.top1:6.4f}  {r.top_doc_id:<40}  {r.probe}")
            lines.append("")
    lines += [
        "--- pooled across both collections ---",
        f"  minimum in-scope top-1      = {summary['min_in_scope']:.4f}",
        f"  maximum out-of-scope top-1  = {summary['max_out_of_scope']:.4f}",
        f"  gap                         = {summary['gap']:.4f}",
        "",
        f"  threshold T = ({summary['min_in_scope']:.4f} + {summary['max_out_of_scope']:.4f}) / 2 "
        f"= {summary['threshold']:.4f}",
        "",
        "An answer is produced only when top-1 similarity is at least T AND at",
        f"least {config.SUPPORT_MIN_SHARED} of the top {config.TOP_K} chunks share one parent document.",
    ]
    return "\n".join(lines) + "\n"
