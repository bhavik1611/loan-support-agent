"""Task 4. Answer only from retrieved context, or say so plainly.

Two mechanisms refuse here, and they are separate on purpose. The product gate
of rag/scope.py decides topicality before anything is retrieved, per D-51; the
threshold and the support rule then decide groundedness, per D-46. An answer
that never reached retrieval reports outcome "refused_gate" and carries the
product name; one that retrieved and failed the rule reports "refused_threshold".

The decision rule lives in rag/retrieve.is_supported and is deliberately not
duplicated here. Part 2 Task 7 wraps answer(), and Part 2 Task 10's output
guardrail reads .supported and .top1_similarity, so those names are a contract.
"""

from dataclasses import dataclass
from typing import Literal

import config
import llm
from rag import retrieve, scope
from rag.retrieve import Hit

FALLBACK_TEXT = (
    "I do not know. The loan support knowledge base does not contain enough "
    "supporting material to answer that question, so I will not guess."
)

SYSTEM_PROMPT = (
    "You are Meridian Bank's loan support assistant. Answer only from the "
    "CONTEXT block. Never add a fact that is not in the context. Cite the "
    "document id of every source you use."
)

# The three outcomes of spec section 9.4, and only three. Part 2 matches on
# these strings, so they are frozen. Per D-54 the sentence a user reads is
# Part 2's job: Part 1 produces the structured refusal and the product name.
OUTCOME_ANSWERED = "answered"
OUTCOME_REFUSED_GATE = "refused_gate"
OUTCOME_REFUSED_THRESHOLD = "refused_threshold"

Outcome = Literal["answered", "refused_gate", "refused_threshold"]


@dataclass(frozen=True)
class GroundedAnswer:
    query: str
    text: str
    citations: tuple[str, ...]
    supported: bool
    top1_similarity: float
    strategy: str
    hits: tuple[Hit, ...]
    outcome: Outcome = OUTCOME_ANSWERED
    product: str = ""


def build_prompt(query: str, hits: list[Hit]) -> tuple[str, str]:
    """The (system, user) pair, in the contract llm.parse_prompt reads back."""
    lines = [f"QUESTION: {query}", "", "CONTEXT:"]
    for hit in hits:
        flattened = " ".join(hit.text.split())
        lines.append(f"[{hit.doc_id}] {flattened}")
    return SYSTEM_PROMPT, "\n".join(lines) + "\n"


def _cited_documents(text: str, hits: list[Hit]) -> tuple[str, ...]:
    """The doc_ids the generated answer actually cited, in retrieval order."""
    if "Sources:" not in text:
        return ()
    tail = text.split("Sources:", 1)[1]
    ordered = []
    for doc_id in retrieve.parent_documents(hits):
        if f"[{doc_id}]" in tail and doc_id not in ordered:
            ordered.append(doc_id)
    return tuple(ordered)


def answer(
    query: str,
    strategy: str = config.STRATEGY_SENTENCES,
    k: int | None = None,
    *,
    gate: bool = True,
) -> GroundedAnswer:
    """Retrieve, decide, and either generate from the context or refuse.

    The default strategy is the one Task 5 measured as better, not a guess:
    kb_sentences scored Precision@3 0.8750 against 0.7917 and Recall@3 0.6528
    against 0.5972, while carrying the higher mean |R|, so the margin is not
    the denominator flattering it. Part 2 consumes this collection.

    `gate=False` turns rag/scope.py off entirely - no refusal before retrieval
    and no D-53 product filter on the search - and it exists for exactly one
    caller: rag/evaluate.py measures the near-domain false-answer rate the
    system would have without the gate, so it can print a before figure beside
    the after figure. Both halves then come out of this one function and the
    comparison cannot go stale.

    The alternative was a second copy of the answer decision inside the scorer,
    computing the before figure from retrieve.is_supported directly. That lost
    because the two copies could then disagree about what the system does,
    which is the same reason decide() reads GroundedAnswer.outcome rather than
    re-deriving it. This is not a second design kept alive: the gate is the
    design, and `gate=False` is the measurement of its absence.
    """
    if config.SIMILARITY_THRESHOLD is None:
        raise RuntimeError(
            "config.SIMILARITY_THRESHOLD is unset. Run the Task 11 calibration and "
            "record the measured value; the brief forbids an untested preset."
        )

    # The gate of spec section 8.5 runs before retrieval, because D-46 measured
    # that similarity cannot decide whether a question is about a product
    # Meridian Bank sells.
    verdict = scope.classify(query)
    if gate and verdict.known_adjacent:
        return GroundedAnswer(
            query=query,
            text=FALLBACK_TEXT,
            citations=(),
            supported=False,
            top1_similarity=0.0,
            strategy=strategy,
            hits=(),
            outcome=OUTCOME_REFUSED_GATE,
            product=verdict.product,
        )

    hits = retrieve.retrieve(
        query,
        strategy,
        k=k,
        product=verdict.product if (gate and verdict.in_catalogue) else None,
    )
    supported = retrieve.is_supported(hits, config.SIMILARITY_THRESHOLD)

    if not supported:
        return GroundedAnswer(
            query=query,
            text=FALLBACK_TEXT,
            citations=(),
            supported=False,
            top1_similarity=retrieve.top1_similarity(hits),
            strategy=strategy,
            hits=tuple(hits),
            outcome=OUTCOME_REFUSED_THRESHOLD,
            product=verdict.product,
        )

    system, user = build_prompt(query, hits)
    text = llm.generate(system, user)
    if not text:
        # The context held nothing usable, which is a refusal, not an answer.
        return GroundedAnswer(
            query=query,
            text=FALLBACK_TEXT,
            citations=(),
            supported=False,
            top1_similarity=retrieve.top1_similarity(hits),
            strategy=strategy,
            hits=tuple(hits),
            outcome=OUTCOME_REFUSED_THRESHOLD,
            product=verdict.product,
        )

    return GroundedAnswer(
        query=query,
        text=text,
        citations=_cited_documents(text, hits),
        supported=True,
        top1_similarity=retrieve.top1_similarity(hits),
        strategy=strategy,
        hits=tuple(hits),
        outcome=OUTCOME_ANSWERED,
        product=verdict.product,
    )
