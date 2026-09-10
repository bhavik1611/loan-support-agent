"""Task 4. Answer only from retrieved context, or say so plainly.

The decision rule lives in rag/retrieve.is_supported and is deliberately not
duplicated here. Part 2 Task 7 wraps answer(), and Part 2 Task 10's output
guardrail reads .supported and .top1_similarity, so those names are a contract.
"""

from dataclasses import dataclass

import config
import llm
from rag import retrieve
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


@dataclass(frozen=True)
class GroundedAnswer:
    query: str
    text: str
    citations: tuple[str, ...]
    supported: bool
    top1_similarity: float
    strategy: str
    hits: tuple[Hit, ...]


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
    query: str, strategy: str = config.STRATEGY_SENTENCES, k: int | None = None
) -> GroundedAnswer:
    """Retrieve, decide, and either generate from the context or refuse.

    The default strategy is the one Task 5 measured as better, not a guess:
    kb_sentences scored Precision@3 0.8750 against 0.7917 and Recall@3 0.6528
    against 0.5972, while carrying the higher mean |R|, so the margin is not
    the denominator flattering it. Part 2 consumes this collection.
    """
    if config.SIMILARITY_THRESHOLD is None:
        raise RuntimeError(
            "config.SIMILARITY_THRESHOLD is unset. Run the Task 11 calibration and "
            "record the measured value; the brief forbids an untested preset."
        )

    hits = retrieve.retrieve(query, strategy, k=k)
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
        )

    return GroundedAnswer(
        query=query,
        text=text,
        citations=_cited_documents(text, hits),
        supported=True,
        top1_similarity=retrieve.top1_similarity(hits),
        strategy=strategy,
        hits=tuple(hits),
    )
