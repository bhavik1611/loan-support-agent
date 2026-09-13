"""Part 3 Task 13. The LLM-as-judge, and what it scores on with no model to ask.

D-74. Under MOCK_LLM the three scores are lexical: n-gram containment between
the query, the answer and the retrieved context. Under a real provider the
same three are asked of the model through llm.generate, so D-59's single
switch still decides who answers.

The signal is deliberately NOT the embedding. Sharing a signal with the
retriever would make the judge agree with retrieval by construction, which is
the failure spec 18.4 catalogues four times: a mechanism deciding on a signal
that does not carry the property it was asked to decide. IU-02 is in the query
set precisely to catch that, and it would score high on all three under an
embedding judge.

These are deterministic proxies, not a model's judgement, and README.md says
so where it reports them.
"""

import json
import re
from dataclasses import dataclass

import config
import llm

_WORD = re.compile(r"[a-z0-9]+")

JUDGE_SYSTEM = (
    "You are grading a retrieval-augmented answer. Return only a JSON object "
    "with three float fields in [0, 1]: context_relevance, how far the "
    "retrieved context bears on the question; groundedness, how far every "
    "claim in the answer is supported by that context; and answer_relevance, "
    "how far the answer addresses the question actually asked. Return no prose."
)


@dataclass(frozen=True)
class TriadScores:
    """One row of the triad table."""
    context_relevance: float
    groundedness: float
    answer_relevance: float


def context_of(hits) -> str:
    """The retrieved context as one string, which is what the judge sees."""
    return "\n".join(hit.text for hit in hits)


def build_judge_prompt(query: str, answer_text: str, context: str) -> tuple[str, str]:
    """The prompt a real provider is asked. Unused under mock, tested anyway."""
    user = (
        f"QUESTION:\n{query}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer_text}\n"
    )
    return JUDGE_SYSTEM, user


def _tokens(text: str) -> list[str]:
    """Content tokens, lowercased, function words dropped."""
    return [w for w in _WORD.findall(text.lower()) if w not in config.TRIAD_STOPWORDS]


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def _share(needles: set, haystack: set) -> float:
    """The share of needles present in haystack, floored at 0 when empty."""
    if not needles:
        return 0.0
    return round(
        len(needles & haystack) / len(needles), config.TRIAD_SCORE_PLACES
    )


def _score_lexically(query: str, answer_text: str, context: str) -> TriadScores:
    query_tokens = set(_tokens(query))
    context_tokens = set(_tokens(context))
    answer_tokens = set(_tokens(answer_text))
    context_bigrams = _bigrams(_tokens(context))
    answer_bigrams = _bigrams(_tokens(answer_text))

    return TriadScores(
        context_relevance=_share(query_tokens, context_tokens),
        # Bigrams rather than tokens, because a bag of words is satisfied by an
        # answer that uses the right vocabulary to say the wrong thing.
        groundedness=_share(answer_bigrams, context_bigrams),
        answer_relevance=_share(query_tokens, answer_tokens),
    )


def score(query: str, answer_text: str, context: str) -> TriadScores:
    """The three scores. Mock computes them; a real provider is asked for them."""
    if config.MOCK_LLM:
        return _score_lexically(query, answer_text, context)

    system, user = build_judge_prompt(query, answer_text, context)
    payload = json.loads(llm.generate(system, user))
    return TriadScores(
        context_relevance=round(
            float(payload["context_relevance"]), config.TRIAD_SCORE_PLACES
        ),
        groundedness=round(
            float(payload["groundedness"]), config.TRIAD_SCORE_PLACES
        ),
        answer_relevance=round(
            float(payload["answer_relevance"]), config.TRIAD_SCORE_PLACES
        ),
    )
