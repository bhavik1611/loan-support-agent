"""Task 13. The three triad scores, and the property that makes them mean something."""

import config
from eval import judge


CONTEXT = (
    "The equated monthly instalment is computed from the principal, the "
    "monthly interest rate and the tenure in months."
)


def test_groundedness_is_high_when_the_answer_restates_the_context():
    scores = judge.score(
        "How is the EMI on a loan calculated?",
        "The equated monthly instalment is computed from the principal, the "
        "monthly interest rate and the tenure in months.",
        CONTEXT,
    )
    assert scores.groundedness > 0.9


def test_groundedness_is_floored_when_there_is_no_context():
    """A refusal is grounded in nothing, and the triad should say so."""
    scores = judge.score("How is the EMI calculated?", "I do not know.", "")
    assert scores.groundedness == 0.0


def test_answer_relevance_falls_when_the_answer_is_about_something_else():
    on_topic = judge.score(
        "How is the EMI calculated?",
        "The EMI is calculated from the principal, the interest rate and the tenure.",
        CONTEXT,
    )
    off_topic = judge.score(
        "How is the EMI calculated?",
        "Neptune has fourteen known moons.",
        CONTEXT,
    )
    assert off_topic.answer_relevance < on_topic.answer_relevance


def test_answer_relevance_is_blind_to_paraphrase_by_construction():
    """D-74 accepted this cost when it chose lexical signals over embeddings.

    The query's content tokens are {emi, calculated}. This answer is correct
    and grounded in CONTEXT, but it paraphrases every one of those terms
    ("equated monthly instalment", "computed") rather than repeating them, so
    the lexical overlap is empty and answer_relevance floors at 0.0. An
    embedding judge would not miss this paraphrase, but it would also not miss
    agreeing with the retriever on IU-02, because embedding similarity is the
    same signal the retriever already scores hits with (D-74). This zero is
    documented behaviour, not a bug to route around.
    """
    scores = judge.score(
        "How is the EMI calculated?",
        "The equated monthly instalment is computed from the principal, the "
        "monthly interest rate and the tenure in months.",
        CONTEXT,
    )
    assert scores.answer_relevance == 0.0


def test_every_score_is_a_rounded_ratio():
    scores = judge.score("How is the EMI calculated?", CONTEXT, CONTEXT)
    for value in (scores.context_relevance, scores.groundedness, scores.answer_relevance):
        assert 0.0 <= value <= 1.0
        assert round(value, config.TRIAD_SCORE_PLACES) == value


def test_the_judge_prompt_carries_the_query_the_answer_and_the_context():
    """Exercised under mock so the real-provider path cannot rot unnoticed."""
    system, user = judge.build_judge_prompt("query text", "answer text", "context text")
    assert "query text" in user
    assert "answer text" in user
    assert "context text" in user
    assert "context_relevance" in system.lower()


def test_scoring_is_deterministic():
    first = judge.score("How is the EMI calculated?", CONTEXT, CONTEXT)
    second = judge.score("How is the EMI calculated?", CONTEXT, CONTEXT)
    assert first == second


# Test 39 (spec section 16) is not implemented: IU-02 measures as answered at
# top1=0.4645 with all three chunks from one parent, which is neither refusal
# path D-84 assumed, so the premise is Bhavik's call to make, not a retune.
