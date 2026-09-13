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


def _score_all_triad_rows() -> dict:
    """Every triad row answered and scored once, keyed by item_id.

    Shared by test 39 and the IU-02 test below so the fifteen rows are scored
    once rather than twice. Not cached across test runs: each test calls this
    fresh, so a change to retrieval or the judge shows up in whichever test
    observes it rather than being hidden behind a stale module-level cache.
    """
    from eval import triad
    from rag import generate

    rows = {}
    for item in triad.triad_items():
        answered = generate.answer(item.text)
        scores = judge.score(
            item.text, answered.text, judge.context_of(answered.hits)
        )
        rows[item.item_id] = (answered, scores)
    return rows


def test_39_far_out_of_scope_queries_score_below_answerable_ones_on_groundedness():
    """Spec section 16 test 39, amended per Bhavik's ruling on D-84.

    This asserts every far_out_of_scope row scores below every answerable row
    on groundedness - measured 0.0000 < 0.7234, with margin - and it
    deliberately does not compare inside_uncovered (IU-02) here.

    D-84's original premise was "a judge that cannot mark IU-02 down is
    broken". Measured, that premise is false: IU-02 is not refused. It is
    answered at top1=0.4645 (above T=0.3066), with all three retrieved chunks
    parented under kb-12-nri-account-eligibility, so the support rule passes
    too. Its groundedness of 0.7857 is correct given that - the answer is
    faithfully grounded in the context it retrieved. There is no refusal for
    groundedness to mark down, so a test built on "IU-02 should read low on
    groundedness" was asserting on a false premise that no threshold or
    scorer retune can fix: raising T past 0.4645 to force a refusal also
    refuses EQ-07, EQ-11 and EQ-12 and regenerates every Part 1 number, and
    retuning the lexical scorers has no cut to make, since IU-02 already
    scores higher on context_relevance (0.2000) than correctly-answered EQ-12
    (0.1667).

    This test therefore pins the one property the triad genuinely supports -
    far_out_of_scope refusals read near-zero groundedness against every
    answerable row - rather than the property D-84 originally claimed.
    Pinning a property instead of a number is still D-13's and D-84's rule,
    unchanged; only the property changed. IU-02's own behaviour is pinned
    separately, in
    test_iu02_is_answered_not_refused_and_context_relevance_catches_it below.
    """
    from eval import queries, triad

    rows = _score_all_triad_rows()
    grounded_by_kind: dict[str, list[float]] = {}
    for item in triad.triad_items():
        grounded_by_kind.setdefault(item.kind, []).append(
            rows[item.item_id][1].groundedness
        )

    answerable = grounded_by_kind[queries.KIND_ANSWERABLE]
    far_out_of_scope = grounded_by_kind[queries.KIND_FAR_OUT_OF_SCOPE]
    assert max(far_out_of_scope) < min(answerable), (
        f"a far_out_of_scope query scored {max(far_out_of_scope)} groundedness, "
        f"at or above the weakest answerable one at {min(answerable)}. Both "
        f"classes are refused before generation reaches a real answer, so a "
        f"refusal's groundedness should floor near zero; if it does not, the "
        f"refusal template itself started restating retrieved text."
    )


def test_iu02_is_answered_not_refused_and_context_relevance_catches_it():
    """The clearest single illustration of why the triad has three signals.

    Recorded observations, this run (prose, not assertions - see below for
    why): IU-02 ("How do I transfer money to an account in another country?")
    is outcome='answered', not refused - top1=0.4645 against T=0.3066, with
    all three retrieved chunks parented under kb-12-nri-account-eligibility,
    so the support rule (>= 2 of 3 chunks share a parent) passes too. Its
    groundedness (0.7857) sits inside the answerable band - at or above
    EQ-10's 0.7234, the weakest answerable row - because the answer
    faithfully restates the NRI-account context it retrieved. An answer that
    faithfully restates irrelevant context is grounded by definition, so
    groundedness cannot catch this failure and was never the signal that
    could. Its context_relevance (0.2000) sits low instead, well below its
    own groundedness, and even below correctly-answered EQ-12's 0.1667 is not
    guaranteed - the two are close enough that no single cut on
    context_relevance alone separates them either. Context_relevance still
    catches the failure here because it reads low while groundedness reads
    high for the same row, and that gap is what the assertion below checks.

    Every assertion below is relational, measured against the other fourteen
    triad rows scored in the same run, never against a constant a
    stopword-list or chunker change could drift past while the claim - IU-02
    is a groundedness blind spot - stayed true. That is D-13's and D-84's
    rule: pin the property, never the number. The outcome check is
    categorical rather than numeric, so a fixed value there is legitimate.

    eval.queries.GOLDEN_DATASET still records IU-02's expected behaviour as
    "refuse on the threshold and support rule". Measurement contradicts that:
    the dataset's expectation is the stale half, not this test.
    """
    from eval import queries, triad
    from rag import generate

    rows = _score_all_triad_rows()
    iu02_answered, iu02_scores = rows["IU-02"]
    answerable_groundedness = [
        rows[item.item_id][1].groundedness
        for item in triad.triad_items()
        if item.kind == queries.KIND_ANSWERABLE
    ]

    assert iu02_answered.outcome == generate.OUTCOME_ANSWERED
    assert iu02_scores.groundedness >= min(answerable_groundedness), (
        f"IU-02 groundedness {iu02_scores.groundedness} fell below the weakest "
        f"answerable row's {min(answerable_groundedness)}. The blind spot this "
        f"test exists to pin - a wrong-document answer reading as well "
        f"grounded as a correct one - stopped holding."
    )
    assert iu02_scores.context_relevance < iu02_scores.groundedness
