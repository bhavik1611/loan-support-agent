"""The metric arithmetic must be right, and both collections must be scored."""

import config
from eval.queries import (
    GOLDEN_DATASET,
    KIND_ANSWERABLE,
    KIND_FAR_OUT_OF_SCOPE,
    KIND_INSIDE_UNCOVERED,
    KIND_OUTSIDE_BOUNDARY,
    KINDS,
)
from rag import evaluate, generate


def test_precision_and_recall_arithmetic_on_a_known_case():
    score = evaluate.score_from_sets(
        query_id="EQ-XX",
        query="synthetic",
        strategy=config.STRATEGY_FIXED,
        retrieved=["kb-01", "kb-02", "kb-03"],
        gold=("kb-01", "kb-02"),
    )
    assert score.overlap == ("kb-01", "kb-02")
    assert score.precision == 2 / 3
    assert score.recall == 2 / 2


def test_a_single_concentrated_parent_scores_one_or_zero():
    hit = evaluate.score_from_sets("EQ-XX", "q", config.STRATEGY_SENTENCES, ["kb-01"], ("kb-01",))
    miss = evaluate.score_from_sets("EQ-XX", "q", config.STRATEGY_SENTENCES, ["kb-09"], ("kb-01",))
    assert hit.precision == 1.0 and hit.recall == 1.0
    assert miss.precision == 0.0 and miss.recall == 0.0


def test_no_retrieval_scores_zero_without_dividing_by_zero():
    score = evaluate.score_from_sets("EQ-XX", "q", config.STRATEGY_FIXED, [], ("kb-01",))
    assert score.precision == 0.0
    assert score.recall == 0.0


def test_every_query_is_scored_for_both_collections(built_index):
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        scores, precision, recall = evaluate.evaluate(strategy)
        assert len(scores) == 12
        assert all(s.strategy == strategy for s in scores)
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0


def test_retrieved_parents_are_deduplicated_and_at_most_three(built_index):
    scores, _, _ = evaluate.evaluate(config.STRATEGY_SENTENCES)
    for score in scores:
        assert len(set(score.retrieved)) == len(score.retrieved)
        assert 1 <= len(score.retrieved) <= config.TOP_K


def test_the_comparison_table_shows_per_query_arithmetic_for_both(built_index):
    report = evaluate.format_comparison()
    assert config.COLLECTION_FIXED in report
    assert config.COLLECTION_SENTENCES in report
    for query_id in [f"EQ-{n:02d}" for n in range(1, 13)]:
        assert query_id in report
    assert "|R|" in report
    assert report.count("/") > 40  # fractions, not decimals, on every row


def test_evaluation_is_deterministic(built_index):
    assert evaluate.format_comparison() == evaluate.format_comparison()


# --- Task 19. The decision table, spec section 9.4 --------------------------
#
# Per D-56 exactly two of the four classes may be asserted here. There is no
# test below on Precision@3, on Recall@3, on the inside_uncovered outcomes or
# count, or on any aggregate decision accuracy: those numbers move when chunk
# parameters are tuned, and a test that fights tuning gets deleted.


def test_the_class_tally_counts_every_outcome_in_a_fixed_order():
    """The arithmetic on its own, with no vector store behind it."""
    rows = [
        evaluate.DecisionRow("EQ-XX", "q", KIND_ANSWERABLE, "s", "answered", 0.5, "", ()),
        evaluate.DecisionRow("OB-XX", "q", KIND_OUTSIDE_BOUNDARY, "s", "refused_gate", 0.0, "p", ()),
        evaluate.DecisionRow(
            "FO-XX", "q", KIND_FAR_OUT_OF_SCOPE, "s", "refused_threshold", 0.1, "", ()
        ),
    ]
    tally = evaluate.counts_by_kind(rows)
    assert list(tally) == list(KINDS)  # declared order, never arrival order
    assert list(tally[KIND_ANSWERABLE]) == list(evaluate.OUTCOMES)
    assert tally[KIND_ANSWERABLE]["answered"] == 1
    assert tally[KIND_OUTSIDE_BOUNDARY]["refused_gate"] == 1
    assert tally[KIND_FAR_OUT_OF_SCOPE]["refused_threshold"] == 1
    assert tally[KIND_INSIDE_UNCOVERED] == {outcome: 0 for outcome in evaluate.OUTCOMES}


def test_the_near_domain_rate_counts_both_near_classes_and_nothing_else():
    rows = [
        evaluate.DecisionRow("EQ-XX", "q", KIND_ANSWERABLE, "s", "answered", 0.9, "", ()),
        evaluate.DecisionRow("OB-XX", "q", KIND_OUTSIDE_BOUNDARY, "s", "refused_gate", 0.0, "p", ()),
        evaluate.DecisionRow("IU-XX", "q", KIND_INSIDE_UNCOVERED, "s", "answered", 0.5, "", ()),
    ]
    answered, readings = evaluate.near_domain_false_answers(rows)
    assert (answered, readings) == (1, 2)  # the answerable row is not near-domain


def test_every_far_out_of_scope_item_is_refused(built_index):
    """Acceptance criterion 29, on both collections.

    Either mechanism counts. These items are not close to anything, so this is
    the robust half of what D-56 lets a test pin.
    """
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        for row in evaluate.decisions(strategy):
            if row.kind == KIND_FAR_OUT_OF_SCOPE:
                assert row.outcome != generate.OUTCOME_ANSWERED, (
                    f"{row.item_id} was answered on {strategy} at {row.top1_similarity:.4f}"
                )


def test_every_inside_uncovered_item_has_a_line_in_the_decision_transcript(built_index):
    """Acceptance criterion 24b, as reworded on 2026-09-12.

    It asserts that the transcript carries a line per inside_uncovered item
    with that item's outcome and its top-1 similarity, and deliberately not
    what the outcome is. IU-02 is answered at 0.4645 against kb_sentences,
    because all three of its top chunks come from kb-12 and the support rule
    therefore agrees with itself about a document that does not answer the
    question. An assertion on the outcome would have been false the day it was
    written, and D-56 keeps this class reported rather than pinned.
    """
    report = evaluate.format_decision_report()
    lines = report.splitlines()
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        for row in evaluate.decisions(strategy):
            if row.kind != KIND_INSIDE_UNCOVERED:
                continue
            matching = [
                line
                for line in lines
                if line.startswith(row.item_id)
                and row.outcome in line
                and f"{row.top1_similarity:.4f}" in line
            ]
            assert matching, (
                f"{row.item_id} on {strategy}: no transcript line carrying "
                f"outcome {row.outcome} and top-1 {row.top1_similarity:.4f}"
            )


def test_the_decision_report_covers_every_golden_item_on_both_collections(built_index):
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        rows = evaluate.decisions(strategy)
        assert len(rows) == len(GOLDEN_DATASET)
        assert [r.item_id for r in rows] == [i.item_id for i in GOLDEN_DATASET]
        assert all(r.outcome in evaluate.OUTCOMES for r in rows)


def test_the_decision_report_prints_the_before_figure_beside_the_after(built_index):
    """The near-domain false-answer rate is the evidence the gate worked.

    The two literals this used to assert, 14 of 30, were carried over from a
    retired 15-item near-domain tier. Today's tier is 12 items and 24 readings,
    so the printed comparison had a different numerator population and a
    different denominator on either side of the word "before", and the test
    locked that in. Both halves are now computed on the run, so what is left to
    assert is the property that makes the comparison honest: the same
    denominator on both sides, over the same items.
    """
    report = evaluate.format_decision_report()
    assert "before the product gate" in report
    assert "after the product gate" in report

    rows = []
    ungated = []
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        rows += evaluate.decisions(strategy)
        ungated += evaluate.decisions(strategy, gate=False)
    after_answered, after_readings = evaluate.near_domain_false_answers(rows)
    before_answered, before_readings = evaluate.near_domain_false_answers(ungated)

    assert before_readings == after_readings, "the two sides must share a denominator"
    assert f"before the product gate : {before_answered} of {before_readings} " in report
    assert f"after the product gate  : {after_answered} of {after_readings} " in report
    # The gate exists to lower this number, so the before figure cannot be the
    # smaller one. Neither figure is pinned, per D-56 and D-13.
    assert before_answered >= after_answered


def test_the_ungated_pass_really_bypasses_the_gate(built_index):
    """The before figure would be worthless if it still ran the gate.

    Every outside_boundary item refuses at the gate with the gate on, so a
    bypass that did not bypass would print the same number twice and look like
    a comparison.
    """
    rows = evaluate.decisions(config.STRATEGY_SENTENCES, gate=False)
    outside = [r for r in rows if r.kind == KIND_OUTSIDE_BOUNDARY]
    assert outside
    assert all(r.outcome != generate.OUTCOME_REFUSED_GATE for r in outside)
    assert all(r.top1_similarity > 0.0 for r in outside), "a bypassed item retrieved"


def test_the_decision_report_is_deterministic(built_index):
    assert evaluate.format_decision_report() == evaluate.format_decision_report()
