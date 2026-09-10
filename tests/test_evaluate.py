"""The metric arithmetic must be right, and both collections must be scored."""

import config
from eval.queries import EvalQuery
from rag import evaluate


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
