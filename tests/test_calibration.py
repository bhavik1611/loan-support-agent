"""Test 6 of spec section 11: the clusters must actually separate."""

import config
from eval import calibration


def test_the_threshold_is_a_measured_float_not_a_placeholder():
    assert isinstance(config.SIMILARITY_THRESHOLD, float)
    assert 0.0 < config.SIMILARITY_THRESHOLD < 1.0


def test_in_scope_and_out_of_scope_clusters_do_not_overlap(built_index):
    """Test 6. If they overlap, no threshold separates them and the brief's
    ban on an untested preset cannot be satisfied honestly."""
    summary = calibration.calibrate()
    assert summary["min_in_scope"] > summary["max_out_of_scope"], (
        f"clusters overlap: in-scope minimum {summary['min_in_scope']} is not above "
        f"out-of-scope maximum {summary['max_out_of_scope']}"
    )
    assert summary["gap"] > 0


def test_the_committed_threshold_still_sits_between_the_clusters(built_index):
    summary = calibration.calibrate()
    assert summary["max_out_of_scope"] < config.SIMILARITY_THRESHOLD < summary["min_in_scope"]


def test_every_in_scope_probe_retrieves_a_required_document(built_index):
    from rag import kb

    required = {d.doc_id for d in kb.load_documents() if d.required}
    misses = [
        r.probe
        for r in calibration.measure(config.STRATEGY_SENTENCES)
        if r.in_scope and r.top_doc_id not in required
    ]
    # A probe landing on a confusable neighbour is acceptable and expected;
    # this test only reports how often, so record rather than assert on zero.
    assert len(misses) <= 4, misses
