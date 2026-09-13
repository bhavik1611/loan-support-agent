"""Tests 17 and 18 of spec section 16. The score must be designed, not a boolean OR."""

import config
import dataset
from agent import escalation
from db.query import OPEN_STATUSES


def test_staleness_runs_at_full_rate_while_the_application_is_open():
    for status in OPEN_STATUSES:
        assert escalation.staleness(21, status) == 1.0
        assert escalation.staleness(0, status) == 0.0


def test_staleness_runs_at_half_rate_once_disbursed():
    assert escalation.staleness(21, "Disbursed") == 0.5


def test_staleness_stops_once_rejected():
    assert escalation.staleness(30, "Rejected") == 0.0


def test_staleness_saturates_and_never_exceeds_one():
    assert escalation.staleness(30, "Submitted") == 1.0
    assert escalation.staleness(999, "Submitted") == 1.0


def test_score_is_bounded_to_the_unit_interval():
    for record in dataset.LOAN_APPLICATIONS:
        assert 0.0 <= escalation.escalation_score(record) <= 1.0


def test_the_score_is_not_a_bare_boolean_or():
    """D-33. A boolean OR on the fraud flag would escalate exactly 16 and 0."""
    above_unflagged = [
        r for r in dataset.LOAN_APPLICATIONS
        if not r["flagged_for_fraud_review"]
        and escalation.escalation_score(r) >= config.ESCALATION_THRESHOLD
    ]
    below_flagged = [
        r for r in dataset.LOAN_APPLICATIONS
        if r["flagged_for_fraud_review"]
        and escalation.escalation_score(r) < config.ESCALATION_THRESHOLD
    ]
    assert above_unflagged, "no unflagged record crosses the threshold on staleness alone"
    assert below_flagged, "no flagged record sits below the threshold"
    assert len(above_unflagged) == 6
    assert len(below_flagged) == 2


def test_the_threshold_sits_at_the_eightieth_percentile():
    """D-34. The justification the brief asks for, asserted rather than claimed."""
    scores = sorted(escalation.escalation_score(r) for r in dataset.LOAN_APPLICATIONS)
    below = sum(1 for s in scores if s < config.ESCALATION_THRESHOLD)
    assert below == 80


def test_recommend_escalation_agrees_with_the_threshold():
    for record in dataset.LOAN_APPLICATIONS:
        expected = escalation.escalation_score(record) >= config.ESCALATION_THRESHOLD
        assert escalation.recommend_escalation(record) is expected


def test_the_score_orders_records_rather_than_bucketing_them():
    distinct = {escalation.escalation_score(r) for r in dataset.LOAN_APPLICATIONS}
    assert len(distinct) == 42
