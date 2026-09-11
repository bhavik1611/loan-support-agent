"""Tests 1 to 4 of spec section 11, one per Part 1 acceptance criterion."""

import hashlib

import config
import dataset


def test_generator_reproduces_the_committed_snapshot():
    """Test 1. D-09: the generator and the snapshot must not drift apart."""
    on_disk = config.DATASET_SNAPSHOT.read_bytes()
    regenerated = dataset.snapshot_bytes()
    assert hashlib.sha256(regenerated).hexdigest() == hashlib.sha256(on_disk).hexdigest(), (
        "dataset.py no longer reproduces data/loan_applications.json. "
        "If the change is intended, rewrite the snapshot in the same commit."
    )


def test_every_category_has_at_least_three_records():
    """Test 2. The brief's category-count floor."""
    counts = dataset.validation_report()["by_category"]
    for category in config.CATEGORIES:
        assert counts.get(category, 0) >= config.MIN_RECORDS_PER_CATEGORY, category


def test_every_status_appears_at_least_once():
    """Test 3. The brief's status-coverage floor."""
    counts = dataset.validation_report()["by_status"]
    for status in config.STATUSES:
        assert counts.get(status, 0) >= config.MIN_RECORDS_PER_STATUS, status


def test_fraud_rate_lands_inside_the_band():
    """Test 4. The brief's 10 to 30 percent band."""
    percentage = dataset.validation_report()["fraud_percentage"]
    assert config.FRAUD_RATE_MIN * 100 <= percentage <= config.FRAUD_RATE_MAX * 100


def test_record_count_and_shape():
    assert len(dataset.LOAN_APPLICATIONS) == config.RECORD_COUNT
    expected_keys = [
        "record_id",
        "category",
        "status",
        "loan_amount_inr",
        "days_since_created",
        "flagged_for_fraud_review",
    ]
    for record in dataset.LOAN_APPLICATIONS:
        assert list(record) == expected_keys


def test_every_value_is_inside_its_declared_domain():
    for record in dataset.LOAN_APPLICATIONS:
        assert record["category"] in config.CATEGORIES
        assert record["status"] in config.STATUSES
        assert isinstance(record["flagged_for_fraud_review"], bool)
        assert config.DAYS_MIN <= record["days_since_created"] <= config.DAYS_MAX
        low, high = config.CATEGORY_BANDS[record["category"]]
        # Rounding to the nearest 1000 can carry a draw 500 past either end.
        assert low - 500 <= record["loan_amount_inr"] <= high + 500
        assert record["loan_amount_inr"] % 1000 == 0


def test_record_ids_are_unique_and_contiguous():
    ids = [r["record_id"] for r in dataset.LOAN_APPLICATIONS]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "LN-1001"
    assert ids[-1] == f"LN-{config.RECORD_ID_START + config.RECORD_COUNT - 1}"


def test_generation_is_deterministic_across_calls():
    assert dataset.generate_applications() == dataset.generate_applications()


def test_lookup_returns_the_record_and_none_for_a_miss():
    assert dataset.get_application("LN-1001")["record_id"] == "LN-1001"
    assert dataset.get_application("LN-9999") is None


# --- the relational store must not move a single existing byte ------------


def test_the_projection_is_byte_identical_to_the_committed_snapshot():
    """The whole promise of the database work: adding it moved no record."""
    on_disk = config.DATASET_SNAPSHOT.read_bytes()
    assert (
        hashlib.sha256(dataset.snapshot_bytes()).hexdigest()
        == hashlib.sha256(on_disk).hexdigest()
    )


def test_the_projection_keeps_exactly_the_six_fields_in_order():
    assert dataset.PROJECTED_FIELDS == (
        "record_id",
        "category",
        "status",
        "loan_amount_inr",
        "days_since_created",
        "flagged_for_fraud_review",
    )
    for record in dataset.LOAN_APPLICATIONS:
        assert tuple(record) == dataset.PROJECTED_FIELDS


def test_the_rich_row_is_a_superset_of_the_projection():
    assert len(dataset.RICH_APPLICATIONS) == len(dataset.LOAN_APPLICATIONS)
    for rich, projected in zip(dataset.RICH_APPLICATIONS, dataset.LOAN_APPLICATIONS):
        for field in dataset.PROJECTED_FIELDS:
            assert rich[field] == projected[field]
        assert rich["product_code"]
        assert rich["tenure_months"] > 0
        assert rich["interest_rate_pct"] > 0


def test_enrichment_does_not_disturb_the_six_field_draw():
    """The enricher must not consume from the stream that draws the six."""
    before = dataset.generate_applications()
    dataset.enrich_applications(dataset.generate_applications())
    after = dataset.generate_applications()
    assert before == after


def test_generate_applications_still_returns_only_the_six_fields():
    """Stream 0's loop is untouched; enrichment happens strictly afterwards."""
    for record in dataset.generate_applications():
        assert tuple(record) == dataset.PROJECTED_FIELDS


def test_tenure_and_rate_sit_inside_the_knowledge_base_bands():
    for row in dataset.RICH_APPLICATIONS:
        low, high = config.RATE_BANDS[row["category"]]
        assert low <= row["interest_rate_pct"] <= high, row["record_id"]
        assert row["tenure_months"] in config.TENURE_CHOICES[row["category"]]
        assert row["product_code"] == config.PRODUCT_CODES[row["category"]]


def test_enrichment_is_deterministic():
    base = dataset.generate_applications()
    assert dataset.enrich_applications(base) == dataset.enrich_applications(base)
