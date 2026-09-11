"""Tests 7 to 12 of spec section 11.

Three of these assert the generated rows cannot contradict the knowledge base.
That is the price of D-16's seven tables and it is paid here rather than
trusted.
"""

import re
from pathlib import Path

import config
import dataset
from db import build, schema

CHILD_FOREIGN_KEYS = [
    ("loan_applications", "customer_id", "customers", "customer_id"),
    ("loan_applications", "product_code", "loan_products", "product_code"),
    ("application_events", "record_id", "loan_applications", "record_id"),
    ("repayments", "record_id", "loan_applications", "record_id"),
    ("support_tickets", "customer_id", "customers", "customer_id"),
    ("support_tickets", "record_id", "loan_applications", "record_id"),
    ("kyc_documents", "customer_id", "customers", "customer_id"),
]


def _kb_text(stem: str) -> str:
    return (config.KB_DIR / f"{stem}.md").read_text(encoding="utf-8").lower()


# --- test 7 ---------------------------------------------------------------


def test_the_committed_snapshot_did_not_move():
    """D-18 and D-19: adding the store changed no existing record."""
    import hashlib

    assert (
        hashlib.sha256(dataset.snapshot_bytes()).hexdigest()
        == hashlib.sha256(config.DATASET_SNAPSHOT.read_bytes()).hexdigest()
    )


# --- test 8 ---------------------------------------------------------------


def test_every_foreign_key_resolves(db_conn):
    for child, column, parent, parent_key in CHILD_FOREIGN_KEYS:
        orphans = db_conn.execute(
            f"""
            SELECT count(*) AS n FROM {child} c
             WHERE c.{column} IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM {parent} p WHERE p.{parent_key} = c.{column})
            """
        ).fetchone()["n"]
        assert orphans == 0, f"{child}.{column} has {orphans} orphans"


def test_sqlite_reports_no_foreign_key_violations(db_conn):
    assert db_conn.execute("PRAGMA foreign_key_check").fetchall() == []


# --- test 9 ---------------------------------------------------------------


def test_every_application_obeys_its_product_row(db_conn):
    rows = db_conn.execute(
        """
        SELECT a.record_id, a.loan_amount_inr, a.interest_rate_pct, a.tenure_months,
               p.min_amount_inr, p.max_amount_inr, p.min_rate_pct, p.max_rate_pct,
               p.max_tenure_months, a.category, p.category AS product_category
          FROM loan_applications a JOIN loan_products p USING (product_code)
        """
    ).fetchall()
    assert len(rows) == config.RECORD_COUNT
    for r in rows:
        assert r["category"] == r["product_category"], r["record_id"]
        # Rounding to the nearest 1000 can carry a draw 500 past either end.
        assert r["min_amount_inr"] - 500 <= r["loan_amount_inr"] <= r["max_amount_inr"] + 500
        assert r["min_rate_pct"] <= r["interest_rate_pct"] <= r["max_rate_pct"]
        assert 0 < r["tenure_months"] <= r["max_tenure_months"]


def test_statuses_and_ages_stay_in_their_vocabularies(db_conn):
    for r in db_conn.execute("SELECT status, days_since_created FROM loan_applications"):
        assert r["status"] in config.STATUSES
        assert config.DAYS_MIN <= r["days_since_created"] <= config.DAYS_MAX


# --- test 10 --------------------------------------------------------------


def test_every_repayment_reproduces_the_kb02_formula(db_conn):
    """Recomputed here independently. Calling the generator's own helper would
    prove only that the helper equals itself."""
    rows = db_conn.execute(
        """
        SELECT r.record_id, r.instalment_no, r.emi_inr,
               a.loan_amount_inr, a.interest_rate_pct, a.tenure_months
          FROM repayments r JOIN loan_applications a USING (record_id)
        """
    ).fetchall()
    assert rows, "no repayment rows to check"
    worst = 0.0
    for row in rows:
        r = row["interest_rate_pct"] / 12.0 / 100.0
        n = row["tenure_months"]
        growth = (1.0 + r) ** n
        expected = row["loan_amount_inr"] * r * growth / (growth - 1.0)
        worst = max(worst, abs(row["emi_inr"] - expected))
        assert abs(row["emi_inr"] - expected) < 1.0, row["record_id"]
    assert worst < 1.0


def test_the_amortisation_split_adds_up(db_conn):
    for row in db_conn.execute("SELECT * FROM repayments"):
        assert abs(row["principal_inr"] + row["interest_inr"] - row["emi_inr"]) < 0.05
        assert row["balance_inr"] >= 0


def test_repayments_exist_only_for_disbursed_applications(db_conn):
    leaked = db_conn.execute(
        """
        SELECT count(*) AS n FROM repayments r
          JOIN loan_applications a USING (record_id)
         WHERE a.status != 'Disbursed'
        """
    ).fetchone()["n"]
    assert leaked == 0


# --- test 11 --------------------------------------------------------------


def test_every_kyc_doc_type_appears_in_the_knowledge_base(db_conn):
    """kb-04 lists the standard proofs; kb-12 adds the passport and visa a
    non-resident submits. A doc_type in neither document is invented."""
    text = _kb_text("kb-04-kyc-documents") + _kb_text("kb-12-nri-account-eligibility")
    for row in db_conn.execute("SELECT DISTINCT doc_type FROM kyc_documents"):
        assert row["doc_type"].lower() in text, row["doc_type"]


def test_every_ticket_channel_appears_in_kb05(db_conn):
    text = _kb_text("kb-05-fraud-dispute")
    for row in db_conn.execute("SELECT DISTINCT channel FROM support_tickets"):
        assert row["channel"].lower() in text, row["channel"]


def test_every_rate_band_matches_kb07(db_conn):
    text = _kb_text("kb-07-interest-rate-slabs")
    for row in db_conn.execute("SELECT category, min_rate_pct, max_rate_pct FROM loan_products"):
        low = f"{row['min_rate_pct']:.2f}".rstrip("0").rstrip(".")
        high = f"{row['max_rate_pct']:.2f}".rstrip("0").rstrip(".")
        assert low in text, (row["category"], low)
        assert high in text, (row["category"], high)


def test_no_loan_holder_is_below_the_kb01_credit_score_floor(db_conn):
    lowest = db_conn.execute("SELECT min(credit_score) AS m FROM customers").fetchone()["m"]
    assert lowest >= config.CREDIT_SCORE_LOAN_FLOOR


# --- test 12 --------------------------------------------------------------


def test_a_fresh_build_matches_the_committed_manifest(built_db, tmp_path):
    manifest = Path(config.DB_MANIFEST).read_text(encoding="utf-8")
    committed = re.search(r"```\n([0-9a-f]{64})\n```", manifest)
    assert committed, "no content hash found in the committed manifest"
    assert build.content_hash(built_db) == committed.group(1)


def test_the_manifest_lists_every_table_and_the_seed():
    manifest = Path(config.DB_MANIFEST).read_text(encoding="utf-8")
    for table in schema.TABLE_ORDER:
        assert f"`{table}`" in manifest
    assert f"`{config.SEED}`" in manifest


def test_two_builds_produce_the_same_content_hash(tmp_path):
    a = tmp_path / "a.db"
    b = tmp_path / "b.db"
    build.build_database(a)
    build.build_database(b)
    assert build.content_hash(a) == build.content_hash(b)
