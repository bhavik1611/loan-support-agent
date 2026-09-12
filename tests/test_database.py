"""Tests 7 to 12 of spec section 11.

Three of these assert the generated rows cannot contradict the knowledge base.
That is the price of D-16's seven tables and it is paid here rather than
trusted.
"""

import re
from pathlib import Path

from datetime import datetime, timedelta

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
    return (config.KB_DIR / f"{stem}.txt").read_text(encoding="utf-8").lower()


RATE_SENTENCE = re.compile(
    r"the (?P<category>[a-z ]+?) at meridian bank carries an interest rate "
    r"between (?P<low>\d+\.\d\d) and (?P<high>\d+\.\d\d) percent per annum"
)


def _kb07_rate_bands() -> dict[str, tuple[float, float]]:
    """The bands kb-07 actually quotes, keyed by the product they belong to."""
    return {
        m["category"]: (float(m["low"]), float(m["high"]))
        for m in RATE_SENTENCE.finditer(_kb_text("kb-07-interest-rate-slabs"))
    }


# --- test 7 ---------------------------------------------------------------


BRIEF_FIELDS = (
    "record_id",
    "category",
    "status",
    "loan_amount_inr",
    "days_since_created",
    "flagged_for_fraud_review",
)


def test_no_brief_field_value_has_ever_moved():
    """D-18, D-19 and D-31: nothing added to this dataset moved an old value.

    The store added six tables and the time axis added two fields, and neither
    was allowed to perturb stream 0. So the brief's six fields as the generator
    draws them today must still equal the six fields the committed snapshot
    carries. Comparing the raw stream-0 output rather than the projection is
    what makes this bite: if a future change draws anything new inside
    generate_applications, every record shifts and this fails.
    """
    import json

    committed = json.loads(config.DATASET_SNAPSHOT.read_text())
    drawn = dataset.generate_applications()
    assert len(committed) == len(drawn)
    for record, raw in zip(committed, drawn):
        assert {field: record[field] for field in BRIEF_FIELDS} == raw


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
    """A bare substring search would pass on any document mentioning "9.85"
    anywhere, so parse kb-07's own sentence and bind both bounds to the
    category they are quoted for."""
    bands = _kb07_rate_bands()
    assert bands, "kb-07 no longer states its rate bands in the parsed form"
    for row in db_conn.execute("SELECT category, min_rate_pct, max_rate_pct FROM loan_products"):
        quoted = bands.get(row["category"].lower())
        assert quoted is not None, (row["category"], sorted(bands))
        assert quoted == (row["min_rate_pct"], row["max_rate_pct"]), (row["category"], quoted)


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


# --- test 13 --------------------------------------------------------------

# Every timestamped column in the store, and whether it may sit past the
# anchor. repayments.due_at is the one that may: a future instalment is
# genuinely in the future (D-29).
STAMPED_COLUMNS = [
    ("loan_applications", "created_at", False),
    ("loan_applications", "updated_at", False),
    ("application_events", "occurred_at", False),
    ("repayments", "due_at", True),
    ("support_tickets", "opened_at", False),
    ("kyc_documents", "submitted_at", False),
]


def test_every_timestamp_is_iso_8601_with_an_offset(db_conn):
    """Test 13. D-26 and D-27: the derivation is total and offset-aware."""
    for table, column, _ in STAMPED_COLUMNS:
        for row in db_conn.execute(f"SELECT {column} AS value FROM {table}"):
            moment = datetime.fromisoformat(row["value"])
            assert moment.utcoffset() == config.IST.utcoffset(None), f"{table}.{column}"


def test_no_timestamp_sits_past_the_anchor_except_a_due_date(db_conn):
    """Test 13. Nothing is dated in the future, and due_at says why it may be."""
    for table, column, may_be_future in STAMPED_COLUMNS:
        if may_be_future:
            continue
        latest = db_conn.execute(f"SELECT max({column}) AS m FROM {table}").fetchone()["m"]
        assert datetime.fromisoformat(latest) <= config.AS_OF, f"{table}.{column}"


def test_no_column_below_loan_applications_still_carries_a_day_integer(db_conn):
    """Test 13. D-26 dropped the four `_days_ago` columns rather than keeping both."""
    for table in schema.TABLE_ORDER:
        columns = [c["name"] for c in db_conn.execute(f"PRAGMA table_info({table})")]
        assert not [c for c in columns if c.endswith("_days_ago")], table


# --- test 14 --------------------------------------------------------------


def test_created_at_is_the_anchor_minus_days_since_created(db_conn):
    """Test 14. D-26, the derivation everything else rests on.

    Recomputed from the anchor here rather than read back from the generator,
    so a change to how created_at is built fails this instead of redefining it.
    """
    rows = db_conn.execute(
        "SELECT record_id, days_since_created, created_at FROM loan_applications"
    ).fetchall()
    assert len(rows) == config.RECORD_COUNT
    for row in rows:
        expected = (config.AS_OF - timedelta(days=row["days_since_created"])).date()
        assert datetime.fromisoformat(row["created_at"]).date() == expected, row["record_id"]


# --- test 15 --------------------------------------------------------------


def test_no_bank_side_event_falls_on_a_weekend(db_conn):
    """Test 15. D-29, the rule that answers the defect anchoring exposed."""
    offenders = [
        row["event_id"]
        for row in db_conn.execute(
            "SELECT event_id, sequence_no, to_status, occurred_at FROM application_events"
        )
        if datetime.fromisoformat(row["occurred_at"]).weekday() >= 5
        and not (row["sequence_no"] == 1 and row["to_status"] == config.CUSTOMER_SIDE_ARRIVAL)
    ]
    assert offenders == []


def test_the_exempt_weekend_rows_are_all_online_submissions(db_conn):
    """Test 15. The exemption is narrow, and this says how narrow."""
    weekend = [
        row
        for row in db_conn.execute(
            "SELECT sequence_no, to_status, occurred_at FROM application_events"
        )
        if datetime.fromisoformat(row["occurred_at"]).weekday() >= 5
    ]
    assert weekend, "the exemption is meant to be exercised, not vacuous"
    for row in weekend:
        assert row["sequence_no"] == 1
        assert row["to_status"] == config.CUSTOMER_SIDE_ARRIVAL


def test_every_drawn_timestamp_sits_inside_business_hours(db_conn):
    """Test 15. D-29's window, on every column whose time was drawn."""
    for table, column, may_be_future in STAMPED_COLUMNS:
        if may_be_future:  # due_at takes the fixed debit time, not a drawn one
            continue
        for row in db_conn.execute(f"SELECT {column} AS value FROM {table}"):
            moment = datetime.fromisoformat(row["value"]).time()
            assert config.BUSINESS_START <= moment <= config.BUSINESS_END, f"{table}.{column}"


# --- test 16 --------------------------------------------------------------


def test_updated_at_is_the_most_recent_event(db_conn):
    """Test 16. D-30: the stored copy agrees with the trail it came from."""
    rows = db_conn.execute(
        """SELECT a.record_id, a.updated_at,
                  (SELECT max(e.occurred_at) FROM application_events e
                    WHERE e.record_id = a.record_id) AS latest
             FROM loan_applications a"""
    ).fetchall()
    assert len(rows) == config.RECORD_COUNT
    for row in rows:
        assert row["latest"] is not None, row["record_id"]
        assert row["updated_at"] == row["latest"], row["record_id"]


def test_paid_agrees_with_the_due_date_against_the_anchor(db_conn):
    """Test 16. D-13 kept the flag stored, so something has to pin it."""
    rows = db_conn.execute("SELECT repayment_id, due_at, paid FROM repayments").fetchall()
    assert rows
    for row in rows:
        expected = datetime.fromisoformat(row["due_at"]) <= config.AS_OF
        assert bool(row["paid"]) is expected, row["repayment_id"]
