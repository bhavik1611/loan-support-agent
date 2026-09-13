"""The read helpers, and the PII contract that customer_context must keep."""

import config
from db import query

FORBIDDEN = {"pan", "aadhaar", "account_number", "email", "phone"}


def test_customer_context_never_returns_pii(db_conn):
    """D-21. The brief masks PII on the input side; this path must not serve it."""
    context = query.customer_context("LN-1001", conn=db_conn)
    assert context is not None
    assert set(context) == set(query.CONTEXT_FIELDS)
    assert FORBIDDEN.isdisjoint(context)


def test_no_application_leaks_pii_through_customer_context(db_conn):
    for record_id in ("LN-1001", "LN-1050", "LN-1100"):
        context = query.customer_context(record_id, conn=db_conn)
        assert FORBIDDEN.isdisjoint(context), record_id


def test_customer_context_is_none_for_an_unknown_record(db_conn):
    assert query.customer_context("LN-9999", conn=db_conn) is None


def test_open_loan_count_counts_only_open_statuses(db_conn):
    context = query.customer_context("LN-1001", conn=db_conn)
    book = query.loan_book(context["customer_id"], conn=db_conn)
    expected = sum(1 for loan in book if loan["status"] in query.OPEN_STATUSES)
    assert context["open_loan_count"] == expected


def test_loan_book_returns_only_that_customers_loans(db_conn):
    context = query.customer_context("LN-1001", conn=db_conn)
    book = query.loan_book(context["customer_id"], conn=db_conn)
    assert book
    assert any(loan["record_id"] == "LN-1001" for loan in book)
    ids = {loan["record_id"] for loan in book}
    rows = db_conn.execute(
        "SELECT record_id FROM loan_applications WHERE customer_id = ?",
        (context["customer_id"],),
    ).fetchall()
    assert ids == {r["record_id"] for r in rows}


def test_every_application_belongs_to_exactly_one_loan_book(db_conn):
    total = sum(
        len(query.loan_book(r["customer_id"], conn=db_conn))
        for r in db_conn.execute("SELECT customer_id FROM customers")
    )
    assert total == config.RECORD_COUNT


def test_application_timeline_is_ordered_and_ends_at_the_current_status(db_conn):
    for row in db_conn.execute("SELECT record_id, status FROM loan_applications LIMIT 20"):
        timeline = query.application_timeline(row["record_id"], conn=db_conn)
        assert timeline, row["record_id"]
        assert [e["sequence_no"] for e in timeline] == list(range(1, len(timeline) + 1))
        assert timeline[-1]["to_status"] == row["status"]


def test_repayment_schedule_is_empty_unless_disbursed(db_conn):
    pending = db_conn.execute(
        "SELECT record_id FROM loan_applications WHERE status != 'Disbursed' LIMIT 5"
    ).fetchall()
    for row in pending:
        assert query.repayment_schedule(row["record_id"], conn=db_conn) == []

    disbursed = db_conn.execute(
        "SELECT record_id FROM loan_applications WHERE status = 'Disbursed' LIMIT 5"
    ).fetchall()
    assert disbursed
    for row in disbursed:
        schedule = query.repayment_schedule(row["record_id"], conn=db_conn)
        assert len(schedule) == config.SCHEDULE_MONTHS
        assert [s["instalment_no"] for s in schedule] == list(
            range(1, config.SCHEDULE_MONTHS + 1)
        )


def test_table_counts_covers_every_table(db_conn):
    counts = query.table_counts(conn=db_conn)
    assert len(counts) == 7
    assert counts["loan_applications"] == config.RECORD_COUNT
    assert counts["customers"] == config.CUSTOMER_COUNT
    assert counts["loan_products"] == len(config.CATEGORIES)
