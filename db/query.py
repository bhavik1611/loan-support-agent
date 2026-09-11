"""Read helpers Part 2 consumes.

customer_context is the one with a contract rather than merely an
implementation: it returns the customer id, name, credit score and open-loan
count, and nothing else. PAN, Aadhaar and account number must never come out
of it. The brief puts PII masking on the INPUT side and makes the output
guardrail a groundedness check, so serving PII through this path would need an
output masker the brief never asked for. Spec D-21, and tests/test_db_query.py
asserts the omission rather than trusting it.
"""

from contextlib import contextmanager

import config
from db import schema

CONTEXT_FIELDS = ("customer_id", "full_name", "credit_score", "open_loan_count")

# Statuses that mean the application is still moving through the funnel.
OPEN_STATUSES = ("Submitted", "Under Review", "Approved")


@contextmanager
def _connection(conn=None):
    """Use the caller's connection, or open and close our own."""
    if conn is not None:
        yield conn
        return
    owned = schema.connect(config.DB_PATH)
    try:
        yield owned
    finally:
        owned.close()


def customer_context(record_id: str, conn=None) -> dict | None:
    """Non-PII context for the customer behind one application.

    Returns exactly CONTEXT_FIELDS. Adding a PII column here is a defect, not
    an enhancement.
    """
    with _connection(conn) as c:
        row = c.execute(
            """
            SELECT cu.customer_id,
                   cu.full_name,
                   cu.credit_score,
                   (SELECT count(*) FROM loan_applications o
                     WHERE o.customer_id = cu.customer_id
                       AND o.status IN (?, ?, ?)) AS open_loan_count
              FROM loan_applications a
              JOIN customers cu USING (customer_id)
             WHERE a.record_id = ?
            """,
            (*OPEN_STATUSES, record_id),
        ).fetchone()
    return {field: row[field] for field in CONTEXT_FIELDS} if row else None


def loan_book(customer_id: str, conn=None) -> list[dict]:
    """Every application this customer holds, newest first."""
    with _connection(conn) as c:
        rows = c.execute(
            """
            SELECT record_id, category, status, loan_amount_inr,
                   tenure_months, interest_rate_pct, days_since_created,
                   flagged_for_fraud_review
              FROM loan_applications
             WHERE customer_id = ?
             ORDER BY days_since_created ASC, record_id ASC
            """,
            (customer_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def application_timeline(record_id: str, conn=None) -> list[dict]:
    """The status audit trail for one application, in sequence order."""
    with _connection(conn) as c:
        rows = c.execute(
            """
            SELECT sequence_no, from_status, to_status, occurred_days_ago, note
              FROM application_events
             WHERE record_id = ?
             ORDER BY sequence_no ASC
            """,
            (record_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def repayment_schedule(record_id: str, conn=None) -> list[dict]:
    """The amortisation schedule. Empty for anything not yet disbursed."""
    with _connection(conn) as c:
        rows = c.execute(
            """
            SELECT instalment_no, due_days_ago, emi_inr,
                   principal_inr, interest_inr, balance_inr, paid
              FROM repayments
             WHERE record_id = ?
             ORDER BY instalment_no ASC
            """,
            (record_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def open_tickets(customer_id: str, conn=None) -> list[dict]:
    """Support tickets this customer has open, newest first."""
    with _connection(conn) as c:
        rows = c.execute(
            """
            SELECT ticket_id, record_id, channel, category,
                   opened_days_ago, status, summary
              FROM support_tickets
             WHERE customer_id = ? AND status != 'Closed'
             ORDER BY opened_days_ago ASC, ticket_id ASC
            """,
            (customer_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def table_counts(conn=None) -> dict[str, int]:
    """Row count per table, in schema order."""
    with _connection(conn) as c:
        return {
            table: c.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]
            for table in schema.TABLE_ORDER
        }
