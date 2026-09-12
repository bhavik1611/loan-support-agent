"""The seven CREATE TABLE statements and nothing else.

SQLite does not enforce foreign keys unless the pragma is set per connection,
so connect() sets it. Without that, the orphan test in tests/test_database.py
would pass happily against a database full of orphans.
"""

import sqlite3

# Insertion order, and it must respect the foreign keys.
TABLE_ORDER = [
    "loan_products",
    "customers",
    "loan_applications",
    "application_events",
    "repayments",
    "support_tickets",
    "kyc_documents",
]

CREATE_STATEMENTS = {
    "loan_products": """
        CREATE TABLE loan_products (
            product_code        TEXT    PRIMARY KEY,
            category            TEXT    NOT NULL UNIQUE,
            min_amount_inr      INTEGER NOT NULL,
            max_amount_inr      INTEGER NOT NULL,
            min_rate_pct        REAL    NOT NULL,
            max_rate_pct        REAL    NOT NULL,
            max_tenure_months   INTEGER NOT NULL,
            is_secured          INTEGER NOT NULL CHECK (is_secured IN (0, 1))
        )
    """,
    "customers": """
        CREATE TABLE customers (
            customer_id         TEXT    PRIMARY KEY,
            full_name           TEXT    NOT NULL,
            city                TEXT    NOT NULL,
            state               TEXT    NOT NULL,
            pan                 TEXT    NOT NULL UNIQUE,
            aadhaar             TEXT    NOT NULL UNIQUE,
            account_number      TEXT    NOT NULL UNIQUE,
            email               TEXT    NOT NULL,
            phone               TEXT    NOT NULL,
            employment_type     TEXT    NOT NULL,
            annual_income_inr   INTEGER NOT NULL,
            credit_score        INTEGER NOT NULL,
            kyc_status          TEXT    NOT NULL,
            is_nri              INTEGER NOT NULL CHECK (is_nri IN (0, 1))
        )
    """,
    "loan_applications": """
        CREATE TABLE loan_applications (
            record_id                TEXT    PRIMARY KEY,
            customer_id              TEXT    NOT NULL REFERENCES customers(customer_id),
            product_code             TEXT    NOT NULL REFERENCES loan_products(product_code),
            category                 TEXT    NOT NULL,
            status                   TEXT    NOT NULL,
            loan_amount_inr          INTEGER NOT NULL,
            days_since_created       INTEGER NOT NULL,
            flagged_for_fraud_review INTEGER NOT NULL CHECK (flagged_for_fraud_review IN (0, 1)),
            created_at               TEXT    NOT NULL,
            updated_at               TEXT    NOT NULL,
            tenure_months            INTEGER NOT NULL,
            interest_rate_pct        REAL    NOT NULL
        )
    """,
    "application_events": """
        CREATE TABLE application_events (
            event_id            INTEGER PRIMARY KEY,
            record_id           TEXT    NOT NULL REFERENCES loan_applications(record_id),
            sequence_no         INTEGER NOT NULL,
            from_status         TEXT,
            to_status           TEXT    NOT NULL,
            occurred_at         TEXT    NOT NULL,
            note                TEXT    NOT NULL,
            UNIQUE (record_id, sequence_no)
        )
    """,
    "repayments": """
        CREATE TABLE repayments (
            repayment_id    INTEGER PRIMARY KEY,
            record_id       TEXT    NOT NULL REFERENCES loan_applications(record_id),
            instalment_no   INTEGER NOT NULL,
            due_at          TEXT    NOT NULL,
            emi_inr         REAL    NOT NULL,
            principal_inr   REAL    NOT NULL,
            interest_inr    REAL    NOT NULL,
            balance_inr     REAL    NOT NULL,
            paid            INTEGER NOT NULL CHECK (paid IN (0, 1)),
            UNIQUE (record_id, instalment_no)
        )
    """,
    "support_tickets": """
        CREATE TABLE support_tickets (
            ticket_id           TEXT    PRIMARY KEY,
            customer_id         TEXT    NOT NULL REFERENCES customers(customer_id),
            record_id           TEXT    REFERENCES loan_applications(record_id),
            channel             TEXT    NOT NULL,
            category            TEXT    NOT NULL,
            opened_at           TEXT    NOT NULL,
            status              TEXT    NOT NULL,
            summary             TEXT    NOT NULL
        )
    """,
    "kyc_documents": """
        CREATE TABLE kyc_documents (
            document_id         TEXT    PRIMARY KEY,
            customer_id         TEXT    NOT NULL REFERENCES customers(customer_id),
            doc_type            TEXT    NOT NULL,
            doc_kind            TEXT    NOT NULL CHECK (doc_kind IN ('identity', 'address')),
            submitted_at        TEXT    NOT NULL,
            verified            INTEGER NOT NULL CHECK (verified IN (0, 1))
        )
    """,
}


def connect(path) -> sqlite3.Connection:
    """Open a connection with foreign keys actually enforced."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def create_all(conn) -> None:
    for table in TABLE_ORDER:
        conn.execute(CREATE_STATEMENTS[table])
