"""The grader's one-command check on the relational store.

Builds the database if it is absent, then runs six checks and prints five
worked example joins. Exits non-zero if any check fails, so it is usable in CI
as well as by eye.

PII is always masked. The values are fabricated, so printing them in the clear
would be harmless, but masking costs nothing and shows the discipline before
Part 2 Task 10 formalises it. No flag is offered to unmask: one behaviour.
"""

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from db import build, schema  # noqa: E402

PII_COLUMNS = ("pan", "aadhaar", "account_number", "email", "phone")

CHILD_FOREIGN_KEYS = [
    ("loan_applications", "customer_id", "customers", "customer_id"),
    ("loan_applications", "product_code", "loan_products", "product_code"),
    ("application_events", "record_id", "loan_applications", "record_id"),
    ("repayments", "record_id", "loan_applications", "record_id"),
    ("support_tickets", "customer_id", "customers", "customer_id"),
    ("support_tickets", "record_id", "loan_applications", "record_id"),
    ("kyc_documents", "customer_id", "customers", "customer_id"),
]

failures: list[str] = []


def mask(value) -> str:
    """Fabricated values, masked anyway. See the module docstring."""
    text = str(value)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}{'*' * (len(text) - 4)}{text[-2:]}"


def heading(number: int, title: str) -> None:
    print()
    print(f"{number}. {title}")
    print("-" * (len(title) + 3))


def check(condition: bool, message: str) -> None:
    print(f"   {'PASS' if condition else 'FAIL'}  {message}")
    if not condition:
        failures.append(message)


def show_rows(conn, sql: str, params=()) -> list:
    """Run a query and print it as a table, masking any PII column."""
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("   (no rows)")
        return rows
    columns = rows[0].keys()
    widths = {
        c: max(len(c), *(len(mask(r[c]) if c in PII_COLUMNS else str(r[c])) for r in rows))
        for c in columns
    }
    print("   " + "  ".join(c.ljust(widths[c]) for c in columns))
    print("   " + "  ".join("-" * widths[c] for c in columns))
    for r in rows:
        cells = [
            (mask(r[c]) if c in PII_COLUMNS else str(r[c])).ljust(widths[c])
            for c in columns
        ]
        print("   " + "  ".join(cells))
    return rows


def main() -> int:
    print("Meridian Bank relational store - grader check")
    print(f"database: {config.DB_PATH}")

    if not Path(config.DB_PATH).exists():
        print("not found, building it now")
        build.build_database()
        build.write_manifest()

    conn = schema.connect(config.DB_PATH)

    # --- 1 -----------------------------------------------------------------
    heading(1, "Schema and row counts")
    total = 0
    for table in schema.TABLE_ORDER:
        cols = [c["name"] for c in conn.execute(f"PRAGMA table_info({table})")]
        n = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]
        total += n
        print(f"   {table:<20} {n:>5} rows   {len(cols)} columns: {', '.join(cols)}")
    print(f"   {'TOTAL':<20} {total:>5} rows")
    check(total > 0, "the database holds rows")

    # --- 2 -----------------------------------------------------------------
    heading(2, "Referential integrity")
    for child, column, parent, parent_key in CHILD_FOREIGN_KEYS:
        orphans = conn.execute(
            f"""SELECT count(*) AS n FROM {child} c
                 WHERE c.{column} IS NOT NULL
                   AND NOT EXISTS (SELECT 1 FROM {parent} p
                                    WHERE p.{parent_key} = c.{column})"""
        ).fetchone()["n"]
        check(orphans == 0, f"{child}.{column} -> {parent}: {orphans} orphans")
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    check(not violations, f"PRAGMA foreign_key_check: {len(violations)} violations")

    # --- 3 -----------------------------------------------------------------
    heading(3, "Domain bounds against the product catalogue")
    bad = conn.execute(
        """SELECT count(*) AS n FROM loan_applications a
             JOIN loan_products p USING (product_code)
            WHERE a.loan_amount_inr < p.min_amount_inr - 500
               OR a.loan_amount_inr > p.max_amount_inr + 500
               OR a.interest_rate_pct < p.min_rate_pct
               OR a.interest_rate_pct > p.max_rate_pct
               OR a.tenure_months > p.max_tenure_months"""
    ).fetchone()["n"]
    check(bad == 0, f"every amount, rate and tenure inside its product row ({bad} outside)")

    out_of_range = conn.execute(
        f"""SELECT count(*) AS n FROM loan_applications
             WHERE days_since_created NOT BETWEEN {config.DAYS_MIN} AND {config.DAYS_MAX}"""
    ).fetchone()["n"]
    check(out_of_range == 0, f"days_since_created within 0 to 30 ({out_of_range} outside)")

    # --- the time axis, D-26 to D-30 ---------------------------------------
    anchor = config.AS_OF
    stamped = {
        "loan_applications": ("created_at", "updated_at"),
        "application_events": ("occurred_at",),
        "repayments": ("due_at",),
        "support_tickets": ("opened_at",),
        "kyc_documents": ("submitted_at",),
    }

    unparseable = []
    future = []
    for table, columns in stamped.items():
        for column in columns:
            for row in conn.execute(f"SELECT {column} AS v FROM {table}"):
                try:
                    moment = datetime.fromisoformat(row["v"])
                except (TypeError, ValueError):
                    unparseable.append(f"{table}.{column}")
                    continue
                if moment.utcoffset() is None:
                    unparseable.append(f"{table}.{column}")
                # due_at is the one column allowed past the anchor: future
                # instalments are genuinely in the future (D-29).
                elif moment > anchor and column != "due_at":
                    future.append(f"{table}.{column}")
    check(not unparseable, f"every timestamp is ISO 8601 with an offset ({len(unparseable)} bad)")
    check(not future, f"no timestamp past the anchor except due_at ({len(future)} ahead)")

    drifted = sum(
        1
        for row in conn.execute(
            "SELECT days_since_created AS d, created_at AS c FROM loan_applications"
        )
        if datetime.fromisoformat(row["c"]).date()
        != (anchor - timedelta(days=row["d"])).date()
    )
    check(drifted == 0, f"created_at is the anchor minus days_since_created ({drifted} off)")

    weekend = conn.execute(
        """SELECT count(*) AS n FROM application_events
            WHERE strftime('%w', occurred_at) IN ('0', '6')
              AND NOT (sequence_no = 1 AND to_status = ?)""",
        (config.CUSTOMER_SIDE_ARRIVAL,),
    ).fetchone()["n"]
    check(weekend == 0, f"no bank-side event on a weekend ({weekend} found)")

    outside = sum(
        1
        for table, columns in stamped.items()
        for column in columns
        if column != "due_at"
        for row in conn.execute(f"SELECT {column} AS v FROM {table}")
        if not (
            config.BUSINESS_START
            <= datetime.fromisoformat(row["v"]).time()
            <= config.BUSINESS_END
        )
    )
    check(outside == 0, f"every drawn timestamp inside business hours ({outside} outside)")

    stale = conn.execute(
        """SELECT count(*) AS n FROM loan_applications a
            WHERE a.updated_at != (SELECT max(e.occurred_at)
                                     FROM application_events e
                                    WHERE e.record_id = a.record_id)"""
    ).fetchone()["n"]
    check(stale == 0, f"updated_at is the most recent event ({stale} disagree)")

    mismatched = sum(
        1
        for row in conn.execute("SELECT due_at AS d, paid FROM repayments")
        if bool(row["paid"]) != (datetime.fromisoformat(row["d"]) <= anchor)
    )
    check(mismatched == 0, f"paid agrees with due_at against the anchor ({mismatched} off)")

    vocab = {r["status"] for r in conn.execute("SELECT DISTINCT status FROM loan_applications")}
    check(vocab <= set(config.STATUSES), f"statuses in vocabulary: {sorted(vocab)}")

    low = conn.execute("SELECT min(credit_score) AS m FROM customers").fetchone()["m"]
    check(
        low >= config.CREDIT_SCORE_LOAN_FLOOR,
        f"every loan holder at or above kb-01's credit floor of "
        f"{config.CREDIT_SCORE_LOAN_FLOOR} (lowest is {low})",
    )

    # --- 4 -----------------------------------------------------------------
    heading(4, "Knowledge-base agreement")
    rows = conn.execute(
        """SELECT r.emi_inr, a.loan_amount_inr, a.interest_rate_pct, a.tenure_months
             FROM repayments r JOIN loan_applications a USING (record_id)"""
    ).fetchall()
    worst = 0.0
    for row in rows:
        rate = row["interest_rate_pct"] / 12.0 / 100.0
        growth = (1.0 + rate) ** row["tenure_months"]
        expected = row["loan_amount_inr"] * rate * growth / (growth - 1.0)
        worst = max(worst, abs(row["emi_inr"] - expected))
    check(
        worst < 1.0,
        f"all {len(rows)} repayments reproduce kb-02's EMI formula "
        f"(largest deviation {worst:.4f} rupees)",
    )

    kb04 = (config.KB_DIR / "kb-04-kyc-documents.txt").read_text(encoding="utf-8").lower()
    kb12 = (config.KB_DIR / "kb-12-nri-account-eligibility.txt").read_text(encoding="utf-8").lower()
    unknown = [
        r["doc_type"]
        for r in conn.execute("SELECT DISTINCT doc_type FROM kyc_documents")
        if r["doc_type"].lower() not in kb04 + kb12
    ]
    check(not unknown, f"every doc_type appears in kb-04 or kb-12 (unknown: {unknown})")

    kb05 = (config.KB_DIR / "kb-05-fraud-dispute.txt").read_text(encoding="utf-8").lower()
    unknown = [
        r["channel"]
        for r in conn.execute("SELECT DISTINCT channel FROM support_tickets")
        if r["channel"].lower() not in kb05
    ]
    check(not unknown, f"every ticket channel appears in kb-05 (unknown: {unknown})")

    # --- 5 -----------------------------------------------------------------
    heading(5, "Manifest match")
    manifest = Path(config.DB_MANIFEST)
    if not manifest.exists():
        check(False, "data/database-manifest.md is missing")
    else:
        committed = re.search(r"```\n([0-9a-f]{64})\n```", manifest.read_text(encoding="utf-8"))
        actual = build.content_hash()
        print(f"   committed {committed.group(1) if committed else 'none'}")
        print(f"   built     {actual}")
        check(bool(committed) and committed.group(1) == actual,
              "the built database matches the committed manifest")

    # --- 6 -----------------------------------------------------------------
    heading(6, "Five worked joins")
    examples = [
        ("A customer's full loan book",
         """SELECT c.customer_id, c.full_name, c.pan, c.credit_score,
                   a.record_id, a.category, a.status, a.loan_amount_inr
              FROM customers c JOIN loan_applications a USING (customer_id)
             WHERE c.customer_id = (SELECT customer_id FROM loan_applications
                                     GROUP BY customer_id ORDER BY count(*) DESC LIMIT 1)
             ORDER BY a.record_id"""),
        ("The status trail of one application",
         """SELECT record_id, sequence_no, from_status, to_status, occurred_at
              FROM application_events WHERE record_id = 'LN-1001' ORDER BY sequence_no"""),
        ("First three instalments of one disbursed loan",
         """SELECT record_id, instalment_no, emi_inr, principal_inr, interest_inr, balance_inr, paid
              FROM repayments
             WHERE record_id = (SELECT record_id FROM loan_applications
                                 WHERE status = 'Disbursed' ORDER BY record_id LIMIT 1)
             ORDER BY instalment_no LIMIT 3"""),
        ("Fraud-flagged applications and their open tickets",
         """SELECT a.record_id, a.category, a.loan_amount_inr, t.ticket_id, t.channel, t.status
              FROM loan_applications a JOIN support_tickets t USING (record_id)
             WHERE a.flagged_for_fraud_review = 1 ORDER BY a.record_id LIMIT 5"""),
        ("KYC documents for one non-resident customer",
         """SELECT c.customer_id, c.full_name, c.aadhaar, k.doc_type, k.doc_kind, k.verified
              FROM customers c JOIN kyc_documents k USING (customer_id)
             WHERE c.is_nri = 1 ORDER BY c.customer_id, k.doc_kind LIMIT 6"""),
    ]
    for title, sql in examples:
        print()
        print(f"   {title}")
        for line in sql.strip().splitlines():
            print(f"     | {line.strip()}")
        show_rows(conn, sql)

    conn.close()

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
