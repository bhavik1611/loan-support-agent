"""Loan-application dataset for the Cred support agent (Part 1, Task 1).

The brief fixes six fields per record. Those six are a *projection*: behind each
one sits an application with an event history, and `status` and
`days_since_created` are both read off that history rather than drawn
independently. That is what stops a loan being `Disbursed` and 25 days old and
simultaneously queued for urgent attention.

Design decisions and the measurements behind them: docs/capstone-dataset-design.md

Run `python dataset.py` to regenerate, validate and print the report.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Design constants. Every number here is argued for in the design doc.
# --------------------------------------------------------------------------

SEED = 3
N_RECORDS = 500
MEMBER_POOL = 2000          # ~1 application in 10 belongs to a repeat member
OBSERVATION_WINDOW = 30     # the brief fixes days_since_created at 0..30

CATEGORIES = ["Personal Loan", "Home Loan", "Auto Loan", "Education Loan", "Business Loan"]
STATUSES = ["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]
OPEN_STATUSES = {"Submitted", "Under Review"}

# Time from submission to a decision, and the plausible amount band, per product.
# Slow secured products need valuation and legal checks; a personal loan does not.
PRODUCTS = {
    "Personal Loan":  {"decision_days": (2, 10),  "amount": (50_000, 25_00_000)},
    "Auto Loan":      {"decision_days": (3, 14),  "amount": (1_00_000, 30_00_000)},
    "Education Loan": {"decision_days": (7, 25),  "amount": (1_00_000, 50_00_000)},
    "Business Loan":  {"decision_days": (10, 35), "amount": (2_00_000, 75_00_000)},
    "Home Loan":      {"decision_days": (20, 50), "amount": (10_00_000, 1_50_00_000)},
}

AMOUNT_SKEW = 2.2           # >1 pulls draws toward the band floor; 1.0 would be uniform
AMOUNT_ROUNDING = 10_000    # lenders sanction round numbers

STALL_RATE = 0.10           # 1 review in 10 drags
STALL_DAYS = (22, 34)
APPROVAL_RATE = 0.70

FRAUD_BASE = 0.15           # gentle correlation: either signal roughly doubles the odds
FRAUD_BUMP_LARGE = 0.13     # amount in the top fifth of its product band
FRAUD_BUMP_REPEAT = 0.13    # member already has an application in this book
LARGE_FOR_PRODUCT = 0.80

DATA_DIR = Path(__file__).parent / "data"
SNAPSHOT_PATH = DATA_DIR / "loan_applications.json"
CHECKSUM_PATH = DATA_DIR / "loan_applications.sha256"
SQLITE_PATH = DATA_DIR / "loan_applications.sqlite"


# --------------------------------------------------------------------------
# The internal record: what the six public fields are a view of.
# --------------------------------------------------------------------------

@dataclass
class Event:
    """One stage change in an application's life."""
    stage: str
    day: int        # days after the application was created
    actor: str

    def as_dict(self) -> dict:
        return {"stage": self.stage, "day": self.day, "actor": self.actor}


@dataclass
class Application:
    record_id: str
    member_ref: str
    category: str
    loan_amount_inr: int
    days_since_created: int
    flagged_for_fraud_review: bool
    history: list[Event] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Whatever stage the application had reached by the observation day.

        Read off the history, never drawn, so it cannot contradict the age.
        """
        return self.history[-1].stage

    def project(self) -> dict:
        """The six fields the brief fixes."""
        return {
            "record_id": self.record_id,
            "category": self.category,
            "status": self.status,
            "loan_amount_inr": self.loan_amount_inr,
            "days_since_created": self.days_since_created,
            "flagged_for_fraud_review": self.flagged_for_fraud_review,
        }


def _schedule(rng: random.Random, category: str) -> list[Event]:
    """Plan the full life of one application, whether or not it has happened yet."""
    to_review = rng.randint(1, 3)
    if rng.random() < STALL_RATE:
        to_decision = to_review + rng.randint(*STALL_DAYS)
    else:
        lo, hi = PRODUCTS[category]["decision_days"]
        to_decision = max(to_review + 1, rng.randint(lo, hi))
    approved = rng.random() < APPROVAL_RATE

    plan = [
        Event("Submitted", 0, "member"),
        Event("Under Review", to_review, "credit officer"),
    ]
    if approved:
        plan.append(Event("Approved", to_decision, "underwriter"))
        plan.append(Event("Disbursed", to_decision + rng.randint(1, 4), "disbursement system"))
    else:
        plan.append(Event("Rejected", to_decision, "underwriter"))
    return plan


def _draw_amount(rng: random.Random, category: str) -> tuple[int, float]:
    """An amount inside the product's band, skewed toward the floor.

    Returns the amount and where it sits in the band, 0.0 at the floor to 1.0 at
    the ceiling. The position is what the fraud signal reads.
    """
    lo, hi = PRODUCTS[category]["amount"]
    raw = lo + (hi - lo) * (rng.random() ** AMOUNT_SKEW)
    amount = int(raw / AMOUNT_ROUNDING) * AMOUNT_ROUNDING
    amount = max(lo, min(hi, amount))
    return amount, (amount - lo) / (hi - lo)


def generate(seed: int = SEED, n: int = N_RECORDS) -> list[Application]:
    """Build the book. Deterministic for a given seed."""
    rng = random.Random(seed)
    seen_members: set[str] = set()
    applications: list[Application] = []

    for i in range(n):
        # The first fifteen round-robin the categories, so every category clears
        # the brief's floor of three regardless of which seed is chosen.
        category = CATEGORIES[i % len(CATEGORIES)] if i < 15 else rng.choice(CATEGORIES)

        amount, band_position = _draw_amount(rng, category)
        days_since_created = rng.randint(0, OBSERVATION_WINDOW)
        plan = _schedule(rng, category)

        member_ref = f"CRED-M{rng.randrange(MEMBER_POOL):05d}"
        is_repeat = member_ref in seen_members
        seen_members.add(member_ref)

        fraud_odds = FRAUD_BASE
        if band_position > LARGE_FOR_PRODUCT:
            fraud_odds += FRAUD_BUMP_LARGE
        if is_repeat:
            fraud_odds += FRAUD_BUMP_REPEAT
        flagged = rng.random() < fraud_odds

        # The audit trail holds what has happened, not what is scheduled to.
        history = [e for e in plan if e.day <= days_since_created]

        applications.append(Application(
            record_id=f"CRED-{1000 + i}",
            member_ref=member_ref,
            category=category,
            loan_amount_inr=amount,
            days_since_created=days_since_created,
            flagged_for_fraud_review=flagged,
            history=history,
        ))

    return applications


# --------------------------------------------------------------------------
# Validation: the structural thresholds the brief sets.
# --------------------------------------------------------------------------

def validate(applications: list[Application]) -> tuple[bool, list[str]]:
    """Check every threshold in Task 1. Returns (passed, list of failures)."""
    failures = []
    by_category = Counter(a.category for a in applications)
    by_status = Counter(a.status for a in applications)
    fraud_pct = 100 * sum(a.flagged_for_fraud_review for a in applications) / len(applications)

    if len(applications) < 40:
        failures.append(f"only {len(applications)} records, brief requires at least 40")
    for c in CATEGORIES:
        if by_category[c] < 3:
            failures.append(f"category {c!r} has {by_category[c]} records, needs at least 3")
    for s in STATUSES:
        if by_status[s] < 1:
            failures.append(f"status {s!r} is absent, needs at least 1")
    if not 10.0 <= fraud_pct <= 30.0:
        failures.append(f"fraud rate {fraud_pct:.1f}% is outside the 10-30% band")

    # Not required by the brief, but this is the defect the whole design exists
    # to remove, so it is checked rather than assumed.
    for a in applications:
        if a.history[-1].day > a.days_since_created:
            failures.append(f"{a.record_id}: status is ahead of its own clock")
        if a.status in OPEN_STATUSES and len(a.history) > 2:
            failures.append(f"{a.record_id}: open but carries a decision event")

    return not failures, failures


def report(applications: list[Application]) -> str:
    """The printed report Task 1 asks for."""
    by_category = Counter(a.category for a in applications)
    by_status = Counter(a.status for a in applications)
    flagged = sum(a.flagged_for_fraud_review for a in applications)
    n = len(applications)

    lines = [
        f"LOAN_APPLICATIONS: {n} records, seed {SEED}",
        "",
        "Records per category (brief requires >= 3 each):",
    ]
    for c in CATEGORIES:
        lines.append(f"  {c:<16} {by_category[c]:>4}")
    lines += ["", "Records per status (brief requires >= 1 each):"]
    for s in STATUSES:
        lines.append(f"  {s:<16} {by_status[s]:>4}")
    lines += [
        "",
        f"Flagged for fraud review: {flagged}/{n} = {100 * flagged / n:.1f}% "
        f"(brief requires 10-30%)",
    ]

    # Evidence that the fraud flag is not a coin flip, and not a lookup either.
    large = [a for a in applications if _band_position(a) > LARGE_FOR_PRODUCT]
    repeats = _repeat_applications(applications)
    repeat_ids = {a.record_id for a in repeats}
    ordinary = [a for a in applications
                if _band_position(a) <= LARGE_FOR_PRODUCT and a.record_id not in repeat_ids]
    lines += [
        "",
        "Fraud flag by slice (it should differ, but not be predictable):",
        f"  ordinary application      {_pct_flagged(ordinary):>5.1f}%",
        f"  large for its product     {_pct_flagged(large):>5.1f}%",
        f"  repeat member             {_pct_flagged(repeats):>5.1f}%",
    ]

    lines += ["", "Amount by product (skewed toward the band floor):"]
    for c in CATEGORIES:
        amounts = sorted(a.loan_amount_inr for a in applications if a.category == c)
        lo, hi = PRODUCTS[c]["amount"]
        median = amounts[len(amounts) // 2]
        lines.append(f"  {c:<16} band {_inr(lo):>9} to {_inr(hi):>9}   median {_inr(median):>9}")

    open_apps = [a for a in applications if a.status in OPEN_STATUSES]
    oldest_open = max(a.days_since_created for a in open_apps)
    lines += [
        "",
        f"Open applications: {len(open_apps)}/{n}, oldest {oldest_open} days.",
        f"Stalled and still open past 25 days: "
        f"{sum(1 for a in open_apps if a.days_since_created >= 25)}",
    ]
    return "\n".join(lines)


def _band_position(a: Application) -> float:
    lo, hi = PRODUCTS[a.category]["amount"]
    return (a.loan_amount_inr - lo) / (hi - lo)


def _repeat_applications(applications: list[Application]) -> list[Application]:
    """Applications belonging to a member who appears more than once."""
    counts = Counter(a.member_ref for a in applications)
    return [a for a in applications if counts[a.member_ref] > 1]


def _pct_flagged(subset: list[Application]) -> float:
    if not subset:
        return 0.0
    return 100 * sum(a.flagged_for_fraud_review for a in subset) / len(subset)


def _inr(n: int) -> str:
    if n >= 1_00_00_000:
        return f"{n / 1_00_00_000:.2f} cr"
    return f"{n / 1_00_000:.1f} L"


# --------------------------------------------------------------------------
# Reproducibility: a snapshot the grader can check, and a queryable copy.
# --------------------------------------------------------------------------

def snapshot_bytes(applications: list[Application]) -> bytes:
    """Canonical serialisation, so the checksum is stable across machines."""
    payload = [
        {**a.project(),
         "member_ref": a.member_ref,
         "history": [e.as_dict() for e in a.history]}
        for a in applications
    ]
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def write_snapshot(applications: list[Application]) -> str:
    DATA_DIR.mkdir(exist_ok=True)
    blob = snapshot_bytes(applications)
    digest = hashlib.sha256(blob).hexdigest()
    SNAPSHOT_PATH.write_bytes(blob)
    CHECKSUM_PATH.write_text(f"{digest}  {SNAPSHOT_PATH.name}\n")
    return digest


def verify_snapshot(applications: list[Application]) -> tuple[bool, str, str]:
    """Re-derive the checksum and compare against the committed one.

    A drift between Python versions fails here loudly instead of quietly
    grading different data.
    """
    if not CHECKSUM_PATH.exists():
        return False, "", "no committed checksum"
    expected = CHECKSUM_PATH.read_text().split()[0]
    actual = hashlib.sha256(snapshot_bytes(applications)).hexdigest()
    return actual == expected, actual, expected


def write_sqlite(applications: list[Application]) -> None:
    """The agent queries this, the way a deployed system queries its records.

    Swapping the generator for a real feed leaves every query untouched.
    """
    DATA_DIR.mkdir(exist_ok=True)
    SQLITE_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(SQLITE_PATH)
    con.executescript("""
        CREATE TABLE applications (
            record_id                TEXT PRIMARY KEY,
            member_ref               TEXT NOT NULL,
            category                 TEXT NOT NULL,
            status                   TEXT NOT NULL,
            loan_amount_inr          INTEGER NOT NULL,
            days_since_created       INTEGER NOT NULL,
            flagged_for_fraud_review INTEGER NOT NULL
        );
        CREATE TABLE application_events (
            record_id TEXT NOT NULL REFERENCES applications(record_id),
            seq       INTEGER NOT NULL,
            stage     TEXT NOT NULL,
            day       INTEGER NOT NULL,
            actor     TEXT NOT NULL,
            PRIMARY KEY (record_id, seq)
        );
        CREATE INDEX idx_applications_member ON applications(member_ref);
    """)
    con.executemany(
        "INSERT INTO applications VALUES (:record_id, :member_ref, :category, :status,"
        " :loan_amount_inr, :days_since_created, :flagged_for_fraud_review)",
        [{**a.project(), "member_ref": a.member_ref,
          "flagged_for_fraud_review": int(a.flagged_for_fraud_review)}
         for a in applications],
    )
    con.executemany(
        "INSERT INTO application_events VALUES (?, ?, ?, ?, ?)",
        [(a.record_id, i, e.stage, e.day, e.actor)
         for a in applications for i, e in enumerate(a.history)],
    )
    con.commit()
    con.close()


APPLICATIONS = generate()
LOAN_APPLICATIONS = [a.project() for a in APPLICATIONS]


def main() -> int:
    print(report(APPLICATIONS))

    passed, failures = validate(APPLICATIONS)
    print("\n" + "-" * 60)
    if passed:
        print("VALIDATION: every threshold in Task 1 is met.")
    else:
        print("VALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")

    digest = write_snapshot(APPLICATIONS)
    write_sqlite(APPLICATIONS)
    print(f"\nSnapshot written: {SNAPSHOT_PATH.name}")
    print(f"SHA-256:          {digest}")
    print(f"SQLite written:   {SQLITE_PATH.name} "
          f"({len(APPLICATIONS)} applications, "
          f"{sum(len(a.history) for a in APPLICATIONS)} events)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
