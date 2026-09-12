"""Every path, tunable and environment flag the project reads.

Nothing else in the repository hard-codes a filesystem path, a collection name
or a chunk parameter. Parts 2 to 4 import from here too.
"""

import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

KB_DIR = REPO_ROOT / "knowledge_base"
DATA_DIR = REPO_ROOT / "data"
CHROMA_DIR = REPO_ROOT / "chroma"
TRANSCRIPT_DIR = REPO_ROOT / "transcripts"
DATASET_SNAPSHOT = DATA_DIR / "loan_applications.json"

# --- Task 1, dataset generation -------------------------------------------

SEED = 1  # first seed in 1..50 whose draw meets every structural invariant
RECORD_COUNT = 100
RECORD_ID_PREFIX = "LN-"
RECORD_ID_START = 1001

CATEGORIES = [
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan",
]

STATUSES = ["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]

# Retail lending mix: unsecured personal lending is the highest-volume product
# by count, business lending the lowest. Spec section 5.2.
CATEGORY_WEIGHTS = {
    "Personal Loan": 0.34,
    "Auto Loan": 0.22,
    "Home Loan": 0.20,
    "Education Loan": 0.12,
    "Business Loan": 0.12,
}

# A support queue is dominated by in-flight and recently completed
# applications, so the terminal states are not the largest buckets.
STATUS_WEIGHTS = {
    "Under Review": 0.24,
    "Approved": 0.22,
    "Disbursed": 0.22,
    "Submitted": 0.18,
    "Rejected": 0.14,
}

# Low and high in rupees. Four bands are sourced to published SBI and HDFC
# limits; the Home Loan band is a deliberate narrowing of a 50,000 to 50 crore
# lender range and is recorded as a modelling choice in README.md.
CATEGORY_BANDS = {
    "Personal Loan": (50_000, 40_00_000),
    "Auto Loan": (1_00_000, 25_00_000),
    "Education Loan": (1_00_000, 50_00_000),
    "Business Loan": (1_00_000, 50_00_000),
    "Home Loan": (10_00_000, 1_50_00_000),
}

DAYS_MIN = 0
DAYS_MAX = 30
DAYS_MODE = 8

# --- Task 1c, the time axis (D-26 to D-29) ---------------------------------

IST = timezone(timedelta(hours=5, minutes=30), "IST")

# The instant every relative offset in this repository is counted back from.
# Frozen, never read from the clock: a wall-clock anchor would break the
# determinism ground rule and move the manifest hash daily (D-27). Close of
# business rather than midnight, because a 0-day-old application exists in the
# committed data and a midnight anchor would date its events in the future.
AS_OF = datetime(2026, 9, 30, 18, 0, tzinfo=IST)

# D-29. Every timestamp lands inside this window, on both sides of the counter.
# Nothing in the knowledge base states branch hours, so this is a declared
# modelling choice rather than a sourced fact, and README.md says so.
BUSINESS_START = time(9, 30)
BUSINESS_END = time(18, 0)

# An auto-debit batch runs at a fixed hour and does not vary per loan, so an
# EMI due date takes this time rather than a drawn one (D-29).
EMI_DEBIT_TIME = time(10, 0)

# The one event the customer creates rather than the bank, and the only row
# exempt from the working-day shift. The generator's own note calls it an
# online submission, and an online form takes a Sunday one (D-29). The
# exemption is also load-bearing: this event is created_at, so moving it would
# break created_at == AS_OF minus days_since_created.
CUSTOMER_SIDE_ARRIVAL = "Submitted"

# Unsecured and business lending carry higher fraud incidence than secured
# retail lending, so the flag carries signal Part 2's escalation score can use.
FRAUD_BASE_PROBABILITY = 0.15
FRAUD_CATEGORY_BONUS = {"Business Loan": 0.10, "Personal Loan": 0.05}

FRAUD_RATE_MIN = 0.10
FRAUD_RATE_MAX = 0.30
MIN_RECORDS_PER_CATEGORY = 3
MIN_RECORDS_PER_STATUS = 1

# --- Task 1b, the relational store ----------------------------------------

DB_PATH = DATA_DIR / "meridian_bank.db"
DB_MANIFEST = DATA_DIR / "database-manifest.md"

# One seeded sub-stream per table. loan_applications keeps offset 0, which is
# literally the existing random.Random(SEED), so every record generated before
# the database existed is byte-identical afterwards. A table added later takes
# the next free offset and disturbs nothing. Spec D-18.
STREAM_OFFSETS = {
    "loan_applications": 0,
    "customers": 1,
    "loan_products": 2,
    "application_terms": 3,
    "application_events": 4,
    "repayments": 5,
    "support_tickets": 6,
    "kyc_documents": 7,
    # Not a table. The hours and minutes every timestamp carries, drawn here so
    # that adding a clock to four existing tables moves none of their committed
    # values (D-28). This is the "next free offset disturbs nothing" property
    # D-18 was designed for, used for the first time.
    "clock": 8,
}

PRODUCT_CODES = {
    "Personal Loan": "PL",
    "Home Loan": "HL",
    "Auto Loan": "AL",
    "Education Loan": "EL",
    "Business Loan": "BL",
}

SECURED_CATEGORIES = {"Home Loan", "Auto Loan"}

# Read straight out of knowledge_base/kb-07-interest-rate-slabs.txt. If a rate
# here disagrees with that document, the document wins and this is the bug.
RATE_BANDS = {
    "Home Loan": (8.40, 9.85),
    "Auto Loan": (9.10, 11.50),
    "Education Loan": (9.50, 12.25),
    "Personal Loan": (10.75, 18.00),
    "Business Loan": (11.00, 16.50),
}

# Personal is stated in kb-13 (12 to 60 months) and Home in kb-14 (30 years).
# The other three are NOT stated anywhere in the knowledge base, so they are
# declared modelling choices rather than presented as sourced facts, the same
# way the Home Loan amount band is. README.md says so.
TENURE_CHOICES = {
    "Personal Loan": (12, 24, 36, 48, 60),
    "Home Loan": (120, 180, 240, 300, 360),
    "Auto Loan": (12, 24, 36, 48, 60, 72, 84),
    "Education Loan": (36, 60, 84, 120, 180),
    "Business Loan": (12, 24, 36, 48, 60, 84),
}
MAX_TENURE_MONTHS = {c: max(t) for c, t in TENURE_CHOICES.items()}

# kb-01: "Meridian Bank requires a minimum credit score of 700 for any loan
# product." kb-10 gives the 300 to 900 range.
CREDIT_SCORE_MIN = 300
CREDIT_SCORE_MAX = 900
CREDIT_SCORE_LOAN_FLOOR = 700

# (loans held, how many customers hold that many). Sums to 100 loans over 66
# customers: most retail customers hold one, a minority two, a few more.
LOANS_PER_CUSTOMER_MIX = [(1, 42), (2, 16), (3, 6), (4, 2)]
CUSTOMER_COUNT = sum(count for _, count in LOANS_PER_CUSTOMER_MIX)

TICKET_COUNT = 40
SCHEDULE_MONTHS = 12  # first year of the amortisation schedule per disbursed loan


def stream(table: str) -> int:
    """The seed for one table's own random stream."""
    try:
        return SEED + STREAM_OFFSETS[table]
    except KeyError:
        raise ValueError(
            f"unknown table {table!r}, expected one of {sorted(STREAM_OFFSETS)}"
        ) from None


# --- Task 3, chunking and indexing ----------------------------------------

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

STRATEGY_FIXED = "fixed"
STRATEGY_SENTENCES = "sentences"

COLLECTION_FIXED = "kb_fixed_400_80"
COLLECTION_SENTENCES = "kb_sentences"

COLLECTION_FOR_STRATEGY = {
    STRATEGY_FIXED: COLLECTION_FIXED,
    STRATEGY_SENTENCES: COLLECTION_SENTENCES,
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_CHUNKS_PER_DOCUMENT = 2

# The default HNSW search breadth returns a non-nearest neighbour on some
# rebuilds, which breaks byte-identical reruns.
SEARCH_EF = 200

# --- Task 4, retrieval and the answer decision ----------------------------

TOP_K = 3
SUPPORT_MIN_SHARED = 2

# Re-derived on 2026-09-12 after pinning hnsw:search_ef (task 17b), over the
# same 12 in-scope and 17 out-of-scope probes pooled across both collections.
# Minimum in-scope top-1 0.3263, maximum out-of-scope top-1 0.2870, gap
# 0.0393, unchanged from the pre-fix measurement since neither pooled extreme
# was the probe the search breadth had destabilised. T is the midpoint. A
# tutorial preset of 0.5 would have wrongly refused 8 of the 24 in-scope
# measurements, which is why the brief bans one. Reproduce with
# scripts/run_part1.py, which writes transcripts/part1-calibration.txt.
SIMILARITY_THRESHOLD = 0.3066

# --- Part 2 Task 6, the escalation score (D-33, D-34) ---------------------

# Weighted so that neither signal alone at a typical value crosses the
# threshold, which is what forces the two to combine. A formula where either
# signal fires on its own is the bare boolean OR the brief forbids: measured
# on this dataset, 0.5*fraud + 0.5*(days/30) selects exactly the 16
# fraud-flagged records at its own 85th percentile and nothing else.
ESCALATION_FRAUD_WEIGHT = 0.45
ESCALATION_STALENESS_WEIGHT = 0.55

# Only 10 of the 100 records sit at or beyond 21 days, so the staleness term
# stops discriminating there. Declared modelling choice, not a sourced fact:
# the knowledge base states no loan-assessment turnaround. kb-01's 3 working
# days is the shortened path for an existing customer, not the standard one.
ESCALATION_SATURATION_DAYS = 21

# A disbursed loan's clock still runs because the money has left the bank,
# but no customer is waiting on a decision. A rejected file's clock stops.
ESCALATION_DISBURSED_RATE = 0.5
ESCALATION_HALF_RATE_STATUS = "Disbursed"

# The 80th percentile of the score over the committed 100 records. Measured,
# not preset. Reproduce with scripts/run_part2.py, which writes
# transcripts/part2-escalation.txt.
ESCALATION_THRESHOLD = 0.50

# --- Part 2 Task 10, guardrails (D-35, D-39) ------------------------------

# Aadhaar is twelve digits and an account number is eleven to sixteen, so the
# two overlap exactly. Masking keys on format alone and is therefore
# fail-safe; the Verhoeff check digit only decides which label to print. On
# the committed data 9 of 66 account numbers are twelve digits and 1 of those
# 9 also passes Verhoeff, so the label is wrong once in 66 and the redaction
# is never wrong. Measured, and stated in README.md.
PII_PLACEHOLDERS = {
    "PAN": "[PAN_REDACTED]",
    "AADHAAR": "[AADHAAR_REDACTED]",
    "ACCOUNT": "[ACCOUNT_REDACTED]",
}

ACCOUNT_DIGITS_MIN = 11
ACCOUNT_DIGITS_MAX = 16
AADHAAR_DIGITS = 12

# --- Part 2 Task 8, persisted memory (D-37) -------------------------------

# One JSON file per thread. Runtime state, so gitignored; the two graded
# demonstration threads are copied into transcripts/ by scripts/run_part2.py,
# which is where committed evidence lives under D-12.
CONVERSATION_DIR = DATA_DIR / "conversations"

# --- Part 2 Task 9, the response envelope (D-36, D-38) --------------------

# Committed, because it is what Part 3's FastAPI layer and the grader read.
# agent/schema.py exports it and scripts/run_part2.py asserts it is current.
RESPONSE_SCHEMA_PATH = REPO_ROOT / "agent" / "response.schema.json"

# --- Environment ----------------------------------------------------------

DEFAULT_PROVIDER = "mock"


def resolve_provider() -> str:
    """The active language-model provider, read fresh so tests can monkeypatch."""
    return os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)


LLM_PROVIDER = resolve_provider()
MOCK_LLM = LLM_PROVIDER == DEFAULT_PROVIDER
