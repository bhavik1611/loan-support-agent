"""Every path, tunable and environment flag the project reads.

Nothing else in the repository hard-codes a filesystem path, a collection name
or a chunk parameter. Parts 2 to 4 import from here too.
"""

import os
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

# Unsecured and business lending carry higher fraud incidence than secured
# retail lending, so the flag carries signal Part 2's escalation score can use.
FRAUD_BASE_PROBABILITY = 0.15
FRAUD_CATEGORY_BONUS = {"Business Loan": 0.10, "Personal Loan": 0.05}

FRAUD_RATE_MIN = 0.10
FRAUD_RATE_MAX = 0.30
MIN_RECORDS_PER_CATEGORY = 3
MIN_RECORDS_PER_STATUS = 1

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

# --- Task 4, retrieval and the answer decision ----------------------------

TOP_K = 3
SUPPORT_MIN_SHARED = 2

# Measured, never preset. Task 11 replaces this with the midpoint between the
# minimum in-scope and the maximum out-of-scope top-1 cosine similarity across
# the 17 probes in eval/calibration.py, and records both clusters in README.md.
SIMILARITY_THRESHOLD = None

# --- Environment ----------------------------------------------------------

DEFAULT_PROVIDER = "mock"


def resolve_provider() -> str:
    """The active language-model provider, read fresh so tests can monkeypatch."""
    return os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)


LLM_PROVIDER = resolve_provider()
MOCK_LLM = LLM_PROVIDER == DEFAULT_PROVIDER
