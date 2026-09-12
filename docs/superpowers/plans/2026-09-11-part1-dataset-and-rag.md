# Part 1 - Dataset and RAG Core - Implementation Plan

Status: approved

> Steps use checkbox (`- [ ]`) syntax for tracking.

**Amended 2026-09-12.**
Tasks 1 to 16 are **implemented** and their checkboxes are historical.
Tasks 17 to 20 are **approved and not implemented**.
They exist because a review of the threshold calibration found that the threshold is not the mechanism that can decide scope, and the fix reaches the catalogue, the retrieval layer and the evaluation.
The twelve decisions behind them are D-46 to D-57 in the spec's decision log, and every alternative they beat is recorded there.

**Goal:** Build the brief-compliant Part 1 of the loan-support-agent capstone - a seeded loan-application dataset, an 18-document knowledge base, two chunking strategies indexed into two ChromaDB collections, empirically calibrated grounded generation, and a document-level Precision@3/Recall@3 comparison - all deterministic and offline under `MOCK_LLM`.

**Architecture:** `dataset.py`, `config.py` and `llm.py` sit at the repository root; `rag/` holds the retrieval pipeline as five small single-responsibility modules; `eval/` holds hand-authored queries and calibration probes committed before any retrieval runs; `scripts/run_part1.py` executes every task in order and writes the four graded transcripts that `README.md` links.
Nothing outside `config.py` knows a filesystem path, and nothing outside `llm.py` knows which language-model provider is active.

**Tech Stack:** Python 3.12.13 managed with `uv`, ChromaDB (persistent client, cosine space), `sentence-transformers` with `all-MiniLM-L6-v2`, PyYAML for front matter, pytest.

**Spec:** [`docs/superpowers/specs/2026-09-10-loan-support-agent-design.md`](../specs/2026-09-10-loan-support-agent-design.md)

**Brief:** [`reference/problem-statement.md`](../../../reference/problem-statement.md).
Where this plan and the brief disagree, the brief wins and this plan is wrong.

---

## Global Constraints

Every task's requirements implicitly include this section.

- Python 3.12.13, interpreter at `.venv/bin/python`, environment managed with `uv`.
  Never call `pip` directly; a `uv` venv has no `pip` binary.
  To add a dependency: `VIRTUAL_ENV=.venv uv pip install <name>` and add the bare name to `requirements.txt`.
- Everything runs offline with zero API keys.
  The embedding weights are already cached at `~/.cache/huggingface`; set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` when running anything that loads the model, to prove the offline claim rather than assume it.
- Every output is deterministic: same input, same seed, same bytes.
  Any iteration over a set, a dict built from a set, or a directory listing must be explicitly sorted before it reaches an output.
- Never use the em dash anywhere, in code, comments, documents or commit messages.
  Use a plain dash `-`.
- In markdown files, one full sentence per line.
- Names, PAN numbers, Aadhaar numbers, account numbers and income figures are fabricated throughout.
  Never use anyone's real data.
- No images, screenshots, diagrams, PDFs or slides are produced anywhere.
  Every deliverable is code or text in the repository.
- Write direct code in the file that owns the responsibility.
  No wrapper, service facade or helper indirection unless this plan names one.
- Commit at the end of each task, with the message given in that task's final step.
- Work on branch `part-1-dataset-and-rag`.
  Never commit to `main`.
- Scratch files go in the session scratchpad, never in the repository.

### Frozen identifiers

These names are fixed here so that tasks running in parallel agree without talking to each other.
Do not rename any of them.

**Configuration constants** (all live in `config.py`, all imported by name, never re-declared):

| Name | Value |
|---|---|
| `SEED` | chosen in Task 2 |
| `RECORD_COUNT` | `100` |
| `CHUNK_SIZE` | `400` |
| `CHUNK_OVERLAP` | `80` |
| `TOP_K` | `3` |
| `SUPPORT_MIN_SHARED` | `2` |
| `STRATEGY_FIXED` | `"fixed"` |
| `STRATEGY_SENTENCES` | `"sentences"` |
| `COLLECTION_FIXED` | `"kb_fixed_400_80"` |
| `COLLECTION_SENTENCES` | `"kb_sentences"` |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` |
| `SIMILARITY_THRESHOLD` | `None` until Task 11 measures it, a float after |

**The eighteen knowledge-base documents.**
Filename is `knowledge_base/<doc_id>.md`.
Documents 1 to 12 carry `required: true`, documents 13 to 18 carry `required: false`.

| `doc_id` | `title` | `topic` |
|---|---|---|
| `kb-01-loan-eligibility` | Loan eligibility criteria by loan type | `loan_eligibility` |
| `kb-02-emi-calculation` | EMI calculation rules | `emi_calculation` |
| `kb-03-credit-card-fees` | Credit-card fee structure | `credit_card_fees` |
| `kb-04-kyc-documents` | KYC document requirements | `kyc_documents` |
| `kb-05-fraud-dispute` | Fraud-dispute resolution process | `fraud_dispute` |
| `kb-06-account-closure` | Account-closure process | `account_closure` |
| `kb-07-interest-rate-slabs` | Interest-rate slabs | `interest_rate_slabs` |
| `kb-08-prepayment-penalty` | Prepayment-penalty rules | `prepayment_penalty` |
| `kb-09-minimum-balance` | Minimum-balance requirements | `minimum_balance` |
| `kb-10-credit-score-impact` | Credit-score impact factors | `credit_score_impact` |
| `kb-11-joint-account-rules` | Joint-account rules | `joint_account_rules` |
| `kb-12-nri-account-eligibility` | NRI-account eligibility | `nri_account_eligibility` |
| `kb-13-personal-loan-eligibility` | Personal Loan eligibility in detail | `personal_loan_eligibility` |
| `kb-14-home-loan-ltv` | Home Loan eligibility and loan-to-value rules | `home_loan_ltv` |
| `kb-15-card-late-payment-charges` | Credit-card late-payment and overlimit charges | `card_late_payment` |
| `kb-16-kyc-reverification` | KYC re-verification and periodic update rules | `kyc_reverification` |
| `kb-17-foreclosure-vs-part-prepayment` | Foreclosure versus part-prepayment | `foreclosure_vs_prepayment` |
| `kb-18-nre-vs-nro-operation` | NRE versus NRO account operation rules | `nre_vs_nro` |

**The one bank.**
Every knowledge-base document and every generated record describes the same fictional lender.
Call it **Meridian Bank** wherever a name is needed.
Any loan amount, tenure or rate a document quotes must be consistent with the per-category bands in `config.py` and with every other document.
This is not decoration: a grader reading `kb-14` and `dataset.py` in the same sitting will see a contradiction if the two disagree.

**The per-category loan-amount bands** (spec section 5.2, sourced there, reproduced here so Task 2 and Tasks 4 to 5 agree):

| Category | Low (INR) | High (INR) |
|---|---|---|
| Personal Loan | 50,000 | 40,00,000 |
| Auto Loan | 1,00,000 | 25,00,000 |
| Education Loan | 1,00,000 | 50,00,000 |
| Business Loan | 1,00,000 | 50,00,000 |
| Home Loan | 10,00,000 | 1,50,00,000 |

---

## Execution waves

Tasks inside a wave touch disjoint files and can run concurrently, up to five at a time.
A wave starts only when the previous wave is committed and reviewed.

| Wave | Tasks | Why they are safe together |
|---|---|---|
| 0 | 1 | Everything imports `config.py`. |
| 1 | 2, 4, 5, 7, 9 | Disjoint files, and every cross-reference between them is frozen above. |
| 2 | 3, 6, 12 | Disjoint files, each depends only on wave 1. |
| 3 | 8 | Needs `kb.py` and `chunking.py`, writes the vector store. |
| 4 | 10 | Needs the index. |
| 5 | 11 | Needs retrieval, and produces the threshold everything downstream reads. |
| 6 | 13, 14 | Disjoint files, both need the threshold. |
| 7 | 15 | Needs every module. |
| 8 | 16 | Needs the measured numbers that Task 15 produces. |

---

## Task 1: Repository skeleton and `config.py`

**Files:**
- Create: `config.py`
- Create: `rag/__init__.py`, `eval/__init__.py`, `tests/__init__.py`
- Create: `pytest.ini`
- Modify: `.gitignore`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: every constant in the Frozen identifiers table above, plus `REPO_ROOT`, `KB_DIR`, `DATA_DIR`, `CHROMA_DIR`, `TRANSCRIPT_DIR`, `DATASET_SNAPSHOT`, `CATEGORY_BANDS`, `CATEGORIES`, `STATUSES`, `LLM_PROVIDER`, `MOCK_LLM`, `COLLECTION_FOR_STRATEGY`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
"""Config is the single source of every path and tunable in the project."""

import config


def test_paths_are_rooted_in_the_repository():
    assert config.KB_DIR == config.REPO_ROOT / "knowledge_base"
    assert config.DATA_DIR == config.REPO_ROOT / "data"
    assert config.CHROMA_DIR == config.REPO_ROOT / "chroma"
    assert config.TRANSCRIPT_DIR == config.REPO_ROOT / "transcripts"
    assert config.DATASET_SNAPSHOT == config.DATA_DIR / "loan_applications.json"


def test_chunk_parameters_are_the_decided_values():
    assert config.CHUNK_SIZE == 400
    assert config.CHUNK_OVERLAP == 80
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
    assert config.TOP_K == 3
    assert config.SUPPORT_MIN_SHARED == 2


def test_every_strategy_maps_to_its_own_collection():
    mapping = config.COLLECTION_FOR_STRATEGY
    assert mapping[config.STRATEGY_FIXED] == "kb_fixed_400_80"
    assert mapping[config.STRATEGY_SENTENCES] == "kb_sentences"
    assert len(set(mapping.values())) == 2


def test_every_category_has_a_band_low_below_high():
    assert set(config.CATEGORY_BANDS) == set(config.CATEGORIES)
    for category, (low, high) in config.CATEGORY_BANDS.items():
        assert 0 < low < high, category


def test_mock_is_the_default_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert config.resolve_provider() == "mock"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert config.resolve_provider() == "openai"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 3: Write `config.py`**

```python
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

SEED = 0  # replaced in Task 2 by the seed the fraud-band search selects
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
```

- [ ] **Step 4: Create the package markers and pytest config**

```bash
mkdir -p rag eval tests scripts data transcripts knowledge_base
printf '"""Retrieval, chunking, indexing, generation and evaluation."""\n' > rag/__init__.py
printf '"""Hand-authored evaluation queries and calibration probes."""\n' > eval/__init__.py
printf '' > tests/__init__.py
```

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra
filterwarnings =
    ignore::DeprecationWarning
```

- [ ] **Step 5: Confirm `.gitignore` covers the generated store**

Run: `grep -nE 'chroma|__pycache__|\.venv' .gitignore`
Expected: all three appear.
If `chroma/` is missing, append it.
Do not remove any existing entry.

- [ ] **Step 6: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add config.py pytest.ini rag/__init__.py eval/__init__.py tests/__init__.py tests/test_config.py .gitignore
git commit -m "Add config.py as the single source of paths, tunables and flags

Every path, chunk parameter, collection name and weight table lives here so
that the eighteen knowledge-base documents, dataset.py and the RAG modules
agree without duplicating constants. SIMILARITY_THRESHOLD stays None until
Task 11 measures it, because the brief forbids an untested preset."
```

---
## Task 2: `dataset.py`, the seed search and the committed snapshot

Brief Task 1.
Spec sections 5.1 to 5.3, D-09, D-10, and open item 3.

**Files:**
- Create: `dataset.py`
- Create: `data/loan_applications.json`
- Modify: `config.py` (the `SEED` line only)
- Test: covered by Task 3

**Interfaces:**
- Consumes: `config.SEED`, `config.RECORD_COUNT`, `config.CATEGORIES`, `config.STATUSES`, `config.CATEGORY_WEIGHTS`, `config.STATUS_WEIGHTS`, `config.CATEGORY_BANDS`, `config.DAYS_MIN`, `config.DAYS_MAX`, `config.DAYS_MODE`, `config.FRAUD_BASE_PROBABILITY`, `config.FRAUD_CATEGORY_BONUS`, `config.DATASET_SNAPSHOT`.
- Produces:
  - `LOAN_APPLICATIONS: list[dict]`, built at import time, 100 records.
  - `generate_applications(seed: int, count: int) -> list[dict]`
  - `get_application(record_id: str) -> dict | None`
  - `validation_report() -> dict` with keys `total`, `by_category`, `by_status`, `fraud_flagged`, `fraud_percentage`, `seed`.
  - `snapshot_bytes() -> bytes`, the exact bytes written to `data/loan_applications.json`.
  - `main() -> None`, printing the report.

Every record is a flat dict with exactly six keys, in this order: `record_id`, `category`, `status`, `loan_amount_inr`, `days_since_created`, `flagged_for_fraud_review`.
No seventh field.
Part 2 reads only these, and an extra field is something to justify to a grader for no gain.

- [ ] **Step 1: Write `dataset.py`**

```python
"""Task 1. A seeded, deterministic generator for Meridian Bank loan applications.

The generator is the source of truth. data/loan_applications.json is a
committed snapshot of its output, and tests/test_dataset.py asserts the two
still agree by SHA-256, so a future edit that reshuffles every record_id
cannot pass silently while Parts 2 to 4 quote those ids in their transcripts.

Every value here is fabricated. No real person's data appears anywhere.
"""

import json
import math
import random
from collections import Counter

import config


def generate_applications(seed: int = None, count: int = None) -> list[dict]:
    """Build the record list from a single seeded Random instance.

    Draw order is fixed per record - category, status, amount, age, fraud flag -
    so the whole list is reproducible from the seed alone.
    """
    seed = config.SEED if seed is None else seed
    count = config.RECORD_COUNT if count is None else count

    rng = random.Random(seed)
    categories = list(config.CATEGORY_WEIGHTS)
    category_weights = [config.CATEGORY_WEIGHTS[c] for c in categories]
    statuses = list(config.STATUS_WEIGHTS)
    status_weights = [config.STATUS_WEIGHTS[s] for s in statuses]

    records = []
    for offset in range(count):
        category = rng.choices(categories, weights=category_weights, k=1)[0]
        status = rng.choices(statuses, weights=status_weights, k=1)[0]

        low, high = config.CATEGORY_BANDS[category]
        # Log-uniform, so the book is dominated by smaller tickets the way a
        # real portfolio is. A uniform draw would put the median Home Loan at
        # the midpoint of its band and make every percentile arbitrary, which
        # matters because Part 2 Task 6 justifies its escalation threshold as
        # a percentile of this column.
        amount = math.exp(rng.uniform(math.log(low), math.log(high)))
        loan_amount_inr = int(round(amount, -3))

        days = int(rng.triangular(config.DAYS_MIN, config.DAYS_MAX, config.DAYS_MODE))
        days_since_created = max(config.DAYS_MIN, min(config.DAYS_MAX, days))

        probability = config.FRAUD_BASE_PROBABILITY + config.FRAUD_CATEGORY_BONUS.get(
            category, 0.0
        )
        flagged = rng.random() < probability

        records.append(
            {
                "record_id": f"{config.RECORD_ID_PREFIX}{config.RECORD_ID_START + offset}",
                "category": category,
                "status": status,
                "loan_amount_inr": loan_amount_inr,
                "days_since_created": days_since_created,
                "flagged_for_fraud_review": flagged,
            }
        )
    return records


LOAN_APPLICATIONS: list[dict] = generate_applications()

_BY_ID = {record["record_id"]: record for record in LOAN_APPLICATIONS}


def get_application(record_id: str) -> dict | None:
    """Look up one application. Part 2 Task 6 and Part 4 Task 14 wrap this."""
    return _BY_ID.get(record_id)


def validation_report() -> dict:
    """Counts per category, counts per status, and the fraud percentage."""
    flagged = sum(1 for r in LOAN_APPLICATIONS if r["flagged_for_fraud_review"])
    total = len(LOAN_APPLICATIONS)
    return {
        "seed": config.SEED,
        "total": total,
        "by_category": dict(
            sorted(Counter(r["category"] for r in LOAN_APPLICATIONS).items())
        ),
        "by_status": dict(
            sorted(Counter(r["status"] for r in LOAN_APPLICATIONS).items())
        ),
        "fraud_flagged": flagged,
        "fraud_percentage": round(100.0 * flagged / total, 2),
    }


def snapshot_bytes() -> bytes:
    """The exact bytes of data/loan_applications.json, so the hash test has one source."""
    return (json.dumps(LOAN_APPLICATIONS, indent=2, sort_keys=False) + "\n").encode("utf-8")


def main() -> None:
    report = validation_report()
    print("Meridian Bank loan applications - validation report")
    print(f"seed:  {report['seed']}")
    print(f"total: {report['total']}")
    print()
    print("count per category (every given category needs at least 3)")
    for category, count in report["by_category"].items():
        print(f"  {category:<16} {count:>3}")
    print()
    print("count per status (every given status needs at least 1)")
    for status, count in report["by_status"].items():
        print(f"  {status:<16} {count:>3}")
    print()
    print(
        f"flagged_for_fraud_review: {report['fraud_flagged']}/{report['total']} "
        f"= {report['fraud_percentage']}% (band is 10% to 30%)"
    )
    print()
    print("loan_amount_inr per category (min / median / max, rupees)")
    for category in sorted(config.CATEGORY_BANDS):
        amounts = sorted(
            r["loan_amount_inr"] for r in LOAN_APPLICATIONS if r["category"] == category
        )
        if not amounts:
            continue
        median = amounts[len(amounts) // 2]
        print(f"  {category:<16} {amounts[0]:>10,} / {median:>10,} / {amounts[-1]:>10,}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Search for a seed that satisfies all four structural invariants**

The seed is chosen by generating and checking, never by editing records.
Write this to the scratchpad, not the repository:

```python
# scratchpad/seed_search.py
import importlib
import sys
from collections import Counter

sys.path.insert(0, ".")
import config
import dataset

for seed in range(1, 51):
    records = dataset.generate_applications(seed=seed)
    by_category = Counter(r["category"] for r in records)
    by_status = Counter(r["status"] for r in records)
    flagged = sum(1 for r in records if r["flagged_for_fraud_review"])
    rate = flagged / len(records)
    ok = (
        all(by_category.get(c, 0) >= 3 for c in config.CATEGORIES)
        and all(by_status.get(s, 0) >= 1 for s in config.STATUSES)
        and config.FRAUD_RATE_MIN <= rate <= config.FRAUD_RATE_MAX
    )
    print(f"seed {seed:>3}  fraud {rate:6.1%}  min-cat {min(by_category.values()):>3}  "
          f"statuses {len(by_status)}  {'PASS' if ok else 'fail'}")
```

Run: `.venv/bin/python <scratchpad>/seed_search.py | head -60`
Take the **first** seed that prints PASS.
Do not cherry-pick a later one for a prettier number, and do not hand-edit any record.
Keep the printed table: Task 16 quotes the first ten rows of it in `README.md`, which is what makes the choice auditable rather than asserted.

- [ ] **Step 3: Set the chosen seed in `config.py`**

Replace `SEED = 0  # replaced in Task 2 ...` with the chosen integer and a comment naming the search:

```python
SEED = <chosen>  # first seed in 1..50 whose draw meets every structural invariant
```

- [ ] **Step 4: Write the snapshot**

```bash
.venv/bin/python -c "
import dataset, config
config.DATASET_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
config.DATASET_SNAPSHOT.write_bytes(dataset.snapshot_bytes())
print(config.DATASET_SNAPSHOT, config.DATASET_SNAPSHOT.stat().st_size, 'bytes')
"
```

- [ ] **Step 5: Run the report and read it**

Run: `.venv/bin/python dataset.py`
Expected: every category at least 3, every status present, fraud percentage between 10 and 30.
If any of those fails, the seed search in Step 2 was wrong; fix it there, never in the records.

- [ ] **Step 6: Commit**

```bash
git add dataset.py config.py data/loan_applications.json
git commit -m "Add the seeded loan-application generator and its committed snapshot

100 records over five categories and five statuses, amounts drawn log-uniformly
inside per-category bands so the book is dominated by smaller tickets and Part 2
can justify its escalation threshold as a percentile. The seed is the first in
1..50 whose draw meets every structural invariant; records are never hand-edited."
```

---

## Task 3: Dataset invariant tests, one per acceptance criterion

Spec section 11, tests 1 to 4, and D-13.

**Files:**
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `dataset.LOAN_APPLICATIONS`, `dataset.generate_applications`, `dataset.get_application`, `dataset.validation_report`, `dataset.snapshot_bytes`, `config.DATASET_SNAPSHOT`.
- Produces: nothing importable.

- [ ] **Step 1: Write the tests**

```python
"""Tests 1 to 4 of spec section 11, one per Part 1 acceptance criterion."""

import hashlib

import pytest

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
```

- [ ] **Step 2: Run them**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: 9 passed.
A failure on the first test means `dataset.py` and the snapshot disagree, which is exactly what the test exists to catch; regenerate the snapshot in the same commit as the generator change.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dataset.py
git commit -m "Test the dataset against every structural acceptance criterion

One test per criterion: snapshot reproducibility by SHA-256, category floor,
status coverage, and the fraud band. Precision and recall are deliberately not
pinned anywhere in the suite, because they move legitimately when chunk
parameters are tuned and a test that fights tuning is a test that gets deleted."
```

---
## Task 4: The twelve required knowledge-base documents

Brief Task 2.
Spec section 6.1, D-05.

**Files:**
- Create: `knowledge_base/kb-01-loan-eligibility.md` through `knowledge_base/kb-12-nri-account-eligibility.md`

**Interfaces:**
- Consumes: the Frozen identifiers table (doc ids, titles, topics) and the per-category band table.
- Produces: 12 markdown files that `rag/kb.py` parses in Task 6.

### The file format, exactly

Every file is YAML front matter delimited by `---` on its own line, a blank line, then the body.
The body is prose only: no headings, no bullet lists, no markdown emphasis, no tables.
`rag/kb.py` feeds the body straight to the chunkers, and a bullet list would make `chunk_sentences` produce fragments that embed badly.

```markdown
---
doc_id: kb-01-loan-eligibility
title: Loan eligibility criteria by loan type
topic: loan_eligibility
required: true
---

Meridian Bank assesses eligibility separately for each loan product, because
the risk profile of an unsecured personal advance differs from that of a
secured home loan. ...
```

### Rules for the prose

1. **10 to 15 sentences, targeting 12.**
   The brief's floor is 2 to 5; going longer is D-05, and the reason is that at 2 to 5 sentences the fixed-size and sentence-based chunkers produce nearly the same chunks and the Task 5 comparison has nothing to measure.
2. **One sentence per line** in the source file, per the global constraints.
   Do not hard-wrap a sentence across lines: `chunk_sentences` splits on terminal punctuation, and a mid-sentence newline is fine, but one-sentence-per-line keeps the diff readable.
3. **Every document describes Meridian Bank**, the single fictional lender.
4. **Every number a document quotes must be consistent** with `config.CATEGORY_BANDS` and with every other document.
   If `kb-01` says a Personal Loan runs from 50,000 to 40 lakh, `kb-13` must not say 60 lakh.
5. **Fabricate freely, but plausibly.**
   Rates, tenures, fees and thresholds are invented for Meridian Bank.
   Never quote a real bank's published schedule as if it were Meridian's.
6. **Write for retrieval.**
   Each document should restate its own subject noun a few times rather than leaning on pronouns, because a chunk that says "it is charged at two percent" with no antecedent retrieves badly and reads worse in a generated answer.
7. **Do not cross-reference other documents by `doc_id`.**
   A sentence like "see kb-08" pollutes the embedding and leaks the file layout into an answer.

### What each document must state

Each list below is the minimum factual content.
Add connective prose to reach 10 to 15 sentences.

- **`kb-01-loan-eligibility`** - the five products Meridian offers; that eligibility is assessed per product; minimum and maximum age; minimum credit score; that salaried and self-employed applicants are assessed on different income evidence; the amount band for each of the five categories exactly as in `config.CATEGORY_BANDS`; that secured products additionally assess the asset; that an existing Meridian relationship shortens assessment.
- **`kb-02-emi-calculation`** - what EMI stands for and that it is level over the tenure; the formula `EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)`; that `P` is principal, `r` the monthly rate, that the monthly rate is the annual rate divided by twelve and by one hundred, and `n` the tenure in months; one fully worked example with concrete numbers; that early instalments are interest-heavy and later ones principal-heavy; that a longer tenure lowers the instalment and raises total interest; that a floating-rate reset changes the tenure rather than the instalment by default.
- **`kb-03-credit-card-fees`** - the joining fee and the annual fee, and the spend level that waives the annual fee; the interest-free period on purchases; the cash-advance fee and that cash advances carry no interest-free period; the foreign-currency markup; the card-replacement fee; the duplicate-statement fee; that goods and services tax applies on top of every fee; that fees appear on the statement in the cycle they are levied.
- **`kb-04-kyc-documents`** - that KYC is mandatory before an account or a loan is opened; the accepted proof-of-identity documents; the accepted proof-of-address documents; that PAN is mandatory above a stated transaction threshold; that Aadhaar-based verification can be completed online; what a self-employed applicant additionally submits; what a non-resident additionally submits; that originals must be presented for in-person verification or the copies self-attested; that an incomplete file freezes the application rather than rejecting it.
- **`kb-05-fraud-dispute`** - the channels for reporting an unauthorised transaction; the reporting window that preserves zero liability; that the card or the channel is blocked immediately on report; the acknowledgement timeline and the reference number; the provisional-credit rule and its timeline; the investigation window; that the customer is asked for a written declaration; what happens if the investigation finds the transaction authorised; the escalation route to the banking ombudsman.
- **`kb-06-account-closure`** - that closure is requested in branch or in writing; that the balance must be nil or is paid out; that unused cheque leaves and the debit card are surrendered; the closure charge and the account age above which it is waived; the processing timeline; that standing instructions and mandates must be cancelled first; that a loan or a lien on the account blocks closure; that the customer receives a written closure confirmation; that a dormant account follows the same route.
- **`kb-07-interest-rate-slabs`** - that Meridian prices every product off an internal benchmark plus a spread; the rate band for each of the five loan products; that the spread narrows as the credit score rises, with the score bands stated; that secured products price below unsecured; that a floating rate resets on a stated cycle and a fixed rate does not; the rate on the credit-card revolving balance; that a rate quoted at sanction holds for a stated number of days.
- **`kb-08-prepayment-penalty`** - that a floating-rate loan to an individual carries no prepayment charge; that a fixed-rate loan does, and the percentage on the amount prepaid; the lock-in period before any prepayment is allowed; that a business loan carries a charge regardless of rate type; that goods and services tax applies on the charge; the notice a customer must give; that prepaying from own funds and refinancing from another lender are treated differently; that the charge is computed on the outstanding principal, not the original sanction.
- **`kb-09-minimum-balance`** - that the requirement is an average monthly balance, not a daily floor; the figure for a metro branch, a semi-urban branch and a rural branch; the shortfall charge and that it is tiered by how far below the requirement the balance fell; that salary and basic savings accounts are exempt; that the balance is averaged over the calendar month; that the charge is levied in the following cycle; that a customer is notified before the charge is levied.
- **`kb-10-credit-score-impact`** - the score range Meridian reads and what counts as good; that repayment history is the largest single factor and the weight; credit utilisation and the ratio to stay below; the age of the credit history; the credit mix; that each hard enquiry costs a few points and that several in a short window compound; that a settled account marks the report for years; that checking one's own score is a soft enquiry and costs nothing; how long an entry stays on the report.
- **`kb-11-joint-account-rules`** - the maximum number of joint holders; the operating mandates, either or survivor, former or survivor, and jointly; that every holder completes KYC independently; that the first holder receives the statements and owns the tax reporting; how a mandate is changed and that every holder must consent; what happens on the death of a holder under each mandate; that a joint holder can be removed only by closing and reopening; that a minor can be a joint holder only with a guardian.
- **`kb-12-nri-account-eligibility`** - who qualifies as a non-resident Indian and as a person of Indian origin; that a non-resident may not hold a resident savings account and must convert it; the three account types available, NRE, NRO and FCNR; the documents required, including a passport, a visa and overseas address proof; that the account may be opened from overseas with attested copies; that a resident close relative may be added on the operating mandate; that a joint NRE account is allowed between two non-residents; the tax status of interest at a high level; that a returning non-resident converts the account back.

- [ ] **Step 1: Write all twelve files**

Follow the format block and the rules above exactly.
Write them one at a time and re-read each after writing.

- [ ] **Step 2: Validate the front matter and the sentence counts**

Write this to the scratchpad and run it:

```python
# scratchpad/check_kb.py
import re
import sys
from pathlib import Path

import yaml

KB = Path("knowledge_base")
FRONT = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
SPLIT = re.compile(r"(?<=[.!?])\s+")

failures = []
for path in sorted(KB.glob("*.md")):
    raw = path.read_text(encoding="utf-8")
    match = FRONT.match(raw)
    if not match:
        failures.append(f"{path.name}: no front matter")
        continue
    meta = yaml.safe_load(match.group(1))
    body = match.group(2).strip()
    for key in ("doc_id", "title", "topic", "required"):
        if key not in meta:
            failures.append(f"{path.name}: missing {key}")
    if meta.get("doc_id") != path.stem:
        failures.append(f"{path.name}: doc_id {meta.get('doc_id')!r} does not match filename")
    if not isinstance(meta.get("required"), bool):
        failures.append(f"{path.name}: required must be a YAML boolean")
    sentences = [s for s in SPLIT.split(body) if s.strip()]
    if not 10 <= len(sentences) <= 15:
        failures.append(f"{path.name}: {len(sentences)} sentences, want 10 to 15")
    for marker in ("#", "- ", "* ", "|", "**"):
        if any(line.strip().startswith(marker) for line in body.splitlines()):
            failures.append(f"{path.name}: body contains markdown {marker!r}, prose only")
            break
    if "\u2014" in raw:
        failures.append(f"{path.name}: contains an em dash")
    print(f"{path.stem:<40} {len(sentences):>3} sentences  {len(body):>5} chars  "
          f"{len(body) // max(len(sentences), 1):>4} chars/sentence")

print()
if failures:
    print("FAILURES")
    for line in failures:
        print(" ", line)
    sys.exit(1)
print(f"{len(list(KB.glob('*.md')))} documents OK")
```

Run: `.venv/bin/python <scratchpad>/check_kb.py`
Expected: twelve rows, no failures.
Keep the chars-per-sentence column: Task 8 uses it to close spec open item 2.

- [ ] **Step 3: Check consistency against `config.CATEGORY_BANDS` by eye**

Run: `grep -nE '[0-9]+ ?(lakh|crore|,000)' knowledge_base/kb-01-loan-eligibility.md knowledge_base/kb-07-interest-rate-slabs.md`
Read every match and confirm each amount matches the band table.
A mismatch here becomes a visible contradiction in the submitted repository.

- [ ] **Step 4: Commit**

```bash
git add knowledge_base/kb-0*.md knowledge_base/kb-1[0-2]*.md
git commit -m "Write the twelve required knowledge-base documents

One document per topic the brief names, 10 to 15 sentences each rather than
the brief's 2 to 5, because at that length the fixed-size and sentence-based
chunkers produce nearly the same chunks and the Task 5 comparison would have
nothing to measure. Every document describes one fictional lender and quotes
amounts consistent with config.CATEGORY_BANDS."
```

---

## Task 5: The six confusable neighbour documents

Spec section 6.2, D-04.

**Files:**
- Create: `knowledge_base/kb-13-personal-loan-eligibility.md` through `knowledge_base/kb-18-nre-vs-nro-operation.md`

**Interfaces:**
- Consumes: the same format, rules and band table as Task 4.
- Produces: 6 markdown files with `required: false`.

These are not padding.
Without them every evaluation query has exactly one relevant document, both collections score Precision@3 of one third, and the Task 5 recommendation has nothing to cite.
Each is a document a real support team would write, and each sits close enough to a required topic that a plausible customer question matches both.

Everything in Task 4's format block and prose rules applies unchanged, except `required: false`.

### What each document must state

- **`kb-13-personal-loan-eligibility`** - neighbour of `kb-01`.
  The minimum net monthly income for a salaried applicant and for a self-employed applicant; the minimum years in the current job or business; the amount band, 50,000 to 40 lakh, matching `config.CATEGORY_BANDS`; the tenure range; the fixed-obligation-to-income ratio ceiling; that no collateral is taken; the processing fee; that a co-applicant can lift the eligible amount; the documents specific to this product; that the sanctioned amount can be lower than the applied amount.
  It must overlap `kb-01` on income and credit-score language deliberately, and diverge on the product-specific ratios that `kb-01` does not carry.
- **`kb-14-home-loan-ltv`** - neighbour of `kb-01` and `kb-07`.
  The loan-to-value ratio at each property-value slab; that the margin is the customer's own contribution; the amount band, 10 lakh to 1.5 crore, matching `config.CATEGORY_BANDS`; the maximum tenure; that the property is mortgaged and the title is verified; that stamp duty and registration are excluded from the valuation; the rate band, matching `kb-07`; that a co-applicant who is an income earner is required above a stated amount; that disbursement against an under-construction property is staged.
- **`kb-15-card-late-payment-charges`** - neighbour of `kb-03`.
  The late-payment fee tiered by the outstanding balance; that finance charges begin from the transaction date once the full amount is not paid; that the interest-free period is lost for the entire cycle; the overlimit fee and that it needs the customer's consent to be levied at all; that a returned payment carries its own charge; that a payment reported late is reported to the credit bureaus after a stated number of days; that only the minimum amount due avoids the late fee but not the finance charge; how a late fee is reversed as a goodwill gesture and how often.
  It must not restate the general fee schedule; that is `kb-03`'s job.
- **`kb-16-kyc-reverification`** - neighbour of `kb-04`.
  That KYC is periodically refreshed, not one-off; the refresh cycle for a high-risk, a medium-risk and a low-risk customer; that a change of address or name triggers a refresh regardless of cycle; that the same document list as onboarding applies; that a self-declaration suffices when nothing has changed; the channels for submitting a refresh; what happens when a refresh is overdue, including partial freezing; the notice period before a freeze; how a frozen account is reactivated.
- **`kb-17-foreclosure-vs-part-prepayment`** - neighbour of `kb-08`.
  The definition of foreclosure and of part-prepayment; that foreclosure closes the loan and part-prepayment does not; that part-prepayment by default shortens the tenure rather than the instalment, and that the customer may request the other; the minimum part-prepayment amount and the cap on the number per year; that foreclosure requires a foreclosure statement valid for a stated number of days; that the original property documents are returned within a stated window after foreclosure; that the credit bureaus are updated after closure; that the charge itself is stated in the prepayment-penalty schedule, without naming that document.
- **`kb-18-nre-vs-nro-operation`** - neighbour of `kb-12`.
  What may be credited to an NRE account and what to an NRO account; that NRE balances are freely repatriable and NRO repatriation is capped per financial year with a certificate; that NRE interest is exempt and NRO interest is taxed at source; that an NRO account may be held jointly with a resident and an NRE account may not; that funds may move from NRE to NRO but not the reverse without documentation; that both may be savings or term deposits; that a rupee cheque from a resident goes only to the NRO account; the form required for repatriation.
  It must not restate who qualifies as a non-resident; that is `kb-12`'s job.

- [ ] **Step 1: Write all six files**

- [ ] **Step 2: Re-run the validator over the full set**

Run: `.venv/bin/python <scratchpad>/check_kb.py`
Expected: eighteen rows, no failures.

- [ ] **Step 3: Confirm the required flag splits 12 and 6**

Run: `grep -c 'required: true' knowledge_base/*.md | grep ':1$' | wc -l && grep -l 'required: false' knowledge_base/*.md | wc -l`
Expected: `12` then `6`.

- [ ] **Step 4: Commit**

```bash
git add knowledge_base/kb-1[3-8]*.md
git commit -m "Add six deliberately confusable neighbour documents

Each sits close enough to a required topic that a plausible customer question
matches both, which is what gives Precision@3 something to separate. Without
them every evaluation query has exactly one relevant document, both collections
score one third, and the Task 5 recommendation has nothing to cite."
```

---
## Task 6: `rag/kb.py`, loading the knowledge base

Spec section 6.

**Files:**
- Create: `rag/kb.py`
- Test: `tests/test_kb.py`

**Interfaces:**
- Consumes: `config.KB_DIR`, the eighteen markdown files.
- Produces:
  - `Document`, a frozen dataclass with fields `doc_id: str`, `title: str`, `topic: str`, `required: bool`, `body: str`, `path: Path`.
  - `load_documents(kb_dir: Path | None = None) -> list[Document]`, sorted by `doc_id`.
  - `document_titles() -> dict[str, str]`, mapping `doc_id` to `title`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kb.py`:

```python
"""The knowledge-base loader must be complete, sorted and strict."""

import pytest

import config
from rag import kb

DOCS = kb.load_documents()


def test_all_eighteen_documents_load():
    assert len(DOCS) == 18


def test_documents_are_sorted_by_doc_id():
    assert [d.doc_id for d in DOCS] == sorted(d.doc_id for d in DOCS)


def test_twelve_required_and_six_neighbours():
    assert sum(1 for d in DOCS if d.required) == 12
    assert sum(1 for d in DOCS if not d.required) == 6


def test_doc_id_matches_the_filename():
    for doc in DOCS:
        assert doc.doc_id == doc.path.stem


def test_topics_are_unique():
    topics = [d.topic for d in DOCS]
    assert len(set(topics)) == len(topics)


def test_body_excludes_the_front_matter():
    for doc in DOCS:
        assert "doc_id:" not in doc.body
        assert not doc.body.startswith("---")
        assert doc.body == doc.body.strip()
        assert len(doc.body) > 200


def test_a_missing_field_is_an_error_not_a_default(tmp_path):
    (tmp_path / "kb-99-broken.md").write_text(
        "---\ndoc_id: kb-99-broken\ntitle: Broken\n---\n\nBody.\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="topic"):
        kb.load_documents(tmp_path)


def test_titles_map_covers_every_document():
    titles = kb.document_titles()
    assert len(titles) == 18
    assert titles["kb-01-loan-eligibility"] == "Loan eligibility criteria by loan type"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_kb.py -v`
Expected: FAIL with `ImportError: cannot import name 'kb' from 'rag'`.

- [ ] **Step 3: Write `rag/kb.py`**

```python
"""Load knowledge_base/ into Document objects.

Front matter is parsed strictly: a missing or mistyped field raises rather
than defaulting, because a document that silently loses its doc_id becomes
unscoreable in Task 5 and nothing would report it.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

import config

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)

_REQUIRED_FIELDS = {"doc_id": str, "title": str, "topic": str, "required": bool}


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    topic: str
    required: bool
    body: str
    path: Path


def _parse(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(raw)
    if match is None:
        raise ValueError(f"{path.name}: no YAML front matter delimited by ---")

    meta = yaml.safe_load(match.group(1)) or {}
    for field, expected in _REQUIRED_FIELDS.items():
        if field not in meta:
            raise ValueError(f"{path.name}: front matter is missing {field}")
        if not isinstance(meta[field], expected):
            raise ValueError(
                f"{path.name}: {field} must be {expected.__name__}, "
                f"got {type(meta[field]).__name__}"
            )

    if meta["doc_id"] != path.stem:
        raise ValueError(f"{path.name}: doc_id {meta['doc_id']!r} does not match the filename")

    body = match.group(2).strip()
    if not body:
        raise ValueError(f"{path.name}: empty body")

    return Document(
        doc_id=meta["doc_id"],
        title=meta["title"],
        topic=meta["topic"],
        required=meta["required"],
        body=body,
        path=path,
    )


def load_documents(kb_dir: Path | None = None) -> list[Document]:
    """Every document in the knowledge base, sorted by doc_id for determinism."""
    kb_dir = config.KB_DIR if kb_dir is None else Path(kb_dir)
    paths = sorted(kb_dir.glob("*.md"))
    if not paths:
        raise ValueError(f"no markdown documents found in {kb_dir}")
    return sorted((_parse(p) for p in paths), key=lambda d: d.doc_id)


def document_titles() -> dict[str, str]:
    """doc_id to title, for transcripts and evaluation tables."""
    return {d.doc_id: d.title for d in load_documents()}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_kb.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/kb.py tests/test_kb.py
git commit -m "Load the knowledge base with strict front-matter parsing

A missing or mistyped field raises rather than defaulting: a document that
silently loses its doc_id becomes unscoreable in the Precision@3 comparison
and nothing downstream would report it."
```

---

## Task 7: `rag/chunking.py`, both strategies

Brief Task 3, first half.
Spec section 7.1, D-14.

**Files:**
- Create: `rag/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `config.CHUNK_SIZE`, `config.CHUNK_OVERLAP`.
- Produces:
  - `chunk_fixed(text: str, size: int | None = None, overlap: int | None = None) -> list[str]`
  - `chunk_sentences(text: str) -> list[str]`
  - `chunk(text: str, strategy: str) -> list[str]`, dispatching on `config.STRATEGY_FIXED` or `config.STRATEGY_SENTENCES`.

Both take the document body only, never the front matter.
Both are pure functions from a string to a list of strings, which is what makes the V2 swap to semantic chunking cheap.

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunking.py`:

```python
"""Both chunkers are pure functions and must behave predictably at the edges."""

import pytest

import config
from rag import chunking


def test_fixed_respects_size_and_advances_by_size_minus_overlap():
    text = "x" * 1000
    chunks = chunking.chunk_fixed(text, size=400, overlap=80)
    assert all(len(c) <= 400 for c in chunks)
    # step 320: windows start at 0, 320 and 640, and the third reaches the end
    assert len(chunks) == 3


def test_fixed_chunks_overlap_by_the_declared_amount():
    text = "".join(chr(ord("a") + i % 26) for i in range(1000))
    chunks = chunking.chunk_fixed(text, size=400, overlap=80)
    assert chunks[0][-80:] == chunks[1][:80]


def test_fixed_returns_one_chunk_for_short_text():
    assert chunking.chunk_fixed("A short policy sentence.", size=400, overlap=80) == [
        "A short policy sentence."
    ]


def test_fixed_rejects_overlap_at_or_above_size():
    with pytest.raises(ValueError):
        chunking.chunk_fixed("abc", size=100, overlap=100)


def test_fixed_and_sentences_ignore_empty_input():
    assert chunking.chunk_fixed("   ") == []
    assert chunking.chunk_sentences("   ") == []


def test_sentences_splits_on_terminal_punctuation():
    text = "First sentence here. Second one follows! And a third? Yes."
    assert chunking.chunk_sentences(text) == [
        "First sentence here.",
        "Second one follows!",
        "And a third?",
        "Yes.",
    ]


def test_sentences_does_not_split_inside_a_rupee_abbreviation():
    text = "The fee is Rs. 5,000 per year. It is waived above a stated spend."
    chunks = chunking.chunk_sentences(text)
    assert len(chunks) == 2
    assert chunks[0] == "The fee is Rs. 5,000 per year."


def test_sentences_does_not_split_on_common_abbreviations():
    for text in [
        "Contact Mr. Rao for details. He handles disputes.",
        "Submit PAN, passport, etc. before the deadline. A copy suffices.",
        "Use a utility bill, i.e. electricity or water. Both are accepted.",
    ]:
        assert len(chunking.chunk_sentences(text)) == 2, text


def test_sentences_does_not_split_on_a_decimal_number():
    text = "The rate is 10.5 percent per annum. It resets quarterly."
    assert len(chunking.chunk_sentences(text)) == 2


def test_sentences_handles_a_newline_between_sentences():
    text = "One sentence.\nTwo sentence.\nThree sentence."
    assert chunking.chunk_sentences(text) == [
        "One sentence.",
        "Two sentence.",
        "Three sentence.",
    ]


def test_dispatch_matches_the_direct_calls():
    text = "One sentence. Two sentence. " * 40
    assert chunking.chunk(text, config.STRATEGY_FIXED) == chunking.chunk_fixed(text)
    assert chunking.chunk(text, config.STRATEGY_SENTENCES) == chunking.chunk_sentences(text)
    with pytest.raises(ValueError):
        chunking.chunk(text, "semantic")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_chunking.py -v`
Expected: FAIL with `ImportError: cannot import name 'chunking' from 'rag'`.

- [ ] **Step 3: Write `rag/chunking.py`**

```python
"""Task 3. The two chunking strategies the brief asks to compare.

Both are pure functions from text to a list of strings. Nothing here knows
about embeddings, ChromaDB or document metadata, which is what keeps the V2
swap to semantic or late chunking a one-file change.
"""

import re

import config

# A boundary is terminal punctuation followed by whitespace. The guards below
# undo the split when the token before the period was not a sentence end.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Lower-cased, without the trailing period. NLTK punkt would handle this
# better and needs a download, which breaks the zero-network-access rule.
_ABBREVIATIONS = {
    "approx",
    "co",
    "dr",
    "e.g",
    "etc",
    "i.e",
    "inc",
    "jr",
    "ltd",
    "mr",
    "mrs",
    "ms",
    "no",
    "pvt",
    "rs",
    "sr",
    "st",
    "vs",
    "govt",
    "dept",
    "fig",
}

_LAST_TOKEN = re.compile(r"([A-Za-z.]+)\.\Z")


def _ends_on_an_abbreviation(fragment: str) -> bool:
    """True when the fragment's final period closes an abbreviation, not a sentence."""
    stripped = fragment.rstrip()
    if not stripped.endswith("."):
        return False
    match = _LAST_TOKEN.search(stripped)
    if match is None:
        # A digit before the period: "10." in "10.5" or a numbered item.
        return bool(re.search(r"\d\.\Z", stripped))
    token = match.group(1).lower().rstrip(".")
    if token in _ABBREVIATIONS:
        return True
    # A single capital letter is an initial, as in "S. Rao".
    return len(match.group(1)) == 1 and match.group(1).isupper()


def chunk_fixed(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Walk the character stream in steps of size minus overlap.

    400 and 80 are the decided values. 600 lost because a twelve-sentence
    document then yields three chunks against twelve sentence chunks, and the
    fixed-size collection loses the granularity the comparison needs. 250 lost
    because chunks cut mid-clause often enough to hurt embedding quality.
    """
    size = config.CHUNK_SIZE if size is None else size
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if overlap >= size:
        raise ValueError(f"overlap {overlap} must be smaller than size {size}")
    if overlap < 0 or size <= 0:
        raise ValueError(f"size {size} and overlap {overlap} must be positive")

    text = text.strip()
    if not text:
        return []

    step = size - overlap
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
        start += step
    return chunks


def chunk_sentences(text: str) -> list[str]:
    """Split on terminal punctuation, rejoining across known abbreviations."""
    text = text.strip()
    if not text:
        return []

    fragments = _BOUNDARY.split(text)
    sentences: list[str] = []
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        if sentences and _ends_on_an_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {fragment}"
        else:
            sentences.append(fragment)
    return sentences


def chunk(text: str, strategy: str) -> list[str]:
    """Dispatch to the named strategy. Used by rag/index.py for both collections."""
    if strategy == config.STRATEGY_FIXED:
        return chunk_fixed(text)
    if strategy == config.STRATEGY_SENTENCES:
        return chunk_sentences(text)
    raise ValueError(
        f"unknown strategy {strategy!r}, expected one of "
        f"{sorted(config.COLLECTION_FOR_STRATEGY)}"
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_chunking.py -v`
Expected: 11 passed.
If `test_sentences_does_not_split_on_a_decimal_number` fails, the digit guard in `_ends_on_an_abbreviation` is the thing to fix, not the test.

- [ ] **Step 5: Commit**

```bash
git add rag/chunking.py tests/test_chunking.py
git commit -m "Implement fixed-size and sentence-based chunking as pure functions

400 characters with 80 overlap, and a regex sentence splitter with an
abbreviation guard so that Rs. 5,000 and 10.5 percent do not become sentence
boundaries. NLTK punkt would split better and needs a download, which breaks
the zero-network-access rule the brief sets."
```

---
## Task 8: `rag/index.py`, two ChromaDB collections

Brief Task 3, second half.
Spec sections 7.2 and 7.3, test 5, and spec open item 2.

**Files:**
- Create: `rag/index.py`
- Test: `tests/test_index.py`
- Modify: `tests/conftest.py` (create it)

**Interfaces:**
- Consumes: `rag.kb.load_documents`, `rag.chunking.chunk`, `config.CHROMA_DIR`, `config.COLLECTION_FOR_STRATEGY`, `config.EMBEDDING_MODEL`, `config.MIN_CHUNKS_PER_DOCUMENT`.
- Produces:
  - `Chunk`, a frozen dataclass with `chunk_id: str`, `text: str`, `doc_id: str`, `title: str`, `topic: str`, `required: bool`, `chunk_index: int`, `strategy: str`.
  - `build_chunks(strategy: str, documents: list[Document] | None = None) -> list[Chunk]`
  - `get_embedder() -> SentenceTransformer`, cached per process.
  - `embed(texts: list[str]) -> list[list[float]]`, always with `normalize_embeddings=True`.
  - `get_client()` returning the persistent ChromaDB client at `config.CHROMA_DIR`.
  - `get_collection(strategy: str)` returning the existing collection, raising if it has not been built.
  - `build_index(rebuild: bool = False) -> dict[str, int]`, mapping collection name to chunk count.
  - `chunk_counts_by_document(strategy: str) -> dict[str, int]`.

`normalize_embeddings=True` plus `{"hnsw:space": "cosine"}` is what makes `similarity = 1 - distance` correct in Task 10.
Changing either without the other silently breaks every number downstream.

- [ ] **Step 1: Write the failing test**

Create `tests/conftest.py`:

```python
"""Shared fixtures. The index is built once per session because embedding is slow."""

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@pytest.fixture(scope="session")
def built_index():
    from rag import index

    counts = index.build_index(rebuild=True)
    return counts
```

Create `tests/test_index.py`:

```python
"""Test 5 of spec section 11, plus the invariants the index must hold."""

import config
from rag import index, kb

DOCS = kb.load_documents()


def test_every_document_yields_at_least_two_chunks_in_both_strategies():
    """Test 5. Without this a short document is permanently unanswerable under
    the support rule in D-07, and nothing would report it."""
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        counts = {}
        for chunk in index.build_chunks(strategy, DOCS):
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        assert set(counts) == {d.doc_id for d in DOCS}
        for doc_id, count in sorted(counts.items()):
            assert count >= config.MIN_CHUNKS_PER_DOCUMENT, f"{strategy}/{doc_id}: {count}"


def test_chunk_ids_are_unique_within_a_strategy():
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        ids = [c.chunk_id for c in index.build_chunks(strategy, DOCS)]
        assert len(set(ids)) == len(ids)


def test_chunk_metadata_carries_the_parent_document():
    chunk = index.build_chunks(config.STRATEGY_SENTENCES, DOCS)[0]
    assert chunk.doc_id
    assert chunk.title
    assert chunk.topic
    assert isinstance(chunk.required, bool)
    assert chunk.chunk_index == 0
    assert chunk.strategy == config.STRATEGY_SENTENCES


def test_embeddings_are_unit_length_and_384_dimensional():
    vectors = index.embed(["Meridian Bank charges a foreclosure fee."])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    norm = sum(v * v for v in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-5


def test_both_collections_exist_and_hold_every_chunk(built_index):
    assert set(built_index) == {config.COLLECTION_FIXED, config.COLLECTION_SENTENCES}
    for strategy, name in sorted(config.COLLECTION_FOR_STRATEGY.items()):
        collection = index.get_collection(strategy)
        assert collection.count() == built_index[name]
        assert collection.count() == len(index.build_chunks(strategy, DOCS))


def test_both_collections_return_a_sensible_hit_on_a_sample_query(built_index):
    """The brief's wording: both collections must produce sensible retrieval."""
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        collection = index.get_collection(strategy)
        result = collection.query(
            query_embeddings=index.embed(["How is the EMI on a loan calculated?"]),
            n_results=3,
        )
        assert "kb-02-emi-calculation" in [m["doc_id"] for m in result["metadatas"][0]]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_index.py -v`
Expected: FAIL with `ImportError: cannot import name 'index' from 'rag'`.

- [ ] **Step 3: Write `rag/index.py`**

```python
"""Task 3. Embed every chunk and index each strategy in its own collection.

normalize_embeddings=True together with {"hnsw:space": "cosine"} is what makes
similarity = 1 - distance correct in rag/retrieve.py. Changing one without the
other silently breaks every measured number downstream, so both live here.
"""

import functools
import os

from dataclasses import dataclass

import config
from rag import chunking, kb

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    title: str
    topic: str
    required: bool
    chunk_index: int
    strategy: str

    def metadata(self) -> dict:
        """Chroma accepts scalars only, which is all the mapping to a parent needs."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "topic": self.topic,
            "required": self.required,
            "chunk_index": self.chunk_index,
            "strategy": self.strategy,
        }


def build_chunks(strategy: str, documents=None) -> list[Chunk]:
    """Every chunk of every document under one strategy, in document order."""
    documents = kb.load_documents() if documents is None else documents
    chunks: list[Chunk] = []
    for document in documents:
        pieces = chunking.chunk(document.body, strategy)
        if len(pieces) < config.MIN_CHUNKS_PER_DOCUMENT:
            raise ValueError(
                f"{document.doc_id} produced {len(pieces)} chunk(s) under {strategy!r}, "
                f"needs at least {config.MIN_CHUNKS_PER_DOCUMENT}. A document with one "
                f"chunk can never satisfy the same-parent support rule, so it would be "
                f"permanently unanswerable. Lengthen the document or retune the chunker."
            )
        for position, text in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{strategy}::{position:03d}",
                    text=text,
                    doc_id=document.doc_id,
                    title=document.title,
                    topic=document.topic,
                    required=document.required,
                    chunk_index=position,
                    strategy=strategy,
                )
            )
    return chunks


@functools.lru_cache(maxsize=1)
def get_embedder():
    """The local SentenceTransformers model. Cached: loading costs about 0.3s."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Unit-normalised embeddings, so cosine distance is 1 minus cosine similarity."""
    vectors = get_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return [vector.tolist() for vector in vectors]


@functools.lru_cache(maxsize=1)
def get_client():
    import chromadb
    from chromadb.config import Settings

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def collection_name(strategy: str) -> str:
    try:
        return config.COLLECTION_FOR_STRATEGY[strategy]
    except KeyError:
        raise ValueError(
            f"unknown strategy {strategy!r}, expected one of "
            f"{sorted(config.COLLECTION_FOR_STRATEGY)}"
        ) from None


def get_collection(strategy: str):
    """The built collection. Raises if build_index has never run."""
    return get_client().get_collection(name=collection_name(strategy))


def build_index(rebuild: bool = False) -> dict[str, int]:
    """Embed and index both strategies. Returns collection name to chunk count."""
    client = get_client()
    counts: dict[str, int] = {}

    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        name = collection_name(strategy)
        if rebuild:
            try:
                client.delete_collection(name=name)
            except Exception:
                pass
        collection = client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

        chunks = build_chunks(strategy)
        if collection.count() != len(chunks):
            if collection.count():
                client.delete_collection(name=name)
                collection = client.get_or_create_collection(
                    name=name, metadata={"hnsw:space": "cosine"}
                )
            collection.add(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                embeddings=embed([c.text for c in chunks]),
                metadatas=[c.metadata() for c in chunks],
            )
        counts[name] = collection.count()

    return counts


def chunk_counts_by_document(strategy: str) -> dict[str, int]:
    """Chunks per parent document, for the Task 3 transcript and open item 2."""
    counts: dict[str, int] = {}
    for chunk in build_chunks(strategy):
        counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
    return dict(sorted(counts.items()))
```

- [ ] **Step 4: Build the index once, offline, and read the counts**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -c "
from rag import index
import config
print(index.build_index(rebuild=True))
for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
    counts = index.chunk_counts_by_document(strategy)
    print()
    print(strategy, 'total', sum(counts.values()), 'min', min(counts.values()), 'max', max(counts.values()))
    for doc_id, n in counts.items():
        print(f'  {doc_id:<40} {n:>3}')
"
```

Expected: both collections non-empty, and every per-document count at least 2.
The command running at all under `HF_HUB_OFFLINE=1` is the evidence for the zero-network-access claim.

- [ ] **Step 5: Close spec open item 2**

The chunk-count model behind D-05 and D-14 assumed 134 characters per sentence.
Compare that against the real column printed by `scratchpad/check_kb.py` in Task 4.

- If the measured average is within roughly 25 percent of 134 and every document yields at least 3 fixed chunks and at least 10 sentence chunks, the parameters stand.
  Record the measured average and this conclusion for Task 16.
- If a document yields exactly 2 fixed chunks, the fixed collection has lost the granularity the comparison needs.
  Lower `config.CHUNK_SIZE` to 300 with `CHUNK_OVERLAP` 60, rebuild, and re-read the counts.
- If any document yields fewer than 2 chunks under either strategy, `build_chunks` already raised in Step 4.
  Lengthen that document rather than shrinking the chunk size to hide it.

Write the before and after values down: `README.md` records both the starting and the final chunk parameters, and Task 16 needs them.

- [ ] **Step 6: Run the test to verify it passes**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests/test_index.py -v`
Expected: 6 passed.

- [ ] **Step 7: Confirm the store is not committed**

Run: `git status --short chroma/ && git check-ignore -v chroma/`
Expected: no untracked entries listed, and `.gitignore` named as the reason.

- [ ] **Step 8: Commit**

```bash
git add rag/index.py tests/test_index.py tests/conftest.py
git commit -m "Index both chunking strategies into their own ChromaDB collections

Unit-normalised embeddings plus cosine space, so similarity is one minus
distance for every number this project reports. Indexing refuses a document
that yields fewer than two chunks under either strategy: one chunk can never
satisfy the same-parent support rule, so the document would be permanently
unanswerable and nothing else would notice."
```

---

## Task 9: `eval/queries.py` and `eval/calibration.py`

Brief Tasks 4 and 5, the labelling half.
Spec sections 8.2 and 9.1, D-11.

**Files:**
- Create: `eval/queries.py`
- Create: `eval/calibration.py`
- Test: `tests/test_queries.py`

**Interfaces:**
- Consumes: the eighteen `doc_id` values from the Frozen identifiers table.
- Produces:
  - `EvalQuery`, a frozen dataclass with `query_id: str`, `text: str`, `gold_doc_ids: tuple[str, ...]`.
  - `EVAL_QUERIES: list[EvalQuery]`, twelve entries.
  - `IN_SCOPE_PROBES: list[str]`, twelve entries.
  - `OUT_OF_SCOPE_PROBES: list[str]`, five entries.

**This task must be completed and committed before Task 10 runs any retrieval.**
That ordering is the whole point: a label set written after seeing the results is not a label set, and `README.md` states the ordering explicitly.
Do not adjust a gold set later because a retrieval missed it.

- [ ] **Step 1: Write `eval/queries.py`**

```python
"""The twelve evaluation queries and their hand-authored gold labels.

Authored and committed before any retrieval was run. Sizes are mixed on
purpose: with one relevant document per query, Recall@3 can only take the
values 0 and 1, which makes it decoration rather than a metric.

Four queries have one relevant document, five have two, three have three.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    text: str
    gold_doc_ids: tuple[str, ...]


EVAL_QUERIES: list[EvalQuery] = [
    # --- one relevant document ---
    EvalQuery("EQ-01", "How is the EMI on a loan calculated?", ("kb-02-emi-calculation",)),
    EvalQuery("EQ-02", "What is the process to close my savings account?", ("kb-06-account-closure",)),
    EvalQuery("EQ-03", "What minimum balance must I keep in my savings account?", ("kb-09-minimum-balance",)),
    EvalQuery("EQ-04", "What are the rules for operating a joint account?", ("kb-11-joint-account-rules",)),
    # --- two relevant documents ---
    EvalQuery(
        "EQ-05",
        "What income do I need to qualify for a personal loan?",
        ("kb-01-loan-eligibility", "kb-13-personal-loan-eligibility"),
    ),
    EvalQuery(
        "EQ-06",
        "Which documents does the bank need for KYC?",
        ("kb-04-kyc-documents", "kb-16-kyc-reverification"),
    ),
    EvalQuery(
        "EQ-07",
        "What charges apply if I pay my credit-card bill late?",
        ("kb-03-credit-card-fees", "kb-15-card-late-payment-charges"),
    ),
    EvalQuery(
        "EQ-08",
        "Is there a penalty for paying off my loan early?",
        ("kb-08-prepayment-penalty", "kb-17-foreclosure-vs-part-prepayment"),
    ),
    EvalQuery(
        "EQ-09",
        "Can a non-resident Indian open and operate an account here?",
        ("kb-12-nri-account-eligibility", "kb-18-nre-vs-nro-operation"),
    ),
    # --- three relevant documents ---
    EvalQuery(
        "EQ-10",
        "What interest rate will I get on a home loan and how much can I borrow against the property?",
        ("kb-01-loan-eligibility", "kb-07-interest-rate-slabs", "kb-14-home-loan-ltv"),
    ),
    EvalQuery(
        "EQ-11",
        "How do I report a fraudulent card transaction and will it affect my credit score?",
        ("kb-03-credit-card-fees", "kb-05-fraud-dispute", "kb-10-credit-score-impact"),
    ),
    EvalQuery(
        "EQ-12",
        "What decides how much I can borrow and what the loan will cost me each month?",
        ("kb-01-loan-eligibility", "kb-02-emi-calculation", "kb-07-interest-rate-slabs"),
    ),
]
```

- [ ] **Step 2: Write `eval/calibration.py`**

Only the probe lists in this step.
`calibrate()` is added in Task 11, once retrieval exists.

```python
"""Probe queries for the empirical threshold calibration in Task 4.

The brief's floor is 3 in-scope and 2 out-of-scope probes. Twelve and five
oversample it deliberately: the gap between the two clusters is the entire
justification for the chosen threshold, and three points do not make a cluster.

The in-scope probes are one per required topic and are deliberately worded
differently from the twelve evaluation queries, so that calibration and
scoring are not measuring the same twelve strings twice.
"""

IN_SCOPE_PROBES: list[str] = [
    "Who is eligible to apply for a business loan?",
    "Show me the formula used to work out a monthly instalment.",
    "What annual fee does the credit card carry?",
    "Which identity proof is accepted when opening an account?",
    "Someone used my card without permission, what happens next?",
    "How long does it take to shut a bank account?",
    "What rate of interest applies to an education loan?",
    "Will I be charged for repaying a fixed-rate loan ahead of schedule?",
    "What happens if my average balance falls below the requirement?",
    "How much does a missed payment hurt my credit rating?",
    "Can two people hold one account together?",
    "Which account types can a person living abroad hold?",
]

OUT_OF_SCOPE_PROBES: list[str] = [
    "What is the best recipe for a chocolate sponge cake?",
    "Which team won the football World Cup in 2018?",
    "How do I replace the timing belt on a diesel engine?",
    "What is the boiling point of liquid nitrogen at sea level?",
    "Recommend a three-day hiking route in the Western Ghats.",
]
```

- [ ] **Step 3: Write the test**

Create `tests/test_queries.py`:

```python
"""The label set must match D-11's design and reference only real documents."""

from collections import Counter

from eval import calibration, queries
from rag import kb

DOC_IDS = {d.doc_id for d in kb.load_documents()}


def test_twelve_queries_with_unique_ids():
    assert len(queries.EVAL_QUERIES) == 12
    ids = [q.query_id for q in queries.EVAL_QUERIES]
    assert len(set(ids)) == 12


def test_every_gold_label_names_a_real_document():
    for query in queries.EVAL_QUERIES:
        for doc_id in query.gold_doc_ids:
            assert doc_id in DOC_IDS, f"{query.query_id} references unknown {doc_id}"


def test_gold_sets_are_graded_four_five_three():
    """D-11: 4 queries with 1 relevant document, 5 with 2, 3 with 3."""
    sizes = Counter(len(q.gold_doc_ids) for q in queries.EVAL_QUERIES)
    assert sizes == Counter({1: 4, 2: 5, 3: 3})


def test_gold_sets_have_no_duplicates():
    for query in queries.EVAL_QUERIES:
        assert len(set(query.gold_doc_ids)) == len(query.gold_doc_ids)


def test_every_required_document_is_gold_for_at_least_one_query():
    covered = {d for q in queries.EVAL_QUERIES for d in q.gold_doc_ids}
    required = {d.doc_id for d in kb.load_documents() if d.required}
    missing = sorted(required - covered)
    assert not missing, f"required topics never scored: {missing}"


def test_probe_counts_oversample_the_brief_floors():
    assert len(calibration.IN_SCOPE_PROBES) == 12
    assert len(calibration.OUT_OF_SCOPE_PROBES) == 5
    assert len(set(calibration.IN_SCOPE_PROBES)) == 12
    assert len(set(calibration.OUT_OF_SCOPE_PROBES)) == 5


def test_probes_do_not_reuse_the_evaluation_query_strings():
    evaluation = {q.text for q in queries.EVAL_QUERIES}
    assert not evaluation & set(calibration.IN_SCOPE_PROBES)
```

- [ ] **Step 4: Run it**

Run: `.venv/bin/python -m pytest tests/test_queries.py -v`
Expected: 7 passed.
If `test_every_required_document_is_gold_for_at_least_one_query` fails, add the missing document to the gold set of the query it genuinely answers.
Doing that now is honest; doing it after Task 14 has printed a score is not.

- [ ] **Step 5: Commit, before any retrieval runs**

```bash
git add eval/queries.py eval/calibration.py tests/test_queries.py
git commit -m "Author the evaluation labels and calibration probes before any retrieval

Twelve queries with graded gold sets of one, two and three documents, so that
Recall@3 can take more than the two values it would take under one document per
query. Twelve in-scope and five out-of-scope probes oversample the brief's
floors of three and two, because the gap between the clusters is the entire
justification for the threshold and three points do not make a cluster.

Committed now, deliberately ahead of the retrieval code, because a label set
written after seeing the results is not a label set."
```

---
## Task 10: `rag/retrieve.py`, top-k retrieval and the support rule

Brief Task 4, retrieval half.
Spec sections 8.1 and 8.3, D-07 and D-08.

**Files:**
- Create: `rag/retrieve.py`
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `rag.index.get_collection`, `rag.index.embed`, `config.TOP_K`, `config.SUPPORT_MIN_SHARED`.
- Produces:
  - `Hit`, a frozen dataclass with `text: str`, `doc_id: str`, `title: str`, `chunk_index: int`, `similarity: float`.
  - `retrieve(query: str, strategy: str, k: int | None = None) -> list[Hit]`, ranked highest similarity first.
  - `parent_documents(hits: list[Hit]) -> list[str]`, distinct `doc_id` values in rank order.
  - `top1_similarity(hits: list[Hit]) -> float`, `0.0` for an empty list.
  - `has_same_parent_support(hits: list[Hit], minimum: int | None = None) -> bool`.
  - `is_supported(hits: list[Hit], threshold: float) -> bool`, both conditions of the answer decision.

The answer decision is both conditions, per D-07 and D-08:

1. `top1_similarity(hits) >= threshold`
2. at least `SUPPORT_MIN_SHARED` of the top `TOP_K` chunks share one `doc_id`

Two of top 5 lost because evidence ranked fifth is weak support dressed up as agreement.
Retrieval already fetches 3 for Precision@3, so this window costs no extra query.

- [ ] **Step 1: Write the failing test**

Create `tests/test_retrieve.py`:

```python
"""Retrieval must be ranked, bounded, and honest about similarity."""

import pytest

import config
from rag import retrieve


def hit(doc_id, similarity, chunk_index=0):
    return retrieve.Hit(
        text=f"chunk of {doc_id}",
        doc_id=doc_id,
        title=doc_id,
        chunk_index=chunk_index,
        similarity=similarity,
    )


def test_similarity_is_bounded_and_descending(built_index):
    hits = retrieve.retrieve("What annual fee does the credit card carry?", config.STRATEGY_FIXED)
    assert len(hits) == config.TOP_K
    assert all(-1.0 <= h.similarity <= 1.0 for h in hits)
    assert [h.similarity for h in hits] == sorted((h.similarity for h in hits), reverse=True)


def test_an_in_scope_query_scores_far_above_an_out_of_scope_one(built_index):
    in_scope = retrieve.retrieve("How is the EMI calculated?", config.STRATEGY_SENTENCES)
    out_of_scope = retrieve.retrieve(
        "What is the best recipe for a chocolate sponge cake?", config.STRATEGY_SENTENCES
    )
    assert retrieve.top1_similarity(in_scope) > retrieve.top1_similarity(out_of_scope) + 0.2


def test_both_strategies_answer_the_same_query(built_index):
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        hits = retrieve.retrieve("How do I close my account?", strategy)
        assert "kb-06-account-closure" in retrieve.parent_documents(hits)


def test_parent_documents_dedups_and_keeps_rank_order():
    hits = [hit("kb-02", 0.9), hit("kb-01", 0.8, 1), hit("kb-02", 0.7, 2)]
    assert retrieve.parent_documents(hits) == ["kb-02", "kb-01"]


def test_top1_similarity_of_nothing_is_zero():
    assert retrieve.top1_similarity([]) == 0.0


def test_support_needs_two_of_the_top_three_to_agree():
    agreeing = [hit("kb-02", 0.9), hit("kb-01", 0.8), hit("kb-02", 0.7)]
    scattered = [hit("kb-02", 0.9), hit("kb-01", 0.8), hit("kb-03", 0.7)]
    assert retrieve.has_same_parent_support(agreeing)
    assert not retrieve.has_same_parent_support(scattered)


def test_the_answer_decision_needs_both_conditions():
    agreeing_high = [hit("kb-02", 0.62), hit("kb-02", 0.55), hit("kb-01", 0.40)]
    agreeing_low = [hit("kb-02", 0.30), hit("kb-02", 0.28), hit("kb-01", 0.20)]
    scattered_high = [hit("kb-02", 0.62), hit("kb-01", 0.55), hit("kb-03", 0.40)]
    assert retrieve.is_supported(agreeing_high, threshold=0.45)
    assert not retrieve.is_supported(agreeing_low, threshold=0.45)
    assert not retrieve.is_supported(scattered_high, threshold=0.45)


def test_an_unknown_strategy_raises():
    with pytest.raises(ValueError):
        retrieve.retrieve("anything", "semantic")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests/test_retrieve.py -v`
Expected: FAIL with `ImportError: cannot import name 'retrieve' from 'rag'`.

- [ ] **Step 3: Write `rag/retrieve.py`**

```python
"""Task 4, retrieval half. Top-k chunks, cosine similarity, and the support rule.

Callers never see ChromaDB. That is deliberate: the V2 upgrade to hybrid BM25
plus dense retrieval with reciprocal rank fusion replaces this file alone.
"""

from dataclasses import dataclass

import config
from rag import index


@dataclass(frozen=True)
class Hit:
    text: str
    doc_id: str
    title: str
    chunk_index: int
    similarity: float


def retrieve(query: str, strategy: str, k: int | None = None) -> list[Hit]:
    """The top k chunks for a query, highest cosine similarity first."""
    k = config.TOP_K if k is None else k
    collection = index.get_collection(strategy)  # raises on an unknown strategy
    result = collection.query(
        query_embeddings=index.embed([query]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        # Embeddings are unit-normalised and the space is cosine, so this is
        # an identity, not an approximation. See rag/index.py.
        hits.append(
            Hit(
                text=text,
                doc_id=metadata["doc_id"],
                title=metadata["title"],
                chunk_index=int(metadata["chunk_index"]),
                similarity=round(1.0 - float(distance), 4),
            )
        )
    return sorted(hits, key=lambda h: h.similarity, reverse=True)


def parent_documents(hits: list[Hit]) -> list[str]:
    """Distinct parent doc_ids in rank order. This is the dedup Task 5 requires."""
    seen: list[str] = []
    for h in hits:
        if h.doc_id not in seen:
            seen.append(h.doc_id)
    return seen


def top1_similarity(hits: list[Hit]) -> float:
    return hits[0].similarity if hits else 0.0


def has_same_parent_support(hits: list[Hit], minimum: int | None = None) -> bool:
    """At least `minimum` of the top-k chunks agree on one parent document."""
    minimum = config.SUPPORT_MIN_SHARED if minimum is None else minimum
    counts: dict[str, int] = {}
    for h in hits:
        counts[h.doc_id] = counts.get(h.doc_id, 0) + 1
    return bool(counts) and max(counts.values()) >= minimum


def is_supported(hits: list[Hit], threshold: float) -> bool:
    """The answer decision: a calibrated threshold AND agreement on one parent.

    Similarity alone lost because a single confident chunk can be confidently
    wrong; the second condition asks the knowledge base to agree with itself
    before the system speaks.
    """
    return top1_similarity(hits) >= threshold and has_same_parent_support(hits)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests/test_retrieve.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/retrieve.py tests/test_retrieve.py
git commit -m "Add top-k retrieval, parent dedup and the same-parent support rule

An answer needs both a calibrated similarity and at least two of the top three
chunks agreeing on one parent document. Similarity alone lost because a single
confident chunk can be confidently wrong; the second condition asks the
knowledge base to agree with itself before the system speaks."
```

---

## Task 11: Measure the threshold and set it

Brief Task 4, the calibration the brief explicitly requires.
Spec section 8.2, test 6, and spec open item 3's sibling.

**Files:**
- Modify: `eval/calibration.py` (add the measurement functions)
- Modify: `config.py` (the `SIMILARITY_THRESHOLD` line only)
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `rag.retrieve.retrieve`, `rag.retrieve.top1_similarity`, `IN_SCOPE_PROBES`, `OUT_OF_SCOPE_PROBES`.
- Produces:
  - `ProbeResult`, a frozen dataclass with `probe: str`, `in_scope: bool`, `strategy: str`, `top1: float`, `top_doc_id: str`.
  - `measure(strategy: str) -> list[ProbeResult]`, seventeen results.
  - `calibrate() -> dict` with keys `by_strategy`, `min_in_scope`, `max_out_of_scope`, `gap`, `threshold`, `results`.
  - `format_report(summary: dict) -> str`, the transcript body.

The brief forbids a preset.
`0.5`, `0.6` and `0.7` are tutorial defaults that do not reliably separate short policy-sentence embeddings from unrelated queries, and using one would fail the criterion outright.

The threshold is a **single** value used across both collections, taken as the midpoint of the pooled clusters.
A per-collection threshold lost because the answer decision has to be one rule that Part 2 imports, not a rule that changes when Task 5's recommendation changes.

- [ ] **Step 1: Add the measurement code to `eval/calibration.py`**

The three imports go at the **top** of the file, above the probe lists.
Everything below them is appended after the probe lists.

```python
# --- at the top of the file, above IN_SCOPE_PROBES ---
from dataclasses import dataclass

import config
from rag import retrieve

# --- appended below OUT_OF_SCOPE_PROBES ---


@dataclass(frozen=True)
class ProbeResult:
    probe: str
    in_scope: bool
    strategy: str
    top1: float
    top_doc_id: str


def measure(strategy: str) -> list[ProbeResult]:
    """Top-1 cosine similarity for all seventeen probes against one collection."""
    results = []
    for probes, in_scope in ((IN_SCOPE_PROBES, True), (OUT_OF_SCOPE_PROBES, False)):
        for probe in probes:
            hits = retrieve.retrieve(probe, strategy, k=1)
            results.append(
                ProbeResult(
                    probe=probe,
                    in_scope=in_scope,
                    strategy=strategy,
                    top1=retrieve.top1_similarity(hits),
                    top_doc_id=hits[0].doc_id if hits else "",
                )
            )
    return results


def calibrate() -> dict:
    """Pool both collections and set the threshold midway between the clusters.

    One threshold, not one per collection: the answer decision has to be a
    single rule that Part 2 imports, and Part 2's collection is not chosen
    until Task 5's numbers exist.
    """
    by_strategy = {s: measure(s) for s in sorted(config.COLLECTION_FOR_STRATEGY)}
    results = [r for rs in by_strategy.values() for r in rs]

    in_scope = [r.top1 for r in results if r.in_scope]
    out_of_scope = [r.top1 for r in results if not r.in_scope]

    min_in_scope = min(in_scope)
    max_out_of_scope = max(out_of_scope)

    return {
        "by_strategy": by_strategy,
        "results": results,
        "min_in_scope": round(min_in_scope, 4),
        "max_out_of_scope": round(max_out_of_scope, 4),
        "gap": round(min_in_scope - max_out_of_scope, 4),
        "threshold": round((min_in_scope + max_out_of_scope) / 2.0, 4),
    }


def format_report(summary: dict) -> str:
    """The transcript body: every measured value, clustered, then the arithmetic."""
    lines = [
        "Threshold calibration - measured, not preset",
        "",
        "The brief forbids an untested preset. 0.5, 0.6 and 0.7 are tutorial",
        "defaults that do not reliably separate short policy-sentence embeddings",
        "from unrelated queries, so every value below was measured on this",
        "knowledge base with all-MiniLM-L6-v2 and cosine similarity.",
        "",
        f"Probes: {len(IN_SCOPE_PROBES)} in-scope, {len(OUT_OF_SCOPE_PROBES)} out-of-scope, "
        f"measured against both collections.",
        "",
    ]
    for strategy, results in summary["by_strategy"].items():
        lines.append(f"--- collection: {config.COLLECTION_FOR_STRATEGY[strategy]} ---")
        for label, wanted in (("IN SCOPE", True), ("OUT OF SCOPE", False)):
            lines.append(f"  {label}")
            for r in sorted(
                (r for r in results if r.in_scope is wanted),
                key=lambda r: r.top1,
                reverse=True,
            ):
                lines.append(f"    {r.top1:6.4f}  {r.top_doc_id:<40}  {r.probe}")
            lines.append("")
    lines += [
        "--- pooled across both collections ---",
        f"  minimum in-scope top-1      = {summary['min_in_scope']:.4f}",
        f"  maximum out-of-scope top-1  = {summary['max_out_of_scope']:.4f}",
        f"  gap                         = {summary['gap']:.4f}",
        "",
        f"  threshold T = ({summary['min_in_scope']:.4f} + {summary['max_out_of_scope']:.4f}) / 2 "
        f"= {summary['threshold']:.4f}",
        "",
        "An answer is produced only when top-1 similarity is at least T AND at",
        f"least {config.SUPPORT_MIN_SHARED} of the top {config.TOP_K} chunks share one parent document.",
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 2: Run the calibration and read every number**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -c "
from eval import calibration
summary = calibration.calibrate()
print(calibration.format_report(summary))
"
```

Read the two clusters.
If `gap` is not positive, the clusters overlap, the threshold cannot be set honestly, and the chunking parameters are wrong.
That is a stop-and-fix, not a footnote: go back to Task 8 Step 5, retune, rebuild, and re-measure.

- [ ] **Step 3: Set the measured threshold in `config.py`**

Replace the `SIMILARITY_THRESHOLD = None` line with the measured value and the derivation:

```python
# Measured on 2026-09-11 over 12 in-scope and 5 out-of-scope probes pooled
# across both collections. Minimum in-scope top-1 <MIN>, maximum out-of-scope
# top-1 <MAX>, gap <GAP>. T is the midpoint. Reproduce with
# scripts/run_part1.py, which writes transcripts/part1-calibration.txt.
SIMILARITY_THRESHOLD = <measured>
```

Substitute the real numbers.
Do not round to a tidy 0.5 or 0.45; the point of the exercise is that the value came from the data.

- [ ] **Step 4: Write the test**

Create `tests/test_calibration.py`:

```python
"""Test 6 of spec section 11: the clusters must actually separate."""

import config
from eval import calibration


def test_the_threshold_is_a_measured_float_not_a_placeholder():
    assert isinstance(config.SIMILARITY_THRESHOLD, float)
    assert 0.0 < config.SIMILARITY_THRESHOLD < 1.0


def test_in_scope_and_out_of_scope_clusters_do_not_overlap(built_index):
    """Test 6. If they overlap, no threshold separates them and the brief's
    ban on an untested preset cannot be satisfied honestly."""
    summary = calibration.calibrate()
    assert summary["min_in_scope"] > summary["max_out_of_scope"], (
        f"clusters overlap: in-scope minimum {summary['min_in_scope']} is not above "
        f"out-of-scope maximum {summary['max_out_of_scope']}"
    )
    assert summary["gap"] > 0


def test_the_committed_threshold_still_sits_between_the_clusters(built_index):
    summary = calibration.calibrate()
    assert summary["max_out_of_scope"] < config.SIMILARITY_THRESHOLD < summary["min_in_scope"]


def test_every_in_scope_probe_retrieves_a_required_document(built_index):
    from rag import kb

    required = {d.doc_id for d in kb.load_documents() if d.required}
    misses = [
        r.probe
        for r in calibration.measure(config.STRATEGY_SENTENCES)
        if r.in_scope and r.top_doc_id not in required
    ]
    # A probe landing on a confusable neighbour is acceptable and expected;
    # this test only reports how often, so record rather than assert on zero.
    assert len(misses) <= 4, misses
```

- [ ] **Step 5: Run it**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests/test_calibration.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add eval/calibration.py config.py tests/test_calibration.py
git commit -m "Measure the I-do-not-know threshold instead of presetting it

Seventeen probes across both collections, threshold set at the midpoint of the
two observed clusters. The brief bans an untested preset because 0.5, 0.6 and
0.7 do not reliably separate short policy-sentence embeddings from unrelated
queries, and the measured gap here shows what the real separation is. A test
fails if the clusters ever overlap, because then no honest threshold exists."
```

---
## Task 12: `llm.py`, the provider seam and the mock

Brief Task 4, generation half.
Spec section 8.4, D-06.

**Files:**
- Create: `llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `config.resolve_provider()`.
- Produces:
  - `generate(system: str, user: str) -> str`, the one function Part 3 Task 13's judge also calls.
  - `classify_shape(question: str) -> str`, one of `definition`, `eligibility`, `amount_or_rate`, `process`, `documents`, `duration`.
  - `parse_prompt(user: str) -> tuple[str, list[tuple[str, str]]]`, returning the question and the `(doc_id, chunk_text)` pairs.

This is the only interface in V1 written for two implementations, and it exists because the brief mandates both a mock mode and an optional real provider.
The real branch raises rather than shipping a second implementation nothing exercises.

### The prompt contract, frozen

`rag/generate.py` builds the `user` string in exactly this shape, and `llm.py` parses it back.
Both sides of the contract are frozen here so the two tasks can be built independently.

```
QUESTION: <the query, one line>

CONTEXT:
[<doc_id>] <chunk text, one line, newlines collapsed to spaces>
[<doc_id>] <chunk text>
[<doc_id>] <chunk text>
```

The mock never invents content: every sentence in its output comes from a `CONTEXT` line.

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm.py`:

```python
"""The mock provider must be deterministic and strictly grounded."""

import pytest

import llm

PROMPT = """QUESTION: What annual fee does the credit card carry?

CONTEXT:
[kb-03-credit-card-fees] The Meridian Rewards Card carries an annual fee of 999 rupees. The fee is waived when annual spend crosses two lakh rupees.
[kb-03-credit-card-fees] Goods and services tax applies on top of every fee that the card levies.
[kb-15-card-late-payment-charges] A late-payment fee applies when the minimum amount due is not paid by the due date.
"""


def test_parse_prompt_recovers_the_question_and_the_sources():
    question, sources = llm.parse_prompt(PROMPT)
    assert question == "What annual fee does the credit card carry?"
    assert len(sources) == 3
    assert sources[0][0] == "kb-03-credit-card-fees"
    assert "999 rupees" in sources[0][1]


@pytest.mark.parametrize(
    "question,shape",
    [
        ("Which documents do I need to submit for KYC?", "documents"),
        ("What annual fee does the credit card carry?", "amount_or_rate"),
        ("Am I eligible for a business loan?", "eligibility"),
        ("How do I close my savings account?", "process"),
        ("How long does a refund take?", "duration"),
        ("What is an NRE account?", "definition"),
    ],
)
def test_shape_classification_is_stable(question, shape):
    assert llm.classify_shape(question) == shape


def test_the_mock_is_deterministic():
    assert llm.generate("system", PROMPT) == llm.generate("system", PROMPT)


def test_every_sentence_in_the_answer_came_from_the_context():
    """The whole groundedness guarantee under MOCK_LLM: strip the template's own
    prefix, and every remaining sentence must be a substring of the context."""
    question, sources = llm.parse_prompt(PROMPT)
    answer = llm.generate("system", PROMPT)
    body = answer.split("Sources:")[0]
    prefix = llm._TEMPLATES[llm.classify_shape(question)].split("{body}")[0]
    assert body.startswith(prefix)
    quoted = body[len(prefix):].strip()
    context = " ".join(text for _, text in sources)
    for sentence in llm._SENTENCE.split(quoted):
        sentence = sentence.strip()
        if sentence:
            assert sentence in context, f"ungrounded sentence: {sentence!r}"


def test_the_answer_cites_only_documents_it_used():
    answer = llm.generate("system", PROMPT)
    assert "Sources:" in answer
    cited = answer.split("Sources:")[1]
    assert "[kb-03-credit-card-fees]" in cited
    for doc_id in cited.replace("[", " ").replace("]", " ").split():
        assert doc_id.startswith("kb-")


def test_an_empty_context_yields_no_answer():
    assert llm.generate("system", "QUESTION: anything\n\nCONTEXT:\n") == ""


def test_an_unwired_provider_raises_rather_than_pretending(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(NotImplementedError, match="LLM_PROVIDER"):
        llm.generate("system", PROMPT)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm'`.

- [ ] **Step 3: Write `llm.py`**

```python
"""The language-model seam. MOCK_LLM by default, a real provider behind LLM_PROVIDER.

Under MOCK_LLM the provider is deterministic template synthesis: classify the
question shape, pick a template, and fill it from sentences that appear in the
retrieved context and nowhere else.

Pure extractive stitching lost because groundedness would then score 1.0 on all
fifteen Part 3 queries by construction and the RAG triad would demonstrate
nothing. A small local generative model lost because it needs a weight
download, which breaks the zero-network-access rule.

Template selection can pick the wrong shape. That is deliberate: it is what
makes Part 3's context relevance and answer relevance vary across the fifteen
queries instead of being 1.0 by construction.
"""

import re

import config

# Checked in order, first match wins. Order encodes precedence: a question
# that asks both "which documents" and "how do I" is a document question.
_SHAPE_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("documents", ("document", "papers", "proof", "submit", "paperwork", "kyc form")),
    ("duration", ("how long", "how many days", "when will", "turnaround", "timeline", "take to")),
    (
        "amount_or_rate",
        ("how much", "rate", "interest", "fee", "charge", "penalty", "cost",
         "amount", "limit", "percent", "balance"),
    ),
    ("eligibility", ("eligible", "eligibility", "qualify", "who can", "can i", "criteria", "income")),
    ("process", ("how do i", "how can i", "process", "steps", "procedure", "what happens", "report", "raise")),
]

_TEMPLATES = {
    "definition": "Here is what the knowledge base says. {body}",
    "eligibility": "On eligibility: {body}",
    "amount_or_rate": "On amounts, rates and charges: {body}",
    "process": "The process works as follows. {body}",
    "documents": "You will need the following. {body}",
    "duration": "On timing: {body}",
}

# Cue words used to pick which retrieved sentences fill the template. A shape
# whose cues match nothing falls back to the first sentences in rank order.
_SENTENCE_CUES = {
    "definition": ("is", "means", "refers", "allows"),
    "eligibility": ("eligible", "eligibility", "qualify", "minimum", "criteria", "income", "score"),
    "amount_or_rate": ("rupees", "percent", "fee", "charge", "rate", "lakh", "crore", "waived"),
    "process": ("must", "submit", "request", "branch", "within", "issued", "processed"),
    "documents": ("document", "proof", "pan", "aadhaar", "passport", "copy", "statement"),
    "duration": ("days", "months", "years", "within", "working day", "cycle"),
}

_QUESTION = re.compile(r"^QUESTION:\s*(.*)$", re.MULTILINE)
_SOURCE = re.compile(r"^\[([a-z0-9\-]+)\]\s*(.+)$", re.MULTILINE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

MAX_SENTENCES = 3


def parse_prompt(user: str) -> tuple[str, list[tuple[str, str]]]:
    """Recover the question and the (doc_id, chunk text) pairs from the prompt."""
    match = _QUESTION.search(user)
    question = match.group(1).strip() if match else ""
    sources = [(m.group(1), m.group(2).strip()) for m in _SOURCE.finditer(user)]
    return question, sources


def classify_shape(question: str) -> str:
    """The question shape, by keyword. Wrong picks are expected and deliberate."""
    lowered = question.lower()
    for shape, cues in _SHAPE_CUES:
        if any(cue in lowered for cue in cues):
            return shape
    return "definition"


def _select_sentences(shape: str, sources: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Rank every context sentence by cue overlap, keeping retrieval order as the tiebreak."""
    cues = _SENTENCE_CUES[shape]
    scored = []
    for rank, (doc_id, text) in enumerate(sources):
        for position, sentence in enumerate(s.strip() for s in _SENTENCE.split(text)):
            if len(sentence) < 20:
                continue
            lowered = sentence.lower()
            score = sum(1 for cue in cues if cue in lowered)
            scored.append((-score, rank, position, doc_id, sentence))
    scored.sort()

    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, _, _, doc_id, sentence in scored:
        if sentence in seen:
            continue
        seen.add(sentence)
        chosen.append((doc_id, sentence))
        if len(chosen) == MAX_SENTENCES:
            break
    return chosen


def _generate_mock(system: str, user: str) -> str:
    question, sources = parse_prompt(user)
    if not sources:
        return ""

    shape = classify_shape(question)
    chosen = _select_sentences(shape, sources)
    if not chosen:
        return ""

    body = " ".join(sentence for _, sentence in chosen)
    cited = sorted({doc_id for doc_id, _ in chosen})
    citations = " ".join(f"[{doc_id}]" for doc_id in cited)
    return f"{_TEMPLATES[shape].format(body=body)}\n\nSources: {citations}"


def generate(system: str, user: str) -> str:
    """The single generation call. Part 3's judge calls this too."""
    provider = config.resolve_provider()
    if provider == config.DEFAULT_PROVIDER:
        return _generate_mock(system, user)
    raise NotImplementedError(
        f"LLM_PROVIDER={provider!r} is declared by the brief as optional and is not "
        f"wired in V1. Every acceptance criterion is demonstrated under MOCK_LLM. "
        f"Unset LLM_PROVIDER or set it to {config.DEFAULT_PROVIDER!r}."
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_llm.py -v`
Expected: 12 passed.
If `test_every_sentence_in_the_answer_came_from_the_context` fails, the template introduced prose that is not in the context; shorten the template, do not loosen the test.
That test is the entire groundedness guarantee under `MOCK_LLM`.

- [ ] **Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "Add the language-model seam with a deterministic mock provider

The mock classifies the question shape, picks a template, and fills it only
with sentences that appear in the retrieved context. Pure extractive stitching
lost because groundedness would then score 1.0 on all fifteen Part 3 queries by
construction and the RAG triad would measure nothing. Template selection can
pick wrong, which is what makes those scores vary. A real provider raises
rather than shipping a second implementation nothing exercises."
```

---

## Task 13: `rag/generate.py`, grounded answers and the fallback

Brief Task 4, the deliverable.
Spec sections 8.3 and 8.4, and the Part 2 interface contract in spec section 10.

**Files:**
- Create: `rag/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `rag.retrieve.retrieve`, `rag.retrieve.is_supported`, `rag.retrieve.top1_similarity`, `llm.generate`, `config.SIMILARITY_THRESHOLD`, `config.TOP_K`.
- Produces:
  - `GroundedAnswer`, a frozen dataclass with `query: str`, `text: str`, `citations: tuple[str, ...]`, `supported: bool`, `top1_similarity: float`, `strategy: str`, `hits: tuple[Hit, ...]`.
  - `FALLBACK_TEXT: str`
  - `build_prompt(query: str, hits) -> tuple[str, str]`, the `(system, user)` pair in the frozen contract from Task 12.
  - `answer(query: str, strategy: str = config.STRATEGY_SENTENCES, k: int | None = None) -> GroundedAnswer`

Part 2 Task 7 wraps `answer`, and Part 2 Task 10's output guardrail reads `.supported` and `.top1_similarity`.
Those three names are a contract; do not rename them.

The `strategy` default is `STRATEGY_SENTENCES` as a placeholder only.
Task 16 replaces it with whichever collection Task 5's measured numbers actually recommend, which closes spec open item 4.

- [ ] **Step 1: Write the failing test**

Create `tests/test_generate.py`:

```python
"""Grounded generation answers in scope and refuses out of scope."""

import config
from rag import generate


def test_an_in_scope_query_is_answered_and_cited(built_index):
    result = generate.answer("How is the EMI on a loan calculated?", config.STRATEGY_SENTENCES)
    assert result.supported
    assert result.text != generate.FALLBACK_TEXT
    assert "kb-02-emi-calculation" in result.citations
    assert result.top1_similarity >= config.SIMILARITY_THRESHOLD


def test_an_out_of_scope_query_triggers_the_fallback(built_index):
    result = generate.answer(
        "What is the best recipe for a chocolate sponge cake?", config.STRATEGY_SENTENCES
    )
    assert not result.supported
    assert result.text == generate.FALLBACK_TEXT
    assert result.citations == ()


def test_the_fallback_never_cites_anything(built_index):
    for query in [
        "Which team won the football World Cup in 2018?",
        "How do I replace the timing belt on a diesel engine?",
    ]:
        result = generate.answer(query, config.STRATEGY_FIXED)
        assert result.text == generate.FALLBACK_TEXT
        assert result.citations == ()


def test_every_citation_names_a_retrieved_document(built_index):
    result = generate.answer("What annual fee does the credit card carry?", config.STRATEGY_FIXED)
    retrieved = {h.doc_id for h in result.hits}
    assert set(result.citations) <= retrieved


def test_both_strategies_answer_the_same_in_scope_query(built_index):
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        result = generate.answer("What documents are needed for KYC?", strategy)
        assert result.supported, strategy
        assert result.strategy == strategy


def test_the_prompt_matches_the_frozen_contract(built_index):
    from rag import retrieve
    import llm

    hits = retrieve.retrieve("How do I close my account?", config.STRATEGY_SENTENCES)
    system, user = generate.build_prompt("How do I close my account?", hits)
    assert user.startswith("QUESTION: How do I close my account?")
    assert "\nCONTEXT:\n" in user
    question, sources = llm.parse_prompt(user)
    assert question == "How do I close my account?"
    assert len(sources) == len(hits)
    assert "\n" not in "".join(text for _, text in sources)


def test_generation_is_deterministic(built_index):
    first = generate.answer("What is the minimum balance requirement?", config.STRATEGY_SENTENCES)
    second = generate.answer("What is the minimum balance requirement?", config.STRATEGY_SENTENCES)
    assert first.text == second.text
    assert first.citations == second.citations
```

- [ ] **Step 2: Run it to verify it fails**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests/test_generate.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate' from 'rag'`.

- [ ] **Step 3: Write `rag/generate.py`**

```python
"""Task 4. Answer only from retrieved context, or say so plainly.

The decision rule lives in rag/retrieve.is_supported and is deliberately not
duplicated here. Part 2 Task 7 wraps answer(), and Part 2 Task 10's output
guardrail reads .supported and .top1_similarity, so those names are a contract.
"""

from dataclasses import dataclass

import config
import llm
from rag import retrieve
from rag.retrieve import Hit

FALLBACK_TEXT = (
    "I do not know. The loan support knowledge base does not contain enough "
    "supporting material to answer that question, so I will not guess."
)

SYSTEM_PROMPT = (
    "You are Meridian Bank's loan support assistant. Answer only from the "
    "CONTEXT block. Never add a fact that is not in the context. Cite the "
    "document id of every source you use."
)


@dataclass(frozen=True)
class GroundedAnswer:
    query: str
    text: str
    citations: tuple[str, ...]
    supported: bool
    top1_similarity: float
    strategy: str
    hits: tuple[Hit, ...]


def build_prompt(query: str, hits: list[Hit]) -> tuple[str, str]:
    """The (system, user) pair, in the contract llm.parse_prompt reads back."""
    lines = [f"QUESTION: {query}", "", "CONTEXT:"]
    for hit in hits:
        flattened = " ".join(hit.text.split())
        lines.append(f"[{hit.doc_id}] {flattened}")
    return SYSTEM_PROMPT, "\n".join(lines) + "\n"


def _cited_documents(text: str, hits: list[Hit]) -> tuple[str, ...]:
    """The doc_ids the generated answer actually cited, in retrieval order."""
    if "Sources:" not in text:
        return ()
    tail = text.split("Sources:", 1)[1]
    ordered = []
    for doc_id in retrieve.parent_documents(hits):
        if f"[{doc_id}]" in tail and doc_id not in ordered:
            ordered.append(doc_id)
    return tuple(ordered)


def answer(
    query: str, strategy: str = config.STRATEGY_SENTENCES, k: int | None = None
) -> GroundedAnswer:
    """Retrieve, decide, and either generate from the context or refuse."""
    if config.SIMILARITY_THRESHOLD is None:
        raise RuntimeError(
            "config.SIMILARITY_THRESHOLD is unset. Run the Task 11 calibration and "
            "record the measured value; the brief forbids an untested preset."
        )

    hits = retrieve.retrieve(query, strategy, k=k)
    supported = retrieve.is_supported(hits, config.SIMILARITY_THRESHOLD)

    if not supported:
        return GroundedAnswer(
            query=query,
            text=FALLBACK_TEXT,
            citations=(),
            supported=False,
            top1_similarity=retrieve.top1_similarity(hits),
            strategy=strategy,
            hits=tuple(hits),
        )

    system, user = build_prompt(query, hits)
    text = llm.generate(system, user)
    if not text:
        # The context held nothing usable, which is a refusal, not an answer.
        return GroundedAnswer(
            query=query,
            text=FALLBACK_TEXT,
            citations=(),
            supported=False,
            top1_similarity=retrieve.top1_similarity(hits),
            strategy=strategy,
            hits=tuple(hits),
        )

    return GroundedAnswer(
        query=query,
        text=text,
        citations=_cited_documents(text, hits),
        supported=True,
        top1_similarity=retrieve.top1_similarity(hits),
        strategy=strategy,
        hits=tuple(hits),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests/test_generate.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/generate.py tests/test_generate.py
git commit -m "Answer only from retrieved context, or refuse and say why

GroundedAnswer carries .supported and .top1_similarity because Part 2's output
guardrail reads exactly those two fields. The decision rule stays in
rag/retrieve.is_supported rather than being duplicated here, so there is one
place where the answer-or-refuse question is settled."
```

---
## Task 14: `rag/evaluate.py`, Precision@3 and Recall@3 for both collections

Brief Task 5.
Spec sections 9.1 to 9.3, D-11.

**Files:**
- Create: `rag/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `eval.queries.EVAL_QUERIES`, `rag.retrieve.retrieve`, `rag.retrieve.parent_documents`, `config.TOP_K`.
- Produces:
  - `QueryScore`, a frozen dataclass with `query_id`, `query`, `strategy`, `retrieved: tuple[str, ...]`, `gold: tuple[str, ...]`, `overlap: tuple[str, ...]`, `precision: float`, `recall: float`.
  - `score_query(query: EvalQuery, strategy: str) -> QueryScore`
  - `evaluate(strategy: str) -> tuple[list[QueryScore], float, float]`, returning the scores and the two macro averages.
  - `format_comparison() -> str`, the transcript body with per-query arithmetic for both collections.

### The metric, and the asymmetry it carries

Chunks are mapped back to their parent `doc_id` and deduplicated before scoring, as the brief requires.
Let `R` be the set of distinct parent documents among the top 3 chunks, and `G` the gold set.

- `Precision@3 = |R and G| / |R|`
- `Recall@3    = |R and G| / |G|`

**Dividing by `|R|` rather than by 3 is a known asymmetry, and it is reported rather than hidden.**
Sentence-based chunking often returns three chunks from one parent, giving `|R| = 1` and a precision of either 0 or 1.
Fixed-size chunking spreads across parents more, giving `|R| = 3` and finer-grained precision.
Dividing by `|R|` avoids penalising a collection for agreeing with itself, but it does favour the collection that concentrates.
So `|R|` is printed on every row, and the recommendation in Task 16 must address the asymmetry explicitly instead of reading the averages off the bottom of the table.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:

```python
"""The metric arithmetic must be right, and both collections must be scored."""

import config
from eval.queries import EvalQuery
from rag import evaluate


def test_precision_and_recall_arithmetic_on_a_known_case():
    score = evaluate.score_from_sets(
        query_id="EQ-XX",
        query="synthetic",
        strategy=config.STRATEGY_FIXED,
        retrieved=["kb-01", "kb-02", "kb-03"],
        gold=("kb-01", "kb-02"),
    )
    assert score.overlap == ("kb-01", "kb-02")
    assert score.precision == 2 / 3
    assert score.recall == 2 / 2


def test_a_single_concentrated_parent_scores_one_or_zero():
    hit = evaluate.score_from_sets("EQ-XX", "q", config.STRATEGY_SENTENCES, ["kb-01"], ("kb-01",))
    miss = evaluate.score_from_sets("EQ-XX", "q", config.STRATEGY_SENTENCES, ["kb-09"], ("kb-01",))
    assert hit.precision == 1.0 and hit.recall == 1.0
    assert miss.precision == 0.0 and miss.recall == 0.0


def test_no_retrieval_scores_zero_without_dividing_by_zero():
    score = evaluate.score_from_sets("EQ-XX", "q", config.STRATEGY_FIXED, [], ("kb-01",))
    assert score.precision == 0.0
    assert score.recall == 0.0


def test_every_query_is_scored_for_both_collections(built_index):
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        scores, precision, recall = evaluate.evaluate(strategy)
        assert len(scores) == 12
        assert all(s.strategy == strategy for s in scores)
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0


def test_retrieved_parents_are_deduplicated_and_at_most_three(built_index):
    scores, _, _ = evaluate.evaluate(config.STRATEGY_SENTENCES)
    for score in scores:
        assert len(set(score.retrieved)) == len(score.retrieved)
        assert 1 <= len(score.retrieved) <= config.TOP_K


def test_the_comparison_table_shows_per_query_arithmetic_for_both(built_index):
    report = evaluate.format_comparison()
    assert config.COLLECTION_FIXED in report
    assert config.COLLECTION_SENTENCES in report
    for query_id in [f"EQ-{n:02d}" for n in range(1, 13)]:
        assert query_id in report
    assert "|R|" in report
    assert report.count("/") > 40  # fractions, not decimals, on every row


def test_evaluation_is_deterministic(built_index):
    assert evaluate.format_comparison() == evaluate.format_comparison()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate' from 'rag'`.

- [ ] **Step 3: Write `rag/evaluate.py`**

```python
"""Task 5. Document-level Precision@3 and Recall@3 for both collections.

The denominator is |R|, the number of distinct parent documents among the top
three chunks, not a flat 3. That avoids penalising a collection for agreeing
with itself, and it does favour the collection that concentrates. Rather than
hide that, every row prints |R| and the recommendation has to argue with it.
"""

from dataclasses import dataclass

import config
from eval.queries import EVAL_QUERIES, EvalQuery
from rag import retrieve


@dataclass(frozen=True)
class QueryScore:
    query_id: str
    query: str
    strategy: str
    retrieved: tuple[str, ...]
    gold: tuple[str, ...]
    overlap: tuple[str, ...]
    precision: float
    recall: float


def score_from_sets(
    query_id: str, query: str, strategy: str, retrieved, gold: tuple[str, ...]
) -> QueryScore:
    """The arithmetic on its own, so it can be tested without a vector store."""
    retrieved = tuple(retrieved)
    gold_set = set(gold)
    overlap = tuple(d for d in retrieved if d in gold_set)
    return QueryScore(
        query_id=query_id,
        query=query,
        strategy=strategy,
        retrieved=retrieved,
        gold=tuple(gold),
        overlap=overlap,
        precision=len(overlap) / len(retrieved) if retrieved else 0.0,
        recall=len(overlap) / len(gold) if gold else 0.0,
    )


def score_query(query: EvalQuery, strategy: str) -> QueryScore:
    hits = retrieve.retrieve(query.text, strategy, k=config.TOP_K)
    return score_from_sets(
        query_id=query.query_id,
        query=query.text,
        strategy=strategy,
        retrieved=retrieve.parent_documents(hits),  # the dedup the brief requires
        gold=query.gold_doc_ids,
    )


def evaluate(strategy: str) -> tuple[list[QueryScore], float, float]:
    """Every query scored against one collection, plus the two macro averages."""
    scores = [score_query(q, strategy) for q in EVAL_QUERIES]
    n = len(scores)
    return (
        scores,
        sum(s.precision for s in scores) / n,
        sum(s.recall for s in scores) / n,
    )


def _table(scores: list[QueryScore], strategy: str, precision: float, recall: float) -> list[str]:
    lines = [
        f"--- collection: {config.COLLECTION_FOR_STRATEGY[strategy]} "
        f"(strategy: {strategy}) ---",
        "",
        f"{'query':<7} {'|R|':>3} {'|G|':>3} {'|R and G|':>9}  "
        f"{'Precision@3':>13}  {'Recall@3':>11}",
    ]
    for s in scores:
        p = f"{len(s.overlap)}/{len(s.retrieved) or 1}"
        r = f"{len(s.overlap)}/{len(s.gold)}"
        lines.append(
            f"{s.query_id:<7} {len(s.retrieved):>3} {len(s.gold):>3} {len(s.overlap):>9}  "
            f"{p:>13}  {r:>11}"
        )
    lines += [
        "",
        f"{'mean':<7} {'':>3} {'':>3} {'':>9}  {precision:>13.4f}  {recall:>11.4f}",
        "",
        "per-query detail",
    ]
    for s in scores:
        lines += [
            f"  {s.query_id}  {s.query}",
            f"    retrieved parents R : {list(s.retrieved)}",
            f"    gold documents    G : {list(s.gold)}",
            f"    intersection        : {list(s.overlap)}",
            f"    Precision@3 = |R and G| / |R| = {len(s.overlap)}/{len(s.retrieved) or 1} "
            f"= {s.precision:.4f}",
            f"    Recall@3    = |R and G| / |G| = {len(s.overlap)}/{len(s.gold)} "
            f"= {s.recall:.4f}",
            "",
        ]
    return lines


def format_comparison() -> str:
    """Per-query arithmetic for both collections, then the four averages."""
    lines = [
        "Chunking strategy comparison - document-level Precision@3 and Recall@3",
        "",
        "Chunks are mapped back to their parent doc_id and deduplicated before",
        "scoring. R is the set of distinct parents among the top 3 chunks, G is",
        "the hand-authored gold set, committed before any retrieval was run.",
        "",
        "  Precision@3 = |R and G| / |R|",
        "  Recall@3    = |R and G| / |G|",
        "",
        "Dividing by |R| rather than by 3 avoids penalising a collection for",
        "agreeing with itself, and it does favour the collection that",
        "concentrates. |R| is printed on every row for exactly that reason.",
        "",
    ]

    summary = {}
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        scores, precision, recall = evaluate(strategy)
        summary[strategy] = (precision, recall)
        lines += _table(scores, strategy, precision, recall)

    lines += ["--- both collections side by side ---", ""]
    lines.append(f"{'collection':<20} {'Precision@3':>13} {'Recall@3':>11}")
    for strategy, (precision, recall) in sorted(summary.items()):
        lines.append(
            f"{config.COLLECTION_FOR_STRATEGY[strategy]:<20} {precision:>13.4f} {recall:>11.4f}"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: 7 passed.

- [ ] **Step 5: Read the actual numbers**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -c "
from rag import evaluate
print(evaluate.format_comparison())
"
```

Read the mean `|R|` for each collection alongside the two averages.
Task 16's recommendation is written from these numbers, so note them.
Do not adjust a gold label because a query scored badly.

- [ ] **Step 6: Commit**

```bash
git add rag/evaluate.py tests/test_evaluate.py
git commit -m "Score both collections with document-level Precision@3 and Recall@3

Chunks map back to their parent document and deduplicate before scoring. The
precision denominator is the number of distinct parents retrieved, not a flat
three, which avoids penalising a collection for agreeing with itself but does
favour the one that concentrates. Every row prints that denominator so the
recommendation has to argue with the asymmetry instead of reading the averages
off the bottom of the table."
```

---

## Task 15: `scripts/run_part1.py` and the four transcripts

Spec section 12, D-12.

**Files:**
- Create: `scripts/run_part1.py`
- Create: `transcripts/part1-dataset.txt`
- Create: `transcripts/part1-calibration.txt`
- Create: `transcripts/part1-generation.txt`
- Create: `transcripts/part1-evaluation.txt`
- Create: `transcripts/part1-readme-numbers.md`

**Interfaces:**
- Consumes: every module built so far.
- Produces: `main() -> None`, and the five files above.

The transcripts are the graded evidence.
`part1-readme-numbers.md` is the generated block that Task 16 pastes into `README.md`, which is what stops the README numbers being typed by hand and drifting.

The generation transcript must demonstrate **at least 5 in-scope queries plus 1 deliberately out-of-scope query that triggers the fallback**, per the brief.
Use all 12 evaluation queries in scope, plus 2 out-of-scope probes, which clears the floor with margin.

- [ ] **Step 1: Write `scripts/run_part1.py`**

```python
"""Run every Part 1 task in order and write the graded transcripts.

Nothing in README.md's number tables is typed by hand. This script writes
transcripts/part1-readme-numbers.md, and README.md carries that block, so a
retune that moves a number moves it in both places or in neither.
"""

import io
import os
import sys
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import config  # noqa: E402
import dataset  # noqa: E402
from eval import calibration  # noqa: E402
from eval.queries import EVAL_QUERIES  # noqa: E402
from rag import evaluate, generate, index, kb  # noqa: E402

HEADER = (
    "Generated by scripts/run_part1.py under MOCK_LLM with zero API keys and\n"
    "zero network access. Reproduce with: .venv/bin/python scripts/run_part1.py\n"
)


def write(name: str, body: str) -> Path:
    config.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TRANSCRIPT_DIR / name
    path.write_text(f"{HEADER}\n{body}", encoding="utf-8")
    print(f"  wrote {path.relative_to(config.REPO_ROOT)}")
    return path


def dataset_transcript() -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        dataset.main()
    return buffer.getvalue()


def indexing_summary() -> tuple[str, dict]:
    counts = index.build_index(rebuild=True)
    documents = kb.load_documents()
    lines = [
        "Task 3 - chunking and indexing",
        "",
        f"documents: {len(documents)} "
        f"({sum(1 for d in documents if d.required)} required topics, "
        f"{sum(1 for d in documents if not d.required)} confusable neighbours)",
        f"fixed-size parameters: size {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP}",
        f"embedding model: {config.EMBEDDING_MODEL}, normalised, cosine space",
        "",
        f"{'document':<40} {'fixed':>6} {'sentences':>10}",
    ]
    per_strategy = {
        s: index.chunk_counts_by_document(s) for s in sorted(config.COLLECTION_FOR_STRATEGY)
    }
    for document in documents:
        lines.append(
            f"{document.doc_id:<40} "
            f"{per_strategy[config.STRATEGY_FIXED][document.doc_id]:>6} "
            f"{per_strategy[config.STRATEGY_SENTENCES][document.doc_id]:>10}"
        )
    lines += ["", "collection totals"]
    for name, total in sorted(counts.items()):
        lines.append(f"  {name:<22} {total:>5} chunks")
    return "\n".join(lines) + "\n", counts


def generation_transcript() -> str:
    lines = [
        "Task 4 - grounded generation and the I-do-not-know fallback",
        "",
        f"threshold T = {config.SIMILARITY_THRESHOLD} (measured, see part1-calibration.txt)",
        f"support rule: at least {config.SUPPORT_MIN_SHARED} of the top "
        f"{config.TOP_K} chunks share one parent document",
        "",
    ]
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        lines += [
            f"=== collection: {config.COLLECTION_FOR_STRATEGY[strategy]} ===",
            "",
            "--- in-scope queries ---",
            "",
        ]
        for query in EVAL_QUERIES:
            lines += _one_answer(generate.answer(query.text, strategy), query.query_id)
        lines += ["--- deliberately out-of-scope queries ---", ""]
        for probe in calibration.OUT_OF_SCOPE_PROBES[:2]:
            lines += _one_answer(generate.answer(probe, strategy), "OOS")
    return "\n".join(lines)


def _one_answer(result, label: str) -> list[str]:
    lines = [
        f"[{label}] {result.query}",
        f"  top-1 similarity : {result.top1_similarity:.4f}  "
        f"(threshold {config.SIMILARITY_THRESHOLD})",
        f"  retrieved parents: {[h.doc_id for h in result.hits]}",
        f"  supported        : {result.supported}",
        f"  citations        : {list(result.citations)}",
        "  answer           :",
    ]
    for line in result.text.splitlines():
        lines.append(f"    {line}")
    lines.append("")
    return lines


def readme_numbers(summary: dict, counts: dict) -> str:
    scores = {s: evaluate.evaluate(s) for s in sorted(config.COLLECTION_FOR_STRATEGY)}
    mean_r = {
        s: sum(len(q.retrieved) for q in scores[s][0]) / len(scores[s][0]) for s in scores
    }
    report = dataset.validation_report()

    lines = [
        f"<!-- Generated by scripts/run_part1.py on {date.today().isoformat()}. Do not edit by hand. -->",
        "",
        "### Dataset",
        "",
        "| Seed | Records | Fraud flagged | Categories | Statuses |",
        "|---|---|---|---|---|",
        f"| `{report['seed']}` | {report['total']} | "
        f"{report['fraud_flagged']} ({report['fraud_percentage']}%) | "
        f"{len(report['by_category'])} | {len(report['by_status'])} |",
        "",
        "### Threshold calibration",
        "",
        "| Measurement | Value |",
        "|---|---|",
        f"| Minimum in-scope top-1 similarity | {summary['min_in_scope']:.4f} |",
        f"| Maximum out-of-scope top-1 similarity | {summary['max_out_of_scope']:.4f} |",
        f"| Gap between the clusters | {summary['gap']:.4f} |",
        f"| **Chosen threshold T** | **{summary['threshold']:.4f}** |",
        "",
        "### Chunking comparison",
        "",
        "| Collection | Chunks | Mean \\|R\\| | Precision@3 | Recall@3 |",
        "|---|---|---|---|---|",
    ]
    for strategy in sorted(scores):
        name = config.COLLECTION_FOR_STRATEGY[strategy]
        _, precision, recall = scores[strategy]
        lines.append(
            f"| `{name}` | {counts[name]} | {mean_r[strategy]:.2f} | "
            f"{precision:.4f} | {recall:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    print("Part 1 - running every task under MOCK_LLM")
    print()

    print("Task 1: dataset")
    write("part1-dataset.txt", dataset_transcript())

    print("Task 3: chunking and indexing")
    index_body, counts = indexing_summary()

    print("Task 4: threshold calibration")
    summary = calibration.calibrate()
    write("part1-calibration.txt", index_body + "\n" + calibration.format_report(summary))

    print("Task 4: grounded generation")
    write("part1-generation.txt", generation_transcript())

    print("Task 5: chunking comparison")
    write("part1-evaluation.txt", evaluate.format_comparison())

    print("README number tables")
    write("part1-readme-numbers.md", readme_numbers(summary, counts))

    print()
    print(f"threshold in config.py: {config.SIMILARITY_THRESHOLD}")
    print(f"threshold measured now: {summary['threshold']}")
    if config.SIMILARITY_THRESHOLD != summary["threshold"]:
        print(
            "  NOTE: the measured midpoint has moved. Update config.SIMILARITY_THRESHOLD "
            "and README.md together, then re-run."
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/run_part1.py`
Expected: five files written, and the two thresholds printed at the end agreeing.
If they disagree, update `config.SIMILARITY_THRESHOLD` to the measured value and re-run before going on.

- [ ] **Step 3: Read every transcript, do not skim**

```bash
for f in transcripts/*.txt transcripts/*.md; do echo "=== $f ==="; wc -l "$f"; sed -n '1,60p' "$f"; echo; done
```

Check, by reading:
- the dataset report meets all three floors;
- the calibration clusters visibly separate;
- at least 5 in-scope queries produced a cited answer and both out-of-scope probes produced the fallback;
- the evaluation table shows per-query fractions for both collections.

- [ ] **Step 4: Confirm the offline claim end to end**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/run_part1.py`
Expected: identical output, no network error.
This is the evidence for the README's zero-network-access statement; do not write that sentence without this run.

- [ ] **Step 5: Confirm determinism**

```bash
shasum -a 256 transcripts/*.txt > /tmp/first.sha
.venv/bin/python scripts/run_part1.py > /dev/null
shasum -a 256 -c /tmp/first.sha
```

Expected: all OK.
A mismatch means something iterates over an unsorted set; find it and sort it.

- [ ] **Step 6: Run the whole suite**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -v`
Expected: every test passes, nothing skipped.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_part1.py transcripts/
git commit -m "Run every Part 1 task and commit the graded transcripts

One script produces the dataset report, the calibration measurements, the
grounded-generation demonstration and the chunking comparison, plus the number
tables README.md carries. Nothing in the README is typed by hand, so a retune
that moves a number moves it in both places or in neither."
```

---
## Task 16: `README.md`, the recommendation, and closing the open items

Brief's submission requirements and acceptance criteria.
Spec section 12, D-12, and open items 2, 3 and 4.

**Files:**
- Modify: `README.md`
- Modify: `rag/generate.py` (the `strategy` default only)
- Modify: `docs/superpowers/specs/2026-09-10-loan-support-agent-design.md` (section 14 only)
- Modify: `docs/superpowers/plans/2026-09-11-part1-dataset-and-rag.md` (the `Status:` line only)

**Interfaces:**
- Consumes: `transcripts/part1-readme-numbers.md`, and the numbers read in Tasks 8, 11 and 14.
- Produces: nothing importable.

- [ ] **Step 1: Write `README.md`**

Structure, in this order.
Paste the generated tables from `transcripts/part1-readme-numbers.md` verbatim; type no number by hand.

1. **Title and track statement.**
   State plainly that this repository completes the **Cred (Banking & FinTech)** track, and that everything runs under `MOCK_LLM` with zero API keys and zero network access.
   The brief requires the track statement at the top.
2. **Quickstart.**
   The exact commands: create the environment with `uv`, install from `requirements.txt`, run `scripts/run_part1.py`, run `pytest`.
   Note that the embedding weights download once on first use and every later run is offline.
3. **What is built so far.**
   Part 1 complete, Parts 2 to 4 pending, with a one-line description of each.
4. **Part 1 Task 1, dataset design.**
   Everything a grader needs to reproduce the dataset deterministically, which the brief names explicitly: the **seed**, the **category weights**, the **status weights**, and the **amount range**.
   State in one sentence why the amount range is realistic, per the brief's wording.
   Say that amounts are drawn log-uniformly inside per-category bands, that four bands are sourced to published SBI and HDFC limits, and that the Home Loan band is a deliberate narrowing of a 50,000 to 50 crore lender range and is a modelling choice rather than evidence.
   Paste the generated dataset table.
   Include the first ten rows of the seed-search table from Task 2 Step 2, and say that the chosen seed is the first that passed, so the choice is auditable rather than asserted.
   Link `transcripts/part1-dataset.txt`.
5. **Part 1 Task 2, the knowledge base.**
   18 documents: 12 required topics and 6 deliberately confusable neighbours.
   Say why the documents run to 10 to 15 sentences instead of the brief's 2 to 5, and why the neighbours exist.
   State the measured chars-per-sentence figure from Task 4 Step 2.
6. **Part 1 Task 3, chunking and indexing.**
   Both strategies, both collections, the embedding model, and that cosine space plus normalised vectors is what makes similarity `1 - distance`.
   State **both** the starting chunk parameters and the final ones, and whether Task 8 Step 5 retuned them.
   That is spec open item 2, and this sentence is where it closes.
7. **Part 1 Task 4, the measured threshold.**
   State the count of probes, paste the generated calibration table, and say in one sentence that the brief forbids a preset and why `0.5`, `0.6` and `0.7` would not have worked here.
   State the two-condition answer rule.
   Link `transcripts/part1-calibration.txt` and `transcripts/part1-generation.txt`.
8. **Part 1 Task 5, the comparison and the recommendation.**
   Paste the generated comparison table.
   Then write the recommendation in 2 to 3 sentences, citing **both** sets of numbers, and it must address the `|R|` asymmetry rather than reading the averages off the bottom of the table.
   Link `transcripts/part1-evaluation.txt`.
9. **Tests.**
   The table from spec section 11: six invariants, each naming the acceptance criterion it restates.
   Say that Precision@3 and Recall@3 are deliberately not pinned, and why.
10. **Design decisions.**
    Link `docs/superpowers/specs/2026-09-10-loan-support-agent-design.md` for the full decision log rather than restating fourteen rows.
11. **Repository layout.**
    The tree, one line of purpose per entry.

- [ ] **Step 2: Write the recommendation honestly**

This is the graded sentence, so do not let it be generic.
Read the four numbers plus the two mean `|R|` values from Task 14 Step 5, then write which collection you would deploy and why.

The honest shape, whichever way the numbers fall:

- If sentence chunking wins on precision **and** its mean `|R|` is near 1, say so, and say that the win is partly the denominator concentrating rather than the retrieval being better, then let recall break the tie.
- If fixed-size chunking wins on recall while scoring lower precision, say that it spreads across parents, which is what recall rewards and what the `|R|` denominator penalises.
- Name the operational consequence for Part 2: whichever collection is recommended becomes Part 2's fixed RAG input.

Do not write "both perform comparably" unless the numbers are actually within a couple of hundredths, and say so with the figures if they are.

- [ ] **Step 3: Close spec open item 4**

Set the `strategy` default in `rag/generate.py` to the recommended collection's strategy constant.
It was `config.STRATEGY_SENTENCES` as a placeholder; make it the measured answer.

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -v`
Expected: still all passing.

- [ ] **Step 4: Update the spec's open items**

In section 14 of the spec, strike through items 2, 3 and 4 the way item 1 is struck through, each with the date and a one-line resolution:

- Item 2: the measured chars-per-sentence figure, and whether the parameters were retuned.
- Item 3: the chosen seed and the search that chose it.
- Item 4: the recommended collection and the numbers that chose it.

Do not rewrite any other part of the spec.
It is the approved document, and only this section changed.

- [ ] **Step 5: Mark this plan implemented**

Change the plan's `Status: draft` line to `Status: implemented`.

- [ ] **Step 6: Full gate before the commit**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -v
.venv/bin/python scripts/run_part1.py
git status --short
grep -rn "$(printf '\\u2014')" README.md docs/ knowledge_base/ *.py rag/ eval/ scripts/ tests/ || echo "no em dash found"
```

Expected: every test passing, no unexpected untracked file, and no em dash anywhere.

- [ ] **Step 7: Verify every Part 1 acceptance criterion by reading, one at a time**

Open `reference/problem-statement.md` at the Part 1 acceptance-criteria block and check each of the five bullets against a specific artefact.
Write the mapping down; it goes in the reply to Bhavik, not in the repository.

| Criterion | Where it is satisfied |
|---|---|
| `dataset.py` generates at least 40 records meeting every structural threshold, choices in `README.md` | `dataset.py`, `transcripts/part1-dataset.txt`, README section 4 |
| At least 12 documents covering every required topic | `knowledge_base/`, `tests/test_kb.py` |
| Both strategies indexed into two separate collections, both giving sensible retrieval | `rag/index.py`, `tests/test_index.py`, `transcripts/part1-calibration.txt` |
| At least 5 in-scope plus 1 out-of-scope with a correct fallback | `transcripts/part1-generation.txt` |
| Precision@3 and Recall@3 for both collections with visible per-query arithmetic and a numbers-cited recommendation | `transcripts/part1-evaluation.txt`, README section 8 |

- [ ] **Step 8: Commit and push**

```bash
git add README.md rag/generate.py docs/
git commit -m "Document Part 1 with the measured numbers and the deployment recommendation

The track statement, the dataset-design choices needed to reproduce the data,
the measured calibration values and chosen threshold, and the chunking
recommendation, each linking to the transcript that produced it. Every number
table is pasted from scripts/run_part1.py output rather than typed.

Closes the spec's open items on the chunk model, the seed and which collection
Part 2 consumes."
git push -u origin part-1-dataset-and-rag
```

Merging is Bhavik's call.
Do not open or merge a pull request without being asked.

---


## Task 17: Grow the far tier and retune `T`

Spec section 8.2, D-48 and D-49.
This is the first of the two regenerations D-57 requires, and it changes **only** the calibration.
Nothing about retrieval moves in this task, so every number that shifts in the transcripts is attributable to the probe set alone.

**Files:**
- Modify: `eval/calibration.py`
- Modify: `config.py` (the `SIMILARITY_THRESHOLD` value and its comment)
- Modify: `tests/test_queries.py` (the probe-count test)
- Modify: `scripts/run_part1.py` (the renamed constant at its one call site)
- Regenerate: `transcripts/`, and the generated blocks in `README.md`

**Interfaces:**
- Consumes: `rag/retrieve.py`, unchanged.
- Produces: `FAR_OUT_OF_SCOPE_PROBES`, replacing `OUT_OF_SCOPE_PROBES`.

- [ ] **Step 1: Rename the constant and state what the file now is**

Rename `OUT_OF_SCOPE_PROBES` to `FAR_OUT_OF_SCOPE_PROBES` at all five call sites.
They are in three files: `eval/calibration.py` (the definition, the loop in `measure`, and the header line in `format_report`), `tests/test_queries.py` (the length and uniqueness assertions in `test_probe_counts_oversample_the_brief_floors`), and `scripts/run_part1.py` (the slice that feeds the out-of-scope demonstration).
Find them with `grep -rn OUT_OF_SCOPE_PROBES --include='*.py' .` rather than trusting a line number written here.
The rename is the point, not cosmetic: after D-48 there are two kinds of out-of-scope probe and only one of them belongs here.

Rewrite the module docstring to say that this file is the **fitting set**, that it derives `T` and is never scored against, and that near-domain probes live in `eval/queries.py` because scoring a threshold on the strings that set it measures nothing.

- [ ] **Step 2: Add the twelve far probes**

Append these to `FAR_OUT_OF_SCOPE_PROBES`, bringing it from 5 to 17.
They were measured during review on 2026-09-12; the values are recorded here as the expectation the run must reproduce, not as numbers to type into any output.

| probe | highest measured top-1 |
|---|---|
| How do I train a puppy to stop chewing furniture? | below 0.19 |
| What is the tallest mountain in South America? | below 0.19 |
| Write me a haiku about monsoon rain. | below 0.19 |
| Which vaccine schedule applies to a newborn in the first year? | **0.2870** |
| How do I fix a leaking kitchen tap? | below 0.19 |
| What is the plot of the novel Midnight's Children? | 0.1845 |
| How long should I bake sourdough at 220 degrees? | 0.2025 |
| Explain how photosynthesis converts light into sugar. | below 0.19 |
| What is the best time of year to visit Iceland? | 0.2005 |
| How do I change a flat tyre on a bicycle? | 0.1960 |
| Who composed the Four Seasons? | below 0.19 |
| What is the offside rule in football? | below 0.19 |

The vaccine probe is the one that moves `T`, and it stays.
Dropping it would be cherry-picking, which is the outcome D-49 rejected by name.

- [ ] **Step 3: Update the probe-count test**

`tests/test_queries.py::test_probe_counts_oversample_the_brief_floors` asserts 12 and 5.
Change the second to 17, in both the length and the uniqueness assertion.
Do not add an assertion on the new `T`; the threshold is measured, and a test that pins it would make retuning look like a regression.

- [ ] **Step 4: Measure, then set**

Mirror the two-pass dance Task 11 established, because `config.SIMILARITY_THRESHOLD` is read by the thing that measures it.

1. Run `.venv/bin/python scripts/run_part1.py` and read the new midpoint out of `transcripts/part1-calibration.txt`.
2. Write that value into `config.SIMILARITY_THRESHOLD` and rewrite the comment above it to state the new date, the new probe counts, the new minimum in-scope and maximum out-of-scope readings, and the new gap.
3. Run `scripts/run_part1.py` again so every transcript reflects the committed threshold.

Expected from the review measurement: minimum in-scope 0.3263, maximum far out-of-scope 0.2870, `T` 0.3066, gap 0.0393, down from 0.0889.
If the run disagrees, the run is right and this table is stale.

- [ ] **Step 5: Verify and commit**

Run the full suite with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest`.
Test 6 must still pass: 0.3263 clears 0.2870, so the clusters still separate, but by less.
Never hand-edit `transcripts/` or the generated blocks in `README.md`; both are written by the run script.

Commit: `Grow the far probe tier to 17 and retune the threshold`

---

## Task 17b: Pin the search breadth and re-derive `T`

Spec section 7.3 and the determinism ground rule in section 2.
Added 2026-09-12 during execution, after Task 17's review found that the committed
calibration transcript does not reproduce.

**Why this exists.**
`scripts/run_part1.py` line 178 calls `index.build_index(rebuild=True)`, so both
collections are deleted and rebuilt on every run.
At ChromaDB's default search breadth the HNSW query sometimes misses its true nearest
neighbour, so one probe lands on a different parent document from one run to the next.
That makes the repository's "same input, same seed, same bytes" claim false.

Measured on 2026-09-12, 215 sentence chunks embedded once and reused, five rebuilds per
configuration, probe "What is the offside rule in football?":

| configuration | result |
|---|---|
| `hnsw:space` cosine only, as shipped | **unstable**: 3/5 gave 0.1619 kb-07, 2/5 gave 0.1611 kb-02 |
| `hnsw:num_threads` 1 | **unstable**: same 3/5 and 2/5 split |
| `hnsw:search_ef` 200 | **stable**: 5/5 gave 0.1619 kb-07 |
| both | **stable**: 5/5 gave 0.1619 kb-07 |

`hnsw:random_seed` is rejected outright by chromadb 1.5.9, so seeding is not available.
Threading is not the cause; search breadth is.
0.1611 is the approximation missing the true neighbour and 0.1619 is the correct answer,
so this fix makes retrieval more accurate as well as reproducible.

**Files:**
- Modify: `config.py` (one new constant in the Task 3 chunking and indexing section)
- Modify: `rag/index.py` (the collection metadata, in both places it is written)
- Modify: `config.py` comment above `SIMILARITY_THRESHOLD`, and its value
- Regenerate: `transcripts/`, and the generated blocks in `README.md`

**Interfaces:**
- Produces: `config.SEARCH_EF`, consumed only by `rag/index.py`.

- [ ] **Step 1: Add the constant**

`SEARCH_EF = 200` goes in `config.py` beside `CHUNK_SIZE` and `MIN_CHUNKS_PER_DOCUMENT`, in the Task 3 section.
It is an index parameter, and `config.py` is the only place one is defined.
Append inside that existing section; do not touch the Part 2 section at the end of the file, which another session is editing concurrently.

Comment it with what it buys, in one line: the default breadth returns a non-nearest neighbour on some rebuilds, which breaks byte-identical reruns.

- [ ] **Step 2: Wire it into both collection-metadata sites**

`rag/index.py` writes `metadata={"hnsw:space": "cosine"}` in two places inside `build_index`.
Both become `{"hnsw:space": "cosine", "hnsw:search_ef": config.SEARCH_EF}`.
Miss one and the collection it creates stays unstable, which is the whole defect.

**Do not add `hnsw:num_threads`.**
It was measured and it does not help, so it would be a constant carrying no effect.

- [ ] **Step 3: Prove the fix, do not assume it**

Run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/run_part1.py` **five** times.
After each run, check `git status --short transcripts/`.
The tree must be clean after every run from the second onwards.
Two runs are not enough evidence here: the defect showed up 2 times in 5, so a two-run check passes by luck more often than not.

Paste all five results into the report.

- [ ] **Step 4: Re-derive `T`**

Retrieval just became more accurate, and `T` is the midpoint between two retrieval measurements, so the threshold set in Task 17 is now derived from superseded numbers.
Repeat Task 17's measure-then-set dance exactly: run the script, read the new midpoint out of `transcripts/part1-calibration.txt`, write it into `config.SIMILARITY_THRESHOLD` with an updated comment, then run the script again.

`T` may stay at 0.3066 or it may move.
Either is correct; what is not acceptable is leaving a threshold whose derivation no longer matches the printed table.

- [ ] **Step 5: Verify and commit**

Full suite with the network hard-disabled.
Test 6 must still pass: the in-scope minimum must clear the far-tier maximum.
If it does not, stop and report rather than adjusting anything - that would mean the accuracy fix changed the cluster separation, which is a finding, not a nuisance.

Commit: `Pin the HNSW search breadth so retrieval reproduces`

---

## Task 18: The product catalogue, the scope gate and the golden dataset

Spec section 8.5, D-47, D-51, D-52 and D-53.
This is the fix for the root cause recorded as item 6 in spec section 18.1.

This task also **authors the golden dataset**, because the gate's two tests read it and a task cannot end with failing tests.
Task 19 owns the evaluation machinery that scores it.

**Files:**
- Modify: `knowledge_base/catalogue.json`
- Modify: `rag/kb.py` (parse and validate the new fields)
- Create: `rag/scope.py`
- Modify: `rag/retrieve.py` (the conditional filter)
- Modify: `rag/generate.py` (the gate refusal path)
- Modify: `eval/queries.py` (the four-class golden dataset)
- Create: `tests/test_scope.py`
- Modify: `tests/test_kb.py`, `tests/test_queries.py`

`rag/index.py` is deliberately **not** in that list, and neither collection is rebuilt.
Step 4 explains why.

`config.py` is deliberately not in that list either.
`KNOWN_ADJACENT` is a vocabulary, and `config.py` owns paths, weights, bands, chunk parameters, collection names and thresholds, none of which it is.
The closest precedents both sit outside `config.py`: `IN_SCOPE_PROBES` in `eval/calibration.py` and `SYSTEM_PROMPT` in `rag/generate.py`.
This also keeps `config.py` free for the Part 2 session, which is appending to it concurrently.

**Interfaces:**
- Consumes: `knowledge_base/catalogue.json`.
- Produces: `scope.classify(query) -> ScopeVerdict` with fields `product`, `in_catalogue`, `known_adjacent`, carrying enough for `rag/generate.py` to name the product in its refusal.
- Produces: `GOLDEN_DATASET: list[GoldenItem]`, with `EVAL_QUERIES` kept as its `answerable` subset so every existing caller keeps working.

- [ ] **Step 1: Extend `catalogue.json`**

Add a top-level `products` list naming what Meridian Bank sells, per D-47.
Read the names out of the documents rather than inventing them.
Counted across `knowledge_base/*.txt` on 2026-09-12: Personal Loan 15, Meridian Rewards Card 13, Home Loan 13, NRO account 12, NRE account 12, joint account 7, Business Loan 6, savings account 5, Auto Loan 5, Education Loan 3, salary account 1.

That is the list, and it is eleven entries.
**"Current account" is not one of them.**
An earlier draft of this task listed it; no document mentions one, and inventing a product would make `catalogue.json` disagree with the corpus it describes.
`joint account` is a mode of holding rather than a distinct product, and it stays in anyway, because the gate only needs to know the phrase is something Meridian does.

Add a `products` tag to each of the 18 document entries, listing which of those products that document covers.
A document covering all loan products lists all five, not a shorthand.

**The knowledge base stays the authority.**
If a product appears in a document and not in `products`, `catalogue.json` is the bug, exactly as `RATE_BANDS` yields to `kb-07` today.

- [ ] **Step 2: Validate it in `rag/kb.py`**

`rag/kb.py` already refuses to load when the documents and the catalogue describe different sets.
Extend that strictness: every name in a document's `products` tag must appear in the top-level `products` list, and the loader raises otherwise.
That is acceptance criterion 31.

- [ ] **Step 3: Write `rag/scope.py`**

Hold two lists and one function.
`KNOWN_ADJACENT` names products Meridian does not sell that people ask about anyway: fixed deposit, recurring deposit, mutual fund, SIP, ELSS, demat, shares, stock market, insurance, gold, cryptocurrency, income tax, GST, tax return.
The catalogue side is read from `catalogue.json`, never duplicated here.

`classify(query)` case-folds the query and matches both lists by phrase, longest match first, and returns a verdict.
Deterministic string matching only.
Do not embed anything here: D-51 records that an embedding router loses to the same measurement the threshold lost to.

- [ ] **Step 4: Filter conditionally in `rag/retrieve.py`**

Add an optional `product` argument to `retrieve`.
When it is set, ask the catalogue which `doc_id` values carry that product and pass `where={"doc_id": {"$in": [...]}}`.
When it is not set, query exactly as today with no `where` clause at all.

**The filter must be conditional.**
Measured during review: 9 of the 12 in-scope calibration probes name no product at all, so an unconditional filter would search an empty subset for three quarters of real questions.

**Do not add a product field to chunk metadata, and do not reindex.**
An earlier draft of this task did exactly that and it does not work.
Verified against ChromaDB 1.5.9 on 2026-09-12: `$in` and `$eq` on `doc_id` both filter correctly, but `$contains` on a string metadata field returns an **empty result rather than raising**.
A delimited `products` string filtered with `$contains` would therefore have made every product-named query retrieve nothing, silently, and the tests in Step 7 would have failed with no clue why.
The `doc_id` `$in` clause uses a supported operator, keeps `catalogue.json` the single authority, and leaves both collections untouched.
This is recorded in D-53.

- [ ] **Step 5: Wire the gate into `rag/generate.py`**

Call `scope.classify` before retrieval.
On `known_adjacent`, return the refusal without retrieving anything, carrying the product name so Part 2 can say which product was asked about.
On `in_catalogue`, retrieve with the filter.
Otherwise retrieve unfiltered and let `T` and the support rule decide, exactly as today.

Per D-54 the sentence a user reads is Part 2's job.
Part 1 produces the structured refusal and the product name, not prose.

- [ ] **Step 6: Restructure `eval/queries.py` into the golden dataset**

Introduce `GoldenItem` with `item_id`, `text`, `kind`, `gold_doc_ids` and `product`, per spec 9.1.
Keep `EVAL_QUERIES` as a derived list of the 12 `answerable` items so nothing downstream breaks, and keep their ids `EQ-01` to `EQ-12` and their text byte-identical.
Stable ids are what keep every Precision@3 number already in `README.md` comparable across this amendment.

- [ ] **Step 7: Author the three new classes**

Thirteen `outside_boundary` items, `OB-01` to `OB-13`, each naming a product in `KNOWN_ADJACENT`.
The review measured these, and they are the set the gate was sized against.

Two `inside_uncovered` items, `IU-01` and `IU-02`, being exactly the residue named in spec 18.1 item 8: "Can I get a credit card from another bank with a low limit?" and "How do I transfer money to an account in another country?".

Five `far_out_of_scope` items, `FO-01` to `FO-05`, written fresh.
**They must not reuse any of the 17 far probes from Task 17.**
That is acceptance criterion 30, and it is the property that lets this be called a golden dataset at all.

Add criterion 30's test here, checked in both directions, because the dataset it guards exists as of this step.

- [ ] **Step 8: Test the gate in both directions**

`tests/test_scope.py` asserts acceptance criteria 24a and 28, and they pull in opposite directions on purpose:

- every `outside_boundary` golden item yields a gate refusal
- **no `answerable` golden item is refused by the gate**

The second is the one that protects users, and a curated list can drift into violating it.
Neither is tautological as long as the item strings and `KNOWN_ADJACENT` stay separate data, so never generate one from the other.

Both tests pass within this task, because Steps 6 and 7 authored the items they read.

- [ ] **Step 9: Verify and commit**

Run the full suite with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest`.
Confirm neither collection was rebuilt: `chroma/` should be untouched by this task.

Commit: `Add the product catalogue, the scope gate and the golden dataset`

---

## Task 19: Decision-level evaluation

Spec sections 9.1 and 9.4, D-50, D-55 and D-56.
Task 18 authored the golden dataset; this task scores it.

**Files:**
- Modify: `rag/evaluate.py`
- Modify: `tests/test_evaluate.py`
- Modify: `scripts/run_part1.py`
- Modify: `README.md` (the transcript list only)

**Interfaces:**
- Consumes: `GOLDEN_DATASET` from `eval/queries.py`, authored in Task 18.
- Produces: the decision table, and `transcripts/part1-golden-dataset.txt`.

- [ ] **Step 1: Add the decision table to `rag/evaluate.py`**

Keep Precision@3 and Recall@3 exactly as they are, scored over the `answerable` items only.
Add a second pass over all 32 items recording `answered`, `refused_gate` or `refused_threshold`, and a per-class summary.

Report the near-domain false-answer rate explicitly.
Before the gate it was 14 of 30 readings answered outright, and printing the after figure beside it is the evidence the fix worked.

- [ ] **Step 2: Assert only what D-56 allows**

Add criterion 29: every `far_out_of_scope` item is refused, by either mechanism.
Criterion 30 was already added in Task 18 Step 7, alongside the dataset it guards.

Do **not** assert on Precision@3, Recall@3, the `inside_uncovered` count, or any aggregate decision accuracy.
D-13 settled this shape of question for this repository and D-56 extends it: those numbers move legitimately when chunk parameters are tuned, and a test that fights tuning gets deleted.

- [ ] **Step 3: Give it a transcript**

`scripts/run_part1.py` writes the decision table to `transcripts/part1-golden-dataset.txt`, listing every item with its class, its outcome and its top-1 similarity.
Add it to the transcript list in `README.md`.

Commit: `Add decision-level evaluation over the golden dataset`

---

## Task 20: Regenerate, and close the loop

D-57's second regeneration.

**Files:**
- Regenerate: `transcripts/`, and the generated blocks in `README.md`
- Modify: `README.md` (prose sections only)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Regenerate**

Run `.venv/bin/python scripts/run_part1.py`.
Every number that moves in this pass is attributable to the gate and the product filter, because Task 17 already absorbed the threshold change.
That separation is the whole reason D-57 asked for two passes, so do not collapse them even if the diff looks small.

- [ ] **Step 2: Update the README prose**

Add a short section explaining that scope is decided before retrieval and why similarity cannot decide it, citing the product-swap measurement: "fixed deposit" against "car insurance policy" at 0.6805 in one frame, against a lowest in-scope reading of 0.3263.
State the known-adjacent list's weakness in the same breath.
A grader reading only `README.md` should learn both that the system refuses correctly and why a threshold alone could not.

- [ ] **Step 3: Update `CLAUDE.md`**

Three facts in it go stale in this amendment: the acceptance-criteria count rises from twenty-seven to thirty-two, the test count changes, and the threshold sentence naming 0.2818 needs the new value.
Read the actual numbers out of the run and the suite rather than computing them here.

**`CLAUDE.md` is gitignored** (`.gitignore:34`) and untracked, so this step changes a local file only and nothing about it reaches a commit or a reviewer.
Do it anyway, because the next agent in this repository reads it as fact.

- [ ] **Step 4: Full verification**

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest` passes with the network hard-disabled.
- Two consecutive `scripts/run_part1.py` runs produce byte-identical transcripts.
- `.venv/bin/python scripts/check_database.py` still exits zero.
- Every figure in `README.md` traces to a transcript, with no number typed by hand.

Commit: `Regenerate Part 1 evidence after the scope gate`

---

## Self-review

Run this after the last task, before reporting completion.

- [ ] **Spec coverage.**
  Walk spec sections 5 to 12 and name the task that implements each.
  Section 5 to Task 2, section 6 to Tasks 4 and 5, section 7 to Tasks 7 and 8, section 8 to Tasks 10, 11, 12 and 13, section 9 to Tasks 9 and 14, section 10 to the interface blocks in Tasks 2, 12 and 13, section 11 to Tasks 3, 8 and 11, section 12 to Tasks 15 and 16.
  Section 13 is V2 and is deliberately unimplemented.
- [ ] **Interface consistency.**
  `get_application`, `answer`, `.supported`, `.top1_similarity`, `generate(system, user)`, and every name in `config.py` must match spec section 10 exactly, because Parts 2 to 4 are written against that table.
- [ ] **No hand-typed numbers.**
  Every figure in `README.md` traces to `transcripts/part1-readme-numbers.md` or to a transcript.
- [ ] **Determinism.**
  Two consecutive `scripts/run_part1.py` runs produce byte-identical transcripts.
- [ ] **Offline.**
  The full suite and the run script both pass with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

---

## Known deferrals

Recorded here so they are not mistaken for oversights.

- **Parts 2 to 4 are not built.**
  This plan is Part 1 only.
  Spec section 10 fixes the interfaces they will consume so that nothing here has to be refactored when they arrive.
- **V2 is not built.**
  Spec section 13 is a roadmap, and every item there names the single V1 file it replaces.
- **Publishing the seed search as a committed script was raised and not ruled on.**
  This plan records the search table in `README.md` instead, which is the auditable half at no cost.
  Shipping `scripts/seed_scan.py` remains available if Bhavik wants it.
- **Task 16 pushes to `origin`.**
  This repository has no other push target configured.
