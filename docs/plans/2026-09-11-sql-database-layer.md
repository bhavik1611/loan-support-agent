# Relational Store Implementation Plan

Status: implemented

> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a seven-table SQLite database, generated from the same seeded generator that already produces `LOAN_APPLICATIONS`, gitignored with a committed text manifest, plus a one-command check script a grader can run.

**Architecture:** `dataset.py` keeps drawing the brief's six fields on the existing stream, then enriches each row on a separate stream, so every record committed before this work stays byte-identical.
`db/` holds schema, generators, build and query as four single-responsibility modules.
Nothing outside `config.py` knows the database path or a stream offset.

**Tech Stack:** Python 3.12.13, `sqlite3` from the standard library (no new dependency), pytest.

**Spec:** [`docs/specs/2026-09-10-loan-support-agent-design.md`](../specs/2026-09-10-loan-support-agent-design.md), section 5.4 and decisions D-15 to D-23.

**Predecessor:** [`2026-09-11-part1-dataset-and-rag.md`](2026-09-11-part1-dataset-and-rag.md), status `implemented`. That plan stays closed; this one is additive and must not regress it.

---

## Global Constraints

Every task's requirements implicitly include this section.

- Python 3.12.13 at `.venv/bin/python`. Never call `pip`. No new third-party dependency: `sqlite3` is stdlib.
- Never use the em dash. Plain dash `-`.
- One full sentence per line in markdown.
- Every value is fabricated. Meridian Bank is fictional. No real person's data.
- **The hard promise of this work:** `data/loan_applications.json` and its SHA-256 test must be unchanged when you are done.
  If a step changes that file, the step is wrong.
  Verify with `git diff --exit-code data/loan_applications.json` before every commit.
- The existing 85 tests must stay green throughout. Run the full suite, not just the new file.
- Prefix any run that loads the embedding model with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- Commit at the end of each task with the message the task gives.
- Branch `part-1-dataset-and-rag`. Never commit to `main`.

### Frozen identifiers

**New `config.py` constants:**

| Name | Value |
|---|---|
| `DB_PATH` | `DATA_DIR / "meridian_bank.db"` |
| `DB_MANIFEST` | `DATA_DIR / "database-manifest.md"` |
| `STREAM_OFFSETS` | `{"loan_applications": 0, "customers": 1, "loan_products": 2, "application_terms": 3, "application_events": 4, "repayments": 5, "support_tickets": 6, "kyc_documents": 7}` |
| `LOANS_PER_CUSTOMER_MIX` | `[(1, 42), (2, 16), (3, 6), (4, 2)]` |
| `CUSTOMER_COUNT` | `66` (derived: `sum(n for _, n in MIX)`) |
| `TICKET_COUNT` | `40` |
| `SCHEDULE_MONTHS` | `12` |

`loan_applications` keeps offset **0**, which is literally the existing `random.Random(SEED)`.
That is what makes the byte-identity promise hold.

**Table names, fixed:** `loan_products`, `customers`, `loan_applications`, `application_events`, `repayments`, `support_tickets`, `kyc_documents`.

**The six projected fields, in this order:** `record_id`, `category`, `status`, `loan_amount_inr`, `days_since_created`, `flagged_for_fraud_review`.

**Knowledge-base vocabularies that generated rows may not exceed.**
Read the document before generating; do not invent a value.

| Column | Must come from |
|---|---|
| `kyc_documents.doc_type` | the identity and address proofs listed in `kb-04-kyc-documents.md` |
| `support_tickets.channel` | the reporting channels named in `kb-05-fraud-dispute.md` |
| `loan_applications.interest_rate_pct` | inside the product's band in `kb-07-interest-rate-slabs.md` |
| `repayments` amounts | `kb-02-emi-calculation.md`'s formula, to the rupee |

---

## Execution waves

This work is more linear than Part 1 was; say so rather than inventing parallelism.

| Wave | Tasks | Notes |
|---|---|---|
| 0 | 1 | `config.py` and `db/schema.py`. Everything imports these. |
| 1 | 2 | `dataset.py` enrichment. Blocks the applications generator. |
| 2 | 3, 4 | Generators and `db/query.py` touch different files and can run together. |
| 3 | 5 | `db/build.py` needs the generators. |
| 4 | 6, 7 | Check script and tests, disjoint files. |
| 5 | 8 | README and spec open items. |

---

## Task 1: `config.py` constants and `db/schema.py`

**Files:**
- Modify: `config.py` (append a new section only)
- Create: `db/__init__.py`, `db/schema.py`
- Modify: `.gitignore`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: every constant in the Frozen identifiers table, plus `db.schema.CREATE_STATEMENTS: dict[str, str]`, `db.schema.TABLE_ORDER: list[str]`, and `db.schema.create_all(conn) -> None`.

`TABLE_ORDER` is insertion order and must respect foreign keys: `loan_products`, `customers`, `loan_applications`, `application_events`, `repayments`, `support_tickets`, `kyc_documents`.

- [ ] **Step 1: Append the constants to `config.py`**

Add below the existing dataset block, above the chunking section:

```python
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
}

# (loans held, how many customers hold that many). Sums to 100 loans over 66
# customers: most retail customers hold one, a minority two, a few more.
LOANS_PER_CUSTOMER_MIX = [(1, 42), (2, 16), (3, 6), (4, 2)]
CUSTOMER_COUNT = sum(count for _, count in LOANS_PER_CUSTOMER_MIX)

TICKET_COUNT = 40
SCHEDULE_MONTHS = 12  # first year of the amortisation schedule per disbursed loan
```

- [ ] **Step 2: Add the stream helper to `config.py`**

```python
def stream(table: str) -> int:
    """The seed for one table's own random stream."""
    try:
        return SEED + STREAM_OFFSETS[table]
    except KeyError:
        raise ValueError(
            f"unknown table {table!r}, expected one of {sorted(STREAM_OFFSETS)}"
        ) from None
```

- [ ] **Step 3: Write `db/schema.py`**

Seven `CREATE TABLE` statements, each with `PRIMARY KEY`, explicit `REFERENCES` on every foreign key, and `NOT NULL` on everything that is not genuinely optional.
Column sets are in spec section 5.4.
Open every connection with `PRAGMA foreign_keys = ON`, otherwise SQLite does not enforce the references and test 8 becomes decorative.

```python
"""The seven CREATE TABLE statements and nothing else.

SQLite does not enforce foreign keys unless the pragma is on per connection,
so connect() sets it. Without that, the orphan test in tests/test_database.py
would pass against a database that has orphans.
"""

import sqlite3

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
            product_code    TEXT PRIMARY KEY,
            category        TEXT NOT NULL UNIQUE,
            min_amount_inr  INTEGER NOT NULL,
            max_amount_inr  INTEGER NOT NULL,
            min_rate_pct    REAL NOT NULL,
            max_rate_pct    REAL NOT NULL,
            max_tenure_months INTEGER NOT NULL,
            is_secured      INTEGER NOT NULL CHECK (is_secured IN (0, 1))
        )
    """,
    # ... the remaining six, same shape
}


def connect(path):
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def create_all(conn) -> None:
    for table in TABLE_ORDER:
        conn.execute(CREATE_STATEMENTS[table])
```

Write out all seven. The remaining six carry, at minimum:

- `customers`: `customer_id` PK, `full_name`, `city`, `state`, `pan`, `aadhaar`, `account_number`, `email`, `phone`, `employment_type`, `annual_income_inr`, `credit_score`, `kyc_status`, `is_nri`.
- `loan_applications`: `record_id` PK, `customer_id` FK, `product_code` FK, `category`, `status`, `loan_amount_inr`, `days_since_created`, `flagged_for_fraud_review`, `tenure_months`, `interest_rate_pct`.
- `application_events`: `event_id` PK, `record_id` FK, `sequence_no`, `from_status`, `to_status`, `occurred_days_ago`, `note`.
- `repayments`: `repayment_id` PK, `record_id` FK, `instalment_no`, `due_days_ago`, `emi_inr`, `principal_inr`, `interest_inr`, `balance_inr`, `paid`.
- `support_tickets`: `ticket_id` PK, `customer_id` FK, `record_id` FK nullable, `channel`, `category`, `opened_days_ago`, `status`, `summary`.
- `kyc_documents`: `document_id` PK, `customer_id` FK, `doc_type`, `doc_kind`, `submitted_days_ago`, `verified`.

- [ ] **Step 4: Let the database through `.gitignore` deliberately**

`*.sqlite` and `*.sqlite3` are already ignored and `meridian_bank.db` is not, so add it explicitly rather than relying on the suffix:

```
# Generated relational store. Rebuild with: .venv/bin/python -m db.build
data/meridian_bank.db
```

- [ ] **Step 5: Write and run the test**

`tests/test_schema.py`: all seven statements parse and create, `TABLE_ORDER` covers exactly `CREATE_STATEMENTS`, the pragma is on after `connect`, and every `stream()` value is distinct.

Run: `.venv/bin/python -m pytest tests/test_schema.py -v`

- [ ] **Step 6: Commit**

```bash
git add config.py db/__init__.py db/schema.py tests/test_schema.py .gitignore
git commit -m "Add the relational schema and one seeded stream per table

Seven tables with enforced foreign keys - SQLite ignores REFERENCES unless the
pragma is set per connection, so connect() sets it and the orphan test means
something. loan_applications keeps stream offset 0, which is the existing
Random(SEED), so no record generated before this commit changes."
```

---

## Task 2: enrich the application row without moving a single existing byte

**Files:**
- Modify: `dataset.py`
- Test: `tests/test_dataset.py` (append)

**Interfaces:**
- Consumes: `config.stream`, `config.CATEGORY_BANDS`.
- Produces:
  - `generate_applications(seed, count) -> list[dict]` - **unchanged behaviour**, still the six fields, still stream 0.
  - `enrich_applications(records) -> list[dict]` - returns rich rows with `product_code`, `tenure_months`, `interest_rate_pct` added. `customer_id` is attached later by the customer generator, which owns the assignment.
  - `RICH_APPLICATIONS: list[dict]`
  - `LOAN_APPLICATIONS: list[dict]` - the six-field projection of `RICH_APPLICATIONS`, in the frozen field order.
  - `PROJECTED_FIELDS: tuple[str, ...]`

This is the task where the byte-identity promise is kept or broken.
The mechanism: the six fields are drawn first, by the untouched existing loop on stream 0.
Enrichment runs afterwards on stream `application_terms`, so it cannot perturb the original draw sequence.

- [ ] **Step 1: Write the failing test first**

Append to `tests/test_dataset.py`:

```python
def test_the_projection_is_byte_identical_to_the_committed_snapshot():
    """The whole promise of the database work: adding it moved no record."""
    import hashlib
    on_disk = config.DATASET_SNAPSHOT.read_bytes()
    assert hashlib.sha256(dataset.snapshot_bytes()).hexdigest() == \
        hashlib.sha256(on_disk).hexdigest()


def test_the_projection_keeps_exactly_the_six_fields_in_order():
    assert dataset.PROJECTED_FIELDS == (
        "record_id", "category", "status",
        "loan_amount_inr", "days_since_created", "flagged_for_fraud_review",
    )
    for record in dataset.LOAN_APPLICATIONS:
        assert tuple(record) == dataset.PROJECTED_FIELDS


def test_the_rich_row_is_a_superset_of_the_projection():
    for rich, projected in zip(dataset.RICH_APPLICATIONS, dataset.LOAN_APPLICATIONS):
        for field in dataset.PROJECTED_FIELDS:
            assert rich[field] == projected[field]
        assert rich["product_code"]
        assert rich["tenure_months"] > 0
        assert rich["interest_rate_pct"] > 0


def test_enrichment_does_not_disturb_the_six_field_draw():
    """Calling the enricher must not consume from the stream that draws the six."""
    before = dataset.generate_applications()
    dataset.enrich_applications(dataset.generate_applications())
    after = dataset.generate_applications()
    assert before == after


def test_tenure_and_rate_sit_inside_the_product_band():
    for row in dataset.RICH_APPLICATIONS:
        low, high = config.RATE_BANDS[row["category"]]
        assert low <= row["interest_rate_pct"] <= high
        assert 0 < row["tenure_months"] <= config.MAX_TENURE_MONTHS[row["category"]]
```

`config.RATE_BANDS` and `config.MAX_TENURE_MONTHS` are new in this task.
Source both from `knowledge_base/kb-07-interest-rate-slabs.md` and `kb-14-home-loan-ltv.md`; read those files, do not invent numbers.

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: the new tests fail on missing attributes; the nine existing ones still pass.

- [ ] **Step 3: Implement**

Leave `generate_applications` **completely untouched**.
Add below it:

```python
PROJECTED_FIELDS = (
    "record_id",
    "category",
    "status",
    "loan_amount_inr",
    "days_since_created",
    "flagged_for_fraud_review",
)


def enrich_applications(records: list[dict]) -> list[dict]:
    """Add the columns the database carries, on their own stream.

    Runs after the six-field draw, never inside it. That ordering is the whole
    reason data/loan_applications.json does not move when the database lands.
    """
    rng = random.Random(config.stream("application_terms"))
    rich = []
    for record in records:
        category = record["category"]
        low_rate, high_rate = config.RATE_BANDS[category]
        max_tenure = config.MAX_TENURE_MONTHS[category]
        rich.append(
            {
                **record,
                "product_code": config.PRODUCT_CODES[category],
                "tenure_months": rng.choice(config.TENURE_CHOICES[category]),
                "interest_rate_pct": round(rng.uniform(low_rate, high_rate), 2),
            }
        )
    return rich


RICH_APPLICATIONS: list[dict] = enrich_applications(generate_applications())

LOAN_APPLICATIONS: list[dict] = [
    {field: row[field] for field in PROJECTED_FIELDS} for row in RICH_APPLICATIONS
]
```

Move the existing `LOAN_APPLICATIONS = generate_applications()` line; do not leave two definitions.
`_BY_ID` keeps indexing `LOAN_APPLICATIONS`, so `get_application` is unchanged and Part 2's contract holds.

- [ ] **Step 4: Prove the snapshot did not move**

```bash
.venv/bin/python -m pytest tests/test_dataset.py -v
git diff --exit-code data/loan_applications.json && echo "SNAPSHOT UNCHANGED"
.venv/bin/python -c "
import dataset, config, hashlib
print('regenerated :', hashlib.sha256(dataset.snapshot_bytes()).hexdigest()[:16])
print('on disk     :', hashlib.sha256(config.DATASET_SNAPSHOT.read_bytes()).hexdigest()[:16])
"
```

Both digests must match and the `git diff` must be clean.
If they do not, the enrichment is drawing from stream 0; fix that, do not rewrite the snapshot.

- [ ] **Step 5: Run the whole suite**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q`
Expected: 85 existing plus the new ones, nothing failing.

- [ ] **Step 6: Commit**

```bash
git add dataset.py config.py tests/test_dataset.py
git commit -m "Enrich the application row without moving the committed snapshot

The six brief fields are still drawn by the original loop on stream 0, and
enrichment runs afterwards on its own stream, so data/loan_applications.json
and its SHA-256 test are untouched. A test asserts the enricher cannot consume
from the stream that draws the six, because that is the failure mode that
would silently rewrite every record_id the transcripts quote."
```

---

## Task 3: `db/generate.py`, the seven row generators

**Files:**
- Create: `db/generate.py`
- Test: `tests/test_db_generate.py`

**Interfaces:**
- Produces one function per table, each opening its own stream:
  `generate_loan_products()`, `generate_customers()`, `assign_customers(applications, customers)`, `generate_application_events(applications)`, `generate_repayments(applications)`, `generate_support_tickets(applications, customers)`, `generate_kyc_documents(customers)`.
  Each returns `list[dict]` whose keys match its table's columns exactly.

**`loan_products`** is seeded from `config.CATEGORY_BANDS`, `config.RATE_BANDS` and `config.MAX_TENURE_MONTHS`, not retyped.
Five rows, one per category, `product_code` from `config.PRODUCT_CODES`.

**`customers`.**
66 rows. Fabricated Indian names from a name pool defined in the module.
PAN matches `[A-Z]{5}[0-9]{4}[A-Z]`, Aadhaar is 12 digits, account number 11 to 16 digits - the fixed formats Part 2 Task 10 masks.
`credit_score` is drawn in 300 to 900, skewed high, and must be at least 700 for any customer holding a loan, because `kb-01` states 700 as the minimum.
`is_nri` true for about 8 percent.

**`assign_customers`** implements D-20: shuffle the 100 applications on the customers stream, then deal them out according to `LOANS_PER_CUSTOMER_MIX`, writing `customer_id` onto each rich application row.
Every customer must end up with at least one loan, and the counts must match the mix exactly.

**`application_events`.**
Between 2 and 5 per application, forming a legal path through the status vocabulary ending at the application's current `status`.
Legal transitions: Submitted to Under Review, Under Review to Approved or Rejected, Approved to Disbursed.
`occurred_days_ago` is monotonically decreasing along the sequence and the first one equals `days_since_created`.

**`repayments`.**
Only for applications with status `Disbursed`.
The first `SCHEDULE_MONTHS` instalments of the amortisation schedule.
This is the one with arithmetic that a test checks against the knowledge base:

```python
def _emi(principal: int, annual_rate_pct: float, tenure_months: int) -> float:
    """kb-02's formula, and nothing else may compute an EMI in this repository."""
    r = annual_rate_pct / 12.0 / 100.0
    growth = (1.0 + r) ** tenure_months
    return principal * r * growth / (growth - 1.0)


def _schedule(principal: int, annual_rate_pct: float, tenure_months: int, months: int):
    """Standard amortisation: interest on the running balance, principal is the rest."""
    r = annual_rate_pct / 12.0 / 100.0
    emi = _emi(principal, annual_rate_pct, tenure_months)
    balance = float(principal)
    for instalment_no in range(1, min(months, tenure_months) + 1):
        interest = balance * r
        principal_part = emi - interest
        balance -= principal_part
        yield {
            "instalment_no": instalment_no,
            "emi_inr": round(emi, 2),
            "interest_inr": round(interest, 2),
            "principal_inr": round(principal_part, 2),
            "balance_inr": round(max(balance, 0.0), 2),
        }
```

`paid` is true only for instalments whose due date has already passed given `days_since_created`, which at 0 to 30 days means at most one.
Say so in a comment rather than fabricating a repayment history the loan age cannot support.

**`support_tickets`.**
40 rows. `channel` must be one of the channels `kb-05` names - read the file.
About half link to an application, the rest are account-level with `record_id` null.
Tickets on fraud-flagged applications are over-represented, which gives Part 2 a signal.

**`kyc_documents`.**
2 to 4 per customer, at least one identity proof and one address proof.
`doc_type` must appear in `kb-04` - read the file.
NRI customers additionally carry a passport and a visa, per `kb-12`.

- [ ] **Step 1: Read the four knowledge-base documents before writing any vocabulary**

```bash
cat knowledge_base/kb-02-emi-calculation.md
cat knowledge_base/kb-04-kyc-documents.md
cat knowledge_base/kb-05-fraud-dispute.md
cat knowledge_base/kb-07-interest-rate-slabs.md
```

Write the permitted values down. Anything not in these files is not a permitted value.

- [ ] **Step 2: Write `tests/test_db_generate.py` first**

Cover: row counts per table, the mix producing exactly 100 assignments over 66 customers with none empty, PAN and Aadhaar format regexes, credit score at least 700 for loan holders, event sequences ending at the application's status with decreasing days, the EMI schedule balance reaching near zero at full tenure, `doc_type` and `channel` drawn from the knowledge-base vocabularies, and determinism (two calls produce identical output).

- [ ] **Step 3: Run it, watch it fail, then implement**

- [ ] **Step 4: Verify determinism explicitly**

```bash
.venv/bin/python -c "
from db import generate
for fn in ['generate_loan_products','generate_customers']:
    a, b = getattr(generate, fn)(), getattr(generate, fn)()
    print(f'{fn:<26} deterministic: {a == b}')
"
```

- [ ] **Step 5: Commit**

```bash
git add db/generate.py tests/test_db_generate.py
git commit -m "Generate the seven tables, each on its own seeded stream

Every vocabulary is read out of the knowledge base rather than invented: KYC
document types from kb-04, ticket channels from kb-05, rate bands from kb-07.
The EMI schedule uses kb-02's formula and nothing else in this repository
computes an EMI, so the database cannot contradict the document that explains
it to the customer."
```

---

## Task 4: `db/query.py`, the read helpers Part 2 consumes

**Files:**
- Create: `db/query.py`
- Test: `tests/test_db_query.py`

**Interfaces:**
- `customer_context(record_id, conn=None) -> dict | None` returning exactly `customer_id`, `full_name`, `credit_score`, `open_loan_count`, per D-21.
  **No PAN, no Aadhaar, no account number.** That is the contract, and a test asserts it.
- `loan_book(customer_id, conn=None) -> list[dict]`
- `application_timeline(record_id, conn=None) -> list[dict]`
- `repayment_schedule(record_id, conn=None) -> list[dict]`
- `open_tickets(customer_id, conn=None) -> list[dict]`
- `table_counts(conn=None) -> dict[str, int]`

- [ ] **Step 1: Write the test that pins the PII contract**

```python
FORBIDDEN = {"pan", "aadhaar", "account_number"}


def test_customer_context_never_returns_pii(built_db):
    context = query.customer_context("LN-1001")
    assert FORBIDDEN.isdisjoint(context)
    assert set(context) == {"customer_id", "full_name", "credit_score", "open_loan_count"}


def test_customer_context_is_none_for_an_unknown_record(built_db):
    assert query.customer_context("LN-9999") is None
```

Plus: `loan_book` returns every loan that customer holds and nothing else, `application_timeline` comes back in sequence order, `repayment_schedule` is empty for a non-disbursed application.

Add a `built_db` session fixture to `tests/conftest.py` that builds the database into a `tmp_path_factory` directory, so tests never depend on a developer having run the build.

- [ ] **Step 2: Implement, run, commit**

```bash
git add db/query.py tests/test_db_query.py tests/conftest.py
git commit -m "Add the read helpers Part 2 consumes, with PII kept out by contract

customer_context returns the customer id, name, credit score and open loan
count and nothing else. A test asserts PAN, Aadhaar and account number cannot
appear in it, because the brief puts PII masking on the input side and serving
PII here would need an output masker the brief never asked for."
```

---

## Task 5: `db/build.py` and the manifest

**Files:**
- Create: `db/build.py`
- Create: `data/database-manifest.md` (generated, then committed)
- Test: `tests/test_database.py`

**Interfaces:**
- `build_database(path=None) -> dict[str, int]` mapping table name to row count. Drops and recreates; never appends.
- `manifest_text(path=None) -> str`
- `write_manifest(path=None) -> Path`
- `content_hash(path=None) -> str` - SHA-256 over a canonically ordered dump, so it is stable across SQLite page layout.
- `main()` so `python -m db.build` works.

`content_hash` must hash the **data**, not the file bytes.
A SQLite file can differ byte for byte while holding identical rows, so hash `SELECT * FROM <table> ORDER BY <pk>` for every table in `TABLE_ORDER`, serialised deterministically.

- [ ] **Step 1: Write tests 7 to 12 from spec section 11**

`tests/test_database.py`: every FK resolves, every amount and rate sits inside its `loan_products` row, every `repayments` row reproduces `kb-02`'s EMI to the rupee, every `doc_type` and `channel` appears in its knowledge-base document, a fresh build matches the committed manifest, and two builds produce the same `content_hash`.

The EMI test must recompute independently rather than call the same helper the generator used, or it proves nothing:

```python
def test_every_repayment_reproduces_the_kb02_formula(built_db):
    conn = schema.connect(built_db)
    rows = conn.execute("""
        SELECT r.*, a.loan_amount_inr, a.interest_rate_pct, a.tenure_months
        FROM repayments r JOIN loan_applications a USING (record_id)
        WHERE r.instalment_no = 1
    """).fetchall()
    assert rows, "no repayment rows to check"
    for row in rows:
        r = row["interest_rate_pct"] / 12.0 / 100.0
        n = row["tenure_months"]
        growth = (1.0 + r) ** n
        expected = row["loan_amount_inr"] * r * growth / (growth - 1.0)
        assert abs(row["emi_inr"] - expected) < 1.0, row["record_id"]
```

- [ ] **Step 2: Implement, build, write the manifest**

```bash
.venv/bin/python -m db.build
cat data/database-manifest.md | head -40
```

- [ ] **Step 3: Prove the build is reproducible**

```bash
.venv/bin/python -c "from db import build; print(build.content_hash())"
.venv/bin/python -m db.build >/dev/null
.venv/bin/python -c "from db import build; print(build.content_hash())"
git diff --exit-code data/database-manifest.md && echo "MANIFEST STABLE"
git diff --exit-code data/loan_applications.json && echo "SNAPSHOT STILL UNCHANGED"
```

- [ ] **Step 4: Commit**

```bash
git add db/build.py data/database-manifest.md tests/test_database.py
git commit -m "Build the database and commit a manifest a test can check

The manifest hashes the rows rather than the file, because a SQLite file can
differ byte for byte while holding identical data. Six invariants land with
it, three of which assert the generated rows cannot contradict the knowledge
base: the EMI arithmetic, the KYC document types and the ticket channels."
```

---

## Task 6: `scripts/check_database.py`, the grader's one command

**Files:**
- Create: `scripts/check_database.py`

Per D-22, six checks, and PII **always masked** with no flag offered.

- [ ] **Step 1: Write it**

Build the database if it is absent, then print, in order:

1. Schema: every table, its columns, its row count.
2. Referential integrity: orphan count per child table, which must be zero.
3. Domain checks: amounts, rates and tenures inside their product row; statuses in the vocabulary; `days_since_created` in 0 to 30.
4. Knowledge-base agreement: recompute every first instalment against `kb-02`'s formula and report the largest deviation in rupees.
5. Manifest match: built hash against the committed hash.
6. Five example joins, each printed with the SQL that produced it.

Masking:

```python
def mask(value: str) -> str:
    """Fabricated values, masked anyway. One behaviour, no flag: showing the
    discipline costs nothing and Part 2 Task 10 formalises it later."""
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
```

Exit non-zero if any check fails, so it works in CI as well as by eye.

- [ ] **Step 2: Run it and read every line**

```bash
.venv/bin/python scripts/check_database.py; echo "exit: $?"
```

Confirm no PAN, Aadhaar or account number appears unmasked:

```bash
.venv/bin/python scripts/check_database.py | grep -nE '[A-Z]{5}[0-9]{4}[A-Z]|\b[0-9]{12}\b' && echo "LEAK - fix the masking" || echo "no unmasked PII in output"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/check_database.py
git commit -m "Add the grader's one-command database check

Six checks: schema, referential integrity, domain bounds, knowledge-base
agreement on the EMI arithmetic, manifest match, and five worked example
joins printed with their SQL. Exits non-zero on failure so it is usable in CI
as well as by eye. PII is always masked and no flag is offered to unmask it."
```

---

## Task 7: full-suite verification

- [ ] Run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q` and confirm every test passes, old and new.
- [ ] Run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/run_part1.py` and confirm the four Part 1 transcripts are unchanged: `git diff --exit-code transcripts/`.
- [ ] Run `git diff --exit-code data/loan_applications.json`.
- [ ] Scan for em dashes across everything this work touched.

Any failure here is a regression in this plan's work, not a pre-existing issue. Fix it before Task 8.

---

## Task 8: `README.md` and the spec's open items

**Files:**
- Modify: `README.md`
- Modify: `docs/specs/2026-09-10-loan-support-agent-design.md` (open items only)
- Modify: this plan's `Status:` line

- [ ] **Step 1: Add a database section to `README.md`**

Place it after the Task 1 dataset section, because it extends that task. It must state:

- Why the database exists: the brief's six fields are a floor, and three knowledge-base documents describe behaviour no six-field record could exercise.
- The seven tables and their row counts, in a table.
- That it is generated and gitignored, with the manifest committed and hash-tested, and the two commands: build, then check.
- That every existing record is byte-identical to before the database landed, and which test proves it.
- That every vocabulary is read out of the knowledge base, naming the three tests that enforce it.
- That PII is fabricated, masked in all output, and never returned by the lookup path.

- [ ] **Step 2: Close the spec's open items**

Add item 6, resolved, recording the measured row counts and the byte-identity result.

- [ ] **Step 3: Mark this plan `implemented`**

- [ ] **Step 4: Commit**

```bash
git add README.md docs/
git commit -m "Document the relational store and close the spec's open items"
```

---

## Self-review

- [ ] Every table in spec section 5.4 has a generator, a schema entry and at least one test.
- [ ] `data/loan_applications.json` is unchanged. This is the one that matters most.
- [ ] No function outside `db/generate.py::_emi` computes an EMI.
- [ ] `customer_context` cannot return PII, and a test says so.
- [ ] The manifest is generated by the same command that builds the database.
- [ ] No new third-party dependency was added.

## Known deferrals

- Part 2 wiring is not in this plan. `db/query.py` exposes what Task 6 of the brief will consume, but nothing calls it yet.
- `support_tickets` and `kyc_documents` earn their place through the check script and the knowledge-base agreement tests. If they end up unused by Parts 2 to 4, they are demo surface, and that was the accepted trade in D-16.
