# Loan Support Agent - design specification

Status: approved design for Part 1, declared interfaces for Parts 2 to 4, roadmap only for V2.
Written 2026-09-10 after two grilling rounds.
Amended 2026-09-12 after three more, adding the time axis in D-26 to D-31.
Authority: `reference/problem-statement.md` is the brief; where this document and the brief disagree, the brief wins and this document is wrong.

## 1. Scope

This repository answers the **Cred (Banking & FinTech)** capstone track with a loan-operations support agent.
V1 is the brief-compliant system: it answers loan-policy questions from a knowledge base written here, looks up loan applications from a dataset generated here, and is orchestrated, deployed, evaluated and hardened across four parts.
V2 is a later, portfolio-grade rebuild that is described in section 10 and implemented nowhere in this repository.

This document specifies Part 1 in full.
Parts 2 to 4 appear only as the interfaces they will consume, so that Part 1 is built against a known contract rather than refactored into one later.

## 2. Ground rules

Every rule below applies to all four parts and is not restated per section.

- The system runs offline with zero API keys under `MOCK_LLM`, which is the default and the mode every graded transcript uses.
- Every output is deterministic: same input, same seed, same bytes.
- A real language model may be wired in behind the `LLM_PROVIDER` environment variable, and no acceptance criterion may depend on it.
- No screenshots, PDFs, slides, images or diagrams are produced anywhere; every deliverable is code or text in the repository.
- Names, PAN numbers, Aadhaar numbers, account numbers and income figures are fabricated throughout.
- Python 3.12.13, environment managed with `uv`.

The Python version is a deliberate choice, not a default.
Python 3.14 is installed on the build machine and is a wheel risk for `torch`, which `sentence-transformers` requires.

## 3. Decision log

Each row records what was decided, why, and what lost.
Decisions marked "overrode recommendation" were taken by Bhavik against the recommendation in the grilling round, and the reason is recorded because the reason is the part worth keeping.

| # | Decision | Chosen | Rejected, and why it lost |
|---|---|---|---|
| D-01 | Scale beyond the brief's floors | 100 records, 18 KB docs, 12 eval queries | Brief minimums of 40/12/5 lost because Precision@3 on 5 queries moves in steps of 0.067 and the two chunking strategies would likely tie, leaving the Task 5 recommendation with nothing to cite. |
| D-02 | Repository layout | `dataset.py` at repository root as real code, packages for everything multi-module | A flat root lost because Part 4 would leave roughly 20 unstructured modules. A `src/` package lost because the brief names `dataset.py` in its submission list, and satisfying that from `src/` needs a re-export shim, which is wrapper indirection for its own sake. |
| D-03 | How V2 appears | A roadmap section in this document, zero V2 code in the repository | Building the seams now lost because a `Retriever` protocol with one implementation is a layer serving code that does not exist. A V1-only spec lost because some V1 choices are cheap now and expensive to unwind later. |
| D-04 | Knowledge base shape | 12 required topics plus 6 deliberately confusable neighbour topics | Six further distinct topics lost because every query would then have exactly one relevant document, both strategies would score Precision@3 of 1/3, and the comparison would degenerate to a tie. |
| D-05 | Document length | 10 to 15 sentences each, targeting 12 | The brief's 2 to 5 sentences lost because at that length the two chunking strategies produce nearly the same chunks. 18 to 25 sentences lost because a document that long stops being one topic and the gold labels blur. |
| D-06 | What `MOCK_LLM` generation produces | Deterministic template synthesis: classify question shape, select template, fill slots from retrieved chunks, cite parent document ids | Pure extractive stitching lost because groundedness would then score 1.0 on all 15 Part 3 queries by construction and the RAG triad would demonstrate nothing. A small local generative model lost because it needs a weight download, which breaks zero network access. |
| D-07 | The "I do not know" decision rule | Calibrated threshold on top-1 cosine similarity, **and** at least 2 of the top 3 chunks sharing a parent document | Overrode recommendation. A single threshold lost once document length grew: the objection to the support rule was that short documents can yield one chunk, and D-05 removes that premise. A two-tier threshold lost because it doubles the calibration to justify for a marginal gain. |
| D-08 | Support rule window | At least 2 of the top **3** chunks share a parent | Two of top 5 lost because evidence ranked fifth is weak support dressed up as agreement. Retrieval already fetches 3 for Precision@3, so this window costs no extra query. |
| D-09 | Dataset persistence | Seeded generator is the source of truth, snapshot committed to `data/loan_applications.json`, one test asserts the generator still reproduces the snapshot hash | Generator alone lost because nothing would detect a future edit reshuffling every `record_id` that Parts 2 to 4 quote in transcripts. Snapshot alone lost because the file and the generator drift apart silently. |
| D-10 | Loan amount distribution | Per-category bands, sampled log-uniformly | One global uniform range lost because it produces a Home Loan of 80 thousand rupees beside a Personal Loan of 48 lakh. Per-category uniform lost because it puts the median Home Loan at the midpoint of its band, and real portfolios are dominated by smaller tickets. |
| D-11 | Gold label design | Graded and mixed: 4 queries with 1 relevant document, 5 with 2, 3 with 3 | One document per query lost because Recall@3 could then only take the values 0 and 1, which makes it decoration rather than a metric. Primary-plus-secondary lost because it scores like the one-document option with extra bookkeeping. |
| D-12 | Where graded evidence lives | Scripts write full transcripts under `transcripts/`, and `README.md` carries the design choices, measured numbers and recommendation, each linking to its transcript | Transcripts alone lost because the brief requires measured values stated in `README.md`. README alone lost because it would run to thousands of lines and be regenerated by hand, which is the drift D-09 exists to prevent. |
| D-13 | Test scope | Invariants mapped one-to-one onto the brief's acceptance criteria | Hash test alone lost because nothing would catch a weight change pushing the fraud rate outside its band. Pinning Precision@3 and Recall@3 lost because those numbers legitimately move when chunk parameters are tuned, so the suite would fight the work. |
| D-14 | Fixed-size chunk parameters | 400 characters, 80 overlap, regex sentence splitter | 600 characters lost because a 12-sentence document then yields 3 chunks against 12 sentence chunks, and the fixed-size collection loses the granularity the comparison needs. 250 lost because chunks cut mid-clause often enough to hurt embedding quality. NLTK punkt lost because it needs a download. |
| D-15 | Whether the dataset stops at the brief's six fields | A seven-table SQLite database alongside the six-field list | **Overrode recommendation, and the recommendation was wrong.** The spec previously argued that extra fields "would need justifying to a grader for no gain". That reads the brief's field list as a ceiling when it is a floor: the brief says each record *needs* those six, not that it may carry no more. Stopping there also left three knowledge-base documents (EMI, prepayment, foreclosure) describing behaviour no data could exercise. |
| D-16 | How far the schema goes | All seven tables: `customers`, `loan_products`, `loan_applications`, `application_events`, `repayments`, `support_tickets`, `kyc_documents` | Overrode recommendation. Five tables lost because the two dropped ones, ticket history and per-document KYC state, are exactly the surfaces a support agent is asked about, and a capstone is judged partly on what it can demonstrate. The cost is accepted explicitly: seven tables are seven chances to contradict the knowledge base, so D-19 exists. |
| D-17 | Database persistence | Generated on demand, gitignored, with a committed text manifest that a test checks | Committing the binary lost because git cannot diff it and every regeneration is an opaque blob in the history. Generating with nothing committed lost because it abandons the D-09 guarantee that a future edit cannot silently change what the repository claims. A committed `.sql` dump lost because hundreds of INSERT lines duplicate the generator as a second source of truth. |
| D-18 | Random-stream isolation | One seeded sub-stream per table, `Random(SEED + offset)` | A single interleaved stream lost because it changes all 100 existing records, and with them the seed choice, the measured fraud rate, the README table, a committed transcript and the snapshot hash, for no gain. Drawing customers strictly last lost because it is safe only until the next table is inserted in the middle. |
| D-19 | Where a record is defined | The generator builds the rich row; `LOAN_APPLICATIONS` is its six-field projection | Keeping the list and the database as parallel generators lost because two places would then decide what a record is. Reading the list from SQLite at import lost because it makes importing `dataset.py` depend on a generated file existing. The projection keeps only the six fields, so `data/loan_applications.json` and its SHA-256 test are unchanged by this work. |
| D-20 | Customers to loans | About 66 customers holding 1 to 4 loans each, skewed toward 1 | A strict one-to-one split lost because the foreign key would then be decorative and "what else do I have open" could not be asked. Heavier concentration lost because a third of the book sitting with a handful of customers is not a retail portfolio. |
| D-21 | What the lookup tool returns | The brief's three keys plus non-PII customer context: `customer_id`, name, credit score, open-loan count | Returning the full record including PAN and Aadhaar lost because the brief puts PII masking on the **input** side and makes the output guardrail a groundedness check; serving PII would require inventing an output masker the brief never asked for. Returning only the three mandated keys lost because the customer tables would then never surface through the agent. |
| D-22 | PII in the grader's check script | Always masked, no flag | Printing fabricated PAN and Aadhaar in the clear is harmless and lost anyway, because masking costs nothing and shows the discipline before Part 2 formally implements it. No flag is offered: one behaviour, chosen. |
| D-23 | How this lands in the documents | Amend this spec, keep the Part 1 plan marked implemented, write a new dated plan | Reopening the Part 1 plan lost because its `implemented` status and self-review are accurate for the work it actually covered. A second spec lost because two specs describing one dataset is precisely the drift the one-spec rule prevents. |
| D-24 | How a fabricated PAN is built | The Income Tax Department's structure, `AAAAA9999A`: three random series letters, holder type `P`, the surname initial, a 0001-9999 serial, then a check letter derived from the first nine characters | Ten independent random characters lost because the PAN could then contradict the name beside it, and Part 2's masking demo is more convincing on data that is wrong only in being fabricated. Reproducing the real check-character formula lost because the department does not publish it; the derivation here is a documented stand-in, deterministic and recomputable, and `tests/test_db_generate.py` recomputes it rather than calling the generator's helper. |
| D-25 | How a fabricated Aadhaar is built | Twelve digits, the first drawn from 2-9, the twelfth a Verhoeff check digit over the other eleven | Twelve independent random digits lost because the number would fail any real validator, which is the wrong thing to demonstrate in a masking exercise. Unlike PAN's check character (D-24), Verhoeff is the published algorithm UIDAI actually uses, so the check digit here is real rather than a stand-in and the tests pin Verhoeff's own worked examples. Storing the 4-4-4 display grouping lost because the grouping is presentation, and a stored separator would break the masking Part 2 Task 10 applies to the raw value. |
| D-26 | How time is represented | `days_since_created` stays the drawn integer; every calendar value in the repository is derived from it against a fixed anchor | Drawing a date and deriving the integer lost because it changes the draw shape inside stream 0, which moves every `record_id` the transcripts quote. Storing only the integer lost because relative offsets are unfalsifiable: nothing can contradict "13 days ago", and anchoring the same data exposed a defect within minutes (D-29). The four child tables keep no integer at all, because nothing outside this repository asks for one; `days_since_created` alone survives because the brief names it and Part 2 Task 6 scores on it, which is the only honest reason to hold two representations of one fact. |
| D-27 | The anchor | `config.AS_OF_DATETIME = 2026-09-30T18:00:00+05:30`, frozen, in IST, not announced in the README | A rolling anchor lost outright: it breaks the determinism ground rule and the manifest hash would change daily. Midnight lost because a day-0 event exists in the committed data and would land in the future. UTC lost because 09:30 IST business hours render as 04:00Z and stop looking like business hours to a reader; a naive timestamp lost because it invites a guess, and the guess is usually UTC. Overrode recommendation on announcing it: the recommendation was to state the as-of date in `README.md` and the manifest so the data's ageing is explicit rather than discovered, and Bhavik chose to keep the constant silent. |
| D-28 | Where the clock's randomness comes from | A new seeded stream, `config.stream("clock")`, at the first free offset | Drawing hours and minutes from each table's own stream lost because it reshuffles every row of all four tables, moving days, amounts and notes for a cosmetic gain. A fixed derived time with no randomness lost because every event would land at the same clock time, which is visibly synthetic. This is the case D-18's offset scheme was built for, and it is the reason a full timestamp costs nothing beyond the manifest rebuild D-26 already requires. |
| D-29 | Working days and business hours | Bank-side transitions move forward to the next working day; `Submitted` never moves; every timestamp falls between 09:30 and 18:00 | Anchoring the existing offsets showed 99 of 393 events on a Saturday or Sunday, against five knowledge-base sentences written in working days (`kb-01`, `kb-05`, `kb-06`), and the knowledge base is the authority. Shifting backwards lost because it says the bank acted before it could have; nearest lost because it mixes both stories with no per-row reason; forward is a monotone map, which is why it produces no order inversions. `Submitted` is exempt because the generator's own note calls it an online submission, and an online form takes a Sunday one; the exemption is also load-bearing, because `Submitted` is `created_at` and moving it would break D-26's derivation. EMI due dates are exempt for the opposite reason: a due date is a contractual date, not an action, and shifting it would distort the amortisation schedule. Bank holidays lost because no document in this repository carries a holiday calendar, and inventing one contradicts the rule that the knowledge base is the authority. Overrode recommendation on the hours: the recommendation was two windows, 24 hours for customer-side rows and 09:30 to 18:00 for the bank, and Bhavik chose one window for one rule. |
| D-30 | Stored or derived, for facts the trail already carries | `updated_at` and `paid` are stored, each pinned by a test asserting it equals what the trail says | Deriving them in `db/query.py` lost because it costs a correlated subquery on every read and hides the column Part 2 wants to quote; storing without a test lost because the two copies can then disagree and nothing notices. The uniqueness invariant on an application's events moves from distinct days to distinct instants, which is what the timestamp in D-26 was for: two events on one Monday morning and afternoon are a working day, not a defect. |
| D-31 | Whether the snapshot carries the new fields | `data/loan_applications.json` gains `created_at` and `updated_at`, appended after the brief's six in the brief's order | Overrode recommendation. Keeping the projection at six fields lost, although it was recommended: it would have left the committed snapshot byte-identical and the hash test untouched. Bhavik chose to carry the timestamps through to the projection, so acceptance criterion 7 is restated rather than deleted, and the fields are appended rather than interleaved so that `PROJECTED_FIELDS`' promise to hold "the brief's six fields, in the brief's order" stays literally true. |

## 4. Repository layout

```
loan-support-agent/
  dataset.py                  Task 1. Seeded generator, LOAN_APPLICATIONS, lookup, validation report.
  config.py                   Paths, chunk parameters, calibrated threshold, environment flags.
  llm.py                      Provider seam. MOCK_LLM default, real provider behind LLM_PROVIDER.
  knowledge_base/             Task 2. 18 markdown documents with YAML front matter.
  rag/
    kb.py                     Load and parse knowledge_base/ into Document objects.
    chunking.py               Task 3. chunk_fixed and chunk_sentences.
    index.py                  Task 3. Embed and index into two ChromaDB collections.
    retrieve.py               Task 4. Top-k retrieval, cosine similarity, support rule.
    generate.py               Task 4. Grounded generation and the fallback.
    evaluate.py               Task 5. Precision@3 and Recall@3 for both collections.
  eval/
    queries.py                12 evaluation queries with graded gold labels.
    calibration.py            In-scope and out-of-scope probe queries for threshold calibration.
  scripts/
    run_part1.py              Runs every Part 1 task and writes the transcripts.
    check_database.py         The grader's one-command database check.
  db/
    schema.py                 The seven CREATE TABLE statements.
    generate.py               One generator per table, each on its own seeded stream.
    build.py                  Build the database and write the manifest.
    query.py                  Read helpers Part 2 consumes.
  data/
    loan_applications.json    Committed snapshot, guarded by a hash test.
    database-manifest.md      Committed DDL, row counts and hash of the database.
    meridian_bank.db          Generated seven-table store, gitignored.
  transcripts/                Committed graded evidence.
  tests/                      Invariant tests, one per acceptance criterion.
  docs/specs/                This design specification.
  reference/problem-statement.md
  chroma/                     Generated vector store, gitignored.
```

Parts 2 to 4 add `agent/`, `api/`, `mcp_server/` and `mcp_client.py` alongside these.
No existing module moves when they arrive, which is the point of D-02.

## 5. Part 1 Task 1 - dataset

### 5.1 Record schema

Every record is a flat dictionary with exactly these keys.

| Field | Type | Domain |
|---|---|---|
| `record_id` | str | `LN-1001` through `LN-1100` |
| `category` | str | Personal Loan, Home Loan, Auto Loan, Education Loan, Business Loan |
| `status` | str | Submitted, Under Review, Approved, Rejected, Disbursed |
| `loan_amount_inr` | int | Per-category band, rounded to the nearest 1000 |
| `days_since_created` | int | 0 to 30 inclusive |
| `flagged_for_fraud_review` | bool | Category-dependent Bernoulli draw |
| `created_at` | str | ISO 8601 in IST, derived from `days_since_created` against D-27's anchor |
| `updated_at` | str | ISO 8601 in IST, the instant of the application's most recent event |

Those six are what the brief names as an acceptance criterion.
Per D-31 `LOAN_APPLICATIONS` carries them plus the two timestamps, appended after the six in the brief's order.
They are not the whole record.
Per D-15 and D-19 the generator builds a richer row - adding `customer_id`, `product_code`, `tenure_months` and `interest_rate_pct` - writes all of it to the database in section 5.4, and exposes the eight above as a projection.
The projection is what `data/loan_applications.json` snapshots.
The database work of D-15 to D-22 left that snapshot byte-identical; D-31 deliberately does not, and acceptance criterion 7 in section 11 is restated to say so.

An earlier version of this section said no field is added beyond the brief's six, on the grounds that extra fields would need justifying to a grader for no gain.
That was wrong twice over.
The brief says each record *needs* those six, which is a floor and not a ceiling, and three knowledge-base documents describe EMI, prepayment and foreclosure behaviour that no six-field record could ever exercise.

### 5.2 Generation

A single `random.Random(SEED)` instance drives every draw, in a fixed field order per record, so the sequence is reproducible from the seed alone.
Seeded `random` was verified identical across Python 3.11.7, 3.12.13, 3.13.5 and 3.14.0 over 200 iterations of `choices`, `randint`, `uniform`, `random` and `sample`, so the snapshot in D-09 protects against future edits to this file rather than against interpreter drift.

**Category weights.**
Personal 0.34, Auto 0.22, Home 0.20, Education 0.12, Business 0.12.
These reflect retail lending mix, where unsecured personal lending is the highest-volume product and business lending the lowest by count.
At 100 records the least frequent category expects 12, far above the brief's floor of 3 per category.

**Status weights.**
Under Review 0.24, Approved 0.22, Disbursed 0.22, Submitted 0.18, Rejected 0.14.
A support queue is dominated by in-flight and recently completed applications, so the terminal states are not the largest buckets.
Every status expects at least 14 records, far above the brief's floor of 1.

**Loan amount.**
Sampled log-uniformly inside a per-category band, then rounded to the nearest 1000:
`amount = round(exp(uniform(log(lo), log(hi))), -3)`.

| Category | Band (INR) | Source |
|---|---|---|
| Personal Loan | 50,000 to 40,00,000 | Verified. Published Indian personal-loan ranges for 2026 quote 50,000 to 40 lakh, extending to 50 lakh for premium profiles. |
| Auto Loan | 1,00,000 to 25,00,000 | Verified. SBI states a 1 lakh minimum with no cap; HDFC states a 25 lakh maximum for a new car. The band takes the tighter of the two ends. |
| Education Loan | 1,00,000 to 50,00,000 | Verified. HDFC lends up to 50 lakh unsecured for premier institutes; SBI's domestic tiers top out at 40 lakh. Foreign-study schemes reach 3 crore and are deliberately excluded as atypical. |
| Business Loan | 1,00,000 to 50,00,000 | Verified. Published MSME ranges run 50,000 to 50 lakh and above. Lowered from an earlier unverified 2 lakh to 75 lakh to match the sourced range. |
| Home Loan | 10,00,000 to 1,50,00,000 | **Modelling choice, not a lender limit.** SBI quotes a 50,000 minimum and a 50 crore maximum, which is too wide to sample from usefully. This band brackets the typical retail case and is narrower than reality on purpose. |

Log-uniform rather than uniform because a real loan book is dominated by smaller tickets, and Part 2 Task 6 asks for an escalation threshold justified as a percentile of this data.
A uniform draw makes every percentile arbitrary.
Four of these bands were unverified when this document was first written and have since been checked against published SBI and HDFC limits.
The Home Loan band remains a deliberate narrowing rather than a sourced limit, and `README.md` says so.

**Days since created.**
`int(rng.triangular(0, 30, 8))`, clamped to 0 to 30.
Skewed toward recent because most items in a live support queue are new, which gives Part 2's recency signal a distribution with a meaningful 80th percentile instead of a flat one.

**Fraud flag.**
A category-dependent Bernoulli draw rather than a flat coin toss: base probability 0.15, plus 0.10 for Business Loan, plus 0.05 for Personal Loan.
Expected rate is approximately 0.179, inside the brief's 10 to 30 percent band with margin on both sides.
Unsecured and business lending carry higher fraud incidence than secured retail lending, so the flag carries signal that Part 2's escalation score can use.
If the realised rate lands outside the band, the seed changes and the dataset regenerates; individual records are never hand-edited.

### 5.3 Public surface

- `LOAN_APPLICATIONS: list[dict]`, built at import time.
- `get_application(record_id: str) -> dict | None`, the lookup Part 2 Task 6 wraps.
- `validation_report() -> dict`, returning counts per category, counts per status, and the fraud percentage.
- `main()`, printing that report, which is what `scripts/run_part1.py` captures into a transcript.

### 5.4 The relational store

Per D-15 to D-22.
A seven-table SQLite database at `data/meridian_bank.db`, built by `db/build.py` from the same seeded generator, gitignored, and described by a committed manifest at `data/database-manifest.md`.

SQLite rather than a server database because the brief requires the project to run at a grader's end with no paid account and no network, and because Part 4 already puts `langgraph-checkpoint-sqlite` in the stack.
The file is named for the bank so that it cannot be confused with Part 4 Task 15's `checkpoints.sqlite`, which is framework state rather than business data.

**Determinism.**
Each table draws from its own stream, `random.Random(SEED + offset)`, with the offsets fixed in `config.py`.
`loan_applications` keeps offset 0, which is the existing `Random(SEED)`, so every record committed before this work is byte-identical afterwards.
A table added later takes the next free offset and disturbs nothing.

**The tables.**

| Table | Rows | Purpose | Knowledge-base document it must not contradict |
|---|---|---|---|
| `loan_products` | 5 | The catalogue every amount, rate and tenure is drawn inside. Seeded from `config.CATEGORY_BANDS`, not duplicated from it. | `kb-01`, `kb-07` |
| `customers` | about 66 | Identity, contact, employment, income, credit score, KYC status, residency. Carries fabricated PAN, Aadhaar and account numbers; the PAN and the Aadhaar follow the real structures (D-24, D-25). | `kb-10`, `kb-12` |
| `loan_applications` | 100 | The six brief fields plus `created_at`, `updated_at`, `customer_id`, `product_code`, `tenure_months`, `interest_rate_pct`. | `kb-01` |
| `application_events` | 2 to 5 per application | The status audit trail, each transition carrying `occurred_at` under D-26 and the working-day rule of D-29. | `kb-06` |
| `repayments` | instalments for disbursed loans | The EMI schedule, with the principal and interest split. `due_at` is exempt from D-29's working-day shift. | `kb-02`, `kb-08`, `kb-17` |
| `support_tickets` | about 40 | Channel, category, status, linked application, `opened_at`. | `kb-05` |
| `kyc_documents` | 2 to 4 per customer | Identity and address proofs with verification state and `submitted_at`. Verification carries no timestamp of its own; that is a known gap, not an oversight. | `kb-04`, `kb-16` |

Every calendar column in those tables is an ISO 8601 string in IST, and no `_days_ago` integer survives anywhere below `loan_applications` (D-26).

**The consistency rule, which is the price of seven tables.**
Every generated row must satisfy the knowledge-base document that describes it.
Three are load-bearing and are asserted by tests rather than trusted:

1. Every `repayments` row reproduces `kb-02`'s EMI formula, `EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)`, to the rupee.
2. Every `kyc_documents.doc_type` is a document `kb-04` actually lists.
3. Every `support_tickets.channel` is a channel `kb-05` actually names.

If a generated row cannot satisfy its document, that is a defect in the generator or the document, and it is raised rather than rounded off.

**Public surface.**

- `db/schema.py` - the seven `CREATE TABLE` statements and nothing else.
- `db/generate.py` - one function per table, each taking its own seeded stream.
- `db/build.py` - `build_database(path) -> dict[str, int]`, and `write_manifest(path)`.
- `db/query.py` - the read helpers Part 2 consumes, including `customer_context(record_id) -> dict` returning only the non-PII fields named in D-21.
- `scripts/check_database.py` - the grader's one-command check, per D-22.

**What the manifest holds.**
The seed and every sub-stream offset, the full DDL for all seven tables, the row count per table, the SHA-256 of a canonically ordered dump, and the generation date.
It is written by the same script that builds the database, so the two cannot drift, and a test asserts a freshly built database still matches it.

## 6. Part 1 Task 2 - knowledge base

18 documents in `knowledge_base/`, one markdown file each, 10 to 15 sentences targeting 12, written from scratch for this brief.

Each file carries YAML front matter that `rag/kb.py` parses:

```
---
doc_id: kb-01-loan-eligibility
title: Loan eligibility criteria by loan type
topic: loan_eligibility
required: true
---
```

`doc_id` is the unit that Precision@3 and Recall@3 score against, and the unit the support rule in D-07 groups by.

### 6.1 The twelve required topics

Documents 1 to 12 cover, one each, the topics the brief names: loan eligibility criteria by loan type, EMI calculation rules, credit-card fee structure, KYC document requirements, fraud-dispute resolution process, account-closure process, interest-rate slabs, prepayment-penalty rules, minimum-balance requirements, credit-score impact factors, joint-account rules, and NRI-account eligibility.
Each carries `required: true`.

### 6.2 The six confusable neighbours

Documents 13 to 18 exist to make retrieval hard in the way real knowledge bases are hard.
Each is a legitimate document a support team would write, and each sits close enough to a required topic that a plausible query can match both.

| Document | Neighbour of | Why it is confusable |
|---|---|---|
| Personal Loan eligibility in detail | Loan eligibility by type | A query about personal-loan income requirements legitimately matches both. |
| Home Loan eligibility and loan-to-value rules | Loan eligibility by type, interest-rate slabs | Carries both eligibility and rate content. |
| Credit-card late-payment and overlimit charges | Credit-card fee structure | Both are card charges; only one is the general fee schedule. |
| KYC re-verification and periodic update rules | KYC document requirements | Same document list, different trigger. |
| Foreclosure versus part-prepayment | Prepayment-penalty rules | The penalty document states the charge; this one states which action incurs it. |
| NRE versus NRO account operation rules | NRI-account eligibility | Eligibility and operation are different questions about the same account family. |

These carry `required: false`.
They are not padding: without them every query has one relevant document and D-04's reasoning applies.

## 7. Part 1 Task 3 - chunking and indexing

### 7.1 Strategies

`chunk_fixed(text, size=400, overlap=80) -> list[str]` walks the character stream in steps of `size - overlap`.
`chunk_sentences(text) -> list[str]` splits on terminal punctuation followed by whitespace, with a guard list for common abbreviations so "Rs. 5,000" does not become a boundary.
Both take the document body only, never the front matter.

Parameters are D-14.
They are starting values and get re-tuned once real retrieval numbers exist; `README.md` records both the starting and the final values.

### 7.2 Collections

Two ChromaDB collections in one persistent client at `chroma/`, both created with `{"hnsw:space": "cosine"}`:

- `kb_fixed_400_80`
- `kb_sentences`

Embeddings come from `all-MiniLM-L6-v2` via `sentence-transformers`, with `normalize_embeddings=True`, so Chroma's cosine distance converts to similarity as `1 - distance`.

Each chunk carries metadata: `doc_id`, `title`, `topic`, `required`, `chunk_index`, `strategy`.
`doc_id` is what makes the chunk-to-parent mapping in Task 5 and the support rule in D-07 possible.

### 7.3 Index-time assertion

Indexing asserts that every document produces at least 2 chunks in **both** collections.
Without this a short document becomes permanently unanswerable under D-07's support rule and nothing reports it.
The assertion is also test 5 in section 11.

## 8. Part 1 Task 4 - grounded generation

### 8.1 Retrieval

`retrieve(query, strategy, k=3) -> list[Hit]`, where `Hit` carries `text`, `doc_id`, `title`, `chunk_index` and `similarity`.

### 8.2 Threshold calibration

The brief forbids a preset threshold and requires measurement.
`eval/calibration.py` holds 12 in-scope probe queries, one per required topic, and 5 deliberately out-of-scope probes.
This oversamples the brief's floors of 3 and 2, which makes the gap between the two clusters far more convincing.

The procedure, run once and recorded:

1. Measure top-1 cosine similarity for all 17 probes against both collections.
2. Print every measured value, clustered.
3. Set `T` at the midpoint between the minimum in-scope value and the maximum out-of-scope value.
4. Record the measured values, the gap, and the chosen `T` in `README.md`.

If the clusters overlap, `T` cannot be set honestly and the chunking parameters are wrong.
That is test 6 in section 11, not a footnote.

### 8.3 The answer decision

An answer is produced only when both conditions hold, per D-07 and D-08:

1. `top1.similarity >= T`
2. At least 2 of the top 3 chunks share the same `doc_id`

Otherwise the fallback fires.
The demonstration covers at least 5 in-scope queries plus 1 deliberately out-of-scope query that must trigger the fallback.

### 8.4 Generation under MOCK_LLM

`llm.py` exposes one function, `generate(system: str, user: str) -> str`, and selects a provider from the `LLM_PROVIDER` environment variable, defaulting to `mock`.
This is the only interface in V1 written for two implementations, and it exists because the brief mandates both a mock mode and an optional real provider.

The mock provider is deterministic template synthesis, per D-06:

1. Classify the question shape from the query by keyword and pattern: definition, eligibility, amount or rate, process or steps, document list, duration.
2. Select the response template for that shape.
3. Fill its slots from sentences in the retrieved chunks, never from anything outside them.
4. Append a `[doc_id]` citation for each source used.

Template selection can pick wrong, which is deliberate.
It means Part 3's context relevance and answer relevance vary across the 15 queries instead of being 1.0 by construction, so the RAG triad measures something.

## 9. Part 1 Task 5 - evaluation

### 9.1 Queries and gold labels

`eval/queries.py` holds the same 12 queries used in Task 4, each with a hand-authored set of relevant `doc_id` values.
Per D-11 the sizes are mixed: 4 queries with 1 relevant document, 5 with 2, and 3 with 3.

Labels are authored and committed **before** any retrieval is run.
That ordering is stated in `README.md`, because a label set written after seeing the results is not a label set.

### 9.2 Metric definitions

Chunks are mapped back to parent `doc_id` and deduplicated before scoring, as the brief requires.
Let `R` be the set of distinct parent documents among the top 3 chunks and `G` the gold set.

- `Precision@3 = |R and G| / |R|`
- `Recall@3    = |R and G| / |G|`

**The denominator choice is a known asymmetry and is reported, not hidden.**
Sentence-based chunking often returns three chunks from one parent, giving `|R| = 1` and a precision of either 0 or 1.
Fixed-size chunking spreads across parents more, giving `|R| = 3` and finer-grained precision.
Dividing by `|R|` rather than by 3 avoids penalising a collection for agreeing with itself, but it does favour the collection that concentrates.
Therefore `|R|` is printed per query alongside the two metrics, and the Task 5 recommendation must address the asymmetry explicitly rather than reading the averages off the bottom of the table.

### 9.3 Output

Per-query arithmetic is printed as fractions, not decimals, for both collections, followed by the averages and a 2 to 3 sentence recommendation citing both sets of numbers.

## 10. Interfaces Parts 2 to 4 consume

These are declared now so Part 1 is built against a fixed contract.
Nothing here is implemented in Part 1 beyond what Part 1 already needs.

| Consumer | Interface | Provided by |
|---|---|---|
| Part 2 Task 6, lookup tool | `get_application(record_id) -> dict \| None` | `dataset.py` |
| Part 2 Task 6, customer context | `customer_context(record_id) -> dict` with `customer_id`, `full_name`, `credit_score`, `open_loan_count`, and no PII, per D-21 | `db/query.py` |
| Part 2 Task 7, RAG tool | `answer(query, strategy) -> GroundedAnswer` with `.text`, `.citations`, `.supported`, `.top1_similarity` | `rag/generate.py` |
| Part 2 Task 10, output guardrail | `.supported` and `.top1_similarity` on `GroundedAnswer` | `rag/generate.py` |
| Part 3 Task 13, LLM judge | `generate(system, user) -> str` under `MOCK_LLM` | `llm.py` |
| Part 3 Task 12, log masking | The same masking function the input guardrail uses | Part 2, `agent/guardrails.py` |
| Part 4 Task 14, MCP tool | `get_application` with a proper docstring | `dataset.py` |
| All parts | `T`, chunk parameters, paths, `MOCK_LLM` flag | `config.py` |

The recommended collection from Task 5 becomes Part 2's fixed RAG input.
Which one that is cannot be decided here, because it depends on numbers that do not exist until Part 1 runs.

## 11. Testing

Per D-13, one test per acceptance criterion, and no more.

| # | Test | Acceptance criterion it restates |
|---|---|---|
| 1 | The generator reproduces `data/loan_applications.json` byte for byte, checked by SHA-256 | D-09, and the brief's reproducibility requirement |
| 2 | Every category has at least 3 records | "category counts" |
| 3 | Every status appears at least once | "status coverage" |
| 4 | The fraud rate is inside 10 to 30 percent | "flagged_for_fraud_review percentage band" |
| 5 | Every knowledge-base document yields at least 2 chunks in both collections | Section 7.3, and the support rule's viability |
| 6 | The minimum in-scope top-1 similarity exceeds the maximum out-of-scope top-1 similarity | Section 8.2, and the brief's ban on an untested preset threshold |
| 7 | Every one of the brief's six fields is byte-identical to the pre-database snapshot, value for value across all 100 records | D-18, D-19 and D-31. Adding the store changed no existing record, and adding the time axis changed no existing *value*; it appended two fields |
| 13 | Every timestamp parses as ISO 8601 with an offset, and none is later than the anchor | D-26 and D-27, the derivation is total and nothing is dated in the future |
| 14 | `created_at` equals the anchor minus `days_since_created`, for all 100 records | D-26, the derivation that everything else rests on |
| 15 | No bank-side event falls on a Saturday or a Sunday, and every timestamp falls inside business hours | D-29, the rule stated as something a grader can watch fail |
| 16 | `updated_at` equals the instant of the application's most recent event, and `paid` equals `due_at <= anchor` | D-30, the two stored facts agree with the trail they were written from |
| 8 | Every foreign key in the six child tables resolves, with no orphans | Section 5.4, the joins are real |
| 9 | Every `loan_applications` amount, rate and tenure sits inside its `loan_products` row | Section 5.4, the data obeys its own catalogue |
| 10 | Every `repayments` row reproduces `kb-02`'s EMI formula to the rupee | Section 5.4 consistency rule 1 |
| 11 | Every `kyc_documents.doc_type` and `support_tickets.channel` appears in `kb-04` and `kb-05` | Section 5.4 consistency rules 2 and 3 |
| 12 | A freshly built database matches the committed `data/database-manifest.md` | D-17, the drift guarantee D-09 established for the JSON snapshot |

Precision@3 and Recall@3 are deliberately not pinned.
They move legitimately when chunk parameters are tuned, and a test that fights tuning is a test that gets deleted.

## 12. Evidence

Per D-12.

`scripts/run_part1.py` runs every Part 1 task in order and writes:

- `transcripts/part1-dataset.txt` - the validation report
- `transcripts/part1-calibration.txt` - all 17 measured similarities, the gap, the chosen `T`
- `transcripts/part1-generation.txt` - 5 or more in-scope queries answered, plus the out-of-scope fallback
- `transcripts/part1-evaluation.txt` - per-query arithmetic for both collections and the averages

`README.md` states the completed track, the exact dataset-design choices needed to reproduce the dataset, the measured calibration values and chosen threshold, the Task 5 recommendation, and a link to each transcript.
Number tables in `README.md` are generated by the same run that writes the transcripts, never typed by hand.

## 13. V2 roadmap

None of this is built in V1.
Each item names the V1 module it replaces and what makes the swap cheap.

| Upgrade | Replaces | Why the swap is cheap from V1 |
|---|---|---|
| Hybrid retrieval: BM25 plus dense, reciprocal rank fusion | `rag/retrieve.py` | `retrieve` already returns a scored `Hit` list; callers never see the store. |
| Cross-encoder reranking of the top 20 | `rag/retrieve.py` | Same seam. Reranking slots between retrieval and the support rule. |
| Real language model for generation and judging | `llm.py` | The provider switch exists in V1 because the brief requires it. |
| Semantic and late chunking | `rag/chunking.py` | Chunking is already a pure function from text to a list of strings. |
| pgvector or Qdrant instead of local ChromaDB | `rag/index.py` | Chunk metadata is already store-agnostic. |
| Learned threshold and calibration on held-out data | `config.py` and section 8.2 | `T` is a single named constant with a documented derivation. |
| Two-signal fallback: support rule plus null-query margin | Section 8.3 | The rule is one function; this is the option that lost in D-07. |
| Docker, CI, and a published image | New | Nothing in V1 assumes a local path outside `config.py`. |
| OpenTelemetry tracing and a metrics endpoint | Part 3's structured logging | Trace IDs already exist in V1's log lines. |
| Streaming responses over server-sent events | Part 3's FastAPI layer | Requires a real model, so it follows the `llm.py` swap. |

## 14. Open items

These are recorded rather than resolved, and each is a task.

1. ~~Four of the five loan-amount bands are unverified.~~ Closed 2026-09-10.
   Auto, Education and Business are now sourced to published SBI and HDFC limits, and Business was lowered from 75 lakh to 50 lakh to match.
   Home stays a deliberate narrowing of a 50,000-to-50-crore lender range, recorded as a modelling choice rather than evidence.
2. ~~The chunk-count model behind D-05 and D-14 assumes 134 characters per sentence.~~ Closed 2026-09-11.
   Measured across the 18 written documents: 26,656 body characters over 215 sentences, 124 characters per sentence.
   That is 7.5 percent below the assumption, every document yields 4 to 6 fixed chunks and 11 to 13 sentence chunks, and the parameters stand unchanged at 400/80.
3. ~~The seed is not yet chosen.~~ Closed 2026-09-11.
   Seed 1, the first in a scan from 1 that met every structural invariant: 16.0 percent fraud rate, smallest category 11 records, all five statuses present.
   No record was hand-edited. Seed 1 passing on the first try reflects the weights placing the expected rate at 17.9 percent near the middle of the band, not luck.
4. ~~Which collection Part 2 consumes depends on Task 5's measured numbers.~~ Closed 2026-09-11.
   `kb_sentences`. Precision@3 0.8750 against 0.7917 and Recall@3 0.6528 against 0.5972, won while carrying the higher mean |R| of 1.33 against 1.25, so the asymmetry documented in section 9.2 runs against the winner rather than for it.

A fifth item opened during implementation and is recorded here rather than fixed.

5. The support rule in D-07 has a false-refusal mode on broad questions that span documents.
   "What documents are needed for KYC?" scores 0.5124 top-1, far above T, but its top three sentence chunks land on three different parents, so no two agree and the system refuses.
   None of the 12 labelled evaluation queries is affected, and `tests/test_generate.py` pins the behaviour so it cannot regress unnoticed.
   The two-signal fallback in section 13 is the V2 upgrade that removes it.

6. ~~The relational store in section 5.4 is specified but not built.~~ Closed 2026-09-11.
   Built: 1,076 rows over seven tables, `loan_products` 5, `customers` 66, `loan_applications` 100, `application_events` 393, `repayments` 276, `support_tickets` 40, `kyc_documents` 196.
   `data/loan_applications.json` is byte-identical to its pre-database state and a test asserts it on every run.
   The knowledge-base agreement tests found a real drift on their first run - the generator had written "Voter ID Card" and "NREGA Job Card" where `kb-04` says "a voter identity card" and "a job card issued under NREGA" - which is what those tests exist for.
   Largest EMI deviation from `kb-02`'s formula across all 276 instalments: 0.0047 rupees.
