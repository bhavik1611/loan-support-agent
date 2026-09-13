# The relational store

The seven-table SQLite database behind `dataset.py`, and the invariants that keep it
from contradicting the knowledge base.

This is Part 1 detail, moved out of `README.md` so that file stays short enough to read.
The decisions behind it, and the alternatives each one beat, are D-16 to D-21 and
section 5.4 of [the design spec](superpowers/specs/2026-09-10-loan-support-agent-design.md).
The generated evidence is [`data/database-manifest.md`](../data/database-manifest.md).

`dataset.py` also feeds a seven-table SQLite database at `data/meridian_bank.db`, built by `db/build.py` from the same seed.

**Why it exists.**
The brief says each record *needs* `record_id`, `category`, `status`, `loan_amount_inr`, `days_since_created` and `flagged_for_fraud_review`.
That is a floor, not a ceiling.
Stopping there also left three knowledge-base documents - EMI calculation, prepayment penalties and foreclosure - describing behaviour that no six-field record could ever exercise.

```bash
.venv/bin/python -m db.build            # build it, about a second
.venv/bin/python scripts/check_database.py   # the grader check, exits non-zero on failure
```

| Table | Rows | What it holds |
|---|---|---|
| `loan_products` | 5 | The catalogue every amount, rate and tenure is drawn inside. |
| `customers` | 66 | Identity, employment, income, credit score, KYC status, residency. |
| `loan_applications` | 100 | The six brief fields plus `created_at`, `updated_at`, customer, product, tenure and rate. |
| `application_events` | 393 | The status audit trail, each transition carrying an `occurred_at` instant. |
| `repayments` | 276 | The first 12 instalments of each disbursed loan's schedule. |
| `support_tickets` | 40 | Channel, category, status, linked application. |
| `kyc_documents` | 196 | Identity and address proofs with verification state. |
| **total** | **1,076** | |

**Nothing that existed before this landed moved.**
Each table draws from its own stream, `random.Random(SEED + offset)`, and `loan_applications` keeps offset 0, which is literally the original `Random(SEED)`.
So every value of the brief's six fields in `data/loan_applications.json` is unchanged, by the store and by the time axis alike, and `tests/test_database.py::test_no_brief_field_value_has_ever_moved` proves it on every run by comparing the committed file against the raw stream-0 draw.
The time axis appended `created_at` and `updated_at` and moved no existing cell; its hours and minutes come from a stream of their own at the next free offset.
A second test asserts the enricher cannot draw from stream 0 at all, because that is the failure mode that would silently rewrite every `record_id` the transcripts quote.

**The database is generated, not committed.**
`data/meridian_bank.db` is gitignored because it is regenerable and git cannot diff a binary.
What *is* committed is [`data/database-manifest.md`](../data/database-manifest.md): the seed, every stream offset, the full DDL, row counts, and a SHA-256 over a canonically ordered dump of every row.
The hash covers the rows rather than the file, because a SQLite file can differ byte for byte while holding identical data.
A test rebuilds and compares.

**The database is not allowed to contradict the knowledge base.**
Seven tables are seven chances to do exactly that, so three invariants are tested rather than trusted:

| Test | Asserts |
|---|---|
| `test_every_repayment_reproduces_the_kb02_formula` | Every one of the 276 instalments matches `kb-02`'s EMI formula. Largest measured deviation: **0.0047 rupees**. It recomputes independently rather than calling the generator's own helper, which would only prove the helper equals itself. |
| `test_every_kyc_doc_type_appears_in_the_knowledge_base` | Every `doc_type` is a document `kb-04` or `kb-12` actually lists. |
| `test_every_ticket_channel_appears_in_kb05` | Every ticket channel is one `kb-05` actually names. |

That first vocabulary check caught a real drift the moment it ran: the generator had written "Voter ID Card" and "NREGA Job Card" where `kb-04` says "a voter identity card" and "a job card issued under NREGA".
The generator now uses the document's own wording.

**Two places the data is deliberately thin, rather than fabricated.**
Every loan is 0 to 30 days old, so no disbursed loan has reached its first instalment due date and no repayment is marked paid.
One Approved application aged zero days carries a single compressed event instead of a three-stage status history, because the days to spread it over have not happened yet.
Both are honest limits of a 30-day window, not oversights.

**PII.**
`customers` carries fabricated PAN, Aadhaar and account numbers in the fixed formats Part 2 Task 10 will mask.
The PAN follows the Income Tax Department's structure, `AAAAA9999A`: three series letters, the holder-type code `P` for an individual, the surname initial, a 0001-9999 serial and a check letter derived from the first nine characters.
The department does not publish the real check-character formula, so that last letter is a documented stand-in, not a claim to be verifiable against anything.
The Aadhaar numbers are twelve digits never beginning 0 or 1, with the twelfth a real Verhoeff check digit over the first eleven - UIDAI's published algorithm, so every fabricated number passes a genuine validator and a mistyped or transposed digit fails it.
They are masked in every line the check script prints, with no flag to unmask them.
They are never returned by `db/query.py::customer_context`, which yields only `customer_id`, `full_name`, `credit_score` and `open_loan_count` - a contract a test enforces, because the brief puts masking on the input side and serving PII here would need an output masker the brief never asked for.

**Rates and tenures.**
Interest-rate bands are read out of `kb-07`, not retyped.
Tenures for Personal (12 to 60 months) and Home (30 years) come from `kb-13` and `kb-14`.
The Auto, Education and Business tenures are stated nowhere in the knowledge base and are recorded in `config.py` as modelling choices rather than presented as sourced facts.

**The time axis.**
Every date in this repository is derived from `days_since_created` against one frozen anchor, `config.AS_OF`, which is 2026-09-30 at close of business IST.
Nothing reads the wall clock, so the same seed still produces the same bytes.

Anchoring those offsets to a calendar is what made them checkable, and the first thing it caught was the dataset's own: 99 of 393 application events had been landing on a Saturday or a Sunday, while `kb-01`, `kb-05` and `kb-06` write their turnaround promises in working days.
A status transition the bank makes now moves forward to the next working day.
The application's own `Submitted` event does not, because `db/generate.py` calls it an online submission and an online form takes a Sunday one.
An EMI due date does not either: it is a contractual date rather than an action, and shifting it would stop the monthly spacing being monthly, so `repayments.due_at` is the one column allowed past the anchor.

Business hours of 09:30 to 18:00 are stated nowhere in the knowledge base and are a modelling choice, recorded here the same way the Home Loan band and the three tenures are.
Indian bank holidays are deliberately out of scope: weekends are computable from a date, a holiday calendar is not, and no document in this repository carries one.
