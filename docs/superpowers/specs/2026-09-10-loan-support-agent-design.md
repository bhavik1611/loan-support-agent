# Loan Support Agent - design specification

Status: approved design for Parts 1 and 2, declared interfaces for Parts 3 and 4, roadmap only for V2.
Written 2026-09-10 after two grilling rounds.
Amended 2026-09-12 after three more, adding the time axis in D-26 to D-31.
Amended again 2026-09-12, adding Part 2 in sections 10 to 14 and D-33 to D-45, approved off the review artifact of that date.
Authority: `reference/problem-statement.md` is the brief; where this document and the brief disagree, the brief wins and this document is wrong.
Section numbering changed in that amendment so the parts read in order; the two dated plans under `docs/superpowers/plans/` cite the numbering as it stood when they were written, and are left alone because a dated plan is a record.

## 1. Scope

This repository answers the **Cred (Banking & FinTech)** capstone track with a loan-operations support agent.
V1 is the brief-compliant system: it answers loan-policy questions from a knowledge base written here, looks up loan applications from a dataset generated here, and is orchestrated, deployed, evaluated and hardened across four parts.
V2 is a later, portfolio-grade rebuild that is described in section 19 and implemented nowhere in this repository.

This document specifies Parts 1 and 2 in full.
Parts 3 and 4 appear only as the interfaces they will consume, so that the parts below them are built against a known contract rather than refactored into one later.
Part 1 is implemented; Part 2 is approved and not yet implemented.

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

### 3.1 Foundations

Three choices everything else rests on.

| # | Decision | Chosen | Rejected, and why it lost |
|---|---|---|---|
| D-01 | Scale beyond the brief's floors | 100 records, 18 KB docs, 12 eval queries | Brief minimums of 40/12/5 lost because Precision@3 on 5 queries moves in steps of 0.067 and the two chunking strategies would likely tie, leaving the Task 5 recommendation with nothing to cite. |
| D-02 | Repository layout | `dataset.py` at repository root as real code, packages for everything multi-module | A flat root lost because Part 4 would leave roughly 20 unstructured modules. A `src/` package lost because the brief names `dataset.py` in its submission list, and satisfying that from `src/` needs a re-export shim, which is wrapper indirection for its own sake. |
| D-03 | How V2 appears | A roadmap section in this document, zero V2 code in the repository | Building the seams now lost because a `Retriever` protocol with one implementation is a layer serving code that does not exist. A V1-only spec lost because some V1 choices are cheap now and expensive to unwind later. |

### 3.2 Part 1, the knowledge base and RAG core

Tasks 2 to 5. Document shape, chunking, the answer decision, and what is measured.

| # | Decision | Chosen | Rejected, and why it lost |
|---|---|---|---|
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

### 3.3 Part 1, the dataset and its relational store

Task 1, and the seven-table store D-15 added beyond the brief's six fields.

| # | Decision | Chosen | Rejected, and why it lost |
|---|---|---|---|
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

### 3.4 Part 1, the time axis and the document format

Added 2026-09-12. Every calendar value in the repository derives from one frozen anchor.

| # | Decision | Chosen | Rejected, and why it lost |
|---|---|---|---|
| D-26 | How time is represented | `days_since_created` stays the drawn integer; every calendar value in the repository is derived from it against a fixed anchor | Drawing a date and deriving the integer lost because it changes the draw shape inside stream 0, which moves every `record_id` the transcripts quote. Storing only the integer lost because relative offsets are unfalsifiable: nothing can contradict "13 days ago", and anchoring the same data exposed a defect within minutes (D-29). The four child tables keep no integer at all, because nothing outside this repository asks for one; `days_since_created` alone survives because the brief names it and Part 2 Task 6 scores on it, which is the only honest reason to hold two representations of one fact. |
| D-27 | The anchor | `config.AS_OF_DATETIME = 2026-09-30T18:00:00+05:30`, frozen, in IST, not announced in the README | A rolling anchor lost outright: it breaks the determinism ground rule and the manifest hash would change daily. Midnight lost because a day-0 event exists in the committed data and would land in the future. UTC lost because 09:30 IST business hours render as 04:00Z and stop looking like business hours to a reader; a naive timestamp lost because it invites a guess, and the guess is usually UTC. Overrode recommendation on announcing it: the recommendation was to state the as-of date in `README.md` and the manifest so the data's ageing is explicit rather than discovered, and Bhavik chose to keep the constant silent. |
| D-28 | Where the clock's randomness comes from | A new seeded stream, `config.stream("clock")`, at the first free offset | Drawing hours and minutes from each table's own stream lost because it reshuffles every row of all four tables, moving days, amounts and notes for a cosmetic gain. A fixed derived time with no randomness lost because every event would land at the same clock time, which is visibly synthetic. This is the case D-18's offset scheme was built for, and it is the reason a full timestamp costs nothing beyond the manifest rebuild D-26 already requires. |
| D-29 | Working days and business hours | Bank-side transitions move forward to the next working day; `Submitted` never moves; every timestamp falls between 09:30 and 18:00 | Anchoring the existing offsets showed 99 of 393 events on a Saturday or Sunday, against five knowledge-base sentences written in working days (`kb-01`, `kb-05`, `kb-06`), and the knowledge base is the authority. Shifting backwards lost because it says the bank acted before it could have; nearest lost because it mixes both stories with no per-row reason; forward is a monotone map, which is why it produces no order inversions. `Submitted` is exempt because the generator's own note calls it an online submission, and an online form takes a Sunday one; the exemption is also load-bearing, because `Submitted` is `created_at` and moving it would break D-26's derivation. EMI due dates are exempt for the opposite reason: a due date is a contractual date, not an action, and shifting it would distort the amortisation schedule. Bank holidays lost because no document in this repository carries a holiday calendar, and inventing one contradicts the rule that the knowledge base is the authority. Overrode recommendation on the hours: the recommendation was two windows, 24 hours for customer-side rows and 09:30 to 18:00 for the bank, and Bhavik chose one window for one rule. |
| D-30 | Stored or derived, for facts the trail already carries | `updated_at` and `paid` are stored, each pinned by a test asserting it equals what the trail says | Deriving them in `db/query.py` lost because it costs a correlated subquery on every read and hides the column Part 2 wants to quote; storing without a test lost because the two copies can then disagree and nothing notices. The uniqueness invariant on an application's events moves from distinct days to distinct instants, which is what the timestamp in D-26 was for: two events on one Monday morning and afternoon are a working day, not a defect. |
| D-31 | Whether the snapshot carries the new fields | `data/loan_applications.json` gains `created_at` and `updated_at`, appended after the brief's six in the brief's order | Overrode recommendation. Keeping the projection at six fields lost, although it was recommended: it would have left the committed snapshot byte-identical and the hash test untouched. Bhavik chose to carry the timestamps through to the projection, so acceptance criterion 7 is restated rather than deleted, and the fields are appended rather than interleaved so that `PROJECTED_FIELDS`' promise to hold "the brief's six fields, in the brief's order" stays literally true. |
| D-32 | What a knowledge-base document is | Plain `.txt` prose, with `title`, `topic` and `required` in a sidecar `catalogue.json` | Markdown with YAML front matter lost because it makes a document carry claims about itself that a real policy document does not: an ingestion pipeline reads files a business already has, and those files are prose. Renaming the extension while keeping the front matter lost too, and lost for the worse reason, because it relabels the format without changing anything. Deriving the metadata instead of storing it lost because `required` records which of the brief's topics a document covers, which is a fact about the corpus rather than about any one file, and `rag/index.py` writes `topic` and `required` into every chunk's metadata, so the retrieval layer needs them as much as evaluation does. The catalogue is the one place they live, and the loader fails rather than defaults if it and the directory disagree. Bodies were migrated programmatically and verified byte-identical, so no measured number moved. |

### 3.5 Part 2, the agent

Tasks 6 to 10. Approved 2026-09-12 off the review artifact of that date.

| # | Decision | Chosen | Rejected, and why it lost |
|---|---|---|---|
| D-33 | The escalation score's shape | `0.45*fraud + 0.55*stale`, where `stale` is `min(days/21, 1)` at full rate while the application is open, half rate once `Disbursed`, and zero once `Rejected` | `0.5*fraud + 0.5*(days/30)` lost on measurement: at its own 85th percentile it selects exactly the 16 fraud-flagged records and no others, so the recency term changes nothing and the score is the bare boolean OR the brief forbids. The same formula with a 21-day saturation but no status awareness lost for the same reason, identically. Zeroing staleness for **both** terminal statuses lost because it leaves 5 `Disbursed` applications carrying a fraud flag permanently below the line, which is money already out of the bank with nobody looking at it. |
| D-34 | The escalation threshold | 0.50, which is the 80th percentile of the score over the committed 100 records | A percentile computed at run time lost because it moves when the dataset does and the transcripts would stop being reproducible. A round 0.5 chosen first and justified afterwards lost because that is the untested preset the brief bans in Task 4 and the same objection applies here. |
| D-35 | How a fixed-format PII field is recognised | Format alone decides that a value is masked; the Verhoeff check digit only decides whether the label reads `AADHAAR` or `ACCOUNT` | Recognising Aadhaar by its 12-digit format alone lost because 9 of the 66 generated account numbers are also 12 digits and would have reached Part 3's log in the clear. Gating the mask on a valid check digit lost because a mask that can fail open is not a mask; this way the label can be wrong, never the redaction. |
| D-36 | The response envelope's shape | One Pydantic model with typed nullable `policy`, `lookup` and `guardrails` blocks, exported to a committed JSON Schema and validated with `jsonschema` as well as by construction | A discriminated union per route lost because Part 3's FastAPI response model would become a union of four and the grader would have four schemas to check. A flat always-present schema lost because it advertises fields a given route can never fill. Trusting Pydantic construction alone lost because only re-validating the serialised dict proves the **exported** schema is the one being met. |
| D-37 | What the persisted memory holds | The turn log the brief asks for, plus a resolved-entity slot carrying `last_record_id` | A bare turn log lost because it can only demonstrate that history is present, and the brief also asks for a transcript showing state correctly absent. With an entity slot the same second-turn question routes to `lookup` on a warm thread and to `clarify` on a fresh one, which is a difference a reader can see. A rolling MOCK_LLM summary lost because the summary would be template output and would demonstrate plumbing rather than memory. |
| D-38 | Where the trace id comes from | `sha256(f"{thread_id}|{turn_index}|{masked_query}")` truncated to 16 hex characters | `uuid4()` lost outright: two runs of `scripts/run_part2.py` would produce different transcript bytes and the determinism ground rule in section 2 would stop holding. A monotonic counter lost because it is not stable when one transcript is regenerated on its own. The input is the **masked** query, so no raw PII reaches the hash. |
| D-39 | What the guardrails are | Three input PII rules, four named injection rules, and two output rules | A single unnamed injection regex lost because a refusal that cannot say which rule fired is not demonstrable, and the brief asks for each guardrail to be shown firing. `delimiter_injection` exists because `llm.py` parses its own prompt back with `^\[([a-z0-9\-]+)\]` and a forged `[kb-07]` line in a query would otherwise be read as retrieved context. `phantom_citation` goes beyond the brief and catches a cited `doc_id` that was never retrieved. |
| D-40 | Where Part 2's evidence lands | Seven transcripts written by `scripts/run_part2.py`, never typed by hand | Same rule as D-12, restated because it is the rule most easily lost when a second script appears. A single combined Part 2 transcript lost because a grader checking one acceptance criterion would have to read all of them. |
| D-41 | How the lookup route speaks | A fixed non-LLM template over the record's own fields, tagged `source: "record"` in the envelope | Sending the record through `llm.generate` lost because that is generation with no retrieval behind it, so the groundedness guardrail has nothing to check and the output would sit in the same envelope as a grounded answer while meaning something different. Returning the structured block with no sentence lost because a support agent reads the sentence. |
| D-42 | How much customer context is spoken aloud | Name and open-loan count in the answer text; credit score only in the structured block, never in prose | Saying the credit score in the answer text lost because a support agent reading a score aloud to a member is a different act from seeing it on screen, and the envelope already carries it for any caller that needs it. Omitting it entirely lost because D-21 put it in `customer_context` deliberately and Part 3 consumes that block. |
| D-43 | How far past the brief Part 2 goes | Nine nodes, two conditional edges and four route outcomes, against the brief's floor of four nodes and one conditional edge | A brief-tight graph of four nodes lost because the conditional edge would then be a binary with nothing to demonstrate beyond itself. A supervisor delegating to specialist sub-agents lost because under `MOCK_LLM` a supervisor is template matching delegating to template matching, and the brief names multi-agent orchestration in its preamble without requiring it in any Part 2 task. Section 18 lists what this deliberately does not build. |
| D-44 | How intent is routed | A record-id regex decides outright; otherwise the query is embedded with the model already loaded and scored against intent exemplar centroids | A scored keyword table lost because it is a second vocabulary to keep in sync with the knowledge base, and it is brittle under paraphrase. A `MOCK_LLM` classifier node lost because under mock the classifier is template matching anyway, so it adds a layer without adding signal. The margin that decides whether the router may commit is measured, not preset, for the reason the brief gives in Task 4. |
| D-45 | What happens when the router cannot commit | A `clarify` node returning one specific question, capped at one per thread | Defaulting to the RAG route lost because the router would silently guess and the groundedness fallback would absorb the mistake, which hides it. Treating low margin as the `both` route lost because it wastes a lookup on queries carrying no record id and blurs what `both` means. Uncapped clarification lost because two ambiguous turns in a row would loop. |

## 4. Repository layout

```
loan-support-agent/
  dataset.py                  Task 1. Seeded generator, LOAN_APPLICATIONS, lookup, validation report.
  config.py                   Paths, chunk parameters, calibrated threshold, environment flags.
  llm.py                      Provider seam. MOCK_LLM default, real provider behind LLM_PROVIDER.
  knowledge_base/             Task 2. 18 plain-text documents plus catalogue.json.
  rag/
    kb.py                     Load and parse knowledge_base/ into Document objects.
    chunking.py               Task 3. chunk_fixed and chunk_sentences.
    index.py                  Task 3. Embed and index into two ChromaDB collections.
    retrieve.py               Task 4. Top-k retrieval, cosine similarity, support rule.
    generate.py               Task 4. Grounded generation and the fallback.
    evaluate.py               Task 5. Precision@3 and Recall@3 for both collections.
  agent/
    state.py                  Task 7. AgentState, the graph's only channel schema.
    escalation.py             Task 6. The designed score and its threshold, pure functions of one record.
    tools.py                  Tasks 6 and 7. check_loan_application_status, and the RAG tool wrapper.
    intents.py                Task 7. Intent exemplars and the embedding router.
    guardrails.py             Task 10. PII masking, injection rules, groundedness check.
    memory.py                 Task 8. JSON thread store and the entity slot.
    schema.py                 Task 9. The response envelope and its exported JSON Schema.
    response.schema.json      Task 9. Committed, and what compose validates against.
    nodes.py                  Task 7. The nine node functions.
    graph.py                  Task 7. Wiring, compile, and the ask() entry point.
  eval/
    queries.py                12 evaluation queries with graded gold labels.
    calibration.py            In-scope and out-of-scope probe queries for threshold calibration.
    routing.py                Part 2. Labelled probe queries for the ROUTE_MARGIN calibration.
  scripts/
    run_part1.py              Runs every Part 1 task and writes the transcripts.
    run_part2.py              Runs every Part 2 task and writes the transcripts.
    check_database.py         The grader's one-command database check.
  db/
    schema.py                 The seven CREATE TABLE statements.
    generate.py               One generator per table, each on its own seeded stream.
    build.py                  Build the database and write the manifest.
    query.py                  Read helpers Part 2 consumes.
  data/
    conversations/            Generated JSON thread store, gitignored.
    loan_applications.json    Committed snapshot, guarded by a hash test.
    database-manifest.md      Committed DDL, row counts and hash of the database.
    meridian_bank.db          Generated seven-table store, gitignored.
  transcripts/                Committed graded evidence.
  tests/                      Invariant tests, one per acceptance criterion.
  docs/superpowers/specs/     This design specification.
  reference/problem-statement.md
  chroma/                     Generated vector store, gitignored.
```

Parts 3 and 4 add `api/`, `mcp_server/` and `mcp_client.py` alongside these.
No existing module moved when `agent/` arrived and none moves when those do, which is the point of D-02.

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
The database work of D-15 to D-22 left that snapshot byte-identical; D-31 deliberately does not, and acceptance criterion 7 in section 16 is restated to say so.

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

18 documents in `knowledge_base/`, one plain-text file each, 10 to 15 sentences targeting 12, written from scratch for this brief.

A document carries no metadata of its own.
It is the prose a support team would write, and nothing else.
What each one is about lives beside it in `knowledge_base/catalogue.json`, keyed by `doc_id`, which is the filename stem (D-32):

```
{
  "kb-01-loan-eligibility": {
    "title": "Loan eligibility criteria by loan type",
    "topic": "loan_eligibility",
    "required": true
  }
}
```

`rag/kb.py` parses both sides strictly and refuses to load if they describe different sets, in either direction.
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
Both take the document body, which since D-32 is the whole file.

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
The assertion is also test 5 in section 16.

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
That is test 6 in section 16, not a footnote.

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

## 10. Part 2 Task 6 - the lookup tool and the escalation score

### 10.1 What the tool returns

`agent/tools.py::check_loan_application_status(record_id: str) -> dict` returns the brief's three keys plus the context D-21 allows.

| Key | Source |
|---|---|
| `status`, `loan_amount_inr` | `dataset.get_application` |
| `escalation_score`, `recommend_escalation` | `agent/escalation.py` |
| `customer_context` | `db/query.py::customer_context`, exactly its four non-PII fields |
| `found` | `False` when the id is unknown, rather than raising |

An unknown id is a normal outcome, not an error.
The graph has to put it in the same envelope as everything else, so the tool returns `found: False` rather than raising and forcing a try block into a node.

### 10.2 The escalation score

Per D-33.

```
fraud = 1.0 if flagged_for_fraud_review else 0.0

raw   = min(days_since_created / 21, 1.0)
stale = raw        if status in {Submitted, Under Review, Approved}
      = 0.5 * raw  if status == Disbursed
      = 0.0        if status == Rejected

escalation_score = round(0.45 * fraud + 0.55 * stale, 4)
```

The open-status set is imported from `db/query.py::OPEN_STATUSES` rather than redefined, so the two cannot drift.

Three of the four constants are declared modelling choices rather than sourced facts, and `README.md` says so, the same way it already does for tenure and business hours.

| Constant | Value | Why |
|---|---|---|
| Saturation | 21 days | Only 10 of the 100 records sit at or beyond it, so the term stops discriminating there. The knowledge base states no loan-assessment turnaround, so nothing could be read out of it: `kb-01`'s 3 working days is the shortened path for an existing customer, not the standard one. |
| Weights | 0.45 fraud, 0.55 staleness | Chosen so neither signal at a typical value crosses the threshold alone, which is what forces the two to combine. |
| Disbursed half-rate | 0.5 | The money has left the bank, so the clock still runs, but no customer is waiting on a decision. |
| Open statuses | Submitted, Under Review, Approved | Imported, not declared. |

### 10.3 The threshold, and the property that justifies it

`ESCALATION_THRESHOLD = 0.50`, in `config.py` with every other tunable.

Measured over the committed 100 records:

| Measure | Value |
|---|---|
| Percentile of 0.50 in the score distribution | 80th |
| Records escalated | 20 |
| Of those, flagged | 14 |
| Of those, unflagged but stale | 6 |
| Flagged records below the threshold | 2, both `Rejected` |
| Distinct score values | 42 |
| Days before a flagged open application fires | 2 |
| Days before an unflagged open application fires | 20 |

The last four rows are the point.
Six unflagged records cross the line on staleness alone and two flagged records stay below it, which is exactly what a bare boolean OR on `flagged_for_fraud_review` cannot produce: an OR yields 16 and 0.
Test 18 asserts both directions, so a future retune that collapses the score back into an OR fails the suite rather than passing quietly.

## 11. Part 2 Task 7 - the graph

Per D-43: nine nodes and two conditional edges, against the brief's floor of four and one.

### 11.1 Nodes

| Node | Does | Writes |
|---|---|---|
| `guard_input` | Masks fixed-format PII, runs the four injection rules | `masked_query`, `guardrails` |
| `recall` | Loads the thread's history and entity slot, resolves ellipsis | `history`, `entities` |
| `route` | Picks the intent and records the margin | `route`, `route_scores` |
| `policy_answer` | `rag.generate.answer` on `kb_sentences` | `policy` |
| `lookup_status` | `check_loan_application_status`, per section 10 | `lookup` |
| `clarify` | Returns one specific question | `clarification` |
| `verify` | Output-side groundedness and the citation check | `guardrails.grounded` |
| `compose` | Builds the envelope, validates it, persists the turn | `response` |
| `refuse` | Short-circuit refusal naming the rule that fired | `refusal` |

`route` is a node and `pick_branch` is the edge's path function.
Keeping them separate means the branch decision is a pure function of state and test 19 can call it without running the graph.

### 11.2 The conditional edges

The first is `guard_input -> refuse` when an injection rule fired, and `guard_input -> recall` otherwise.

The second is the one the brief asks for.
`pick_branch(state)` returns `"policy"`, `"lookup"`, `"clarify"`, or the two-element sequence `["policy_answer", "lookup_status"]`.

That sequence is why the `both` route is one return value rather than two edges: `StateGraph.add_conditional_edges` types its path as `Callable[..., Hashable | Sequence[Hashable]]`, so LangGraph runs both nodes in the same superstep and merges.

The merge is safe because the two nodes write disjoint keys, `policy` and `lookup`.
No reducer runs, so merge order cannot change the bytes, and the determinism ground rule survives the only concurrency in the system.
Test 27 asserts the disjointness rather than assuming it.

Every branch, including `refuse`, converges on `compose`.
A refusal is a response, so it carries a trace id, validates against the same schema, and is persisted to the thread like any other turn.

### 11.3 The router

Three stages, cheapest first, per D-44 and D-45.

1. A record id in the query, `\bLN-\d{4}\b`, is decisive. With policy language alongside it, the route is `both`; alone, `lookup`.
2. Otherwise the query is embedded with `rag/index.py::embed` and scored against intent exemplar centroids. The embeddings are already unit-normalised, so cosine is a dot product and no new dependency appears.
3. If the best intent's margin over the second is below `config.ROUTE_MARGIN`, the route is `clarify`.

`ROUTE_MARGIN` is measured, not chosen.
`eval/routing.py` holds labelled probe queries, `scripts/run_part2.py` measures the in-scope and ambiguous margins, and `transcripts/part2-routing.txt` records them, exactly as `SIMILARITY_THRESHOLD` 0.2818 was derived in section 8.2.
The value cannot be written here because it does not exist until the code runs; section 18 carries it as an open item.

### 11.4 State

`AgentState` is a `TypedDict` and the graph's only channel schema.

`thread_id`, `trace_id`, `turn`, `query`, `masked_query`, `history`, `entities`, `route`, `route_scores`, `policy`, `lookup`, `clarification`, `refusal`, `guardrails`, `response`.

`trace_id` is `sha256(f"{thread_id}|{turn}|{masked_query}")[:16]`, per D-38.
It is derived from the masked query, so no raw PII reaches the hash input, and Part 3 Task 12 can log it beside the masked text without a second decision.

## 12. Part 2 Task 8 - memory

Per D-37.

One JSON file per thread under `data/conversations/`, gitignored as runtime state.
`scripts/run_part2.py` copies the two demonstration threads into `transcripts/`, which is where committed evidence lives under D-12.

The store holds the turn log the brief asks for, plus an `entities` object whose only member in V1 is `last_record_id`.
`lookup_status` sets it; `recall` injects it when the query carries a pronoun or ellipsis and no explicit record id.

The two transcripts are the same second-turn question on two threads:

| Thread | Turn 1 | Turn 2, "Is it flagged for fraud?" | Route |
|---|---|---|---|
| `demo-multiturn` | "What is the status of LN-1042?" | resolves `it` to LN-1042 from the entity slot | `lookup` |
| `demo-fresh` | none | nothing to resolve `it` to | `clarify` |

That difference is what makes "state correctly absent" visible.
A bare turn log would show an empty list in one file and a populated one in the other, and the agent would behave identically either way.

This store is not Part 4's checkpointer.
The JSON store is the conversation, readable and diffable; the SQLite checkpointer in Task 15 is graph execution state for resuming a half-finished run.
Both key on `thread_id` and neither reads the other.

## 13. Part 2 Task 9 - the response envelope

Per D-36.

`agent/schema.py` defines one Pydantic model, `AgentResponse`, and exports `agent/response.schema.json`, which is committed.

Top level: `trace_id`, `thread_id`, `turn`, `route`, `answer`, then the nullable blocks `policy`, `lookup` and `guardrails`.

| `route` | `policy` | `lookup` | `answer` |
|---|---|---|---|
| `policy` | set | null | grounded text with citations |
| `lookup` | null | set | the record template of D-41 |
| `both` | set | set | both, joined |
| `clarify` | null | null | one specific question |
| `refused` | null | null | the refusal, and the rule that caused it |

`compose` validates twice on purpose.
Pydantic builds the object, then `jsonschema.validate` checks the serialised dict against the committed schema file.
Only the second check proves the exported schema is the one being met, and the exported schema is what Part 3's FastAPI layer and the grader both read.

The `lookup` block carries `source: "record"`, per D-41.
A lookup answer is a fixed template over the record's own fields with no model call behind it, so it must be distinguishable in the envelope from a grounded answer that went through retrieval.

`customer_context` inside `lookup` carries the four fields D-21 permits.
Per D-42, the answer text names the customer and their open-loan count; the credit score appears only in the block and never in prose.

## 14. Part 2 Task 10 - guardrails

Per D-39. Three input rules, four injection rules, two output rules.

| Side | Rule | Fires on | Effect |
|---|---|---|---|
| input, PII | `PAN` | `[A-Z]{5}[0-9]{4}[A-Z]` | `[PAN_REDACTED]` |
| input, PII | `AADHAAR` | 12 digits with a valid Verhoeff check digit | `[AADHAAR_REDACTED]` |
| input, PII | `ACCOUNT` | 11 to 16 digits, not a valid Aadhaar | `[ACCOUNT_REDACTED]` |
| input, injection | `instruction_override` | "ignore previous instructions", "disregard the above" | refuse, rule named |
| input, injection | `role_reassignment` | "you are now", "act as", "pretend to be" | refuse, rule named |
| input, injection | `exfiltration` | "reveal your system prompt", "print your instructions" | refuse, rule named |
| input, injection | `delimiter_injection` | a forged `CONTEXT:` or `[kb-NN]` block in the query | refuse, rule named |
| output | `unsupported` | `GroundedAnswer.supported` is false | the Part 1 fallback text |
| output | `phantom_citation` | a cited `doc_id` that was not among the retrieved hits | drop to refusal |

The masker is one function, `mask_pii`, and Part 3 Task 12 calls the same one over what it logs.
That is a contract, not a convenience: the brief requires that a fixed-format PII field never reach disk in the clear.

`delimiter_injection` exists because of a real surface in this repository.
`llm.py` recovers its own prompt with `_SOURCE = ^\[([a-z0-9\-]+)\]\s*(.+)$` applied to the whole user string, so a query containing a forged `[kb-07] ...` line would be parsed as retrieved context.

`AADHAAR` and `ACCOUNT` overlap by format, and the Verhoeff digit is what separates them, per D-35.
Measured over the 66 generated customers: account-number lengths are `{11: 11, 12: 9, 13: 10, 14: 14, 15: 10, 16: 12}`, so 9 collide with the Aadhaar format exactly, and 1 of those 9 also passes the Verhoeff check and is labelled `AADHAAR` instead of `ACCOUNT`.
All 66 are redacted either way.
The label can be wrong once in 66; the redaction cannot be wrong at all, which is the property that matters.

## 15. Interfaces Parts 3 and 4 consume

Parts 1 and 2 are specified above; this table is what Parts 3 and 4 may rely on and nothing more.
The Part 1 rows are kept because Part 2 consumes them, and because they are what Part 1 was built against.

| Consumer | Interface | Provided by |
|---|---|---|
| Part 2 Task 6, lookup tool | `get_application(record_id) -> dict \| None` | `dataset.py` |
| Part 2 Task 6, customer context | `customer_context(record_id) -> dict` with `customer_id`, `full_name`, `credit_score`, `open_loan_count`, and no PII, per D-21 | `db/query.py` |
| Part 2 Task 7, RAG tool | `answer(query, strategy) -> GroundedAnswer` with `.text`, `.citations`, `.supported`, `.top1_similarity` | `rag/generate.py` |
| Part 2 Task 10, output guardrail | `.supported` and `.top1_similarity` on `GroundedAnswer` | `rag/generate.py` |
| Part 3 Task 13, LLM judge | `generate(system, user) -> str` under `MOCK_LLM` | `llm.py` |
| Part 3 Task 11, request and response models | `AgentResponse`, and the committed `agent/response.schema.json` it exports | `agent/schema.py` |
| Part 3 Task 11, the one call behind both endpoints | `ask(query, thread_id) -> AgentResponse` | `agent/graph.py` |
| Part 3 Task 12, log masking | `mask_pii(text) -> (masked, rules_fired)`, the same function the input guardrail uses | `agent/guardrails.py` |
| Part 3 Task 12, trace id | `AgentResponse.trace_id`, deterministic per D-38 | `agent/schema.py` |
| Part 4 Task 14, MCP tool | `get_application` with a proper docstring, and `escalation_score` beside it | `dataset.py`, `agent/escalation.py` |
| Part 4 Task 15, checkpointing | The compiled graph, which takes a checkpointer at `compile()` time and is keyed by the same `thread_id` the memory store uses | `agent/graph.py` |
| Part 4 Task 16, the node that retries | A node boundary that already exists, so the retry policy attaches without reshaping the graph | `agent/graph.py` |
| All parts | `T`, chunk parameters, paths, `MOCK_LLM` flag | `config.py` |

The recommended collection from Task 5 becomes Part 2's fixed RAG input.
Which one that is cannot be decided here, because it depends on numbers that do not exist until Part 1 runs.

## 16. Testing

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

Part 2 continues the numbering.

| # | Test | Acceptance criterion it restates |
|---|---|---|
| 17 | `check_loan_application_status` returns status, amount and score for a known id, and a typed miss for an unknown one | "correctly looks up a record" |
| 18 | At least one unflagged record scores at or above the threshold and at least one flagged record scores below it | "a designed, justified escalation score, not a bare boolean OR". This is the test D-33 exists for, and the rejected formulas fail it |
| 19 | The compiled graph reports nine nodes, and two sample queries take different branches | "4 or more nodes and a conditional edge that routes to both tools" |
| 20 | Turn 2 of a seeded thread resolves the record id from memory, and the same turn on a fresh thread routes to `clarify` | "multi-turn memory demonstrated, with a separate transcript showing it correctly absent" |
| 21 | Every response from every route validates against `agent/response.schema.json` | "every agent response validates against the declared schema" |
| 22 | A PAN, an Aadhaar and a 14-digit account number are each masked, and no raw value appears anywhere in the emitted state | "PII masking demonstrated firing on the fixed-format fields" |
| 23 | Each of the four injection rules fires on its own probe and yields `route = refused` with the rule named | "prompt-injection detection demonstrated firing" |
| 24 | An out-of-scope query yields the groundedness refusal rather than an answer | "an output-side groundedness check that refuses" |
| 25 | The measured in-scope routing margin exceeds the measured ambiguous margin | D-44's calibration discipline, and the brief's ban on an untested preset |
| 26 | The same thread and turn yields the same `trace_id` twice | D-38, and the determinism ground rule in section 2 |
| 27 | The `both` route writes `policy` and `lookup` from different nodes, and no key twice | Section 11.2, the claim that the fan-out cannot reorder |

Precision@3 and Recall@3 are deliberately not pinned.
They move legitimately when chunk parameters are tuned, and a test that fights tuning is a test that gets deleted.
The measured `ROUTE_MARGIN` is pinned only by test 25, which asserts the separation rather than the value, for the same reason.

## 17. Evidence

Per D-12.

`scripts/run_part1.py` runs every Part 1 task in order and writes:

- `transcripts/part1-dataset.txt` - the validation report
- `transcripts/part1-calibration.txt` - all 17 measured similarities, the gap, the chosen `T`
- `transcripts/part1-generation.txt` - 5 or more in-scope queries answered, plus the out-of-scope fallback
- `transcripts/part1-evaluation.txt` - per-query arithmetic for both collections and the averages

`scripts/run_part2.py` runs every Part 2 task in order and writes:

- `transcripts/part2-escalation.txt` - the formula, the full score distribution, the threshold's percentile, and the escalated set
- `transcripts/part2-routing.txt` - the `ROUTE_MARGIN` calibration, then both tool routes firing on different queries
- `transcripts/part2-graph.txt` - the node list, the edge list, and one run per route end to end
- `transcripts/part2-memory.txt` - the two-turn thread with state carried
- `transcripts/part2-memory-fresh.txt` - the separate fresh thread with state absent
- `transcripts/part2-schema.txt` - the exported JSON Schema, then every response validated against it
- `transcripts/part2-guardrails.txt` - one deliberate case per rule, with the text before and after

`README.md` states the completed track, the exact dataset-design choices needed to reproduce the dataset, the measured calibration values and chosen threshold, the Task 5 recommendation, the escalation formula with its threshold percentile, and a link to each transcript.
Number tables in `README.md` are generated by the same run that writes the transcripts, never typed by hand.

## 18. Open items and carried risks

### 18.1 Part 1

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
   The two-signal fallback in section 19 is the V2 upgrade that removes it.

6. ~~The relational store in section 5.4 is specified but not built.~~ Closed 2026-09-11.
   Built: 1,076 rows over seven tables, `loan_products` 5, `customers` 66, `loan_applications` 100, `application_events` 393, `repayments` 276, `support_tickets` 40, `kyc_documents` 196.
   `data/loan_applications.json` is byte-identical to its pre-database state and a test asserts it on every run.
   The knowledge-base agreement tests found a real drift on their first run - the generator had written "Voter ID Card" and "NREGA Job Card" where `kb-04` says "a voter identity card" and "a job card issued under NREGA" - which is what those tests exist for.
   Largest EMI deviation from `kb-02`'s formula across all 276 instalments: 0.0047 rupees.

### 18.2 Part 2

Recorded here rather than resolved, at Bhavik's instruction on the review artifact of 2026-09-12, so that implementation acts on them deliberately.

1. `config.ROUTE_MARGIN` is not chosen.
   It is measured at implementation time over `eval/routing.py`, written to `transcripts/part2-routing.txt`, and pinned only by test 25, which asserts the separation rather than the value.
   Writing a number here would be the untested preset the brief bans in Task 4, and the same objection applies to a router.

2. **Carried risk.** The known false refusal of item 5 in 18.1 gets louder in Part 2.
   "What documents are needed for KYC?" scores 0.5124 top-1, far above `T`, but its top three sentence chunks land on three different parents, so the support rule refuses.
   In Part 1 that sat in a transcript; in Part 2 it is what a user sees.
   Decision: leave it. It is honest behaviour, `tests/test_generate.py` pins it, and changing the support rule now would move every measured Part 1 number.
   The two-signal fallback in section 19 is the V2 upgrade that removes it, and it is the option that lost in D-07.

3. **Carried risk.** The `clarify` route can loop if a user answers an ambiguous question ambiguously.
   Decision: one clarify per thread, per D-45.
   A second consecutive ambiguous turn falls through to the policy route and lets the groundedness guardrail handle the outcome, so the graph cannot ask twice in a row.
   `recall` reads the previous turn's route to enforce this, which is a second use for state the thread already carries.

4. The `both` route's answer joins two texts whose tones differ, one grounded and cited, one a record template.
   No decision is needed before implementation, but the join is the first thing to read in `transcripts/part2-graph.txt`, and if it reads badly the fix is the template in D-41 rather than the graph.

### 18.3 What Part 2 deliberately does not build

Each line is a thing the brief's preamble mentions or a reviewer might expect, and the reason it is absent.

| Not building | Why not, in V1 |
|---|---|
| A supervisor or multi-agent pattern | The brief's preamble names multi-agent orchestration; no Part 2 task requires it. Under `MOCK_LLM` a supervisor is template matching delegating to template matching. |
| A rolling conversation summary | The summary would be template output, so it would demonstrate plumbing rather than memory. |
| A reflection or self-critique loop | Nothing under `MOCK_LLM` can judge its own output, so the loop would always agree with itself. |
| Tool calling through a real language model | `llm.py` raises for any provider but mock, by design. The deterministic router of section 11.3 is the stand-in. |
| Streaming or async nodes | Part 3 territory, and streaming needs a real model, so it follows the `llm.py` swap in section 19. |

## 19. V2 roadmap

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
