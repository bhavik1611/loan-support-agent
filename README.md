# Cred Domain Support Agent — Banking & FinTech track

Track completed: **Cred (Banking & FinTech)**.

Everything below runs under `MOCK_LLM` with **zero API keys and zero network calls**
once the SentenceTransformers model is cached. Part 1 is complete; Parts 2 to 4 are not
yet built.

```bash
python run_part1.py     # every Part 1 task, in order, exits non-zero on any failure
```

## Part 1 status

| Task | What it asks for | Where | State |
|---|---|---|---|
| 1 | Seeded loan-application dataset, validated | `dataset.py` | done |
| 2 | At least 12 knowledge-base documents | `knowledge_base/` (18 documents) | done |
| 3 | Two chunking strategies, two ChromaDB collections | `chunking.py`, `index_kb.py` | done |
| 4 | Grounded generation with a calibrated threshold | `rag.py` | done |
| 5 | Precision@3 and Recall@3 for both collections | `evaluate_retrieval.py` | done |

---

## Task 1 — dataset design choices, for deterministic reproduction

**Seed: 3. Records: 500.** `python dataset.py` regenerates, validates and writes a
snapshot plus its SHA-256.

The brief fixes six fields. Those six are a **projection**, not the source. Behind each
record sits an application with an event history, and `status` and `days_since_created`
are both read off that history rather than drawn independently. That is what makes it
impossible for a loan to be `Disbursed`, 25 days old, and queued for urgent attention at
the same time. The full reasoning and the measurements behind every constant are in
[`../../docs/capstone-dataset-design.md`](../../docs/capstone-dataset-design.md).

### Category and status weights

Categories are **not** weighted. The first 15 records round-robin the five categories so
every category clears the brief's floor of three for any seed; the remaining 485 are drawn
uniformly. Observed: Personal 96, Home 105, Auto 104, Education 98, Business 97.

Statuses are **not drawn at all**. A status is whatever stage the application had reached
by its observation day, which makes the distribution a consequence of the timeline model
rather than a parameter. Observed: Submitted 32, Under Review 247, Approved 15,
Rejected 66, Disbursed 140.

`Under Review` dominates because slow products dominate the open book: a Home Loan takes
20 to 50 days to a decision and the brief caps `days_since_created` at 30. `Approved` is
rare because it is a 1-to-4 day window between the decision and disbursal.

### Amount ranges, and the one-sentence reasoning the brief asks for

Each product draws from its own band, because one flat range wide enough to cover a home
loan puts 57% of records at an amount their product could never carry — a 1.02 crore
personal loan, a 91.8 lakh auto loan.

| Product | Band | Days to decision | Median drawn |
|---|---|---|---|
| Personal Loan | ₹50,000 – ₹25 L | 2–10 | ₹4.4 L |
| Auto Loan | ₹1 L – ₹30 L | 3–14 | ₹7.8 L |
| Education Loan | ₹1 L – ₹50 L | 7–25 | ₹13.3 L |
| Business Loan | ₹2 L – ₹75 L | 10–35 | ₹13.4 L |
| Home Loan | ₹10 L – ₹1.5 cr | 20–50 | ₹42.2 L |

Amounts are drawn skewed toward the band floor (`floor + range * u**2.2`), because a
uniform draw puts 35% of home loans above a crore with a median of 79 lakh, which no
Indian home-loan book resembles.

### Fraud flag: 20.4%, inside the required 10–30% band

The flag is not a coin flip. Base odds are 15%, rising 13 points for an amount in the top
fifth of its product band and 13 points for a member who already has an application in the
book. Both are genuine fraud signals and both read fields the record already has.

| Slice | Flagged |
|---|---|
| ordinary application | 16.6% |
| large for its product | 40.0% |
| repeat member | 24.8% |

The correlation is deliberately gentle. Measured over 40 seeds, a simple rule reading only
those two signals predicts 31% of the flags, against 19% by chance. A stronger correlation
reached 57% — at which point over half the flag is a restatement of columns already
visible, and the escalation score's two signals collapse into one.

### Reproducibility

`data/loan_applications.json` and its SHA-256 are committed. `dataset.py` re-derives the
hash on every run, so a drift on a grader's machine fails loudly instead of quietly grading
different data. Verified identical across two fresh Python 3.14.0 interpreters:

```
c3af1099f14637e6e564c77875373b696f6b53600e9d2deadb6602cc497e589b
```

Cross-**version** stability of `random.randint` and `random.choice` is not verified, and
CPython has changed those internals before. The committed hash is what turns that
unverified assumption into something that announces itself.

`dataset.py` also writes `data/loan_applications.sqlite` (500 applications, 1,329 events),
which is what the Part 2 lookup tool will query, the way a deployed system queries records
rather than holding the loan book in memory.

---

## Task 2 — knowledge base

18 documents in `knowledge_base/`, 5 sentences each, mean 712 characters. All 12 required
topics are covered, plus 6 more that overlap them naturally: application stages and
timelines, home-loan documents, late payment and default, disbursal, co-applicants, and
escalation.

The count and length are both deliberate. At the short end of the brief's 2-to-5 sentence
range, fixed-size chunking at 400 characters produces roughly one chunk per document, and
the Task 5 comparison degenerates into "chunked" versus "not chunked". The six extra
documents exist to make retrieval discriminate: with no overlapping material, every query
has an obvious single answer and Precision@3 measures nothing.

---

## Task 3 — two chunking strategies, two collections

| Collection | Strategy | Chunks | Mean chars |
|---|---|---|---|
| `kb_fixed_size` | fixed 200 chars, 40 overlap | 84 | 184 |
| `kb_sentence` | one chunk per sentence | 90 | 142 |

Embeddings: `all-MiniLM-L6-v2`, local, 384 dimensions. Both collections use cosine space,
so reported similarity is `1 - distance` and the threshold below means what it says.

**200/40 was measured, not defaulted.** A strategy that emits more chunks per document gets
more chances to place one in a top-3, so comparing strategies with very different chunk
counts measures the count rather than the boundaries. 200/40 is the row closest to parity
with sentence chunking:

| size/overlap | chunks | vs sentence |
|---|---|---|
| 150/30 | 111 | 1.23× |
| **200/40** | **84** | **0.93×** |
| 250/50 | 70 | 0.78× |
| 400/80 | 44 | 0.49× |

`python chunking.py` reproduces this table. The common 400/80 tutorial default would have
been 0.49× — barely a comparison.

---

## Task 4 — grounded generation and the "I don't know" threshold

**Measured top-1 cosine similarity on the deployed (`fixed`) collection:**

| Cluster | Query | Top-1 cosine |
|---|---|---|
| in-scope | Q1 credit score for a home loan | 0.613 |
| in-scope | Q2 EMI and prepayment | 0.632 |
| in-scope | Q3 home loan documents | 0.499 |
| in-scope | Q4 unauthorised card transaction | 0.463 |
| in-scope | Q5 savings account charges | 0.502 |
| in-scope | Q6 home loan timeline | 0.608 |
| in-scope | Q7 joint application | 0.460 |
| in-scope | Q8 moved abroad, which account | 0.409 |
| out, far | X1 weather in Mumbai | 0.285 |
| out, adjacent | X2 fixed deposit rates | **0.466** |
| out, adjacent | X3 travel insurance | 0.297 |

**Chosen threshold: 0.40.**

**The clusters do not separate.** In-scope spans 0.409 to 0.632; out-of-scope spans 0.285
to 0.466. X2 outscores three genuine questions. No similarity threshold refuses "what are
your fixed deposit rates?" without also refusing real ones, so 0.40 is the best available
trade rather than a clean boundary: **0 of 8 real questions refused, 1 of 3 out-of-scope
answered**.

This is a property of the signal, not a tuning failure, and it is worth stating plainly
because the brief asks for a threshold "between the two clusters" and on this corpus there
is no such gap. The banking-adjacent query is the one that shows it; a weather question
flatters any threshold. Part 2's output-side groundedness check is where X2 has to be
caught.

The presets the brief warns against, measured here: **0.5 would refuse 4 of 8 real
questions, 0.6 would refuse 5, and 0.7 would refuse all 8.**

Generation under `MOCK_LLM` is extractive by construction. Retrieved chunks are split into
sentences, incomplete sentences left by fixed-size cuts are discarded, and the best-matching
whole sentences are returned with their source document tagged. Nothing is paraphrased, so
groundedness is a property of the code rather than a hope about a model.

`python rag.py` demonstrates 5 in-scope queries answering and 2 of 3 out-of-scope queries
triggering the fallback.

---

## Task 5 — Precision@3 and Recall@3, both collections

Scoring is at **document level**: retrieved chunks are mapped to parent documents and
deduplicated before counting. Both strategies are given the same 12-chunk pool and the
first three distinct parents are taken, so neither is advantaged by emitting more chunks.

**Ground truth was committed before any retrieval was run** (commit `8c933ae`, `queries.py`).
Each query has exactly two relevant documents, which lifts the Precision@3 ceiling from
0.333 to **0.667**. With one relevant document, both strategies would score 0.333
identically and the comparison would measure noise.

| Query | fixed P@3 | sentence P@3 | fixed R@3 | sentence R@3 |
|---|---|---|---|---|
| Q1 | 0.667 | 0.667 | 1.000 | 1.000 |
| Q2 | 0.667 | 0.333 | 1.000 | 0.500 |
| Q3 | 0.333 | 0.667 | 0.500 | 1.000 |
| Q4 | 0.333 | 0.333 | 0.500 | 0.500 |
| Q5 | 0.667 | 0.667 | 1.000 | 1.000 |
| Q6 | 0.667 | 0.667 | 1.000 | 1.000 |
| Q7 | 0.667 | 0.667 | 1.000 | 1.000 |
| Q8 | 0.667 | 0.333 | 1.000 | 0.500 |
| **mean** | **0.583** | **0.542** | **0.875** | **0.812** |

`python evaluate_retrieval.py` prints the per-query arithmetic for both collections.

### Recommendation

**Deploy fixed-size 200/40.** It beats sentence chunking on both metrics on my own numbers
— Precision@3 0.583 against 0.542, Recall@3 0.875 against 0.812 — and finds both relevant
documents for 6 of 8 queries against 5 of 8. The margin is one query wide, so this is a
preference rather than a rout, and the reason is visible in the failures: a single sentence
averaging 142 characters often carries a fact without the context that makes it match a
question, whereas a 200-character window usually carries the fact and its surrounding
clause together. The cost is that fixed-size cuts leave fragments, which `rag.py` has to
discard explicitly before composing an answer.

---

## What is not done

Parts 2, 3 and 4 — the LangGraph agent, memory, guardrails, FastAPI deployment, RAG-triad
evaluation, MCP, checkpointing, timeouts and retries — are not built yet.

The `escalation_score` described in the design document belongs to Part 2 Task 6 and is
deliberately absent from Part 1.
