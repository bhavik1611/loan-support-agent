# Groq provider and observability

Status: approved

Approved 2026-09-12 off three grilling rounds, closed on the artifact of that date.
Implements D-58 to D-70 and sections 20 and 21 of [`../specs/2026-09-10-loan-support-agent-design.md`](../specs/2026-09-10-loan-support-agent-design.md), which is the authority; where this plan and the spec disagree, the spec wins.

Section numbers below cite the spec as it stands after the 2026-09-12 amendment that added sections 20 and 21.

## What this changes, in one line

`LLM_PROVIDER=groq` becomes a working value instead of a raise, and every boundary in the pipeline starts emitting one JSON log line, without a single graded byte moving.

## The constraint that shapes every task

`MOCK_LLM` stays the default and stays the graded mode.
The measure of success is that `.venv/bin/python -m pytest` still reports the same count it did before this work started, plus the new tests, and that `transcripts/` and the README number blocks are byte-identical.
Baseline measured on the committed tree at `2d347e7`: **339 passed**.

## Tasks

### Task 1 - `config.py`: the constants and the `.env` load

Add, in the environment section beside `DEFAULT_PROVIDER`:

- `PROVIDER_GROQ = "groq"`
- `GROQ_BASE_URL`, `GROQ_MODEL` (default `openai/gpt-oss-120b`), `GROQ_MAX_TOKENS`, `GROQ_TIMEOUT_SECONDS`, `GROQ_USER_AGENT`
- `LOG_LEVEL` and `LOG_DIR`
- a `.env` load at import that never overrides a variable already set in the real environment

Nothing else in the repository hard-codes any of these, per the architecture rule.

Add `python-dotenv` to `requirements.txt`, and write a committed `.env.example` naming the four variables with no values.

**Done when** `config.resolve_provider()` still returns `mock` with no environment set, and `GROQ_MODEL` reads back the default.

### Task 2 - `llm.py`: the Groq branch

Add `_generate_groq(system, user)` beside `_generate_mock`, and extend `generate` to dispatch to it.
Every other `LLM_PROVIDER` value keeps raising, with the existing message shape.

Per spec 20.2 and 20.3:

- stdlib `urllib` POST, explicit `User-Agent` with a comment naming the 403 it prevents
- `temperature` 0, `seed` pinned, `max_completion_tokens` from config
- raise on a missing key, on any transport or HTTP error, on `finish_reason == "length"`, and on empty content
- return `content` only; the `reasoning` field is never part of the answer

**Done when** the branch raises a message naming `GROQ_API_KEY` with no key set, without touching the network.

### Task 3 - `rag/generate.py`: the citation contract

One sentence appended to `SYSTEM_PROMPT` requiring a closing `Sources: [doc-id]` line.
`_cited_documents` is not touched.

**Done when** test 32 shows mock output byte-identical across the change.

### Task 4 - `obs.py`: the logging spine

New module, stdlib only: a JSON formatter, `get_logger`, and a `timed` context manager emitting one line per boundary with `event`, `trace_id`, `duration_ms` and an outcome.
`LOG_LEVEL` sets the level; the default keeps an ordinary run as quiet as it is today.
Lines go to stderr and to a gitignored `logs/`.

Redaction lives here, not at the call sites: a helper that runs a query through `guardrails.mask_pii`, and a rule that the key is never logged and context is logged as doc ids, counts and lengths.

**Done when** a line round-trips through `json.loads` and carries `trace_id`.

### Task 5 - instrument the seven boundaries

One call each in `rag/scope.py`, `rag/retrieve.py`, `rag/generate.py`, `llm.py`, `agent/intents.py`, `agent/tools.py` and `agent/guardrails.py`, per spec 21.2.
No new boundary is invented to hold a log line, and no function changes signature.

**Done when** the suite is unchanged in count and outcome, because logging is not supposed to be observable from a test that is not about logging.

### Task 6 - the guards

- `tests/conftest.py`: an autouse fixture forcing `LLM_PROVIDER=mock`, so `.env` can never reach the suite
- `scripts/run_part1.py` and `scripts/run_part2.py`: exit non-zero under a non-mock provider, before writing anything

**Done when** running either script with `LLM_PROVIDER=groq` exits non-zero and leaves `transcripts/` untouched.

### Task 7 - `scripts/run_groq_demo.py`

The only end-to-end Groq entry point.
Runs a handful of queries, stamps provider and model in its header per D-65, writes to a gitignored directory, and refuses politely when no key is set.

**Done when** it runs against a live key and its output is not tracked by git.

### Task 8 - tests 32 to 35

One test per criterion, named in spec section 16:

| # | Test |
|---|---|
| 32 | Mock output is byte-identical before and after the `SYSTEM_PROMPT` change |
| 33 | `LLM_PROVIDER=groq` with no key raises naming the variable, offline; the suite stays on mock whatever `.env` says |
| 34 | `scripts/run_part1.py` exits non-zero under a non-mock provider and writes nothing |
| 35 | A log line is valid JSON, carries `trace_id`, and contains no unmasked PAN or Aadhaar and no API key |

### Task 9 - README

A section documenting the two commands and the four variables, naming variables and never a value.
No number block is touched.

### Task 10 - verify

`.venv/bin/python -m pytest` at 339 plus the new tests, all passing.
`git status` showing `transcripts/` and `README.md`'s generated blocks unmodified.
One live demo run against the key, reported with its real output.

## Order

1, 2 and 3 are independent of 4 and 5.
6 depends on 1 and 2. 7 depends on 2. 8 depends on everything it tests. 9 and 10 last.

## What this plan does not do

No metrics endpoint, no exporter, no retry layer: the first two wait for Part 3's server per D-68, and retries are Part 4's resilience task per D-61.
No change to `transcripts/`, the README number blocks, `data/`, the seed, or any threshold.
