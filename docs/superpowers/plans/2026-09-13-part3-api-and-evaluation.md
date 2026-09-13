# Part 3 API and Evaluation Implementation Plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap Part 2's agent in a FastAPI deployment with two Pydantic endpoints, one structured log line per request, and a 15-query RAG-triad evaluation, all deterministic under `MOCK_LLM`.

**Architecture:** One new package, `api/`, holding a single module.
Two new modules under the existing `eval/` package.
Nothing existing moves and nothing existing is edited except `config.py`, which gains constants, and `README.md`, which gains a generated block.
The API imports exactly one call from `agent/`, `ask`, because spec section 15 contracted exactly one.

**Tech Stack:** Python 3.12.13, `fastapi` 0.141.1, `starlette` 1.6.0, `uvicorn`, `pydantic`, `pytest`.
No new dependency: every one is already in `requirements.txt` and installed.

**Spec:** [`docs/superpowers/specs/2026-09-10-loan-support-agent-design.md`](../specs/2026-09-10-loan-support-agent-design.md).
Sections 22 to 24 and decisions D-72 to D-75, D-80 and D-84 specify this part.
Section 15 is the interface table this part is contracted to consume.

**Sibling plan:** [`2026-09-13-part4-mcp-and-resilience.md`](2026-09-13-part4-mcp-and-resilience.md).
The two are independent and may land in either order, per D-80.
Part 4 changes `agent/graph.py` and `agent/nodes.py`; this plan touches neither, and `ask()` keeps its signature through both, which is what makes them separable.

## Global Constraints

Copied from spec section 2. Every task's requirements implicitly include these.

**Two task numberings are in play, and both are correct.**
This plan numbers its own tasks 1 to 6.
The spec and the brief number Part 3 as Tasks 11 to 13.
Module docstrings cite the **brief's** numbers, because that is what a grader will be holding, so a module opening "Task 13" under plan Task 2 is following this convention rather than making a mistake.

- Everything runs under `MOCK_LLM` with zero API keys and zero network access at inference time. `llm.py` raises for any other `LLM_PROVIDER`.
- Every output is deterministic: same input, same seed, same bytes. No `uuid4`, no wall-clock reads in anything that reaches a transcript.
- `config.py` is the only place a path, port, weight, threshold or collection name is defined. If you need a constant, add it there.
- Python 3.12 specifically, run as `.venv/bin/python`. Never `python` or `python3`.
- Never hand-edit `transcripts/` or the generated number blocks in `README.md`. `scripts/run_part3.py` writes both.
- Tests map one-to-one onto acceptance criteria. This plan adds tests 36 to 40 of spec section 16 and no others.
- Run the whole suite, `.venv/bin/python -m pytest`, before every commit. It stands at 372 passing before this plan starts.

---

### Task 1: The triad query set

**Files:**
- Create: `eval/triad.py`
- Test: `tests/test_triad.py`

**Interfaces:**
- Consumes: `eval.queries.GOLDEN_DATASET` and `eval.queries.GoldenItem`, which carries `item_id`, `text`, `kind`, `gold_doc_ids` and `product`.
- Produces: `TRIAD_ITEM_IDS: tuple[str, ...]` of length 15, and `triad_items() -> list[GoldenItem]` returning them in that order.

- [ ] **Step 1: Write the failing test**

```python
"""Task 13. The query set the RAG triad is scored on."""

import eval.queries as queries
from eval import triad


def test_the_triad_set_holds_fifteen_items_in_a_fixed_order():
    items = triad.triad_items()
    assert len(items) == 15
    assert [item.item_id for item in items] == list(triad.TRIAD_ITEM_IDS)


def test_every_required_topic_appears_at_least_once():
    """The brief wants at least one query per required KB topic, kb-01 to kb-12."""
    covered = {
        doc_id
        for item in triad.triad_items()
        for doc_id in item.gold_doc_ids
    }
    required = {
        document.doc_id
        for document in __import__("rag.kb", fromlist=["kb"]).load_documents()
        if document.required
    }
    assert required <= covered, f"uncovered required topics: {sorted(required - covered)}"


def test_the_set_carries_at_least_two_non_answerable_queries():
    """The brief's floor is two. D-73 spends the third on IU-02 deliberately."""
    kinds = [item.kind for item in triad.triad_items()]
    assert sum(1 for kind in kinds if kind != queries.KIND_ANSWERABLE) >= 2
    assert "IU-02" in triad.TRIAD_ITEM_IDS


def test_the_set_is_a_selection_and_never_a_copy():
    """D-73. Selected by item_id, so the two sets cannot drift apart."""
    by_id = {item.item_id: item for item in queries.GOLDEN_DATASET}
    for item in triad.triad_items():
        assert item is by_id[item.item_id]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_triad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.triad'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Part 3 Task 13. The fifteen queries the RAG triad is scored on.

Selected by item_id out of eval.queries.GOLDEN_DATASET rather than copied as
strings, per D-73, so the evaluation set and the golden dataset cannot drift
apart. A typo here raises at import rather than silently scoring fourteen.

The twelve answerable items between them carry gold documents kb-01 to kb-12,
which is every required topic in the brief. The three that remain are the
negative class, and the third of them is deliberate: IU-02 is the probe of
spec 18.2 item 5 that reads 0.4645 top-1 and answers confidently from the
wrong document. A triad that cannot mark IU-02 down is not measuring
groundedness, so it is in the set rather than avoided by it.
"""

from eval.queries import GOLDEN_DATASET, GoldenItem

TRIAD_ITEM_IDS: tuple[str, ...] = (
    "EQ-01", "EQ-02", "EQ-03", "EQ-04", "EQ-05", "EQ-06",
    "EQ-07", "EQ-08", "EQ-09", "EQ-10", "EQ-11", "EQ-12",
    "FO-01",
    "FO-02",
    "IU-02",
)


def triad_items() -> list[GoldenItem]:
    """The fifteen items, in TRIAD_ITEM_IDS order."""
    by_id = {item.item_id: item for item in GOLDEN_DATASET}
    missing = [item_id for item_id in TRIAD_ITEM_IDS if item_id not in by_id]
    if missing:
        raise KeyError(
            f"TRIAD_ITEM_IDS names {missing}, which is not in GOLDEN_DATASET. "
            f"Either the id is a typo or the golden dataset dropped an item; "
            f"D-73 makes this a selection so the drift is caught here."
        )
    return [by_id[item_id] for item_id in TRIAD_ITEM_IDS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_triad.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest -q
git add eval/triad.py tests/test_triad.py
git commit -m "Select the fifteen triad queries out of the golden dataset"
```

---

### Task 2: The judge and the three scores

**Files:**
- Create: `eval/judge.py`
- Modify: `config.py` (append a Part 3 section)
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `config.MOCK_LLM`, `llm.generate(system, user) -> str`, `rag.generate.answer(query, strategy) -> GroundedAnswer` with `.text` and `.hits`.
- Produces: `TriadScores` (a frozen dataclass with `context_relevance`, `groundedness`, `answer_relevance`, all `float`), `score(query, answer_text, context) -> TriadScores`, `build_judge_prompt(query, answer_text, context) -> tuple[str, str]`, and `context_of(hits) -> str`.

**Why this shape.** D-74 puts the judge behind the provider switch, so `score` dispatches on `config.MOCK_LLM` and calls `llm.generate` only when a real provider is configured.
`llm.py` itself is **not edited**: `_generate_mock` must keep ignoring `system`, because D-63 and test 32 rest on that and every graded byte would move otherwise.

- [ ] **Step 1: Add the constants to `config.py`**

Append this block at the end of `config.py`:

```python
# ---------------------------------------------------------------------------
# Part 3, the RAG triad. Spec section 24.
# ---------------------------------------------------------------------------

# Scores are ratios in [0, 1]. Four places is enough to separate the classes
# and few enough that a transcript diff never moves on floating-point noise.
TRIAD_SCORE_PLACES = 4

# Tokens carrying no topical content. Deliberately short and closed: the judge
# measures overlap, and a long stop list would start deleting the words that
# distinguish one policy question from another. Drawn from the query set's own
# function words, not from a general English stop list.
TRIAD_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "me", "much", "must",
    "my", "of", "on", "or", "the", "to", "what", "when", "which", "will",
    "with", "you", "your",
})
```

- [ ] **Step 2: Write the failing test**

```python
"""Task 13. The three triad scores, and the property that makes them mean something."""

import config
from eval import judge


CONTEXT = (
    "The equated monthly instalment is computed from the principal, the "
    "monthly interest rate and the tenure in months."
)


def test_groundedness_is_high_when_the_answer_restates_the_context():
    scores = judge.score(
        "How is the EMI on a loan calculated?",
        "The equated monthly instalment is computed from the principal, the "
        "monthly interest rate and the tenure in months.",
        CONTEXT,
    )
    assert scores.groundedness > 0.9


def test_groundedness_is_floored_when_there_is_no_context():
    """A refusal is grounded in nothing, and the triad should say so."""
    scores = judge.score("How is the EMI calculated?", "I do not know.", "")
    assert scores.groundedness == 0.0


def test_answer_relevance_falls_when_the_answer_is_about_something_else():
    on_topic = judge.score("How is the EMI calculated?", CONTEXT, CONTEXT)
    off_topic = judge.score(
        "How is the EMI calculated?",
        "Neptune has fourteen known moons.",
        CONTEXT,
    )
    assert off_topic.answer_relevance < on_topic.answer_relevance


def test_every_score_is_a_rounded_ratio():
    scores = judge.score("How is the EMI calculated?", CONTEXT, CONTEXT)
    for value in (scores.context_relevance, scores.groundedness, scores.answer_relevance):
        assert 0.0 <= value <= 1.0
        assert round(value, config.TRIAD_SCORE_PLACES) == value


def test_the_judge_prompt_carries_the_query_the_answer_and_the_context():
    """Exercised under mock so the real-provider path cannot rot unnoticed."""
    system, user = judge.build_judge_prompt("query text", "answer text", "context text")
    assert "query text" in user
    assert "answer text" in user
    assert "context text" in user
    assert "context relevance" in system.lower()


def test_scoring_is_deterministic():
    first = judge.score("How is the EMI calculated?", CONTEXT, CONTEXT)
    second = judge.score("How is the EMI calculated?", CONTEXT, CONTEXT)
    assert first == second
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.judge'`

- [ ] **Step 4: Write minimal implementation**

```python
"""Part 3 Task 13. The LLM-as-judge, and what it scores on with no model to ask.

D-74. Under MOCK_LLM the three scores are lexical: n-gram containment between
the query, the answer and the retrieved context. Under a real provider the
same three are asked of the model through llm.generate, so D-59's single
switch still decides who answers.

The signal is deliberately NOT the embedding. Sharing a signal with the
retriever would make the judge agree with retrieval by construction, which is
the failure spec 18.4 catalogues four times: a mechanism deciding on a signal
that does not carry the property it was asked to decide. IU-02 is in the query
set precisely to catch that, and it would score high on all three under an
embedding judge.

These are deterministic proxies, not a model's judgement, and README.md says
so where it reports them.
"""

import json
import re
from dataclasses import dataclass

import config
import llm

_WORD = re.compile(r"[a-z0-9]+")

JUDGE_SYSTEM = (
    "You are grading a retrieval-augmented answer. Return only a JSON object "
    "with three float fields in [0, 1]: context_relevance, how far the "
    "retrieved context bears on the question; groundedness, how far every "
    "claim in the answer is supported by that context; and answer_relevance, "
    "how far the answer addresses the question actually asked. Return no prose."
)


@dataclass(frozen=True)
class TriadScores:
    """One row of the triad table."""
    context_relevance: float
    groundedness: float
    answer_relevance: float


def context_of(hits) -> str:
    """The retrieved context as one string, which is what the judge sees."""
    return "\n".join(hit.text for hit in hits)


def build_judge_prompt(query: str, answer_text: str, context: str) -> tuple[str, str]:
    """The prompt a real provider is asked. Unused under mock, tested anyway."""
    user = (
        f"QUESTION:\n{query}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer_text}\n"
    )
    return JUDGE_SYSTEM, user


def _tokens(text: str) -> list[str]:
    """Content tokens, lowercased, function words dropped."""
    return [w for w in _WORD.findall(text.lower()) if w not in config.TRIAD_STOPWORDS]


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def _share(needles: set, haystack: set) -> float:
    """The share of needles present in haystack, floored at 0 when empty."""
    if not needles:
        return 0.0
    return round(
        len(needles & haystack) / len(needles), config.TRIAD_SCORE_PLACES
    )


def _score_lexically(query: str, answer_text: str, context: str) -> TriadScores:
    query_tokens = set(_tokens(query))
    context_tokens = set(_tokens(context))
    answer_tokens = set(_tokens(answer_text))
    context_bigrams = _bigrams(_tokens(context))
    answer_bigrams = _bigrams(_tokens(answer_text))

    return TriadScores(
        context_relevance=_share(query_tokens, context_tokens),
        # Bigrams rather than tokens, because a bag of words is satisfied by an
        # answer that uses the right vocabulary to say the wrong thing.
        groundedness=_share(answer_bigrams, context_bigrams),
        answer_relevance=_share(query_tokens, answer_tokens),
    )


def score(query: str, answer_text: str, context: str) -> TriadScores:
    """The three scores. Mock computes them; a real provider is asked for them."""
    if config.MOCK_LLM:
        return _score_lexically(query, answer_text, context)

    system, user = build_judge_prompt(query, answer_text, context)
    payload = json.loads(llm.generate(system, user))
    return TriadScores(
        context_relevance=round(
            float(payload["context_relevance"]), config.TRIAD_SCORE_PLACES
        ),
        groundedness=round(
            float(payload["groundedness"]), config.TRIAD_SCORE_PLACES
        ),
        answer_relevance=round(
            float(payload["answer_relevance"]), config.TRIAD_SCORE_PLACES
        ),
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_judge.py -v`
Expected: 6 passed

- [ ] **Step 6: Write test 39, the one property that is asserted**

Append to `tests/test_judge.py`:

```python
def test_39_non_answerable_queries_score_below_answerable_ones_on_groundedness():
    """Spec section 16 test 39, and D-84.

    The numbers move when the chunker or the scorers are tuned and no test
    pins them, per D-13. This pins the ordering, which is the whole claim the
    triad makes: a judge that cannot mark IU-02 down is broken whatever its
    averages read.
    """
    import eval.queries as queries
    from eval import triad
    from rag import generate

    grounded_by_kind: dict[str, list[float]] = {}
    for item in triad.triad_items():
        answered = generate.answer(item.text)
        scores = judge.score(
            item.text, answered.text, judge.context_of(answered.hits)
        )
        grounded_by_kind.setdefault(item.kind, []).append(scores.groundedness)

    answerable = grounded_by_kind[queries.KIND_ANSWERABLE]
    others = [
        value
        for kind, values in grounded_by_kind.items()
        if kind != queries.KIND_ANSWERABLE
        for value in values
    ]
    assert max(others) < min(answerable), (
        f"a non-answerable query scored {max(others)} groundedness, at or above "
        f"the weakest answerable one at {min(answerable)}. The judge is "
        f"agreeing with retrieval rather than checking it."
    )
```

- [ ] **Step 7: Run it**

Run: `.venv/bin/python -m pytest tests/test_judge.py -v`
Expected: 7 passed.
If test 39 fails, **do not weaken the assertion**. It is the only thing standing between this judge and the failure mode D-74 exists to prevent. Retune `_score_lexically` or `TRIAD_STOPWORDS` until the classes separate, and record what moved.

- [ ] **Step 8: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest -q
git add eval/judge.py tests/test_judge.py config.py
git commit -m "Score the RAG triad on lexical containment, not on the retriever's own signal"
```

---

### Task 3: `POST /ask`

**Files:**
- Create: `api/__init__.py` (empty), `api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `agent.graph.ask(query, thread_id) -> dict`, the validated response payload.
- Produces: `app` (the FastAPI instance), `AskRequest`, `AskResponse`.

**Why the endpoint is `def` and not `async def`.** Part 4 makes `ask` wrap `asyncio.run`, per D-78.
`asyncio.run` raises inside a running event loop, so an `async def` endpoint would break the moment Part 4 lands.
A `def` endpoint runs in FastAPI's threadpool, where there is no loop, and works identically before and after.
This is the one place D-78 is visible from outside `agent/`, and it is why it is written this way from the start.

- [ ] **Step 1: Write the failing test**

```python
"""Task 11. The FastAPI deployment."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_36_ask_answers_a_policy_question_through_the_api():
    """Spec section 16 test 36, Part 3 criterion 1."""
    response = client.post("/ask", json={"query": "How is the EMI on a loan calculated?"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "policy"
    assert body["answer"]
    assert body["policy"]["citations"]


def test_ask_routes_a_lookup_question_to_the_record_tool():
    response = client.post("/ask", json={"query": "What is the status of LN-1042?"})
    assert response.status_code == 200
    assert response.json()["route"] == "lookup"


def test_ask_defaults_the_thread_id():
    response = client.post("/ask", json={"query": "How is the EMI calculated?"})
    assert response.json()["thread_id"] == "default"


def test_ask_rejects_an_empty_query_with_422():
    assert client.post("/ask", json={"query": ""}).status_code == 422


def test_the_response_matches_the_committed_schema():
    """The envelope is AgentResponse unchanged, per D-66. Nothing is widened."""
    import json

    import config

    schema = json.loads(config.RESPONSE_SCHEMA_PATH.read_text())
    body = client.post("/ask", json={"query": "How is the EMI calculated?"}).json()
    import jsonschema

    jsonschema.validate(body, schema)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api'`

- [ ] **Step 3: Write minimal implementation**

Create an empty `api/__init__.py`, then `api/main.py`:

```python
"""Part 3 Task 11. The FastAPI deployment, and Task 12's log line.

Two endpoints over one call. Spec section 15 contracted exactly one entry
point from agent/, `ask`, and this module imports exactly that.

Both endpoints are declared `def` rather than `async def` deliberately. Part 4
makes `ask` wrap asyncio.run, per D-78, and asyncio.run raises inside a
running event loop. A `def` endpoint runs in FastAPI's threadpool where there
is no loop, so it works identically before and after Part 4 lands.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.graph import ask

app = FastAPI(
    title="Meridian Bank loan support agent",
    description="Part 3 Task 11. Every response is the Part 2 envelope, unchanged.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    """One turn of a conversation."""
    query: str = Field(min_length=1, description="The member's question.")
    thread_id: str = Field(
        default="default",
        min_length=1,
        description="Conversation key. The same id carries memory across turns.",
    )


@app.post("/ask")
def ask_endpoint(request: AskRequest) -> dict:
    """Answer one question, through the full nine-node graph."""
    return ask(request.query, thread_id=request.thread_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest -q
git add api/ tests/test_api.py
git commit -m "Expose the agent behind POST /ask with Pydantic models"
```

---

### Task 4: `POST /add-document`

**Files:**
- Modify: `api/main.py`
- Modify: `config.py`
- Modify: `.gitignore`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `rag.kb.catalogue_products()`, `rag.index.get_collection(strategy)`, `rag.index.embed(texts)`, `rag.chunking.chunk(text, strategy)`, `config.STRATEGY_SENTENCES`, `config.MIN_CHUNKS_PER_DOCUMENT`.
- Produces: `AddDocumentRequest`, `AddDocumentResponse` with `doc_id`, `chunks_added`, `collection`.

**The contract, from D-72.** `knowledge_base/` is never written.
The file lands in a gitignored `data/uploads/` and the chunks are upserted into `kb_sentences` only.
Rollback needs no code: `rag.index.build_index` rebuilds any collection whose `count()` does not match the corpus, so the next ordinary index build drops every upload.

- [ ] **Step 1: Add the constants to `config.py`**

Append to the Part 3 block added in Task 2:

```python
# Runtime-uploaded documents. Gitignored, and never inside knowledge_base/,
# per D-72: writing there would move Precision@3, Recall@3 and every chunk
# count in the Part 1 transcripts.
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DOC_PREFIX = "kb-up-"

# Only kb_sentences takes uploads. It is the collection Part 2 consumes and
# the one Task 5 recommended; upserting into both would double the work for a
# collection nothing downstream reads.
UPLOAD_STRATEGY = STRATEGY_SENTENCES
```

- [ ] **Step 2: Add `data/uploads/` to `.gitignore`**

```bash
printf 'data/uploads/\n' >> .gitignore
```

- [ ] **Step 3: Write the failing test**

Append to `tests/test_api.py`:

```python
def _clear_uploads():
    import shutil

    import config

    if config.UPLOAD_DIR.exists():
        shutil.rmtree(config.UPLOAD_DIR)


def test_add_document_makes_a_previously_refused_question_answerable():
    """The loop the endpoint exists to demonstrate: refuse, add, answer.

    The query passes the scope gate as Home Loan and is refused today at
    top-1 0.4534 with no citations, measured 2026-09-13. The gap is real.
    """
    _clear_uploads()
    query = "How long does a home loan take to disburse after approval?"

    before = client.post("/ask", json={"query": query, "thread_id": "up-a"}).json()
    assert before["policy"]["outcome"].startswith("refused")

    added = client.post("/add-document", json={
        "title": "Home loan disbursal timeline",
        "body": (
            "A sanctioned home loan is disbursed within seven working days of "
            "the signed agreement reaching the branch. Disbursal happens in "
            "one tranche for a ready property and in stages for a property "
            "under construction. The applicant is notified by SMS on the day "
            "each tranche is released."
        ),
        "products": ["Home Loan"],
    })
    assert added.status_code == 200
    assert added.json()["doc_id"].startswith("kb-up-")
    assert added.json()["chunks_added"] >= 2

    after = client.post("/ask", json={"query": query, "thread_id": "up-b"}).json()
    assert after["policy"]["outcome"] == "answered"
    assert any(c.startswith("kb-up-") for c in after["policy"]["citations"])


def test_add_document_rejects_a_product_outside_the_catalogue():
    """A document nothing can reach is worse than no document.

    rag/scope.py refuses any question naming an out-of-catalogue product
    before retrieval runs, so the document would be permanently unreachable.
    """
    response = client.post("/add-document", json={
        "title": "Fixed deposit early closure",
        "body": "A fixed deposit closed early earns a reduced rate. " * 3,
        "products": ["Fixed Deposit"],
    })
    assert response.status_code == 422


def test_40_add_document_never_writes_into_the_knowledge_base():
    """Spec section 16 test 40, and the whole of D-72."""
    import hashlib

    import config

    def fingerprint() -> str:
        digest = hashlib.sha256()
        for path in sorted(config.KB_DIR.iterdir()):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    before = fingerprint()
    client.post("/add-document", json={
        "title": "Auto loan top up",
        "body": "An existing auto loan may be topped up after twelve paid EMIs. " * 3,
        "products": ["Auto Loan"],
    })
    assert fingerprint() == before


def test_a_rebuild_drops_every_upload():
    """The rollback D-72 relies on, which is behaviour build_index already has."""
    from rag import index

    client.post("/add-document", json={
        "title": "Education loan moratorium",
        "body": "An education loan carries a moratorium until course completion. " * 3,
        "products": ["Education Loan"],
    })
    index.build_index(rebuild=True)
    collection = index.get_collection(config.UPLOAD_STRATEGY)
    stored = collection.get(include=["metadatas"])
    assert not [
        m for m in stored["metadatas"]
        if m["doc_id"].startswith(config.UPLOAD_DOC_PREFIX)
    ]
    _clear_uploads()
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api.py -k add_document -v`
Expected: FAIL with 404, because the route does not exist yet.

- [ ] **Step 5: Write minimal implementation**

Add to `api/main.py`:

```python
import config
from rag import chunking, index, kb


class AddDocumentRequest(BaseModel):
    """A document to add to the live collection for the rest of this process."""
    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    products: list[str] = Field(
        min_length=1,
        description="Products this document covers. Each must already be in "
                    "knowledge_base/catalogue.json.",
    )


class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks_added: int
    collection: str


def _next_upload_id() -> str:
    """Stable within a run, and visibly an upload in any citation."""
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    used = {
        int(path.stem.removeprefix(config.UPLOAD_DOC_PREFIX))
        for path in config.UPLOAD_DIR.glob(f"{config.UPLOAD_DOC_PREFIX}*.txt")
    }
    return f"{config.UPLOAD_DOC_PREFIX}{max(used, default=0) + 1:02d}"


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document_endpoint(request: AddDocumentRequest) -> AddDocumentResponse:
    """Add a document to the live kb_sentences collection.

    knowledge_base/ is never written, per D-72: a document there would move
    Precision@3, Recall@3 and every chunk count in the Part 1 transcripts.
    The next rag.index.build_index drops everything added here, because it
    rebuilds any collection whose count does not match the corpus on disk.
    """
    from fastapi import HTTPException

    catalogue = set(kb.catalogue_products())
    unknown = [p for p in request.products if p not in catalogue]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{unknown} is not in the catalogue. rag/scope.py refuses any "
                f"question naming an out-of-catalogue product before retrieval "
                f"runs, so this document would be permanently unreachable."
            ),
        )

    pieces = chunking.chunk(request.body, config.UPLOAD_STRATEGY)
    if len(pieces) < config.MIN_CHUNKS_PER_DOCUMENT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"body produced {len(pieces)} chunk(s), needs at least "
                f"{config.MIN_CHUNKS_PER_DOCUMENT}. A single-chunk document can "
                f"never satisfy the same-parent support rule, so it would be "
                f"permanently unanswerable."
            ),
        )

    doc_id = _next_upload_id()
    (config.UPLOAD_DIR / f"{doc_id}.txt").write_text(
        f"{request.title}\n\n{request.body}\n", encoding="utf-8"
    )

    collection = index.get_collection(config.UPLOAD_STRATEGY)
    collection.add(
        ids=[f"{doc_id}::{config.UPLOAD_STRATEGY}::{i:03d}" for i in range(len(pieces))],
        documents=pieces,
        embeddings=index.embed(pieces),
        metadatas=[{
            "doc_id": doc_id,
            "title": request.title,
            "topic": "uploaded",
            "required": False,
            "chunk_index": i,
            "strategy": config.UPLOAD_STRATEGY,
        } for i in range(len(pieces))],
    )
    return AddDocumentResponse(
        doc_id=doc_id,
        chunks_added=len(pieces),
        collection=index.collection_name(config.UPLOAD_STRATEGY),
    )
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: 9 passed.

If `test_add_document_makes_a_previously_refused_question_answerable` fails on the second `/ask` still refusing, the added body is not producing three chunks that agree on the parent.
Lengthen the body rather than weakening the support rule: the support rule is D-07 and moving it moves every Part 1 number.

- [ ] **Step 7: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest -q
git add api/main.py config.py .gitignore tests/test_api.py
git commit -m "Add documents at runtime without ever writing the graded corpus"
```

---

### Task 5: The per-request log line

**Files:**
- Modify: `api/main.py`
- Test: `tests/test_api_logging.py`

**Interfaces:**
- Consumes: `obs.event(name, **fields)`, `agent.guardrails.mask_pii(text) -> (masked, rules_fired)`, `config.LOG_FILE`.
- Produces: one JSON line per request, carrying `event`, `trace_id`, `path`, `status`, `duration_ms` and `query_masked`.

**The trace id, per D-75.** `AgentResponse.trace_id` for `/ask`, so the line joins to the transcript that produced it.
`sha256` of the canonical body, truncated to 16 hex characters the same way D-38 does, for `/add-document`.
Both deterministic, because `uuid4` would make two runs of `scripts/run_part3.py` differ and the determinism ground rule would stop holding.

- [ ] **Step 1: Write the failing test**

```python
"""Task 12. One structured log line per request, with nothing unmasked in it."""

import json
import re

import config
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

PAN = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")


def _read_new_lines(before: int) -> list[dict]:
    text = config.LOG_FILE.read_text(encoding="utf-8") if config.LOG_FILE.exists() else ""
    return [json.loads(line) for line in text.splitlines()[before:] if line.strip()]


def _line_count() -> int:
    if not config.LOG_FILE.exists():
        return 0
    return len([l for l in config.LOG_FILE.read_text(encoding="utf-8").splitlines() if l.strip()])


def test_37_one_request_writes_exactly_one_line_carrying_the_response_trace_id():
    """Spec section 16 test 37, Part 3 criterion 2, and D-75."""
    before = _line_count()
    body = client.post("/ask", json={"query": "How is the EMI calculated?"}).json()
    lines = [l for l in _read_new_lines(before) if l.get("event") == "http_request"]
    assert len(lines) == 1
    assert lines[0]["trace_id"] == body["trace_id"]
    assert lines[0]["path"] == "/ask"
    assert lines[0]["status"] == 200
    assert isinstance(lines[0]["duration_ms"], (int, float))


def test_the_logged_query_never_carries_an_unmasked_pan():
    """D-71. The query is the input side, so the input-side mask applies to it."""
    before = _line_count()
    client.post("/ask", json={"query": "My PAN is FXZPG5049K, what is the minimum balance?"})
    lines = [l for l in _read_new_lines(before) if l.get("event") == "http_request"]
    assert lines
    assert not PAN.search(lines[0]["query_masked"])
    assert "FXZPG5049K" not in json.dumps(lines[0])


def test_the_trace_id_is_deterministic_for_add_document():
    """D-38's rule, applied to a request that produces no AgentResponse."""
    payload = {
        "title": "Business loan collateral",
        "body": "A business loan above twenty lakh rupees requires collateral. " * 3,
        "products": ["Business Loan"],
    }
    before = _line_count()
    client.post("/add-document", json=payload)
    first = [l for l in _read_new_lines(before) if l.get("event") == "http_request"][0]

    before = _line_count()
    client.post("/add-document", json=payload)
    second = [l for l in _read_new_lines(before) if l.get("event") == "http_request"][0]

    assert first["trace_id"] == second["trace_id"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api_logging.py -v`
Expected: FAIL with `assert 0 == 1`, because no middleware writes a line yet.

- [ ] **Step 3: Write minimal implementation**

Add to `api/main.py`:

```python
import hashlib
import json
import time

from starlette.requests import Request

import obs
from agent import guardrails

obs.configure()


@app.middleware("http")
async def log_request(request: Request, call_next):
    """Task 12. One JSON line per request, per D-75.

    The middleware is async because Starlette's middleware contract is; the
    endpoints below it stay sync for the reason in the module docstring.
    """
    body = await request.body()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive

    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round(
        (time.perf_counter() - started) * 1000, config.LOG_DURATION_PLACES
    )

    payload = {}
    try:
        payload = json.loads(body) if body else {}
    except ValueError:
        payload = {}

    raw_query = payload.get("query", "")
    masked_query, _ = guardrails.mask_pii(raw_query) if raw_query else ("", ())

    if request.url.path == "/ask" and raw_query:
        # The response's own trace id, so the log line joins to the transcript
        # that produced it. D-38 already made it deterministic and tested.
        from agent import memory, schema

        thread_id = payload.get("thread_id", "default")
        turn = len(memory.load(thread_id).turns)
        trace_id = schema.trace_id(thread_id, turn, masked_query)
    else:
        # No AgentResponse exists for this request, so the id comes from the
        # body. Canonical JSON, so key order in the request cannot change it.
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        trace_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    obs.event(
        "http_request",
        trace_id=trace_id,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
        query_masked=masked_query,
    )
    return response
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_api_logging.py -v`
Expected: 3 passed.

If `test_37` fails because `trace_id` does not match, the turn index is off by one: the middleware runs **before** the endpoint, so `memory.load(thread_id).turns` has not yet grown, while `ask` computes `turn = len(...) + 1`.
Read `agent/graph.py:78` and `agent/schema.py:101` and make the two agree.
Do not change `agent/schema.py`; it is pinned by test 26.

- [ ] **Step 5: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest -q
git add api/main.py tests/test_api_logging.py
git commit -m "Log one masked JSON line per request on the response's own trace id"
```

---

### Task 6: The runner, the transcripts and the README block

**Files:**
- Create: `scripts/run_part3.py`
- Create (by running): `transcripts/part3-api.txt`, `transcripts/part3-logging.txt`, `transcripts/part3-triad.txt`, `transcripts/part3-readme-numbers.md`
- Modify: `README.md`
- Test: `tests/test_part3_transcripts.py`

**Interfaces:**
- Consumes: everything built in Tasks 1 to 5.
- Produces: four committed transcripts and one generated README block.

**Follow `scripts/run_part2.py` exactly.** Same `write()` helper, same header, same idempotence, same refusal to run under a non-mock provider per D-68.
Read it before writing this file; do not invent a second house style.

- [ ] **Step 1: Write the failing test**

```python
"""The Part 3 transcripts, and the claims they have to keep."""

import re

import config


def _read(name: str) -> str:
    return (config.TRANSCRIPT_DIR / name).read_text(encoding="utf-8")


PAN = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")


def test_38_the_triad_transcript_reports_fifteen_rows_and_three_averages():
    """Spec section 16 test 38, Part 3 criterion 3."""
    from eval import triad

    text = _read("part3-triad.txt")
    for item in triad.triad_items():
        assert item.item_id in text
    assert "context relevance" in text.lower()
    assert "groundedness" in text.lower()
    assert "answer relevance" in text.lower()
    assert text.lower().count("average") >= 1


def test_the_logging_transcript_shows_a_masked_pan_and_no_raw_one():
    text = _read("part3-logging.txt")
    assert config.PII_PLACEHOLDERS["pan"] in text
    assert not PAN.search(text)


def test_the_api_transcript_shows_both_endpoints():
    text = _read("part3-api.txt")
    assert "/ask" in text
    assert "/add-document" in text


def test_the_runner_is_idempotent():
    """Two runs, same bytes. The determinism ground rule, applied to Part 3."""
    import subprocess
    import sys

    names = ["part3-api.txt", "part3-logging.txt", "part3-triad.txt"]
    first = {name: _read(name) for name in names}
    subprocess.run(
        [sys.executable, "scripts/run_part3.py"],
        cwd=config.REPO_ROOT, check=True, capture_output=True,
    )
    assert {name: _read(name) for name in names} == first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_part3_transcripts.py -v`
Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: Write `scripts/run_part3.py`**

Model it on `scripts/run_part2.py`. It must:

1. Refuse to run under a non-mock provider, exiting non-zero and writing nothing, per D-68. Copy the guard from `run_part2.py` verbatim.
2. Clear `config.UPLOAD_DIR` at the start, so a second run starts from the same corpus as the first. **This is what makes the runner idempotent**; without it the second run's `/add-document` allocates `kb-up-02` and every byte after it moves.
3. Rebuild the index if uploads were cleared, so the collection matches the corpus on disk.
4. `transcripts/part3-api.txt`: one `TestClient` call to each endpoint, printing the request and the response envelope, plus the refuse-add-answer loop of Task 4.
5. `transcripts/part3-logging.txt`: the `http_request` lines produced by three requests, one of them carrying a fabricated PAN, read back from `config.LOG_FILE`.
6. `transcripts/part3-triad.txt`: a 15-row table of `item_id`, `kind`, the three scores, and the three averages beneath it, with per-query arithmetic visible.
7. `transcripts/part3-readme-numbers.md`: the three averages, for the README block.

- [ ] **Step 4: Run it**

```bash
.venv/bin/python scripts/run_part3.py
.venv/bin/python -m pytest tests/test_part3_transcripts.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Paste the generated block into `README.md`**

Add a Part 3 section wrapped in the same marker the other two blocks carry:

```markdown
<!-- Generated by scripts/run_part3.py. Do not edit by hand. -->
```

Update the status table at `README.md:21` so Part 3 reads as done.
State beside the triad table, in your own words, that under `MOCK_LLM` the three scores are deterministic lexical proxies rather than a model's judgement, per D-74.

- [ ] **Step 6: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest -q
git add scripts/run_part3.py transcripts/part3-*.txt transcripts/part3-readme-numbers.md README.md tests/test_part3_transcripts.py
git commit -m "Write the Part 3 transcripts and the README number block"
```

---

## Self-review

**Spec coverage.** Section 22.1 is Task 3; 22.2 is Task 4; section 23 is Task 5; section 24.1 is Task 1; 24.2 and 24.3 are Task 2; section 17's evidence requirement is Task 6.
D-72 is Task 4, D-73 Task 1, D-74 Task 2, D-75 Task 5, D-84 Task 2 step 6.
D-81 to D-83, the screen, are deliberately **not** here: D-81 puts the app last, after both parts are green, and it is the final task of the Part 4 plan.

**Interfaces.** `triad_items()` is defined in Task 1 and consumed in Tasks 2 and 6.
`TriadScores` and `score()` are defined in Task 2 and consumed in Task 6.
`app` is defined in Task 3, extended in Tasks 4 and 5, and consumed in Task 6.
`config.UPLOAD_DIR`, `UPLOAD_DOC_PREFIX` and `UPLOAD_STRATEGY` are added in Task 4 step 1 and read in Task 4 step 5 and Task 6 step 3.

**Known risk, carried deliberately.** Task 5's trace-id reconstruction duplicates the turn arithmetic in `agent/graph.py:78`.
If either side changes, test 37 fails loudly rather than silently, which is the failure mode spec 18.4 recommends choosing.
The alternative, returning the trace id from the endpoint to the middleware, means either a response header the schema does not describe or request-scoped state, and both are more machinery than one arithmetic line.
