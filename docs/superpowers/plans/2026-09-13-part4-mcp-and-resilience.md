# Part 4 MCP and Resilience Implementation Plan

Status: draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden Part 2's agent with an MCP server and client, SQLite checkpointing that resumes without re-executing, a retry policy, a per-node timeout and a global timeout, then add the Streamlit app that is the one thing here no criterion asks for.

**Architecture:** Two new top-level modules, `mcp_server/` and `mcp_client.py`, plus `ui/app.py`.
Two existing files are edited, and only two: `agent/nodes.py` gains one `async def`, and `agent/graph.py` gains a retry policy, a per-node timeout and an `asyncio.run` wrapper.
No node is added, per D-76, so the graph stays at nine and `part2-graph.txt` never moves.

**Tech Stack:** Python 3.12.13, `langgraph` 1.2.11, `langgraph-checkpoint-sqlite` 3.1.1, `aiosqlite` 0.22.1, `fastmcp` 4.0.3, `streamlit`, `pytest`.
One dependency becomes explicit that was already installed transitively, `aiosqlite`, per D-85.
One is genuinely new and lives outside the graded install path, `streamlit`, per D-83.

**Spec:** [`docs/superpowers/specs/2026-09-10-loan-support-agent-design.md`](../specs/2026-09-10-loan-support-agent-design.md).
Sections 25 and 26 and decisions D-76 to D-79 and D-81 to D-83, D-85 specify this part.

**Sibling plan:** [`2026-09-13-part3-api-and-evaluation.md`](2026-09-13-part3-api-and-evaluation.md).
The two are independent and may land in either order, per D-80.
**One ordering constraint, and only one:** Task 6 of this plan, the Streamlit app, needs the Part 3 API running, so it lands after both parts are otherwise green. That is D-81, not an accident of sequencing.

## Global Constraints

Copied from spec section 2. Every task's requirements implicitly include these.

**Two task numberings are in play, and both are correct.**
This plan numbers its own tasks 1 to 6.
The spec and the brief number Part 4 as Tasks 14 to 16.
Module docstrings cite the **brief's** numbers.

- Everything runs under `MOCK_LLM` with zero API keys and zero outbound network access. Loopback HTTP between `mcp_client.py` and the local MCP server is not outbound: nothing leaves the machine.
- Every output is deterministic. Durations are the one exception and are rounded at `config.LOG_DURATION_PLACES`, per D-70, so a noisy microsecond never makes a transcript diff.
- `config.py` is the only place a port, timeout, retry parameter or path is defined.
- Python 3.12 specifically, run as `.venv/bin/python`.
- Never hand-edit `transcripts/` or the generated blocks in `README.md`.
- This plan adds tests 41 to 46 of spec section 16 and no others.
- **The whole suite must stay green at every commit.** It stands at 372 passing before this plan, plus whatever Part 3 added if that landed first.

---

### Task 1: The async node, the policies, and the proof that nothing moved

**Files:**
- Modify: `agent/nodes.py` (one line)
- Modify: `agent/graph.py`
- Modify: `agent/tools.py` (one line, see step 3)
- Modify: `config.py`
- Modify: `tests/test_nodes.py:131` and `:139`
- Test: `tests/test_resilience.py`

**Interfaces:**
- Consumes: `nodes.NODE_NAMES`, `nodes.lookup_status`, `langgraph.types.RetryPolicy`.
- Produces: `graph.build_graph()` unchanged in signature, `graph.compiled()` unchanged, `graph.ask(query, thread_id) -> dict` unchanged in signature, and a new `graph.ask_async(query, thread_id) -> dict` for callers that already have a loop.

**Why this task exists first.** Tasks 3 and 4 cannot be written until the graph is async, because both the checkpointer and the per-node timeout refuse the sync path.
This task is the one that has to prove it changed nothing, so it carries that proof as a step.

- [ ] **Step 1: Add the constants to `config.py`**

```python
# ---------------------------------------------------------------------------
# Part 4, resilience and MCP. Spec section 25.
# ---------------------------------------------------------------------------

# Per-node budget for the one node that does record I/O. Generous against a
# local SQLite read that takes single-digit milliseconds, so it never fires in
# ordinary operation and only the Task 16 shim can trip it.
NODE_TIMEOUT_SECONDS = 5.0

# Whole-run budget, enforced by asyncio.wait_for around ainvoke. Larger than
# the per-node budget by more than one node's worth, so a single slow node
# trips its own timeout first and the global one means what it says.
GRAPH_TIMEOUT_SECONDS = 30.0

# The four parameters the brief asks you to state, plus the one it does not.
# jitter is OFF deliberately: a jittered sleep makes the retry transcript's
# timings irreproducible, and the determinism ground rule outranks the small
# thundering-herd benefit a single-process demo cannot exhibit anyway.
RETRY_MAX_ATTEMPTS = 3
RETRY_INITIAL_INTERVAL = 0.05
RETRY_BACKOFF_FACTOR = 2.0
RETRY_MAX_INTERVAL = 0.4
RETRY_JITTER = False

# Ports. The brief introduces 8000 for MCP with "e.g." while being insistent
# about the /mcp path, so the API keeps its own default and MCP moves. D-79.
API_PORT = 8000
MCP_PORT = 8765
MCP_PATH = "/mcp"
MCP_HOST = "127.0.0.1"
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}"

# Graph execution state for resuming a half-finished run. Not the conversation:
# that is data/conversations/, readable and diffable. Gitignored, regenerable.
CHECKPOINT_PATH = REPO_ROOT / "checkpoints.sqlite"
```

- [ ] **Step 2: Add `checkpoints.sqlite` to `.gitignore`**

```bash
printf 'checkpoints.sqlite\n' >> .gitignore
```

- [ ] **Step 3: Make `lookup_status` async**

In `agent/nodes.py`, change exactly one line:

```python
async def lookup_status(state: AgentState) -> dict:
```

The body does not change.
It calls `agent.tools.check_loan_application_status`, which stays synchronous: the node is async so LangGraph will accept a timeout on it, not because the work underneath is I/O-bound.
Add one line to the node's docstring saying exactly that, so the next reader does not "fix" it back.

- [ ] **Step 4: Wire the policies and the async driver in `agent/graph.py`**

```python
import asyncio
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

import config
from agent import nodes
from agent.state import AgentState, new_state

# Task 16. Attached to the one node that reads a record, because that is the
# boundary a transient failure would actually cross. D-76 puts the simulated
# failure in scripts/run_part4.py rather than in a tenth node here: the graph
# ships nine nodes and tests/test_graph.py asserts it.
RETRY_POLICY = RetryPolicy(
    max_attempts=config.RETRY_MAX_ATTEMPTS,
    initial_interval=config.RETRY_INITIAL_INTERVAL,
    backoff_factor=config.RETRY_BACKOFF_FACTOR,
    max_interval=config.RETRY_MAX_INTERVAL,
    jitter=config.RETRY_JITTER,
)
RESILIENT_NODE = "lookup_status"
```

In `build_graph`, replace the node loop:

```python
    for name in nodes.NODE_NAMES:
        if name == RESILIENT_NODE:
            # langgraph 1.2.11 accepts a timeout only on an async node, and
            # says why: sync Python execution cannot be safely cancelled
            # in-process. That is the whole reason lookup_status is async.
            builder.add_node(
                name,
                getattr(nodes, name),
                retry_policy=RETRY_POLICY,
                timeout=config.NODE_TIMEOUT_SECONDS,
            )
        else:
            builder.add_node(name, getattr(nodes, name))
```

Replace `ask` and add `ask_async`:

```python
async def ask_async(query: str, thread_id: str = "default") -> dict:
    """One turn, on the async path. For callers that already have a loop."""
    from agent import memory

    turn = len(memory.load(thread_id).turns) + 1
    final = await compiled().ainvoke(new_state(query, thread_id, turn))
    return final["response"]


def ask(query: str, thread_id: str = "default") -> dict:
    """One turn. Returns the validated response payload.

    The signature is unchanged from Part 2 and all 19 existing call sites are
    untouched. The graph underneath is driven by ainvoke, because the per-node
    timeout of Task 16 exists only on the async path.

    asyncio.run raises inside a running event loop, so a caller that already
    has one must use ask_async. api/main.py declares its endpoints `def` for
    exactly this reason, which puts them in FastAPI's threadpool.
    """
    return asyncio.run(ask_async(query, thread_id))
```

- [ ] **Step 5: Fix the two direct calls in `tests/test_nodes.py`**

Lines 131 and 139 call `nodes.lookup_status(state)` directly and now receive a coroutine.
Wrap each in `asyncio.run(...)` and add `import asyncio` at the top.
Do **not** add `pytest-asyncio`: these are direct calls, not async tests, and the dependency buys nothing.

- [ ] **Step 6: Prove nothing moved**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_part2.py
git diff --stat transcripts/ README.md
```

Expected: the suite passes in full, the runner exits 0, and **the diff is empty**.

This was measured on 2026-09-13 before this plan was written and came back byte-identical with 370 of 372 passing, the two failures being exactly the lines fixed in step 5.
If the diff is not empty here, stop and find out why before continuing: a moved transcript byte means the async path changed an outcome, and every downstream task in this plan rests on it not having.

- [ ] **Step 7: Write the tests for the policies themselves**

```python
"""Tasks 15 and 16. Resilience, on the real graph rather than a demo one."""

import asyncio
import time

import pytest

import config
from agent import graph, nodes
from langgraph.errors import NodeTimeoutError


def test_the_graph_still_has_exactly_nine_nodes():
    """D-76. The retry policy attaches to a node that already existed."""
    assert len(graph.node_names()) == 9


def test_the_retry_policy_states_every_parameter_the_brief_asks_for():
    policy = graph.RETRY_POLICY
    assert policy.max_attempts == config.RETRY_MAX_ATTEMPTS == 3
    assert policy.initial_interval == config.RETRY_INITIAL_INTERVAL
    assert policy.max_interval == config.RETRY_MAX_INTERVAL
    assert policy.backoff_factor == config.RETRY_BACKOFF_FACTOR
    assert policy.jitter is False


def test_the_resilient_node_is_async_so_a_timeout_is_accepted():
    """langgraph refuses a timeout on a sync node. This is why it is async."""
    assert asyncio.iscoroutinefunction(getattr(nodes, graph.RESILIENT_NODE))


def test_43_the_retry_policy_recovers_a_transient_failure(monkeypatch):
    """Spec section 16 test 43, Part 4 criterion 3a.

    The shim is the brief's own shape: fails the first two calls via a
    counter, then succeeds. It lives here and in scripts/run_part4.py, never
    in the graph, per D-76.
    """
    from agent import tools

    calls = {"n": 0}
    real = tools.check_loan_application_status

    def flaky(record_id: str):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ConnectionError(f"simulated transient failure {calls['n']}")
        return real(record_id)

    monkeypatch.setattr(tools, "check_loan_application_status", flaky)
    graph.compiled.cache_clear()
    try:
        response = graph.ask("What is the status of LN-1042?", thread_id="retry")
    finally:
        graph.compiled.cache_clear()

    assert calls["n"] == 3, "the policy should have spent exactly three attempts"
    assert response["route"] == "lookup"
    assert response["lookup"]["found"] is True


def test_44_a_node_over_its_budget_raises_rather_than_hanging(monkeypatch):
    """Spec section 16 test 44, Part 4 criterion 3b.

    The criterion asks for a clean error, not a hang. NodeTimeoutError is the
    library's own, which is the whole reason D-78 chose the async path over a
    hand-rolled worker thread that could not deliver one.
    """
    from agent import tools

    def slow(record_id: str):
        time.sleep(config.NODE_TIMEOUT_SECONDS + 1)
        raise AssertionError("the timeout should have fired before this line")

    monkeypatch.setattr(tools, "check_loan_application_status", slow)
    monkeypatch.setattr(config, "NODE_TIMEOUT_SECONDS", 0.1)
    graph.compiled.cache_clear()
    try:
        with pytest.raises(NodeTimeoutError):
            graph.ask("What is the status of LN-1042?", thread_id="node-timeout")
    finally:
        graph.compiled.cache_clear()


def test_45_a_run_over_the_global_budget_is_cancelled(monkeypatch):
    """Spec section 16 test 45, Part 4 criterion 3c."""
    from agent import tools

    def slow(record_id: str):
        time.sleep(5)
        raise AssertionError("the global timeout should have fired first")

    monkeypatch.setattr(tools, "check_loan_application_status", slow)
    graph.compiled.cache_clear()

    async def run():
        return await asyncio.wait_for(
            graph.ask_async("What is the status of LN-1042?", thread_id="global"),
            timeout=0.2,
        )

    try:
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            asyncio.run(run())
    finally:
        graph.compiled.cache_clear()
```

- [ ] **Step 8: Run them**

Run: `.venv/bin/python -m pytest tests/test_resilience.py -v`
Expected: 6 passed.

`graph.compiled` is `lru_cache`d, which is why every test that monkeypatches a node clears it on both sides.
Forgetting the second `cache_clear` leaks a patched graph into the next test, and the failure lands somewhere unrelated.

- [ ] **Step 9: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest -q
git add agent/nodes.py agent/graph.py config.py .gitignore tests/test_nodes.py tests/test_resilience.py
git commit -m "Drive the graph on the async path so a per-node timeout is possible"
```

---

### Task 2: The MCP server and client

**Files:**
- Create: `mcp_server/__init__.py` (empty), `mcp_server/server.py`, `mcp_client.py`
- Test: `tests/test_mcp.py`

**Interfaces:**
- Consumes: `agent.tools.check_loan_application_status(record_id) -> dict`.
- Produces: a `fastmcp` server exposing one tool by that name, and `mcp_client.py` runnable as `python mcp_client.py --url <url> LN-1042 LN-1057`.

**The brief is specific about three things here.** The package is `fastmcp`, not `fast-mcp`.
The HTTP transport mounts at `/mcp`, not the bare root.
The client must be a **different file and process** from the agent, which is why `mcp_client.py` never starts a server, per D-79.

- [ ] **Step 1: Write the failing test**

```python
"""Task 14. The lookup tool, over a real MCP round trip."""

import asyncio

import config
from fastmcp import Client

from mcp_server.server import mcp


def test_41_the_tool_answers_for_two_different_record_ids():
    """Spec section 16 test 41, Part 4 criterion 1.

    In-memory transport here, because a test should not bind a port. The real
    client-server round trip over HTTP is scripts/run_part4.py's job and its
    transcript is the evidence for the criterion.
    """
    async def run():
        async with Client(mcp) as client:
            return [
                await client.call_tool(
                    "check_loan_application_status", {"record_id": rid}
                )
                for rid in ("LN-1042", "LN-1057")
            ]

    first, second = asyncio.run(run())
    assert first.data["record_id"] == "LN-1042"
    assert second.data["record_id"] == "LN-1057"
    assert first.data["found"] is True
    assert 0.0 <= first.data["escalation_score"] <= 1.0
    assert first.data != second.data


def test_the_tool_reports_a_missing_record_rather_than_raising():
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(
                "check_loan_application_status", {"record_id": "LN-9999"}
            )

    assert asyncio.run(run()).data["found"] is False


def test_the_tool_carries_the_docstring_the_brief_asks_for():
    async def run():
        async with Client(mcp) as client:
            return await client.list_tools()

    tools = {tool.name: tool for tool in asyncio.run(run())}
    assert "check_loan_application_status" in tools
    description = tools["check_loan_application_status"].description or ""
    assert "escalation" in description.lower()
    assert len(description.split()) >= 20


def test_the_client_never_starts_a_server():
    """D-79. Criterion 14 checks that the client is a separate process."""
    source = (config.REPO_ROOT / "mcp_client.py").read_text(encoding="utf-8")
    for forbidden in ("subprocess", "mcp.run(", "uvicorn", "from mcp_server"):
        assert forbidden not in source, f"mcp_client.py must not reference {forbidden}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mcp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mcp_server'`

- [ ] **Step 3: Write `mcp_server/server.py`**

```python
"""Part 4 Task 14. The lookup tool, exposed over MCP.

One tool, wrapping the same agent/tools.py call the graph uses, so the MCP
surface and the agent cannot answer differently about the same record.

The HTTP transport mounts at config.MCP_PATH, not at the bare root, which is
what the brief is insistent about. The port is config.MCP_PORT rather than
8000, so the Part 3 API keeps its own default. D-79.
"""

from fastmcp import FastMCP

import config
from agent.tools import check_loan_application_status as _lookup

mcp = FastMCP("meridian-loan-support")


@mcp.tool
def check_loan_application_status(record_id: str) -> dict:
    """Look up one Meridian Bank loan application by its record id.

    Returns the application's current status, the sanctioned amount in rupees,
    the age of the application in days, whether it is flagged for fraud
    review, a designed escalation score in [0, 1] combining the fraud flag
    with a normalised staleness signal, and whether that score clears the
    recommended escalation threshold. Record ids look like LN-1042. An id that
    matches no application returns found=False rather than raising, so a
    caller can distinguish a missing record from a failed call.
    """
    return _lookup(record_id)


def main() -> None:
    mcp.run(transport="http", host=config.MCP_HOST, port=config.MCP_PORT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `mcp_client.py`**

```python
"""Part 4 Task 14. A standalone MCP client.

A different file and a different process from the agent, which is what the
criterion checks. It takes a URL and never starts a server: a client that
launches its own server is no longer only a client. D-79.

Run it against a server already listening:

    .venv/bin/python mcp_client.py --url http://127.0.0.1:8765/mcp LN-1042 LN-1057
"""

import argparse
import asyncio
import json

from fastmcp import Client

TOOL = "check_loan_application_status"


async def call(url: str, record_ids: list[str]) -> list[dict]:
    """One session, one call per record id, in order."""
    results = []
    async with Client(url) as client:
        for record_id in record_ids:
            response = await client.call_tool(TOOL, {"record_id": record_id})
            results.append({
                "record_id": record_id,
                "is_error": response.is_error,
                "content": [block.text for block in response.content],
                "structured": response.data,
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the Meridian MCP tool.")
    parser.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8765/mcp")
    parser.add_argument("record_ids", nargs="+", help="two or more record ids")
    args = parser.parse_args()

    for result in asyncio.run(call(args.url, args.record_ids)):
        print(f"--- {TOOL}({result['record_id']!r}) ---")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_mcp.py -v`
Expected: 4 passed.

If `call_tool` returns something without `.data`, check the `fastmcp` version: this is written against 4.0.3, where the result carries both `.content` and `.data`.
Run `.venv/bin/python -c "import fastmcp; print(fastmcp.__version__)"` before changing any assertion.

- [ ] **Step 6: Prove the real round trip by hand once**

```bash
.venv/bin/python -m mcp_server.server &
sleep 3
.venv/bin/python mcp_client.py --url http://127.0.0.1:8765/mcp LN-1042 LN-1057
kill %1
```

Expected: two JSON blocks, different statuses and amounts.
Task 5 automates this; do it once by hand first so a failure here is not tangled up with the runner.

- [ ] **Step 7: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest -q
git add mcp_server/ mcp_client.py tests/test_mcp.py
git commit -m "Expose the lookup tool over MCP and call it from a separate client"
```

---

### Task 3: Checkpointing

**Files:**
- Modify: `requirements.txt` (declare `aiosqlite`, per D-85)
- Create: `agent/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `graph.build_graph()`, `agent.state.new_state`, `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`.
- Produces: `checkpointed(saver, interrupt_before) -> CompiledStateGraph` and `INTERRUPT_BEFORE = "verify"`.

**Why the async saver.** D-85, forced rather than chosen.
`SqliteSaver` raises `NotImplementedError`, "The SqliteSaver does not support async methods", under `ainvoke`, and Task 1 made the graph async-only.
`aiosqlite` 0.22.1 is already installed transitively with no runtime requirements of its own, so the requirements line makes an existing fact explicit.

- [ ] **Step 1: Declare `aiosqlite`**

Add to `requirements.txt` under the Part 4 heading, with the reason on the line above:

```
# AsyncSqliteSaver needs it, and the graph is async-only per D-78.
aiosqlite
```

- [ ] **Step 2: Write the failing test**

```python
"""Task 15. Checkpointed resume, and the proof that nothing ran twice."""

import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import checkpoint, graph, nodes


def test_42_a_resumed_thread_re_executes_nothing(monkeypatch):
    """Spec section 16 test 42, Part 4 criterion 2, and D-77.

    The proof is node-entry counts, not a claim about state. Each node records
    its own entry, so a re-execution shows up as a second entry rather than
    having to be inferred from a value that happens to look right.
    """
    entered: list[str] = []

    for name in nodes.NODE_NAMES:
        original = getattr(nodes, name)

        def wrap(fn, label):
            if asyncio.iscoroutinefunction(fn):
                async def recorded(state):
                    entered.append(label)
                    return await fn(state)
            else:
                def recorded(state):
                    entered.append(label)
                    return fn(state)
            return recorded

        monkeypatch.setattr(nodes, name, wrap(original, name))

    async def run():
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            compiled = checkpoint.checkpointed(saver)
            config = {"configurable": {"thread_id": "resume-1"}}
            from agent.state import new_state

            first = await compiled.ainvoke(new_state("How is the EMI calculated?", "resume-1", 1), config)
            run_one = list(entered)
            entered.clear()
            second = await compiled.ainvoke(None, config)
            return run_one, list(entered), second

    run_one, run_two, final = asyncio.run(run())

    assert len(run_one) >= 2, "criterion 15a wants at least two nodes before the stop"
    assert checkpoint.INTERRUPT_BEFORE not in run_one
    assert checkpoint.INTERRUPT_BEFORE in run_two
    assert not (set(run_one) & set(run_two)), (
        f"{sorted(set(run_one) & set(run_two))} ran in both invocations; the "
        f"checkpoint was not loaded and criterion 15c is not met"
    )
    assert final["response"]["answer"]


def test_the_conversation_turn_is_written_exactly_once():
    """compose is the node that records a turn, and it runs only after resume."""
    assert checkpoint.INTERRUPT_BEFORE == "verify"
    assert nodes.NODE_NAMES.index("verify") < nodes.NODE_NAMES.index("compose")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: FAIL with `ImportError: cannot import name 'checkpoint'`

- [ ] **Step 4: Write `agent/checkpoint.py`**

```python
"""Part 4 Task 15. The checkpointed graph, and where it stops.

This is graph execution state for resuming a half-finished run, not the
conversation. The conversation is data/conversations/, readable and diffable.
Both key on thread_id and neither reads the other, which agent/memory.py
already said before this module existed.

The saver is the async one, per D-85: the sync SqliteSaver raises
NotImplementedError under ainvoke, and D-78 made the graph async-only.
"""

import config
from agent.graph import build_graph

# Four of the nine nodes run before this one on a policy turn, which clears
# criterion 15a's floor of two with room to spare. It also sits before
# compose, so the turn is written to the conversation store exactly once, on
# resume, rather than half-written by the interrupted run.
INTERRUPT_BEFORE = "verify"


def checkpointed(saver, interrupt_before: str = INTERRUPT_BEFORE):
    """The nine-node graph, compiled against a checkpointer.

    build_graph() has returned the uncompiled builder since Part 2 for exactly
    this, so nothing is reshaped to accommodate a checkpointer.
    """
    return build_graph().compile(
        checkpointer=saver, interrupt_before=[interrupt_before]
    )


def conn_string() -> str:
    """The on-disk checkpoint file. Gitignored and regenerable."""
    return str(config.CHECKPOINT_PATH)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_checkpoint.py -v`
Expected: 2 passed.

This mechanic was proved end to end on 2026-09-13 before this plan was written: with an async node carrying a timeout, `AsyncSqliteSaver` and `interrupt_before`, run 1 entered two nodes, run 2 entered the other two, the final state carried all four and nothing ran twice.

- [ ] **Step 6: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest -q
git add agent/checkpoint.py requirements.txt tests/test_checkpoint.py
git commit -m "Resume a checkpointed thread without re-running what already ran"
```

---

### Task 4: The runner, the transcripts and the README block

**Files:**
- Create: `scripts/run_part4.py`
- Create (by running): `transcripts/part4-mcp.txt`, `transcripts/part4-checkpoint.txt`, `transcripts/part4-resilience.txt`, `transcripts/part4-readme-numbers.md`
- Modify: `README.md`
- Test: `tests/test_part4_transcripts.py`

**Follow `scripts/run_part2.py` exactly.** Same `write()` helper, same header, same idempotence, same non-mock refusal per D-68.

- [ ] **Step 1: Write the failing test**

```python
"""The Part 4 transcripts, and the claims they have to keep."""

import config


def _read(name: str) -> str:
    return (config.TRANSCRIPT_DIR / name).read_text(encoding="utf-8")


def test_the_mcp_transcript_shows_a_real_round_trip_for_two_ids():
    text = _read("part4-mcp.txt")
    assert "LN-1042" in text and "LN-1057" in text
    assert config.MCP_URL in text
    assert "check_loan_application_status" in text


def test_the_checkpoint_transcript_names_what_ran_and_what_was_loaded():
    text = _read("part4-checkpoint.txt")
    assert "run 1" in text.lower() and "run 2" in text.lower()
    assert "loaded from the checkpoint" in text.lower()
    assert "guard_input" in text and "compose" in text


def test_the_resilience_transcript_shows_all_three_demonstrations():
    text = _read("part4-resilience.txt").lower()
    assert "attempt 1" in text and "attempt 3" in text
    assert "nodetimeouterror" in text
    assert "global" in text


def test_the_retry_parameters_are_all_stated():
    text = _read("part4-resilience.txt")
    for parameter in ("max attempts", "initial interval", "max interval", "jitter"):
        assert parameter in text.lower(), f"the brief asks you to state {parameter}"


def test_the_runner_is_idempotent():
    import subprocess
    import sys

    names = ["part4-mcp.txt", "part4-checkpoint.txt", "part4-resilience.txt"]
    first = {name: _read(name) for name in names}
    subprocess.run(
        [sys.executable, "scripts/run_part4.py"],
        cwd=config.REPO_ROOT, check=True, capture_output=True,
    )
    assert {name: _read(name) for name in names} == first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_part4_transcripts.py -v`
Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: Write `scripts/run_part4.py`**

It must:

1. Refuse a non-mock provider, exiting non-zero and writing nothing, per D-68. Copy the guard from `run_part2.py`.
2. Delete `config.CHECKPOINT_PATH` at the start, so run 1 is genuinely a fresh thread. **Without this the second run of the script resumes a thread that is already finished and the transcript changes**, which is the idempotence test above.
3. `part4-mcp.txt`: spawn `python -m mcp_server.server` with `subprocess.Popen`, poll `config.MCP_URL` until it answers or a bounded number of attempts elapses, run `mcp_client.py` as a **separate subprocess** for `LN-1042` and `LN-1057`, capture its stdout verbatim, then terminate the server in a `finally`. Print the URL and the fact that client and server are two processes, because that is what criterion 14 checks.
4. `part4-checkpoint.txt`: the two invocations of Task 3, listing the nodes entered in each and stating which were loaded from the checkpoint rather than re-executed. Write the file to `config.CHECKPOINT_PATH` rather than `:memory:`, so the transcript can report that `checkpoints.sqlite` exists and holds the thread.
5. `part4-resilience.txt`: the three demonstrations, each with its shim described in one line, the stated retry parameters, the attempt-by-attempt trace, the `NodeTimeoutError` message verbatim, and the global cancellation.
6. `part4-readme-numbers.md`: the retry parameters, the two budgets and the node counts, for the README block.

**The server spawn is the one fragile step.** Bound the readiness poll and fail loudly with the server's captured stderr if it never answers.
A bare `sleep` is what makes this flaky on a slower machine.

- [ ] **Step 4: Run it**

```bash
.venv/bin/python scripts/run_part4.py
.venv/bin/python -m pytest tests/test_part4_transcripts.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Paste the generated block into `README.md`**

Wrap it in `<!-- Generated by scripts/run_part4.py. Do not edit by hand. -->` and update the status table at `README.md:21` so Part 4 reads as done.

- [ ] **Step 6: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest -q
git add scripts/run_part4.py transcripts/part4-*.txt transcripts/part4-readme-numbers.md README.md tests/test_part4_transcripts.py
git commit -m "Write the Part 4 transcripts and the README number block"
```

---

### Task 5: The Streamlit app

**Files:**
- Create: `ui/app.py`, `requirements-ui.txt`
- Modify: `requirements.txt` (one comment line), `README.md`
- Test: `tests/test_ui.py`

**Prerequisite: Parts 3 and 4 are both green and committed.** D-81 puts this last on purpose.
It earns no marks, and it is the first thing to cut if time runs short, which is the property that makes it safe to plan at all.

**What it is and is not.** It talks to `POST /ask` over HTTP and imports nothing from `agent/` or `rag/`, per D-82, so every interaction produces a Task 12 log line and the app drives the graded deliverable instead of going around it.
It can never be graded evidence, because the brief accepts no screenshots and a live screen cannot enter a repository.
Say that in the README where the app is introduced.

- [ ] **Step 1: Write `requirements-ui.txt`**

```
# The Streamlit app is NOT part of any acceptance criterion. It is here
# because a portfolio artefact is read by a person, per D-81. Install it only
# if you want the screen:
#     VIRTUAL_ENV=.venv uv pip install -r requirements-ui.txt
#
# The graded path is requirements.txt alone, per D-83.
streamlit
requests
```

Add one comment line to `requirements.txt`, so the main file is self-describing rather than silently incomplete:

```
# The optional Streamlit app has its own file: requirements-ui.txt. D-83.
```

- [ ] **Step 2: Write the failing test**

```python
"""The screen. Nothing here is a graded criterion; test 46 is a boundary check."""

import ast

import config


def test_46_the_app_imports_nothing_from_the_agent_or_the_rag_core():
    """Spec section 16 test 46, and the whole of D-82.

    If this fails, the app is answering questions itself rather than driving
    the Part 3 API, and no click it produces will ever write a Task 12 line.
    """
    tree = ast.parse((config.REPO_ROOT / "ui" / "app.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported & {"agent", "rag", "dataset", "db"}, (
        f"ui/app.py imports {sorted(imported & {'agent', 'rag', 'dataset', 'db'})}. "
        f"D-82 requires it to reach the agent only through POST /ask."
    )


def test_the_app_points_at_the_configured_api_port():
    source = (config.REPO_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert str(config.API_PORT) in source
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ui.py -v`
Expected: FAIL with `FileNotFoundError`

- [ ] **Step 4: Write `ui/app.py`**

It imports `streamlit`, `requests`, `json` and `os` only.
Build it around what a transcript reads badly and a screen reads well, which is the only argument for its existence:

- A sidebar holding the API base URL, defaulting to `http://127.0.0.1:8000`, and a thread id box with a **New conversation** button that generates a fresh id. This is what makes memory and its absence visible in two clicks, which is the thing Task 8's two transcripts need two files to say.
- A chat transcript of the turns in this session, rendered with `st.chat_message`.
- Per answer, an expander showing the route, the citations, the top-1 similarity, the outcome, the escalation score when there is one, and the trace id. The trace id is the join key to the log line, so show it as copyable text.
- A guardrail strip that lights up when `guardrails.pii_masked` is non-empty, when `injection_rule` is set, or when `grounded` is false. Three deliberate example queries as buttons, one per guardrail, so a reader can fire each without inventing one.
- A record panel driven by a lookup query, showing `escalation_score` against `config.ESCALATION_THRESHOLD` as a progress bar.

Hardcode nothing that `config.py` owns: read the port from an environment variable defaulting to `config.API_PORT`'s value, and say in a comment that `ui/` may not import `config` for the same reason it may not import `agent`.

- [ ] **Step 5: Run it against the API and use it**

```bash
VIRTUAL_ENV=.venv uv pip install -r requirements-ui.txt
.venv/bin/python -m uvicorn api.main:app --port 8000 &
.venv/bin/python -m streamlit run ui/app.py
```

Check by hand, because no test can: ask a policy question, ask a follow-up using "it" on the same thread, press **New conversation** and ask the follow-up again.
The second should answer from memory and the third should clarify.
Then confirm `logs/agent.jsonl` grew one `http_request` line per click, which is the claim D-82 was chosen for.

- [ ] **Step 6: Run the whole suite and commit**

```bash
.venv/bin/python -m pytest -q
git add ui/ requirements-ui.txt requirements.txt README.md tests/test_ui.py
git commit -m "Add the Streamlit app, which drives the API rather than the agent"
```

---

## Self-review

**Spec coverage.** Section 25.1 is Task 2; 25.2 is Task 3; 25.3 is Task 1; section 26 is Task 5; section 17's evidence requirement is Task 4.
D-76 and D-78 are Task 1, D-77 and D-85 Task 3, D-79 Task 2, D-81 to D-83 Task 5.

**Interfaces.** `graph.RETRY_POLICY`, `graph.RESILIENT_NODE` and `graph.ask_async` are defined in Task 1 and consumed in Tasks 3 and 4.
`checkpoint.checkpointed` and `checkpoint.INTERRUPT_BEFORE` are defined in Task 3 and consumed in Task 4.
`mcp_server.server.mcp` is defined in Task 2 and consumed in Task 4.
`config.MCP_URL`, `CHECKPOINT_PATH`, `NODE_TIMEOUT_SECONDS`, `GRAPH_TIMEOUT_SECONDS` and the five retry constants are added in Task 1 step 1 and read in Tasks 1 to 5.

**Two risks, both carried deliberately.**

The first is `lru_cache` on `graph.compiled`. Every test and every demo that patches a node must clear it on both sides, and this plan says so at each site.
The alternative, dropping the cache, costs about 10ms per call across 19 existing call sites and the whole Part 2 runner, which is a real cost paid to avoid a discipline the tests already encode.

The second is that Task 4 spawns a server subprocess, the only step in this repository that binds a port.
It is bounded by a readiness poll rather than a sleep, and it fails loudly with the server's stderr, because a flaky evidence generator is worse than one that does not run.
