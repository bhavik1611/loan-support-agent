"""Task 7. Wiring, and the single call Parts 3 and 4 consume.

This module is deliberately thin. Every decision lives in agent/nodes.py as a
pure function, so what is left here is topology: nine nodes, two conditional
edges, and one entry point.

Part 4 Task 15 attaches a SQLite checkpointer at compile() time and Part 4
Task 16 attaches a retry policy to a node, so both slot into build_graph()
without reshaping anything.
"""

import asyncio
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

import config
from agent import nodes
from agent.state import AgentState, new_state

# Where the injection check sends a turn, and where it does not.
_GUARD_BRANCHES = {"refuse": "refuse", "continue": "recall"}

# What the intent edge may return. The `both` case returns a two-element
# sequence instead, which LangGraph runs in one superstep.
_ROUTE_BRANCHES = {
    "policy_answer": "policy_answer",
    "lookup_status": "lookup_status",
    "clarify": "clarify",
}

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


def _guard_branch(state: AgentState) -> str:
    """The first conditional edge: refuse, or carry on."""
    return "refuse" if state.get("injection_rule") else "continue"


def build_graph() -> StateGraph:
    """The graph, uncompiled, so a caller can attach a checkpointer."""
    builder = StateGraph(AgentState)

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

    builder.add_edge(START, "guard_input")
    builder.add_conditional_edges("guard_input", _guard_branch, _GUARD_BRANCHES)
    builder.add_edge("recall", "route")
    builder.add_conditional_edges("route", nodes.pick_branch, _ROUTE_BRANCHES)

    for name in ("policy_answer", "lookup_status", "clarify"):
        builder.add_edge(name, "verify")

    builder.add_edge("refuse", "compose")
    builder.add_edge("verify", "compose")
    builder.add_edge("compose", END)
    return builder


@lru_cache(maxsize=1)
def compiled():
    """The compiled graph. Cached: compiling costs about 10ms."""
    return build_graph().compile()


def node_names() -> list[str]:
    """The nine nodes. LangGraph adds __start__ and __end__, which are not ours."""
    return [
        name
        for name in compiled().get_graph().nodes
        if not name.startswith("__")
    ]


async def ask_async(query: str, thread_id: str = "default") -> dict:
    """One turn, on the async path. For callers that already have a loop.

    Carries the whole-run budget, config.GRAPH_TIMEOUT_SECONDS, via
    asyncio.wait_for around ainvoke, so ask() inherits it and so does any
    caller that already has a loop and calls this directly. Read at call
    time rather than bound as a default argument, so a monkeypatched budget
    in a test takes effect - the same property NODE_TIMEOUT_SECONDS relies
    on at the per-node level.
    """
    from agent import memory

    turn = len(memory.load(thread_id).turns) + 1
    final = await asyncio.wait_for(
        compiled().ainvoke(new_state(query, thread_id, turn)),
        timeout=config.GRAPH_TIMEOUT_SECONDS,
    )
    return final["response"]


def ask(query: str, thread_id: str = "default") -> dict:
    """One turn. Returns the validated response payload.

    The signature is unchanged from Part 2 and all 19 existing call sites are
    untouched. The graph underneath is driven by ainvoke, because the per-node
    timeout of Task 16 exists only on the async path.

    asyncio.run raises inside a running event loop, so a caller that already
    has one must use ask_async. api/main.py declares its endpoints `def` for
    exactly this reason, which puts them in FastAPI's threadpool.

    A cost this wrapping carries, and does not hide: asyncio.run() tears
    down its loop by calling shutdown_default_executor(), which waits for
    every thread it handed work to, including one from a to_thread call a
    NODE_TIMEOUT_SECONDS or GRAPH_TIMEOUT_SECONDS timeout has already fired
    on and abandoned. CPython cannot kill a running thread, so that worker
    keeps executing the original blocking call to completion; ask() cannot
    return to its caller until it does. Both timeouts still bound the
    graph's own execution exactly as documented - the abandoned attempt's
    result is discarded either way - but they do not bound ask()'s wall
    clock past that point. A private executor torn down without waiting, or
    a process pool that actually could be killed, would remove this, at a
    cost of real complexity bought for a local SQLite read of single-digit
    milliseconds. Not worth it here; see tests/test_resilience.py for the
    measured shape of the wait.
    """
    return asyncio.run(ask_async(query, thread_id))
