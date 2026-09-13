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
        time.sleep(config.NODE_TIMEOUT_SECONDS + 0.2)
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
    """Spec section 16 test 45, Part 4 criterion 3c.

    Exercises the shipped path: config.GRAPH_TIMEOUT_SECONDS wraps ainvoke
    inside ask_async itself (agent/graph.py), not a private wait_for built
    here to imitate it - the same shape as test_44 monkeypatching
    NODE_TIMEOUT_SECONDS rather than constructing its own timeout.

    Elapsed time is measured inside the loop, at the instant ask_async's own
    wait_for raises, not around asyncio.run(). asyncio.run's own teardown
    calls shutdown_default_executor(), which waits for the orphaned worker
    thread from the timed-out to_thread call to finish before the loop
    closes - a real, separately-documented cost (see ask()'s docstring), but
    one that has nothing to do with whether the budget itself was honoured.
    A bound taken outside the loop cannot tell a tight cancellation from a
    slow one, so it would pass for the wrong reason.
    """
    from agent import tools

    def slow(record_id: str):
        time.sleep(1)
        raise AssertionError("the global timeout should have fired first")

    monkeypatch.setattr(tools, "check_loan_application_status", slow)
    monkeypatch.setattr(config, "GRAPH_TIMEOUT_SECONDS", 0.2)
    graph.compiled.cache_clear()

    async def run():
        started = time.perf_counter()
        try:
            await graph.ask_async("What is the status of LN-1042?", thread_id="global")
        except (asyncio.TimeoutError, TimeoutError):
            return time.perf_counter() - started
        raise AssertionError("the global budget should have been exceeded")

    try:
        elapsed = asyncio.run(run())
    finally:
        graph.compiled.cache_clear()

    assert elapsed < 1.0, (
        f"the global timeout fired after {elapsed:.3f}s, not at its "
        f"{config.GRAPH_TIMEOUT_SECONDS}s budget"
    )
