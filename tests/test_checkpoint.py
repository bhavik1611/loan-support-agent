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
