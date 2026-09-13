"""Task 7. Wiring, and the single call Parts 3 and 4 consume.

This module is deliberately thin. Every decision lives in agent/nodes.py as a
pure function, so what is left here is topology: nine nodes, two conditional
edges, and one entry point.

Part 4 Task 15 attaches a SQLite checkpointer at compile() time and Part 4
Task 16 attaches a retry policy to a node, so both slot into build_graph()
without reshaping anything.
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

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


def _guard_branch(state: AgentState) -> str:
    """The first conditional edge: refuse, or carry on."""
    return "refuse" if state.get("injection_rule") else "continue"


def build_graph() -> StateGraph:
    """The graph, uncompiled, so a caller can attach a checkpointer."""
    builder = StateGraph(AgentState)

    for name in nodes.NODE_NAMES:
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


def ask(query: str, thread_id: str = "default") -> dict:
    """One turn. Returns the validated response payload.

    Part 3 Task 11 puts both its endpoints behind this call.
    """
    from agent import memory

    turn = len(memory.load(thread_id).turns) + 1
    final = compiled().invoke(new_state(query, thread_id, turn))
    return final["response"]
