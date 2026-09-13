"""Task 7. The graph's only channel schema.

Every node reads this and returns a fragment of it. Keeping the shape in one
place is what makes the fan-out of section 11.2 safe to reason about: two
nodes run in the same superstep and the only question that matters is whether
they write the same key, which is answerable by reading this file.
"""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """One turn, from raw query to validated response."""

    # Identity, set before the graph runs.
    thread_id: str
    turn: int
    query: str

    # guard_input
    masked_query: str
    pii_masked: list[str]
    injection_rule: str | None

    # recall
    history: list[dict]
    entities: dict
    clarify_used: bool

    # route
    route: str
    record_id: str | None
    route_scores: dict[str, float]
    route_reason: str

    # the branches. policy_answer and lookup_status write one key each, and
    # they are different keys, which is why running both concurrently is safe.
    policy: dict | None
    lookup: dict | None
    clarification: str | None

    # verify
    grounded: bool | None
    output_rule: str | None

    # compose
    response: dict[str, Any] | None


def new_state(query: str, thread_id: str, turn: int) -> AgentState:
    """The starting state for one turn."""
    return AgentState(query=query, thread_id=thread_id, turn=turn)
