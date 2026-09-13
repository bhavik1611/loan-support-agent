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
    """Answer one question, through the full nine-node graph.

    `ask` wraps asyncio.run (D-78). A slow tool call ties up this threadpool
    worker for its full duration even after the graph's own timeout has
    fired, because asyncio.run's shutdown waits for every worker thread it
    handed work to and CPython cannot kill a running thread. The graph's
    timeouts bound the graph, not this endpoint's wall clock. This is a known,
    documented cost, not something fixed here.
    """
    return ask(request.query, thread_id=request.thread_id)
