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
