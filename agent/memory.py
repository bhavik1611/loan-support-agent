"""Task 8. Conversation history, and the entity slot that makes it do work.

The brief asks for history persisted to a JSON file. A bare turn log would
satisfy that literally while demonstrating nothing: the agent would behave
identically whether the log were full or empty, and the brief also wants a
transcript showing state correctly absent.

So the store carries one resolved entity, `last_record_id`. The same
second-turn question then routes to lookup on a warm thread and to clarify on
a fresh one, which is a difference a reader can see. That is D-37.

This is not Part 4's checkpointer. This file is the conversation, readable
and diffable; the SQLite checkpointer in Task 15 is graph execution state for
resuming a half-finished run. Both key on thread_id and neither reads the
other.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import config

# Words that make a question depend on the turn before it. Deliberately short:
# a false positive costs one clarify, a false negative costs a wrong answer.
ELLIPSIS_CUES = (
    "it", "its", "that one", "this one", "the same", "them", "they", "those",
)

_RECORD_ID = re.compile(r"\bLN-\d{4}\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z']+")


@dataclass
class Thread:
    """One conversation. `entities` holds what later turns may refer back to."""

    thread_id: str
    turns: list[dict] = field(default_factory=list)
    entities: dict = field(default_factory=dict)


def path_for(thread_id: str, root: Path | None = None) -> Path:
    root = config.CONVERSATION_DIR if root is None else root
    return root / f"{thread_id}.json"


def load(thread_id: str, root: Path | None = None) -> Thread:
    """The stored thread, or an empty one. A missing file is a fresh thread."""
    path = path_for(thread_id, root)
    if not path.exists():
        return Thread(thread_id=thread_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Thread(
        thread_id=payload["thread_id"],
        turns=payload.get("turns", []),
        entities=payload.get("entities", {}),
    )


def save(thread: Thread, root: Path | None = None) -> Path:
    """Write the thread. Sorted keys and a fixed indent, so bytes are stable."""
    path = path_for(thread.thread_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
        {
            "thread_id": thread.thread_id,
            "turns": thread.turns,
            "entities": thread.entities,
        },
        indent=2,
        sort_keys=True,
    )
    path.write_text(body + "\n", encoding="utf-8")
    return path


def record_turn(
    thread: Thread,
    query: str,
    route: str,
    answer: str,
    record_id: str | None = None,
) -> Thread:
    """Append one turn, and update the entity slot when the turn resolved an id."""
    thread.turns.append(
        {
            "turn": len(thread.turns) + 1,
            "query": query,
            "route": route,
            "answer": answer,
            "record_id": record_id,
        }
    )
    if record_id:
        thread.entities["last_record_id"] = record_id
    return thread


def last_route(thread: Thread) -> str | None:
    """The route the previous turn took, which the clarify cap in D-45 reads."""
    return thread.turns[-1]["route"] if thread.turns else None


def needs_resolution(query: str) -> bool:
    """Whether this question leans on the turn before it.

    A query naming its own record id never needs resolution, whatever else it
    contains, so that check comes first.
    """
    if _RECORD_ID.search(query):
        return False
    lowered = query.lower()
    words = set(_WORD.findall(lowered))
    return any(
        cue in words if " " not in cue else cue in lowered for cue in ELLIPSIS_CUES
    )
