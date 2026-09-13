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

# A boundary a same-sentence antecedent can sit behind: a coordinating
# conjunction or a comma closing off an earlier clause. Splitting on these,
# rather than scoring the sentence as one bag of words, is what tells
# "...transaction and will it affect..." (EQ-11, an antecedent one clause
# back) apart from "Is it flagged...?" or "And its status?" (nothing but
# filler, or nothing at all, before the cue).
_CLAUSE_BREAK = re.compile(r"\b(?:and|but|or|so|yet)\b|,|;")


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


_CUE_PATTERNS = tuple(
    (cue, re.compile(r"\b" + re.escape(cue) + r"\b")) for cue in ELLIPSIS_CUES
)


def _earliest_cue_start(lowered: str) -> int | None:
    """The start index of whichever cue in `ELLIPSIS_CUES` appears first, or None."""
    starts = [match.start() for _, pattern in _CUE_PATTERNS for match in pattern.finditer(lowered)]
    return min(starts) if starts else None


def _has_earlier_antecedent_clause(lowered: str, cue_start: int) -> bool:
    """Whether a clause before the cue's own clause introduces a noun phrase.

    Splits everything before the cue on a coordinating conjunction, comma or
    semicolon. The piece immediately holding the cue is the last split; any
    piece before that is a candidate antecedent clause, and it counts only if
    it holds one of `config.NOUN_PHRASE_MARKERS` - a determiner, possessive
    or quantifier. That closed class is what carries "a noun phrase starts
    here", which a bare word count does not: filler like "is" or "will" sits
    in the cue's own clause and was never a candidate, but a clause with
    words and no marker in it - "Loans affect credit scores" - has no
    determiner naming the thing "they" would refer to either, and this
    function correctly does not count it.
    """
    clauses = _CLAUSE_BREAK.split(lowered[:cue_start])
    earlier_clauses = clauses[:-1]
    return any(
        any(word in config.NOUN_PHRASE_MARKERS for word in _WORD.findall(clause))
        for clause in earlier_clauses
    )


def needs_resolution(query: str) -> bool:
    """Whether this question leans on the turn before it.

    A query naming its own record id never needs resolution, whatever else it
    contains, so that check comes first.

    Past that, the signal is a cue word's position relative to a noun-phrase
    marker, not merely the cue's presence: a cue needs resolution only when
    no earlier clause in the same sentence - split on a coordinating
    conjunction, comma or semicolon - already contains a determiner,
    possessive or quantifier (`config.NOUN_PHRASE_MARKERS`), the closed class
    of words that in English overwhelmingly introduces the noun phrase a
    pronoun would point back to. "Is it flagged for fraud?" and "And its
    status?" have no earlier clause at all; "Thanks, and is it approved?" has
    one, but "thanks" carries no marker, so it still resolves from the turn
    before. EQ-11, "How do I report a fraudulent card transaction and will it
    affect my credit score?", has "a fraudulent card transaction" one clause
    earlier - "a" is a marker - so it does not.

    Measured three ways against eval/routing.py::ELLIPTICAL_PROBES (14) and
    ::SELF_CONTAINED_PROBES (12, including EQ-11 and three bare-plural
    probes): a bare cue-word match scored 8/8 and 0/9 (it never saw an
    opener-led or bare-plural probe - that blind spot is what let it ship
    uncalibrated); a clause break with no antecedent check scored 8/14 and
    9/9, fixing EQ-11's shape but reading every conversational opener as
    content; a clause break with a curated filler-word list scored 13/14 and
    9/9, fixing the openers it had a word for but leaving "Well, what about
    it?" uncaught, because a list of interjections has no edge - it is only
    ever as complete as what someone thought to add. This rule, checking for
    a noun-phrase marker instead of any word, scores 14/14 and 9/12 - "Well,
    what about it?" and every other opener now correctly resolve with no
    special case, because none of them carries a determiner.

    The stated residue is a named grammatical gap rather than an open list: a
    bare plural or mass noun takes no determiner in English, so a clause
    whose only antecedent is one - "Loans affect credit scores, do they
    not?", "Interest accrues monthly and does it compound?", "Fraud reviews
    take time, so how long do they run?" - reads as having no antecedent and
    is wrongly treated as elliptical. These three are
    eval/routing.py::SELF_CONTAINED_KNOWN_UNCAUGHT, recorded rather than
    chased: catching them needs a part-of-speech signal this repository does
    not have under MOCK_LLM, not a bigger word list.
    """
    if _RECORD_ID.search(query):
        return False
    lowered = query.lower()
    cue_start = _earliest_cue_start(lowered)
    if cue_start is None:
        return False
    return not _has_earlier_antecedent_clause(lowered, cue_start)
