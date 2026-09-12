"""Task 10. Input-side masking and detection, output-side groundedness.

mask_pii is a contract rather than a helper: Part 3 Task 12 runs the same
function over what it writes to disk, because the brief requires that a
fixed-format PII field never reach a log in the clear.

Aadhaar and account numbers overlap by format. An Aadhaar is twelve digits;
an account number is eleven to sixteen, so nine of the sixty-six generated
customers collide exactly. The mask therefore keys on format alone and is
fail-safe, and the Verhoeff check digit only chooses which label to print.
One of those nine also passes Verhoeff, so the label is wrong once in
sixty-six and the redaction is never wrong. That is the trade D-35 makes.
"""

import re

import config
from db.generate import is_valid_aadhaar
from rag import retrieve

_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

# Grouped form first, because the bare-digits pattern would otherwise match
# only the first group of four and leave the rest of the number in the clear.
_GROUPED_12 = re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b")
_BARE_DIGITS = re.compile(
    rf"\b\d{{{config.ACCOUNT_DIGITS_MIN},{config.ACCOUNT_DIGITS_MAX}}}\b"
)


def _label_for(digits: str) -> str:
    """AADHAAR when the check digit agrees, ACCOUNT otherwise. Both are masked."""
    return "AADHAAR" if is_valid_aadhaar(digits) else "ACCOUNT"


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Replace every fixed-format PII value, and name the rules that fired.

    Returns the masked text and the sorted distinct rule names, so a caller
    can report which guardrail fired without re-running the patterns.
    """
    fired: set[str] = set()

    def _pan(match: re.Match) -> str:
        fired.add("PAN")
        return config.PII_PLACEHOLDERS["PAN"]

    def _digits(match: re.Match) -> str:
        bare = re.sub(r"[ -]", "", match.group(0))
        label = _label_for(bare)
        fired.add(label)
        return config.PII_PLACEHOLDERS[label]

    masked = _PAN.sub(_pan, text)
    masked = _GROUPED_12.sub(_digits, masked)
    masked = _BARE_DIGITS.sub(_digits, masked)
    return masked, sorted(fired)


# Checked in order, first match wins, so detection is deterministic when a
# query trips two rules at once. Each name is returned to the caller and
# printed in the refusal, because a guardrail that cannot say what it caught
# cannot be demonstrated firing.
INJECTION_RULES: tuple[tuple[str, re.Pattern], ...] = (
    (
        # Three alternatives, because "Disregard the above" carries no noun and
        # a single pattern broad enough to catch it also catches "Can I ignore
        # the minimum balance rule?", which is an ordinary customer question.
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^.]{0,30}"
            r"\b(?:previous|prior|earlier|above|all)\b[^.]{0,20}"
            r"\b(?:instruction|instructions|prompt|prompts|rule|rules|message|messages|context)\b"
            r"|\b(?:ignore|disregard|forget|override)\s+(?:the\s+)?(?:above|previous|prior|earlier)\b"
            r"|\b(?:ignore|disregard|forget|override)\b[^.]{0,30}"
            r"\byour\s+(?:instruction|instructions|prompt|rules)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\b(you are now|act as|pretend to be|from now on you|"
            r"you must now behave)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "exfiltration",
        re.compile(
            r"\b(reveal|print|show|repeat|output|dump)\b[^.]{0,30}"
            r"\b(system prompt|your instructions|your prompt|your rules)\b",
            re.IGNORECASE,
        ),
    ),
    (
        # llm.py recovers its own prompt with ^\[([a-z0-9\-]+)\] applied to the
        # whole user string, so a forged source line in a query would be read
        # back as retrieved context. This rule exists for that surface.
        "delimiter_injection",
        re.compile(r"(?m)(^|\s)(CONTEXT:|QUESTION:)|\[kb-\d{2}\]", re.IGNORECASE),
    ),
)


def detect_injection(text: str) -> str | None:
    """The name of the first rule that matches, or None."""
    for name, pattern in INJECTION_RULES:
        if pattern.search(text):
            return name
    return None


def check_grounded(
    supported: bool, citations, retrieved_doc_ids
) -> str | None:
    """The output side. Returns the rule that fired, or None if the answer stands.

    Takes plain values rather than a GroundedAnswer on purpose: these three
    are what the graph carries in state, and Part 4 Task 15 attaches a SQLite
    checkpointer that serialises state. A dataclass holding Hit objects would
    not survive that round trip, so nothing unserialisable ever goes in.

    `unsupported` delegates to Part 1's decision, which produced `supported`,
    rather than restating the threshold and the shared-parent rule here.

    `phantom_citation` is beyond the brief. Under MOCK_LLM the generator
    builds its citation list from the chunks it was handed, so this cannot
    currently fire; it exists because the moment a real provider is wired in
    behind LLM_PROVIDER a fabricated citation becomes possible, and this is
    the check that catches it.
    """
    if not supported:
        return "unsupported"
    retrieved = set(retrieved_doc_ids)
    if any(doc_id not in retrieved for doc_id in citations):
        return "phantom_citation"
    return None


def grounded_rule_for(answer) -> str | None:
    """check_grounded applied to a GroundedAnswer, for callers holding one."""
    return check_grounded(
        answer.supported, answer.citations, retrieve.parent_documents(list(answer.hits))
    )
