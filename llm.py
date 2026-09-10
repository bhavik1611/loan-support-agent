"""The language-model seam. MOCK_LLM by default, a real provider behind LLM_PROVIDER.

Under MOCK_LLM the provider is deterministic template synthesis: classify the
question shape, pick a template, and fill it from sentences that appear in the
retrieved context and nowhere else.

Pure extractive stitching lost because groundedness would then score 1.0 on all
fifteen Part 3 queries by construction and the RAG triad would demonstrate
nothing. A small local generative model lost because it needs a weight
download, which breaks the zero-network-access rule.

Template selection can pick the wrong shape. That is deliberate: it is what
makes Part 3's context relevance and answer relevance vary across the fifteen
queries instead of being 1.0 by construction.
"""

import re

import config

# Checked in order, first match wins. Order encodes precedence: a question
# that asks both "which documents" and "how do I" is a document question.
_SHAPE_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("documents", ("document", "papers", "proof", "submit", "paperwork", "kyc form")),
    ("duration", ("how long", "how many days", "when will", "turnaround", "timeline", "take to")),
    (
        "amount_or_rate",
        ("how much", "rate", "interest", "fee", "charge", "penalty", "cost",
         "amount", "limit", "percent", "balance"),
    ),
    ("eligibility", ("eligible", "eligibility", "qualify", "who can", "can i", "criteria", "income")),
    ("process", ("how do i", "how can i", "process", "steps", "procedure", "what happens", "report", "raise")),
]

_TEMPLATES = {
    "definition": "Here is what the knowledge base says. {body}",
    "eligibility": "On eligibility: {body}",
    "amount_or_rate": "On amounts, rates and charges: {body}",
    "process": "The process works as follows. {body}",
    "documents": "You will need the following. {body}",
    "duration": "On timing: {body}",
}

# Cue words used to pick which retrieved sentences fill the template. A shape
# whose cues match nothing falls back to the first sentences in rank order.
_SENTENCE_CUES = {
    "definition": ("is", "means", "refers", "allows"),
    "eligibility": ("eligible", "eligibility", "qualify", "minimum", "criteria", "income", "score"),
    "amount_or_rate": ("rupees", "percent", "fee", "charge", "rate", "lakh", "crore", "waived"),
    "process": ("must", "submit", "request", "branch", "within", "issued", "processed"),
    "documents": ("document", "proof", "pan", "aadhaar", "passport", "copy", "statement"),
    "duration": ("days", "months", "years", "within", "working day", "cycle"),
}

_QUESTION = re.compile(r"^QUESTION:\s*(.*)$", re.MULTILINE)
_SOURCE = re.compile(r"^\[([a-z0-9\-]+)\]\s*(.+)$", re.MULTILINE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

MAX_SENTENCES = 3


def parse_prompt(user: str) -> tuple[str, list[tuple[str, str]]]:
    """Recover the question and the (doc_id, chunk text) pairs from the prompt."""
    match = _QUESTION.search(user)
    question = match.group(1).strip() if match else ""
    sources = [(m.group(1), m.group(2).strip()) for m in _SOURCE.finditer(user)]
    return question, sources


def classify_shape(question: str) -> str:
    """The question shape, by keyword. Wrong picks are expected and deliberate."""
    lowered = question.lower()
    for shape, cues in _SHAPE_CUES:
        if any(cue in lowered for cue in cues):
            return shape
    return "definition"


def _select_sentences(shape: str, sources: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Rank every context sentence by cue overlap, keeping retrieval order as the tiebreak."""
    cues = _SENTENCE_CUES[shape]
    scored = []
    for rank, (doc_id, text) in enumerate(sources):
        for position, sentence in enumerate(s.strip() for s in _SENTENCE.split(text)):
            if len(sentence) < 20:
                continue
            lowered = sentence.lower()
            score = sum(1 for cue in cues if cue in lowered)
            scored.append((-score, rank, position, doc_id, sentence))
    scored.sort()

    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, _, _, doc_id, sentence in scored:
        if sentence in seen:
            continue
        seen.add(sentence)
        chosen.append((doc_id, sentence))
        if len(chosen) == MAX_SENTENCES:
            break
    return chosen


def _generate_mock(system: str, user: str) -> str:
    question, sources = parse_prompt(user)
    if not sources:
        return ""

    shape = classify_shape(question)
    chosen = _select_sentences(shape, sources)
    if not chosen:
        return ""

    body = " ".join(sentence for _, sentence in chosen)
    cited = sorted({doc_id for doc_id, _ in chosen})
    citations = " ".join(f"[{doc_id}]" for doc_id in cited)
    return f"{_TEMPLATES[shape].format(body=body)}\n\nSources: {citations}"


def generate(system: str, user: str) -> str:
    """The single generation call. Part 3's judge calls this too."""
    provider = config.resolve_provider()
    if provider == config.DEFAULT_PROVIDER:
        return _generate_mock(system, user)
    raise NotImplementedError(
        f"LLM_PROVIDER={provider!r} is declared by the brief as optional and is not "
        f"wired in V1. Every acceptance criterion is demonstrated under MOCK_LLM. "
        f"Unset LLM_PROVIDER or set it to {config.DEFAULT_PROVIDER!r}."
    )
