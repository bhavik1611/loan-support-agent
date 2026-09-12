"""The language-model seam. MOCK_LLM by default, Groq behind LLM_PROVIDER.

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

import json
import os
import re
import urllib.error
import urllib.request

import config
import obs

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


class ProviderError(RuntimeError):
    """The provider failed. Deliberately not a refusal.

    Per D-61 nothing here degrades to the mock provider: output that claims one
    provenance and has another is the failure the ground rules exist to prevent.
    Retries and timeouts are Part 4's resilience task and wrap this.
    """


def _groq_request(system: str, user: str) -> dict:
    """POST the prompt and return the decoded payload, or raise ProviderError."""
    key = os.environ.get(config.GROQ_API_KEY_VAR)
    if not key:
        raise ProviderError(
            f"LLM_PROVIDER={config.PROVIDER_GROQ!r} needs {config.GROQ_API_KEY_VAR} "
            f"and it is not set. Put it in .env, or set LLM_PROVIDER="
            f"{config.DEFAULT_PROVIDER!r} to run offline."
        )

    body = json.dumps(
        {
            "model": config.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "seed": config.SEED,
            "max_completion_tokens": config.GROQ_MAX_TOKENS,
        }
    ).encode()

    request = urllib.request.Request(
        config.GROQ_BASE_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # Not decoration. Groq's edge returns 403 for urllib's default
            # "Python-urllib/3.12" and 200 for any explicit agent; measured with
            # the same body from the same process, changing only this line.
            "User-Agent": config.GROQ_USER_AGENT,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=config.GROQ_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise ProviderError(f"Groq returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Groq unreachable: {exc.reason}") from exc


def _generate_groq(system: str, user: str) -> str:
    """The real provider. Every failure raises; none becomes a refusal."""
    with obs.timed("llm.generate", provider=config.PROVIDER_GROQ, model=config.GROQ_MODEL) as line:
        payload = _groq_request(system, user)

        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Groq returned no choices: {str(payload)[:200]}") from exc

        usage = payload.get("usage") or {}
        line["prompt_tokens"] = usage.get("prompt_tokens")
        line["completion_tokens"] = usage.get("completion_tokens")
        line["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        )
        line["finish_reason"] = choice.get("finish_reason")

        # A reasoning model bills its thinking against the same budget, so a
        # budget set too low returns empty content with finish_reason "length".
        # rag/generate.py reads an empty string as "the context held nothing
        # usable", which would record a transport problem as a principled
        # refusal. Both cases raise instead (D-61).
        if choice.get("finish_reason") == "length":
            raise ProviderError(
                f"Groq truncated the completion at {config.GROQ_MAX_TOKENS} tokens "
                f"({line['reasoning_tokens']} of them reasoning). Raise "
                f"config.GROQ_MAX_TOKENS; an empty answer is not a refusal."
            )

        # The reasoning channel is never part of the answer.
        text = (message.get("content") or "").strip()
        if not text:
            raise ProviderError(
                "Groq returned empty content. That is a provider failure, not a "
                "refusal, so it is not passed off as one."
            )

        line["answer_chars"] = len(text)
        return text


def generate(system: str, user: str) -> str:
    """The single generation call. Part 3's judge calls this too."""
    provider = config.resolve_provider()
    if provider == config.DEFAULT_PROVIDER:
        return _generate_mock(system, user)
    if provider == config.PROVIDER_GROQ:
        return _generate_groq(system, user)
    raise NotImplementedError(
        f"LLM_PROVIDER={provider!r} is not a provider this repository implements. "
        f"Use {config.DEFAULT_PROVIDER!r} for the offline default, or "
        f"{config.PROVIDER_GROQ!r}."
    )
