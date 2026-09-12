"""Task 7. Which tool a query wants, decided cheapest-signal-first.

Two stages, per D-44 and D-45.

1. A record id is decisive, because it is evidence rather than a similarity
   judgement. With policy language beside it the route is `both`; alone it is
   `lookup`. An elliptical query resolves its id from the entity slot.
2. Otherwise the query is embedded with the model already loaded for
   retrieval and scored against three centroids. The highest wins, and
   `vague` winning is what routes to `clarify`.

There is no margin and no threshold, and that is a result rather than an
omission. An earlier design scored two centroids and clarified when the top
two were within a calibrated margin. Measured over these probes, the minimum
labelled margin was 0.0009 and the maximum vague margin 0.2069: a gap of
minus 0.2060, because with two classes the margin measures which way a query
leans, not how sure the router is. A third centroid models the thing being
detected instead of inferring it. D-44 records this.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

from agent import memory
from rag import index
import obs

RECORD_ID = re.compile(r"\bLN-\d{4}\b", re.IGNORECASE)

# Words that mean the asker also wants the rule, not only the record.
POLICY_CUES = (
    "policy", "rule", "rules", "eligibility", "eligible", "criteria",
    "how long", "sla", "timeline", "turnaround", "usually", "normally",
    "process", "penalty", "charge", "fee", "interest rate",
    "requirement", "requirements", "why does", "how does",
)

INTENT_EXEMPLARS: dict[str, list[str]] = {
    "policy": [
        "What are the eligibility criteria for this loan product?",
        "How is the equated monthly instalment calculated?",
        "Which documents satisfy the know your customer requirement?",
        "What interest rate band applies to this category of borrowing?",
        "What penalty applies when a loan is repaid ahead of schedule?",
        "What is the process for disputing a fraudulent transaction?",
        "What minimum balance must the account hold?",
        "Which factors affect a credit score?",
        # These two carry the timing language that would otherwise read as a
        # status question: without them, "How long does a fraud dispute take
        # to settle?" wins on the lookup centroid by 0.0009.
        "How many working days does the bank take to resolve a dispute?",
        "What is the standard turnaround time for this procedure?",
    ],
    "lookup": [
        "What is the current status of my loan application?",
        "Has my application been approved or rejected yet?",
        "How much money was sanctioned against my file?",
        "Which stage of assessment is my application sitting at?",
        "Has my application been flagged for review?",
        "When was my application submitted and how old is it now?",
        "Tell me where my request has reached.",
        "Is there a decision on my file yet?",
    ],
    # The clarify trigger. Modelling "I cannot tell what you want" directly
    # beats inferring it from the gap between the other two.
    "vague": [
        "Can you help me with something?",
        "I have a question.",
        "What should I do?",
        "Tell me more.",
        "Give me some information.",
        "I need assistance please.",
        "Tell me about it.",
        "What are my options?",
        "Anything you can tell me?",
        "Help.",
    ],
}


@dataclass(frozen=True)
class RouteDecision:
    """What the router chose, and enough of why for the transcript to show it."""

    route: str
    record_id: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""


def find_record_id(query: str) -> str | None:
    """The record id in the query, upper-cased, or None."""
    match = RECORD_ID.search(query)
    return match.group(0).upper() if match else None


def has_policy_language(query: str) -> bool:
    """Whether the asker also wants the rule, not only the record."""
    lowered = query.lower()
    return any(cue in lowered for cue in POLICY_CUES)


@lru_cache(maxsize=1)
def centroids() -> dict[str, tuple[float, ...]]:
    """One unit-length mean vector per intent. Cached: embedding costs about 0.3s."""
    result = {}
    for intent, exemplars in INTENT_EXEMPLARS.items():
        vectors = index.embed(exemplars)
        mean = [sum(column) / len(vectors) for column in zip(*vectors)]
        norm = sum(value * value for value in mean) ** 0.5
        result[intent] = tuple(value / norm for value in mean)
    return result


def intent_scores(query: str) -> dict[str, float]:
    """Cosine similarity to each centroid. Both sides are unit vectors."""
    vector = index.embed([query])[0]
    return {
        intent: round(sum(a * b for a, b in zip(vector, centroid)), 4)
        for intent, centroid in centroids().items()
    }


def classify(query: str, entities: dict, clarify_used: bool = False) -> RouteDecision:
    """The route for this query, given what the thread already knows."""
    decision = _classify(query, entities, clarify_used)
    obs.event(
        "agent.route",
        route=decision.route,
        reason=decision.reason,
        top_score=max(decision.scores.values()) if decision.scores else None,
    )
    return decision


def _classify(query: str, entities: dict, clarify_used: bool = False) -> RouteDecision:
    record_id = find_record_id(query)

    if record_id is None and memory.needs_resolution(query):
        resolved = entities.get("last_record_id")
        if resolved is None:
            if clarify_used:
                return RouteDecision("policy", None, {}, "clarify already used")
            return RouteDecision(
                "clarify", None, {}, "elliptical query with nothing to resolve"
            )
        return RouteDecision("lookup", resolved, {}, "resolved from the entity slot")

    if record_id is not None:
        if has_policy_language(query):
            return RouteDecision(
                "both", record_id, {}, "record id plus policy language"
            )
        return RouteDecision("lookup", record_id, {}, "record id present")

    scores = intent_scores(query)
    winner = max(scores, key=scores.get)

    if winner == "vague":
        if clarify_used:
            return RouteDecision("policy", None, scores, "clarify already used")
        return RouteDecision("clarify", None, scores, "vague centroid won")

    return RouteDecision(winner, None, scores, "nearest intent centroid")
