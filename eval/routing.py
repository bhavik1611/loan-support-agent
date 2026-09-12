"""Probe queries for the router in Part 2 Task 7.

Deliberately worded differently from eval/queries.py and eval/calibration.py,
so routing, scoring and threshold calibration are not all measuring the same
strings.

The labelled probes carry no record id, because a record id short-circuits
the router before any embedding happens. These exercise the only stage where
the centroids are consulted.
"""

from agent import intents

# (query, the intent a human would assign). No record ids, by design.
LABELLED_PROBES: list[tuple[str, str]] = [
    ("What is the minimum credit score Meridian asks for?", "policy"),
    ("How is the monthly instalment worked out?", "policy"),
    ("Which papers prove my identity when opening an account?", "policy"),
    ("What does Meridian charge to close an account early?", "policy"),
    ("How long does a fraud dispute take to settle?", "policy"),
    ("Can a person living abroad hold a savings account here?", "policy"),
    ("Where has my application got to?", "lookup"),
    ("Has my loan been approved yet?", "lookup"),
    ("Is my application still sitting with the assessor?", "lookup"),
    ("Tell me how much was sanctioned on my file.", "lookup"),
    ("Has anything been flagged on my application?", "lookup"),
    ("Which stage is my file at right now?", "lookup"),
]

# Queries a human could not confidently route either. The router must not commit.
VAGUE_PROBES: list[str] = [
    "Can you help me?",
    "What is going on?",
    "Tell me about the loan.",
    "I need some information please.",
    "What should I do next?",
]

# The one vague probe the router does not catch, recorded rather than removed.
# Spec section 18.2 item 1 carries the reasoning: it scores policy 0.566,
# lookup 0.408, vague 0.262, so it routes to policy. The failure mode is a
# narrow answer rather than a wrong one, and widening the vague exemplars far
# enough to catch it starts swallowing real questions.
KNOWN_UNCAUGHT: tuple[str, ...] = ("Tell me about the loan.",)


def measure_routing() -> dict:
    """Score every probe, and report what the router did with it."""
    labelled = []
    for query, expected in LABELLED_PROBES:
        scores = intents.intent_scores(query)
        winner = max(scores, key=scores.get)
        labelled.append(
            {
                "query": query,
                "expected": expected,
                "winner": winner,
                "correct": winner == expected,
                "scores": scores,
            }
        )

    vague = []
    for query in VAGUE_PROBES:
        scores = intents.intent_scores(query)
        winner = max(scores, key=scores.get)
        vague.append(
            {
                "query": query,
                "winner": winner,
                "caught": winner == "vague",
                "known_uncaught": query in KNOWN_UNCAUGHT,
                "scores": scores,
            }
        )

    return {
        "labelled": labelled,
        "vague": vague,
        "labelled_correct": sum(1 for row in labelled if row["correct"]),
        "labelled_total": len(labelled),
        "vague_caught": sum(1 for row in vague if row["caught"]),
        "vague_total": len(vague),
    }
