"""Probe queries for the router in Part 2 Task 7.

Deliberately worded differently from eval/queries.py and eval/calibration.py,
so routing, scoring and threshold calibration are not all measuring the same
strings.

The labelled probes carry no record id, because a record id short-circuits
the router before any embedding happens. These exercise the only stage where
the centroids are consulted.
"""

from agent import intents, memory
from eval import queries

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


# --- The ellipsis matcher's own calibration set ----------------------------
#
# Spec section 18.4 names this matcher as one of four deciding on a signal
# that never carried the property it was asked to decide, found because no
# probe set of any kind exercised it. These are new strings, not copied from
# eval/queries.py: that file is the scored golden dataset and mixing the two
# would destroy the separation tests/test_queries.py enforces between the
# golden set and every fitting or probe set in this repository.
#
# The one exception is deliberate and structural rather than a copy: EQ-11 is
# referenced by id from eval.queries, not retyped, because it is the actual
# defect report (spec 18.4, this file's docstring) and the matcher must be
# judged against the literal query that exposed it, not a paraphrase of it.

_EQ_11_TEXT = next(item.text for item in queries.EVAL_QUERIES if item.item_id == "EQ-11")

# Genuinely elliptical: the cue's antecedent lives in the turn before this
# one, so the matcher must ask for it. The first round of probes below has no
# conversational opener before the cue, and passed a rule that then regressed
# on the second round: a leading filler like "Thanks," or "Hi," closes over a
# word, and a rule that counts any word as an antecedent reads that as one.
# A curated filler-word list fixed those but left "Well, what about it?"
# uncaught, because a list of interjections has no edge. The noun-phrase-
# marker rule in agent/memory.py catches all fourteen with no special case:
# none of these openers, "well" included, carries a determiner.
ELLIPTICAL_PROBES: tuple[str, ...] = (
    "Is it flagged for fraud?",
    "What about that one?",
    "Is the same true for mine?",
    "Can you check its status?",
    "What about this one?",
    "Are they still under review?",
    "What happened to them?",
    "Are those still pending?",
    # Ordinary second-turn phrasing: a conversational opener, then the same
    # bare cue as above.
    "Thanks, and is it approved?",
    "OK, what about it?",
    "Sorry, is it flagged for fraud?",
    "Hi, can you tell me about them?",
    "Hey, is the same true for mine?",
    # What sank the curated filler-word list: an opener it had no word for.
    # The noun-phrase-marker rule needs no word for it either, because "well"
    # carries no determiner - this is why the two rules are not the same fix
    # wearing a different vocabulary.
    "Well, what about it?",
)

# Self-contained: a cue word is present, but its antecedent is earlier in the
# same sentence, so the query needs nothing from a previous turn.
SELF_CONTAINED_PROBES: tuple[str, ...] = (
    _EQ_11_TEXT,
    "I have two personal loans open right now, and are they both affecting my credit score?",
    "My card got blocked after three failed PIN attempts, and how do I unblock it?",
    "I paid my credit card bill late this month, and will that affect my credit score the same way a missed EMI would?",
    "I have a savings account and a fixed deposit, and are the maturity rules the same for both?",
    "I submitted my passport and my utility bill as KYC proof, but will these documents also work for reactivating my account, or do I need to submit them again?",
    "I closed my old savings account last year and opened a new one, so does the same minimum balance rule apply to it?",
    "I have pending queries on two loan applications and want an update on those.",
    "My spouse and I hold a joint account, and do the same KYC rules apply to us both?",
    # The noun-phrase-marker rule's own residue: a bare plural or mass noun
    # takes no determiner in English, so a clause whose only antecedent is
    # one of these reads as having none. See SELF_CONTAINED_KNOWN_UNCAUGHT.
    "Loans affect credit scores, do they not?",
    "Interest accrues monthly and does it compound?",
    "Fraud reviews take time, so how long do they run?",
)

# The three self-contained probes the noun-phrase-marker rule does not catch,
# recorded rather than chased. Each antecedent - "loans", "interest", "fraud
# reviews" - is a bare plural or mass noun with no determiner in front of it,
# so no clause before the cue contains a marker and the query reads as
# elliptical. Catching these needs a part-of-speech signal this repository
# does not have under MOCK_LLM, not a bigger marker set: unlike the filler
# list this rule replaced, there is no word to add here that would not also
# have to flag every ordinary plural noun as a marker, which defeats the rule.
SELF_CONTAINED_KNOWN_UNCAUGHT: tuple[str, ...] = (
    "Loans affect credit scores, do they not?",
    "Interest accrues monthly and does it compound?",
    "Fraud reviews take time, so how long do they run?",
)


def measure_ellipsis_matcher() -> dict:
    """Score `memory.needs_resolution` against its own labelled probe set.

    Each probe carries the label a human would assign - True for the
    elliptical class, False for the self-contained one - and the function
    reports where the matcher's verdict disagrees with it. The three probes
    in `SELF_CONTAINED_KNOWN_UNCAUGHT` are expected to disagree; they are
    flagged rather than hidden, the same way `KNOWN_UNCAUGHT` is handled
    above for the router.
    """
    elliptical = []
    for query in ELLIPTICAL_PROBES:
        predicted = memory.needs_resolution(query)
        elliptical.append(
            {"query": query, "expected": True, "predicted": predicted, "correct": predicted is True}
        )

    self_contained = []
    for query in SELF_CONTAINED_PROBES:
        predicted = memory.needs_resolution(query)
        self_contained.append(
            {
                "query": query,
                "expected": False,
                "predicted": predicted,
                "correct": predicted is False,
                "known_uncaught": query in SELF_CONTAINED_KNOWN_UNCAUGHT,
            }
        )

    return {
        "elliptical": elliptical,
        "self_contained": self_contained,
        "elliptical_correct": sum(1 for row in elliptical if row["correct"]),
        "elliptical_total": len(elliptical),
        "self_contained_correct": sum(1 for row in self_contained if row["correct"]),
        "self_contained_total": len(self_contained),
    }
