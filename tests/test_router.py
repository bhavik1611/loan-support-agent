"""Test 25 of spec section 16, and the routing behaviour D-44 and D-45 specify."""

from agent import intents


def test_a_record_id_is_found():
    assert intents.find_record_id("What is the status of LN-1042?") == "LN-1042"


def test_a_lowercase_record_id_is_found_and_normalised():
    assert intents.find_record_id("status of ln-1042 please") == "LN-1042"


def test_no_record_id_returns_none():
    assert intents.find_record_id("What is the minimum credit score?") is None


def test_a_record_id_alone_routes_to_lookup(built_index):
    decision = intents.classify("What is the status of LN-1042?", {})
    assert decision.route == "lookup"
    assert decision.record_id == "LN-1042"


def test_a_record_id_with_policy_language_routes_to_both(built_index):
    decision = intents.classify(
        "Why is LN-1042 still under review, and what is the eligibility rule?", {}
    )
    assert decision.route == "both"
    assert decision.record_id == "LN-1042"


def test_an_elliptical_question_resolves_from_the_entity_slot(built_index):
    decision = intents.classify("Is it flagged for fraud?", {"last_record_id": "LN-1042"})
    assert decision.route == "lookup"
    assert decision.record_id == "LN-1042"


def test_an_elliptical_question_with_no_entity_asks_for_clarification(built_index):
    decision = intents.classify("Is it flagged for fraud?", {})
    assert decision.route == "clarify"
    assert decision.record_id is None


def test_a_leading_filler_still_resolves_from_the_entity_slot(built_index):
    """Ordinary second-turn phrasing must not skip resolution.

    Without config.ELLIPSIS_FILLER_WORDS, "Thanks," reads as an antecedent
    clause, the query is treated as self-contained, and the centroid stage
    happens to land on "lookup" anyway - but with record_id=None, so the
    entity slot is silently never consulted. That is the regression this
    guards: the route must come from the entity slot, not a lucky centroid.
    """
    decision = intents.classify("Thanks, and is it approved?", {"last_record_id": "LN-1042"})
    assert decision.route == "lookup"
    assert decision.record_id == "LN-1042"


def test_a_same_sentence_antecedent_is_not_treated_as_elliptical(built_index):
    """EQ-11's shape must reach the centroids rather than clarify.

    Spec 18.4: the ellipsis matcher used to fire on the bare word "it" and
    route this query to clarify, though the corpus answers it.
    """
    query = "How do I report a fraudulent card transaction and will it affect my credit score?"
    decision = intents.classify(query, {})
    assert decision.route == "policy"
    assert decision.record_id is None


def test_a_policy_question_routes_to_policy(built_index):
    decision = intents.classify("What is the minimum credit score for a home loan?", {})
    assert decision.route == "policy"


def test_a_vague_question_routes_to_clarify(built_index):
    decision = intents.classify("Can you help me?", {})
    assert decision.route == "clarify"


def test_the_clarify_cap_falls_through_to_policy(built_index):
    """D-45. Two ambiguous turns in a row must not clarify twice."""
    decision = intents.classify("Can you help me?", {}, clarify_used=True)
    assert decision.route == "policy"


def test_three_centroids_are_declared():
    assert set(intents.INTENT_EXEMPLARS) == {"policy", "lookup", "vague"}


def test_every_labelled_probe_routes_to_its_label(built_index):
    """Test 25, first half. The router decides rather than guesses."""
    from eval import routing

    result = routing.measure_routing()
    wrong = [r["query"] for r in result["labelled"] if not r["correct"]]
    assert not wrong, wrong
    assert result["labelled_correct"] == result["labelled_total"] == 12


def test_every_vague_probe_is_caught_except_the_one_on_record(built_index):
    """Test 25, second half. The known exception cannot silently grow to two."""
    from eval import routing

    result = routing.measure_routing()
    missed = [r["query"] for r in result["vague"] if not r["caught"]]
    assert missed == list(routing.KNOWN_UNCAUGHT)


def test_no_real_repository_query_is_swallowed_by_the_vague_centroid(built_index):
    """The vague class must not eat questions the rest of the repo treats as real."""
    from eval import calibration
    from eval.queries import EVAL_QUERIES

    texts = [q.text for q in EVAL_QUERIES] + list(calibration.IN_SCOPE_PROBES)
    swallowed = [
        text for text in texts
        if max(intents.intent_scores(text), key=intents.intent_scores(text).get) == "vague"
    ]
    assert not swallowed, swallowed


def test_a_status_question_with_no_id_anywhere_asks_rather_than_crashing(built_index):
    """The lookup centroid can win with no record id in reach.

    Routing to lookup anyway calls the tool with None, which returns a miss
    whose record_id is None, which LookupBlock rejects. Measured 2026-09-13:
    five of six ordinary no-id status phrasings raised ValidationError in
    compose, and POST /ask answered HTTP 500.
    """
    for query in (
        "Why is my application rejected?",
        "What is the status of my loan?",
        "Has my loan been approved?",
        "When will my application be processed?",
        "How much was sanctioned to me?",
    ):
        decision = intents.classify(query, {})
        assert decision.route == "clarify", query
        assert decision.record_id is None, query


def test_a_possessive_follow_up_resolves_from_the_entity_slot(built_index):
    """Turn 2 of an ordinary support conversation.

    "my application" is a complete noun phrase, so memory.needs_resolution is
    right to call it self-contained and is deliberately left alone. It is
    still a reference to the one record the thread is about.
    """
    for query in (
        "Why was my application rejected?",
        "Why is my loan denied?",
        "Why is my home loan application still pending?",
    ):
        decision = intents.classify(query, {"last_record_id": "LN-1042"})
        assert decision.route == "lookup", query
        assert decision.record_id == "LN-1042", query


def test_the_own_record_rule_leaves_the_calibrated_probes_alone(built_index):
    """OWN_RECORD must not reclassify anything eval/routing.py measures.

    The twelve self-contained probes say "my credit score", "my card" or a
    bare "loan applications"; none says "my loan". This test is what keeps
    that true when the vocabulary is next edited.
    """
    from eval import routing

    for query in routing.SELF_CONTAINED_PROBES + routing.ELLIPTICAL_PROBES:
        assert intents.OWN_RECORD.search(query) is None, query
