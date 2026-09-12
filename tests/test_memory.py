"""Test 20 of spec section 16, the half that does not need the graph."""

import json

from agent import memory


def test_a_fresh_thread_has_no_turns_and_no_entities(tmp_path):
    thread = memory.load("demo-fresh", root=tmp_path)
    assert thread.thread_id == "demo-fresh"
    assert thread.turns == []
    assert thread.entities == {}


def test_a_saved_thread_round_trips(tmp_path):
    thread = memory.load("t", root=tmp_path)
    thread = memory.record_turn(thread, "q1", "lookup", "a1", record_id="LN-1042")
    memory.save(thread, root=tmp_path)

    reloaded = memory.load("t", root=tmp_path)
    assert len(reloaded.turns) == 1
    assert reloaded.turns[0]["query"] == "q1"
    assert reloaded.turns[0]["route"] == "lookup"
    assert reloaded.entities["last_record_id"] == "LN-1042"


def test_the_entity_slot_carries_the_most_recent_record_id(tmp_path):
    thread = memory.load("t", root=tmp_path)
    thread = memory.record_turn(thread, "q1", "lookup", "a1", record_id="LN-1001")
    thread = memory.record_turn(thread, "q2", "lookup", "a2", record_id="LN-1042")
    assert thread.entities["last_record_id"] == "LN-1042"


def test_a_turn_without_a_record_id_leaves_the_slot_alone(tmp_path):
    thread = memory.load("t", root=tmp_path)
    thread = memory.record_turn(thread, "q1", "lookup", "a1", record_id="LN-1042")
    thread = memory.record_turn(thread, "q2", "policy", "a2")
    assert thread.entities["last_record_id"] == "LN-1042"


def test_turns_are_numbered_from_one(tmp_path):
    thread = memory.load("t", root=tmp_path)
    thread = memory.record_turn(thread, "q1", "policy", "a1")
    thread = memory.record_turn(thread, "q2", "policy", "a2")
    assert [t["turn"] for t in thread.turns] == [1, 2]


def test_the_file_is_readable_json(tmp_path):
    """The brief asks for conversation history persisted to a JSON file."""
    thread = memory.load("t", root=tmp_path)
    thread = memory.record_turn(thread, "q1", "policy", "a1")
    path = memory.save(thread, root=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["thread_id"] == "t"
    assert payload["turns"][0]["query"] == "q1"


def test_saving_is_byte_stable(tmp_path):
    """Determinism: the same thread written twice produces the same bytes."""
    thread = memory.load("t", root=tmp_path)
    thread = memory.record_turn(thread, "q1", "policy", "a1")
    first = memory.save(thread, root=tmp_path).read_bytes()
    second = memory.save(thread, root=tmp_path).read_bytes()
    assert first == second


def test_an_elliptical_question_needs_resolution():
    for query in ("Is it flagged for fraud?", "What about that one?", "And its status?"):
        assert memory.needs_resolution(query) is True


def test_a_self_contained_question_does_not():
    for query in (
        "What is the status of LN-1042?",
        "What is the minimum credit score for a home loan?",
    ):
        assert memory.needs_resolution(query) is False


def test_a_cue_word_with_its_antecedent_in_the_same_sentence_does_not_resolve():
    """EQ-11's shape: a cue word whose antecedent is one clause earlier.

    The old matcher fired on the bare presence of a cue word and misrouted
    exactly this query (spec 18.4). The fix reads the cue's position instead.
    """
    query = "How do I report a fraudulent card transaction and will it affect my credit score?"
    assert memory.needs_resolution(query) is False


def test_a_leading_filler_does_not_supply_an_antecedent():
    """A second failure of the same shape (spec 18.4): a clause break alone

    reads "Thanks," or "Hi," as content, and would wrongly treat these as
    self-contained. config.NOUN_PHRASE_MARKERS is what tells them apart from
    EQ-11's shape above - none of these openers carries a determiner.
    """
    for query in (
        "Thanks, and is it approved?",
        "OK, what about it?",
        "Sorry, is it flagged for fraud?",
        "Hi, can you tell me about them?",
        "Hey, is the same true for mine?",
        "Well, what about it?",
    ):
        assert memory.needs_resolution(query) is True


def test_a_bare_plural_antecedent_is_the_stated_residue():
    """The noun-phrase-marker rule's own named weakness.

    A bare plural or mass noun takes no determiner, so a clause whose only
    antecedent is one reads as elliptical. This is
    eval/routing.py::SELF_CONTAINED_KNOWN_UNCAUGHT, not a bug to chase.
    """
    for query in (
        "Loans affect credit scores, do they not?",
        "Interest accrues monthly and does it compound?",
        "Fraud reviews take time, so how long do they run?",
    ):
        assert memory.needs_resolution(query) is True


def test_the_ellipsis_matcher_scores_against_its_own_probe_set():
    """The probe set spec 18.4 says never existed before this fix.

    Three self-contained probes, each with a bare plural or mass noun as its
    antecedent, are an honest known miss
    (eval/routing.py::SELF_CONTAINED_KNOWN_UNCAUGHT) rather than a reason to
    grow config.NOUN_PHRASE_MARKERS beyond the closed class it names.
    """
    from eval import routing

    result = routing.measure_ellipsis_matcher()
    elliptical_missed = [r["query"] for r in result["elliptical"] if not r["correct"]]
    assert elliptical_missed == []
    assert result["elliptical_correct"] == result["elliptical_total"] == 14

    missed = [r["query"] for r in result["self_contained"] if not r["correct"]]
    assert missed == list(routing.SELF_CONTAINED_KNOWN_UNCAUGHT)
    assert result["self_contained_correct"] == result["self_contained_total"] - 3
    assert result["self_contained_total"] == 12


def test_the_last_turn_route_is_readable(tmp_path):
    """The clarify cap in D-45 reads this."""
    thread = memory.load("t", root=tmp_path)
    assert memory.last_route(thread) is None
    thread = memory.record_turn(thread, "q1", "clarify", "a1")
    assert memory.last_route(thread) == "clarify"
