"""Test 17 of spec section 16, plus the contract the envelope depends on."""

import dataset
from agent import escalation, tools

FORBIDDEN = {"pan", "aadhaar", "account_number", "email", "phone"}


def test_lookup_returns_the_briefs_three_keys(db_conn):
    result = tools.check_loan_application_status("LN-1001", conn=db_conn)
    assert result["found"] is True
    assert result["status"] == dataset.get_application("LN-1001")["status"]
    assert result["loan_amount_inr"] == dataset.get_application("LN-1001")["loan_amount_inr"]
    assert 0.0 <= result["escalation_score"] <= 1.0


def test_lookup_score_matches_the_scoring_module(db_conn):
    for record_id in ("LN-1001", "LN-1042", "LN-1100"):
        record = dataset.get_application(record_id)
        result = tools.check_loan_application_status(record_id, conn=db_conn)
        assert result["escalation_score"] == escalation.escalation_score(record)
        assert result["recommend_escalation"] is escalation.recommend_escalation(record)


def test_lookup_returns_a_miss_rather_than_raising(db_conn):
    result = tools.check_loan_application_status("LN-9999", conn=db_conn)
    assert result["found"] is False
    assert result["record_id"] == "LN-9999"
    assert result["status"] is None
    assert result["escalation_score"] is None


def test_lookup_never_returns_pii(db_conn):
    """D-21 again, at the tool boundary this time."""
    result = tools.check_loan_application_status("LN-1001", conn=db_conn)
    assert FORBIDDEN.isdisjoint(result)
    assert FORBIDDEN.isdisjoint(result["customer_context"])


def test_lookup_shape_is_the_same_on_a_hit_and_a_miss(db_conn):
    """The envelope has one lookup block, so both outcomes carry the same keys."""
    hit = tools.check_loan_application_status("LN-1001", conn=db_conn)
    miss = tools.check_loan_application_status("LN-9999", conn=db_conn)
    assert set(hit) == set(miss) == set(tools.LOOKUP_FIELDS)


def test_lookup_has_a_docstring_for_the_mcp_wrapper():
    """Part 4 Task 14 wraps this and requires a proper docstring."""
    doc = tools.check_loan_application_status.__doc__
    assert doc and len(doc.strip()) > 80


def test_policy_tool_answers_an_in_scope_question(built_index):
    # "for a loan" on the end of this question makes it refuse. See the note
    # below the test file: the longer form spans three documents and the
    # support rule declines. That is spec 18.1 item 5, measured again here.
    result = tools.answer_policy_question("What is the minimum credit score?")
    assert result.supported is True
    assert result.outcome == "answered"
    assert result.citations


def test_policy_tool_refuses_an_out_of_scope_question(built_index):
    result = tools.answer_policy_question("What is the best pizza topping?")
    assert result.supported is False
    # Names no product in KNOWN_ADJACENT, so it falls past the gate and is
    # refused on the threshold. That is the distinction the next test makes.
    assert result.outcome == "refused_threshold"


def test_policy_tool_refuses_an_adjacent_product_at_the_gate(built_index):
    """[D-54] Spec 8.5. The gate decides before retrieval, so nothing comes back."""
    result = tools.answer_policy_question(
        "What is the interest rate on a fixed deposit for 5 years?"
    )
    assert result.outcome == "refused_gate"
    assert result.product == "fixed deposit"
    assert result.hits == ()
    assert result.supported is False
