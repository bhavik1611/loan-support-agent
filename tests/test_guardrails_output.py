"""Test 24 of spec section 16, and the phantom-citation rule beyond the brief."""

from agent import guardrails, tools
from rag.generate import GroundedAnswer
from rag.retrieve import Hit


def _hit(doc_id, similarity=0.7):
    return Hit(text="body", doc_id=doc_id, title="t", chunk_index=0, similarity=similarity)


def _answer(citations, hits, supported=True):
    return GroundedAnswer(
        query="q", text="a", citations=citations, supported=supported,
        top1_similarity=0.7, strategy="sentences", hits=hits,
    )


def test_a_supported_answer_citing_retrieved_documents_passes():
    assert guardrails.check_grounded(True, ["kb-01"], ["kb-01", "kb-04"]) is None


def test_an_unsupported_answer_is_caught():
    assert guardrails.check_grounded(False, [], ["kb-01"]) == "unsupported"


def test_a_citation_that_was_never_retrieved_is_caught():
    assert guardrails.check_grounded(True, ["kb-11"], ["kb-01"]) == "phantom_citation"


def test_unsupported_is_reported_before_phantom_citation():
    """Deterministic ordering: an unsupported answer's citations are moot."""
    assert guardrails.check_grounded(False, ["kb-11"], ["kb-01"]) == "unsupported"


def test_the_wrapper_agrees_with_the_plain_call():
    answer = _answer(("kb-01",), (_hit("kb-01"), _hit("kb-04")))
    assert guardrails.grounded_rule_for(answer) is None


def test_an_out_of_scope_query_is_caught_end_to_end(built_index):
    """The brief's acceptance criterion, against the real index."""
    answer = tools.answer_policy_question("What is the best pizza topping?")
    assert guardrails.grounded_rule_for(answer) == "unsupported"


def test_an_in_scope_query_passes_end_to_end(built_index):
    answer = tools.answer_policy_question("What is the minimum credit score?")
    assert guardrails.grounded_rule_for(answer) is None
