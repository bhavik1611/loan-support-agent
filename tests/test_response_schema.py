"""Test 21 and test 26 of spec section 16. The envelope, and its trace id."""

import json

import jsonschema
import pytest

import config
from agent import schema


def _guardrails(**kwargs):
    base = {"pii_masked": [], "injection_rule": None, "grounded": True}
    return schema.GuardrailBlock(**(base | kwargs))


def _response(route, **kwargs):
    base = {
        "trace_id": "0123456789abcdef",
        "thread_id": "t",
        "turn": 1,
        "route": route,
        "answer": "text",
        "guardrails": _guardrails(),
    }
    return schema.AgentResponse(**(base | kwargs))


def test_every_route_is_declared():
    assert set(schema.ROUTES) == {"policy", "lookup", "both", "clarify", "refused"}


def test_an_unknown_route_is_rejected():
    with pytest.raises(Exception):
        _response("wandering")


def test_a_policy_response_validates():
    response = _response(
        "policy",
        policy=schema.PolicyBlock(
            citations=["kb-01"], top1_similarity=0.51, supported=True, strategy="sentences"
        ),
    )
    assert schema.validate_response(response)["route"] == "policy"


def test_a_lookup_response_validates_and_is_tagged_as_a_record():
    response = _response(
        "lookup",
        lookup=schema.LookupBlock(
            record_id="LN-1042", found=True, status="Under Review",
            loan_amount_inr=1450000, escalation_score=0.7381,
            recommend_escalation=True, customer_context={"customer_id": "CU-001"},
        ),
    )
    payload = schema.validate_response(response)
    assert payload["lookup"]["source"] == "record"


def test_a_both_response_carries_both_blocks():
    response = _response(
        "both",
        policy=schema.PolicyBlock(
            citations=["kb-01"], top1_similarity=0.51, supported=True, strategy="sentences"
        ),
        lookup=schema.LookupBlock(record_id="LN-1042", found=True),
    )
    payload = schema.validate_response(response)
    assert payload["policy"] and payload["lookup"]


def test_a_clarify_response_carries_neither_block():
    payload = schema.validate_response(_response("clarify"))
    assert payload["policy"] is None
    assert payload["lookup"] is None


def test_a_refusal_is_a_valid_response():
    """Every path converges on compose, so a refusal validates like anything else."""
    response = _response(
        "refused",
        guardrails=_guardrails(injection_rule="instruction_override", grounded=None),
    )
    payload = schema.validate_response(response)
    assert payload["guardrails"]["injection_rule"] == "instruction_override"


def test_a_gate_refusal_is_distinguishable_from_a_threshold_refusal():
    """[D-54] Criterion 24a against 24b. `supported` is False for both."""
    gated = _response(
        "policy",
        policy=schema.PolicyBlock(
            citations=[], top1_similarity=0.0, supported=False,
            strategy="sentences", outcome="refused_gate", product="fixed deposit",
        ),
        guardrails=_guardrails(grounded=None),
    )
    ungrounded = _response(
        "policy",
        policy=schema.PolicyBlock(
            citations=[], top1_similarity=0.19, supported=False,
            strategy="sentences", outcome="refused_threshold",
        ),
        guardrails=_guardrails(grounded=False),
    )
    assert schema.validate_response(gated)["policy"]["product"] == "fixed deposit"
    assert schema.validate_response(ungrounded)["policy"]["product"] == ""
    assert gated.policy.outcome != ungrounded.policy.outcome


def test_an_unknown_outcome_is_rejected():
    """The Literal is the whole guard; spec 9.4 names exactly three values."""
    with pytest.raises(Exception):
        schema.PolicyBlock(
            citations=[], top1_similarity=0.0, supported=False,
            strategy="sentences", outcome="refused_because_i_felt_like_it",
        )


def test_the_committed_schema_file_is_current():
    """A drifted schema file would validate against nothing the code produces."""
    committed = json.loads(config.RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == schema.AgentResponse.model_json_schema()


def test_validation_uses_the_committed_file_not_just_pydantic():
    """D-36. The exported schema must be the one actually being met."""
    committed = json.loads(config.RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))
    payload = schema.validate_response(_response("clarify"))
    jsonschema.validate(instance=payload, schema=committed)


def test_trace_id_is_deterministic():
    first = schema.trace_id("thread-a", 2, "what is the status of LN-1042")
    second = schema.trace_id("thread-a", 2, "what is the status of LN-1042")
    assert first == second
    assert len(first) == 16


def test_trace_id_changes_with_thread_turn_and_query():
    base = schema.trace_id("thread-a", 1, "q")
    assert schema.trace_id("thread-b", 1, "q") != base
    assert schema.trace_id("thread-a", 2, "q") != base
    assert schema.trace_id("thread-a", 1, "other") != base
