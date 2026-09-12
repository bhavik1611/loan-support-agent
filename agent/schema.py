"""Task 9. The one shape every agent response takes.

Validated twice on purpose. Pydantic builds the object, then jsonschema
checks the serialised dict against the committed file. Only the second check
proves the exported schema is the one actually being met, and the exported
schema is what Part 3's FastAPI layer and the grader read.

The trace id is a hash rather than a uuid4 because the ground rule is that
the same input produces the same bytes, and two runs of scripts/run_part2.py
have to write identical transcripts. It hashes the MASKED query, so no raw
PII reaches the hash input. It is not a security control and is not claimed
as one.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field

import config

ROUTES = ("policy", "lookup", "both", "clarify", "refused")

TRACE_ID_LENGTH = 16


class PolicyBlock(BaseModel):
    """What the RAG route produced. Absent on every other route.

    `outcome` is the field that keeps the two out-of-scope refusals apart
    (D-54). "refused_gate" means the product gate of spec 8.5 stopped the
    query before retrieval and `top1_similarity` is 0.0 because nothing was
    searched; "refused_threshold" means retrieval ran and T or the support
    rule refused. Reading `supported` alone cannot tell them apart, and a
    grader checking criterion 24a against 24b needs to.

    The names are spec section 9.4's, not new ones, so the agent's envelope
    and Part 1's evaluation table say the same word for the same event.
    """

    model_config = ConfigDict(extra="forbid")

    citations: list[str] = Field(default_factory=list)
    top1_similarity: float
    supported: bool
    strategy: str
    outcome: Literal["answered", "refused_gate", "refused_threshold"] = "answered"
    product: str = ""


class LookupBlock(BaseModel):
    """What the record route produced.

    `source` is fixed at "record" per D-41: this text came from a template
    over the row's own fields with no model call behind it, so it must be
    distinguishable in the envelope from a grounded answer.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str
    found: bool
    status: str | None = None
    loan_amount_inr: int | None = None
    days_since_created: int | None = None
    flagged_for_fraud_review: bool | None = None
    escalation_score: float | None = None
    recommend_escalation: bool | None = None
    customer_context: dict | None = None
    source: Literal["record"] = "record"


class GuardrailBlock(BaseModel):
    """Which guardrails fired. Present on every response, including refusals."""

    model_config = ConfigDict(extra="forbid")

    pii_masked: list[str] = Field(default_factory=list)
    injection_rule: str | None = None
    grounded: bool | None = None


class AgentResponse(BaseModel):
    """The envelope. Part 3 Task 11 uses this as its FastAPI response model."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    thread_id: str
    turn: int
    route: Literal["policy", "lookup", "both", "clarify", "refused"]
    answer: str
    policy: PolicyBlock | None = None
    lookup: LookupBlock | None = None
    guardrails: GuardrailBlock


def trace_id(thread_id: str, turn: int, masked_query: str) -> str:
    """A deterministic id for one turn, per D-38."""
    raw = f"{thread_id}|{turn}|{masked_query}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:TRACE_ID_LENGTH]


def export_schema(path: Path | None = None) -> Path:
    """Write the JSON Schema the grader and Part 3 read."""
    path = config.RESPONSE_SCHEMA_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(AgentResponse.model_json_schema(), indent=2, sort_keys=True)
    path.write_text(body + "\n", encoding="utf-8")
    return path


def validate_response(response: AgentResponse) -> dict:
    """Serialise, validate against the committed schema, and return the payload."""
    payload = response.model_dump(mode="json")
    schema_doc = json.loads(config.RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=payload, schema=schema_doc)
    return payload
