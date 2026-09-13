"""Task 7. Nine nodes, each a pure function from state to a state fragment.

Nothing here imports langgraph. That is deliberate: every node is testable by
calling it with a dict, and Task 10's graph module is then only wiring. It
also means the fan-out safety property of section 11.2 is checkable by
reading the return value of two functions rather than by running a graph.
"""

import asyncio

import config
from agent import guardrails, intents, memory, schema, tools
from agent.state import AgentState
from rag import retrieve

NODE_NAMES = (
    "guard_input",
    "recall",
    "route",
    "policy_answer",
    "lookup_status",
    "clarify",
    "verify",
    "compose",
    "refuse",
)

CLARIFY_QUESTION = (
    "I can answer a policy question or look up one application, but I am not "
    "sure which you need. Which application id should I check, or which policy "
    "would you like?"
)

REFUSAL_TEXT = (
    "I cannot act on that request. It matched the {rule} guardrail, so I have "
    "not run any retrieval or looked up any record."
)

# D-54. The product gate lives in rag/scope.py and decides; the sentence a
# support agent reads is Part 2's, and this is it. Part 1 deliberately returns
# the structured verdict and the product name rather than prose, because only
# the agent knows it is talking to a person.
#
# It names the product and says what Meridian does cover, because "I cannot
# help with that" sends the asker back to a queue, and the one fact that
# actually redirects them is that we do not sell the thing they asked about.
OUT_OF_SCOPE_TEXT = (
    "Meridian Bank does not offer {product}, so I have nothing on file about "
    "it and did not search the knowledge base. I can help with loans, cards, "
    "accounts, KYC and credit scores."
)


def guard_input(state: AgentState) -> dict:
    """Mask fixed-format PII, then look for an injection attempt."""
    masked, fired = guardrails.mask_pii(state["query"])
    return {
        "masked_query": masked,
        "pii_masked": fired,
        "injection_rule": guardrails.detect_injection(masked),
    }


def recall(state: AgentState) -> dict:
    """Load the thread, and whether it has already clarified once."""
    thread = memory.load(state["thread_id"])
    return {
        "history": thread.turns,
        "entities": thread.entities,
        "clarify_used": memory.last_route(thread) == "clarify",
    }


def route(state: AgentState) -> dict:
    """Pick the intent. The conditional edge reads what this writes."""
    decision = intents.classify(
        state["masked_query"],
        state.get("entities", {}),
        clarify_used=state.get("clarify_used", False),
    )
    return {
        "route": decision.route,
        "record_id": decision.record_id,
        "route_scores": decision.scores,
        "route_reason": decision.reason,
    }


def pick_branch(state: AgentState):
    """The conditional edge. Returning a list is what fans out to both nodes."""
    chosen = state["route"]
    if chosen == "both":
        return ["policy_answer", "lookup_status"]
    return {
        "policy": "policy_answer",
        "lookup": "lookup_status",
        "clarify": "clarify",
    }[chosen]


def policy_answer(state: AgentState) -> dict:
    """The RAG route. Writes state['policy'] and nothing else."""
    answer = tools.answer_policy_question(state["masked_query"])
    return {
        "policy": {
            "text": answer.text,
            "citations": list(answer.citations),
            "top1_similarity": answer.top1_similarity,
            "supported": answer.supported,
            "strategy": answer.strategy,
            # D-54. Which mechanism decided, and what it matched. On a gate
            # refusal hits is empty, so retrieved_doc_ids is [] and nothing
            # downstream has to special-case a missing key.
            "outcome": answer.outcome,
            "product": answer.product,
            # Plain doc ids, not Hit objects. Everything in state has to
            # survive the SQLite checkpointer Part 4 Task 15 attaches.
            "retrieved_doc_ids": retrieve.parent_documents(list(answer.hits)),
        }
    }


async def lookup_status(state: AgentState) -> dict:
    """The record route. Writes state['lookup'] and nothing else.

    Async so LangGraph will accept a timeout on this node, and wrapped in
    asyncio.to_thread so that timeout can actually fire: a coroutine that
    never awaits cannot be preempted by any watchdog, so async def alone
    only satisfies LangGraph's build-time check, not the run-time one.
    agent.tools stays synchronous by design; the wrapper lives here, at the
    call site, not there. The cost is that when the timeout fires, this
    coroutine is cancelled but the worker thread keeps running until the
    underlying call returns (LangGraph's own _runnable_has_native_async
    docstring says the same) - acceptable here because the call is a local
    SQLite read of single-digit milliseconds.
    """
    return {
        "lookup": await asyncio.to_thread(
            tools.check_loan_application_status, state["record_id"]
        )
    }


def clarify(state: AgentState) -> dict:
    """Ask one specific question instead of guessing."""
    return {"clarification": CLARIFY_QUESTION}


def refuse(state: AgentState) -> dict:
    """Short-circuit. Names the rule, and runs no retrieval and no lookup."""
    return {
        "clarification": REFUSAL_TEXT.format(rule=state["injection_rule"]),
        "grounded": None,
    }


def verify(state: AgentState) -> dict:
    """The output side. Only a retrieved answer can be ungrounded."""
    policy = state.get("policy")
    if not policy:
        return {"grounded": None, "output_rule": None}
    if policy["outcome"] == "refused_gate":
        # D-54. The gate refused before retrieval, so there is no answer to
        # check the grounding of. Reporting grounded=False here would have the
        # agent claim a guardrail fired that never ran, which is exactly the
        # thing the brief asks us to demonstrate and therefore the thing we
        # must not fake. None means "did not run", the same value refuse()
        # writes for the same reason.
        return {"grounded": None, "output_rule": None}
    rule = guardrails.check_grounded(
        policy["supported"], policy["citations"], policy["retrieved_doc_ids"]
    )
    return {"grounded": rule is None, "output_rule": rule}


def _lookup_sentence(lookup: dict) -> str:
    """D-41. A fixed template over the record's own fields, no model call.

    Per D-42 the credit score is not spoken; it stays in the structured block.
    """
    if not lookup["found"]:
        return f"I have no application on file with the id {lookup['record_id']}."

    context = lookup.get("customer_context") or {}
    who = context.get("full_name", "the applicant")
    open_loans = context.get("open_loan_count")
    amount = f"{lookup['loan_amount_inr']:,}"
    parts = [
        f"Application {lookup['record_id']} for {who} is {lookup['status']}, "
        f"for {amount} rupees, and was created {lookup['days_since_created']} days ago."
    ]
    if open_loans is not None:
        parts.append(f"{who} has {open_loans} open application(s) with us.")
    if lookup["recommend_escalation"]:
        parts.append(
            f"Its escalation score is {lookup['escalation_score']}, at or above the "
            f"{config.ESCALATION_THRESHOLD} threshold, so I recommend escalating it."
        )
    else:
        parts.append(
            f"Its escalation score is {lookup['escalation_score']}, below the "
            f"{config.ESCALATION_THRESHOLD} threshold, so no escalation is needed."
        )
    return " ".join(parts)


def _answer_text(state: AgentState) -> str:
    """The prose half of the envelope, assembled per route."""
    if state.get("injection_rule") or state.get("clarification"):
        return state["clarification"]

    pieces = []
    policy = state.get("policy")
    lookup = state.get("lookup")
    if lookup:
        pieces.append(_lookup_sentence(lookup))
    if policy:
        if policy["outcome"] == "refused_gate":
            # D-54. The one case where Part 2 overrides Part 1's wording, and
            # the only one: Part 1's fallback says the knowledge base holds
            # nothing, which is true but misleading here. The knowledge base
            # will never hold it, because Meridian does not sell it.
            pieces.append(OUT_OF_SCOPE_TEXT.format(product=policy["product"]))
        else:
            # When the answer is ungrounded, rag.generate.answer has already
            # put the Part 1 fallback text in here, so there is one refusal
            # wording in the repository rather than two that can drift apart.
            pieces.append(policy["text"])
    return "\n\n".join(pieces) if pieces else CLARIFY_QUESTION


def compose(state: AgentState) -> dict:
    """Build the envelope, validate it, and persist the turn."""
    chosen = "refused" if state.get("injection_rule") else state.get("route", "clarify")

    policy = state.get("policy")
    lookup = state.get("lookup")

    policy_block = None
    if policy and chosen != "refused":
        policy_block = schema.PolicyBlock(
            citations=policy["citations"],
            top1_similarity=policy["top1_similarity"],
            supported=policy["supported"],
            strategy=policy["strategy"],
            outcome=policy["outcome"],
            product=policy["product"],
        )

    lookup_block = None
    if lookup and chosen != "refused":
        lookup_block = schema.LookupBlock(
            **{key: lookup[key] for key in schema.LookupBlock.model_fields if key in lookup}
        )

    masked_query = state["masked_query"]
    response = schema.AgentResponse(
        trace_id=schema.trace_id(state["thread_id"], state["turn"], masked_query),
        thread_id=state["thread_id"],
        turn=state["turn"],
        route=chosen,
        answer=_answer_text(state),
        policy=policy_block,
        lookup=lookup_block,
        guardrails=schema.GuardrailBlock(
            pii_masked=state.get("pii_masked", []),
            injection_rule=state.get("injection_rule"),
            grounded=state.get("grounded"),
        ),
    )
    payload = schema.validate_response(response)

    thread = memory.load(state["thread_id"])
    thread = memory.record_turn(
        thread,
        query=masked_query,
        route=chosen,
        answer=response.answer,
        record_id=state.get("record_id"),
    )
    memory.save(thread)

    return {"response": payload}
