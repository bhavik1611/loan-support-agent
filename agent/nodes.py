"""Task 7. Nine nodes, each a pure function from state to a state fragment.

Nothing here imports langgraph. That is deliberate: every node is testable by
calling it with a dict, and Task 10's graph module is then only wiring. It
also means the fan-out safety property of section 11.2 is checkable by
reading the return value of two functions rather than by running a graph.
"""

import asyncio

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

# The rule name is deliberately absent. It is already in the same envelope as
# guardrails.injection_rule, so the sentence was repeating a structured field
# in worse words, and naming the rule to whoever tripped it tells them which
# pattern to write around next. Everything an operator needs to see a
# guardrail fire is still in the JSON; only the prose changed.
REFUSAL_TEXT = (
    "I cannot help with that request. I can answer a question about Meridian "
    "Bank's loan policies, or check an application's status if you give me "
    "its id."
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
    "it. I can help with loans, cards, accounts, KYC and credit scores."
)

# Which record field a follow-up is asking about, matched cheapest-signal-first
# over a closed vocabulary. First match wins, so the choice is deterministic
# when a query carries two cues, and "status" is the fallback rather than a
# rule. This exists because turn 2 of transcripts/part2-memory.txt answered
# "Is it flagged for fraud?" with turn 1's status sentence byte for byte,
# while flagged_for_fraud_review sat unread in the same response.
LOOKUP_FOCUS_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fraud", ("fraud", "flagged", "flag ", "suspicious")),
    ("amount", ("how much", "amount", "sanction", "disburse", "how large")),
    ("age", ("when ", "how long", "how old", "submitted", "created", "filed")),
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
    """Short-circuit. Runs no retrieval and no lookup, and names no rule."""
    return {"clarification": REFUSAL_TEXT, "grounded": None}


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


def _lookup_focus(query: str) -> str:
    """The record field this query is asking about, or "status" by default."""
    lowered = query.lower()
    for focus, cues in LOOKUP_FOCUS_CUES:
        if any(cue in lowered for cue in cues):
            return focus
    return "status"


def _lookup_sentence(lookup: dict, query: str) -> str:
    """D-41 still holds: a fixed template over the record's own fields, no
    model call, so the sentence is byte-reproducible for a given record.

    Deterministic and question-aware are not in conflict. Which field leads is
    chosen by _lookup_focus over a closed cue vocabulary, which is exactly as
    reproducible as always leading with the status was, and it means a
    follow-up about one field is answered with that field.

    Per D-42 the credit score is not spoken; it stays in the structured block.
    The escalation score and its threshold are no longer spoken either: both
    are already in the lookup block, and a number a support agent has to
    interpret against a threshold reads worse than the recommendation it
    produces. Nothing was dropped, only moved to where it already was.
    """
    if not lookup["found"]:
        return f"I have no application on file with the id {lookup['record_id']}."

    record_id = lookup["record_id"]
    context = lookup.get("customer_context") or {}
    who = context.get("full_name", "the applicant")
    open_loans = context.get("open_loan_count")
    amount = f"{lookup['loan_amount_inr']:,}"
    days = lookup["days_since_created"]
    focus = _lookup_focus(query)

    parts = []
    if focus == "fraud":
        flagged = "is" if lookup["flagged_for_fraud_review"] else "is not"
        parts.append(f"Application {record_id} {flagged} flagged for fraud review.")
    elif focus == "amount":
        parts.append(f"Application {record_id} is for {amount} rupees.")
    elif focus == "age":
        parts.append(f"Application {record_id} was created {days} days ago.")

    if focus == "status":
        parts.append(
            f"Application {record_id} for {who} is {lookup['status']}, "
            f"for {amount} rupees, and was created {days} days ago."
        )
    else:
        parts.append(f"It is {lookup['status']}, and it belongs to {who}.")

    if open_loans is not None:
        parts.append(f"{who} has {open_loans} open application(s) with us.")
    if lookup["recommend_escalation"]:
        parts.append("I am escalating this one to a human colleague.")
    return " ".join(parts)


def _answer_text(state: AgentState) -> str:
    """The prose half of the envelope, assembled per route."""
    if state.get("injection_rule") or state.get("clarification"):
        return state["clarification"]

    pieces = []
    policy = state.get("policy")
    lookup = state.get("lookup")
    if lookup:
        pieces.append(_lookup_sentence(lookup, state["masked_query"]))
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
