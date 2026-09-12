"""Tasks 6 and 7. The two tools the conditional edge routes between.

check_loan_application_status returns a miss rather than raising, because an
unknown record id is a normal outcome that has to reach the user through the
same response envelope as a hit. A node that has to wrap a tool call in a try
block is a node that has two shapes, and the schema only has one.

Part 4 Task 14 wraps this function as an MCP tool, so its docstring is part
of the interface rather than a comment.
"""

import config
from agent import escalation
import dataset
from db import query
from rag import generate
import obs

# Every key the lookup block carries, on a hit and on a miss alike.
LOOKUP_FIELDS = (
    "found",
    "record_id",
    "status",
    "loan_amount_inr",
    "days_since_created",
    "flagged_for_fraud_review",
    "escalation_score",
    "recommend_escalation",
    "customer_context",
)


def check_loan_application_status(record_id: str, conn=None) -> dict:
    """Look up one Meridian Bank loan application and score its urgency.

    Returns the application's current status and sanctioned amount, together
    with a designed escalation score in [0, 1] that combines the fraud-review
    flag with a status-aware recency signal, and a boolean saying whether that
    score clears the recommended escalation threshold.

    The score weights the fraud flag at 0.45 and staleness at 0.55, where
    staleness is the application's age capped at 21 days, counted at full rate
    while the application is open, at half rate once it is disbursed, and not
    at all once it is rejected. Escalation is recommended at 0.50 and above,
    which is the 80th percentile of the score across the dataset.

    An unknown record id returns the same keys with `found` set to False
    rather than raising, so every outcome fits one response shape.

    Args:
        record_id: The application id, formatted `LN-1001`.
        conn: An open sqlite3 connection, or None to open and close one.

    Returns:
        A dict carrying every key in LOOKUP_FIELDS. No PII: the customer
        context is the four non-PII fields of db.query.CONTEXT_FIELDS.
    """
    record = dataset.get_application(record_id)
    if record is None:
        obs.event("agent.lookup", record_id=record_id, found=False)
        return dict.fromkeys(LOOKUP_FIELDS) | {"found": False, "record_id": record_id}

    obs.event(
        "agent.lookup",
        record_id=record_id,
        found=True,
        status=record["status"],
        recommend_escalation=escalation.recommend_escalation(record),
    )
    return {
        "found": True,
        "record_id": record_id,
        "status": record["status"],
        "loan_amount_inr": record["loan_amount_inr"],
        "days_since_created": record["days_since_created"],
        "flagged_for_fraud_review": record["flagged_for_fraud_review"],
        "escalation_score": escalation.escalation_score(record),
        "recommend_escalation": escalation.recommend_escalation(record),
        "customer_context": query.customer_context(record_id, conn=conn),
    }


def answer_policy_question(query_text: str) -> generate.GroundedAnswer:
    """The RAG tool, pinned to the collection Task 5 recommended."""
    return generate.answer(query_text, strategy=config.STRATEGY_SENTENCES)
