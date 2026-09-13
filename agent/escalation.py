"""Task 6. The designed escalation score, per spec D-33 and D-34.

The brief forbids "a bare boolean OR". That is a measurable property, not a
style note: a score whose threshold selects exactly the fraud-flagged records
is an OR wearing a formula. Two shapes were measured on this dataset and
rejected for that reason, both recorded in D-33.

What makes this one a blend is the status term. Staleness runs at full rate
while an application is open, at half rate once it is disbursed because the
money has left the bank but nobody is waiting on a decision, and stops
entirely once it is rejected. So a fresh flagged application can sit below
the line while a stale unflagged one crosses it, which is the whole point.

The open-status set is imported rather than redefined, so this file and
db/query.py cannot drift apart.
"""

import config
from db.query import OPEN_STATUSES


def staleness(days_since_created: int, status: str) -> float:
    """How much the clock counts against this application, in [0, 1]."""
    raw = min(days_since_created / config.ESCALATION_SATURATION_DAYS, 1.0)
    if status in OPEN_STATUSES:
        return raw
    if status == config.ESCALATION_HALF_RATE_STATUS:
        return config.ESCALATION_DISBURSED_RATE * raw
    return 0.0


def escalation_score(record: dict) -> float:
    """How urgently a human should look at this application, in [0, 1]."""
    fraud = 1.0 if record["flagged_for_fraud_review"] else 0.0
    stale = staleness(record["days_since_created"], record["status"])
    return round(
        config.ESCALATION_FRAUD_WEIGHT * fraud
        + config.ESCALATION_STALENESS_WEIGHT * stale,
        4,
    )


def recommend_escalation(record: dict) -> bool:
    """Whether the score clears the threshold measured in D-34."""
    return escalation_score(record) >= config.ESCALATION_THRESHOLD
