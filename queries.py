"""Evaluation queries and their ground-truth labels (Part 1, Tasks 4 and 5).

READ THIS BEFORE CHANGING ANYTHING HERE.

These labels were written by reading the knowledge base and deciding which
documents genuinely answer each question. They were committed to git BEFORE any
retrieval was run. That order is the whole point: labels derived from what a
retriever returned would make Precision@3 a measure of nothing.

Each in-scope query has TWO relevant documents, deliberately. With one relevant
document the best achievable Precision@3 is 1/3 for every strategy, so the
Task 5 comparison would measure noise. With two, the ceiling is 2/3 and the
strategies have room to separate.

Out-of-scope queries are split into two clusters, because they test different
things. A question about the weather is trivially far and any threshold catches
it. A question about a banking product Cred does not document is the one that
tells you whether your threshold is real.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    relevant_docs: tuple[str, ...]   # empty means out of scope
    rationale: str
    cluster: str = "in-scope"        # or "far", or "adjacent"


IN_SCOPE: list[Query] = [
    Query(
        "Q1",
        "What credit score do I need for a home loan and how does it affect my rate?",
        ("KB-01", "KB-07"),
        "KB-01 states the 750 floor for Home Loans; KB-07 states that the score "
        "positions the borrower within the 8.25 to 10.5 percent band.",
    ),
    Query(
        "Q2",
        "How is my EMI calculated, and does prepaying early actually save me money?",
        ("KB-02", "KB-08"),
        "KB-02 gives the reducing-balance formula and says early instalments are "
        "mostly interest; KB-08 gives the prepayment charges that decide whether "
        "the saving survives the fee.",
    ),
    Query(
        "Q3",
        "What documents do I need to apply for a home loan?",
        ("KB-14", "KB-04"),
        "KB-14 lists the property papers specific to a Home Loan; KB-04 lists the "
        "identity, address and income documents every applicant needs.",
    ),
    Query(
        "Q4",
        "Someone made a transaction on my card that I did not authorise. What happens now?",
        ("KB-05", "KB-18"),
        "KB-05 is the dispute process, liability window and provisional credit; "
        "KB-18 covers what happens if the member is unhappy with the outcome.",
    ),
    Query(
        "Q5",
        "What charges can hit my savings account?",
        ("KB-09", "KB-06"),
        "KB-09 has the minimum-balance penalty; KB-06 has the closure fee within "
        "twelve months. Those are the two charges a savings account can attract.",
    ),
    Query(
        "Q6",
        "How long does a home loan take to approve, and why is it slower than a personal loan?",
        ("KB-13", "KB-14"),
        "KB-13 gives the per-product decision windows; KB-14 explains the "
        "valuation and title checks that make a Home Loan the slowest product.",
    ),
    Query(
        "Q7",
        "Can my spouse and I apply for a loan together and hold the account jointly?",
        ("KB-17", "KB-11"),
        "KB-17 covers co-applicants and shared liability on the loan; KB-11 "
        "covers joint holding and operating mandates on the account.",
    ),
    Query(
        "Q8",
        "I have moved abroad for work. What account can I open and what must I provide?",
        ("KB-12", "KB-04"),
        "KB-12 covers NRE, NRO and FCNR eligibility and the visa test; KB-04 is "
        "the KYC pack that still applies.",
    ),
]

OUT_OF_SCOPE: list[Query] = [
    Query(
        "X1",
        "What is the weather in Mumbai today?",
        (),
        "Nothing in a banking knowledge base could answer this. Any threshold "
        "catches it, which is why it proves very little on its own.",
        cluster="far",
    ),
    Query(
        "X2",
        "What are your fixed deposit interest rates?",
        (),
        "Plausible banking language, and Cred has no fixed-deposit document. "
        "This is the query that tells you whether the threshold is real.",
        cluster="adjacent",
    ),
    Query(
        "X3",
        "Do you offer travel insurance for international trips?",
        (),
        "Adjacent again: a product a bank might well sell, absent from this "
        "knowledge base. KB-03 mentions foreign currency markup, which is the "
        "nearest wrong answer a retriever could reach for.",
        cluster="adjacent",
    ),
]

ALL_QUERIES = IN_SCOPE + OUT_OF_SCOPE


def summary() -> str:
    lines = [
        f"{len(IN_SCOPE)} in-scope queries, {len(OUT_OF_SCOPE)} out-of-scope "
        f"({sum(1 for q in OUT_OF_SCOPE if q.cluster == 'far')} trivially far, "
        f"{sum(1 for q in OUT_OF_SCOPE if q.cluster == 'adjacent')} banking-adjacent)",
        "",
        "Ground truth, written before retrieval ever ran:",
    ]
    for q in IN_SCOPE:
        lines.append(f"  {q.query_id}  {len(q.relevant_docs)} relevant: "
                     f"{', '.join(q.relevant_docs)}")
        lines.append(f"      {q.text}")
    ceiling = sum(min(len(q.relevant_docs), 3) / 3 for q in IN_SCOPE) / len(IN_SCOPE)
    lines += [
        "",
        f"Best achievable Precision@3 with these labels: {ceiling:.3f}",
        "Best achievable Recall@3: 1.000",
        "(With one relevant document per query the precision ceiling would be "
        "0.333 for both strategies, and the comparison would be meaningless.)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
