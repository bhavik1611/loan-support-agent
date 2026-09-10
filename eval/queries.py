"""The twelve evaluation queries and their hand-authored gold labels.

Authored and committed before any retrieval was run. Sizes are mixed on
purpose: with one relevant document per query, Recall@3 can only take the
values 0 and 1, which makes it decoration rather than a metric.

Four queries have one relevant document, five have two, three have three.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    text: str
    gold_doc_ids: tuple[str, ...]


EVAL_QUERIES: list[EvalQuery] = [
    # --- one relevant document ---
    EvalQuery("EQ-01", "How is the EMI on a loan calculated?", ("kb-02-emi-calculation",)),
    EvalQuery("EQ-02", "What is the process to close my savings account?", ("kb-06-account-closure",)),
    EvalQuery("EQ-03", "What minimum balance must I keep in my savings account?", ("kb-09-minimum-balance",)),
    EvalQuery("EQ-04", "What are the rules for operating a joint account?", ("kb-11-joint-account-rules",)),
    # --- two relevant documents ---
    EvalQuery(
        "EQ-05",
        "What income do I need to qualify for a personal loan?",
        ("kb-01-loan-eligibility", "kb-13-personal-loan-eligibility"),
    ),
    EvalQuery(
        "EQ-06",
        "Which documents does the bank need for KYC?",
        ("kb-04-kyc-documents", "kb-16-kyc-reverification"),
    ),
    EvalQuery(
        "EQ-07",
        "What charges apply if I pay my credit-card bill late?",
        ("kb-03-credit-card-fees", "kb-15-card-late-payment-charges"),
    ),
    EvalQuery(
        "EQ-08",
        "Is there a penalty for paying off my loan early?",
        ("kb-08-prepayment-penalty", "kb-17-foreclosure-vs-part-prepayment"),
    ),
    EvalQuery(
        "EQ-09",
        "Can a non-resident Indian open and operate an account here?",
        ("kb-12-nri-account-eligibility", "kb-18-nre-vs-nro-operation"),
    ),
    # --- three relevant documents ---
    EvalQuery(
        "EQ-10",
        "What interest rate will I get on a home loan and how much can I borrow against the property?",
        ("kb-01-loan-eligibility", "kb-07-interest-rate-slabs", "kb-14-home-loan-ltv"),
    ),
    EvalQuery(
        "EQ-11",
        "How do I report a fraudulent card transaction and will it affect my credit score?",
        ("kb-03-credit-card-fees", "kb-05-fraud-dispute", "kb-10-credit-score-impact"),
    ),
    EvalQuery(
        "EQ-12",
        "What decides how much I can borrow and what the loan will cost me each month?",
        ("kb-01-loan-eligibility", "kb-02-emi-calculation", "kb-07-interest-rate-slabs"),
    ),
]
