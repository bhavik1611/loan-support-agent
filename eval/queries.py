"""The golden dataset: 29 hand-authored items in four classes, per spec 9.1.

This file is the **scored** half of the corpus and it fits nothing. The
calibration probes in eval/calibration.py are the fitting half, and no string
appears in both. tests/test_queries.py enforces that boundary in both
directions, and it is the reason this can be called a golden dataset at all.

Labels were authored and committed before any retrieval was run. Sizes in the
answerable class are mixed on purpose: with one relevant document per query,
Recall@3 can only take the values 0 and 1, which makes it decoration rather
than a metric. Four answerable items have one relevant document, five have two,
three have three.

The four classes pull on different mechanisms, which is the point of splitting
them (D-55):

| class              | ids           | n  | correct behaviour                       |
|--------------------|---------------|----|-----------------------------------------|
| answerable         | EQ-01..EQ-12  | 12 | answer, citing its parent               |
| outside_boundary   | OB-01..OB-10  | 10 | refuse at the gate, before retrieval    |
| inside_uncovered   | IU-01..IU-02  |  2 | refuse on the threshold and support rule|
| far_out_of_scope   | FO-01..FO-05  |  5 | refuse on the threshold                 |

The 12 answerable items keep the ids and the byte-identical text they were
committed with, so every Precision@3 and Recall@3 number already in README.md
stays comparable across this amendment.

The outside_boundary class was thirteen items and is ten. Fix round 1 removed
"shares", "tax return", "GST" and "income tax" from rag/scope.KNOWN_ADJACENT,
because each refused a question the corpus answers, and the three items that
exercised the removed phrases were retired with them rather than reworded onto a
surviving phrase. An item whose only job was to test a vocabulary entry has no
job once the entry is gone, and moving one into far_out_of_scope would have been
worse: the retired tax-return item reads 0.3703 under sentence chunking, above T,
and is refused by the support rule alone, so criterion 29 would have been
asserting on a coin flip. The class is now exactly one item per surviving phrase,
and ten phrases mean ten items.
"""

from dataclasses import dataclass

KIND_ANSWERABLE = "answerable"
KIND_OUTSIDE_BOUNDARY = "outside_boundary"
KIND_INSIDE_UNCOVERED = "inside_uncovered"
KIND_FAR_OUT_OF_SCOPE = "far_out_of_scope"

KINDS = (
    KIND_ANSWERABLE,
    KIND_OUTSIDE_BOUNDARY,
    KIND_INSIDE_UNCOVERED,
    KIND_FAR_OUT_OF_SCOPE,
)


@dataclass(frozen=True)
class GoldenItem:
    item_id: str
    text: str
    kind: str
    gold_doc_ids: tuple[str, ...] = ()  # non-empty only when kind is answerable
    product: str = ""  # the product the text names, "" when it names none


GOLDEN_DATASET: list[GoldenItem] = [
    # --- answerable: one relevant document ---
    GoldenItem(
        "EQ-01",
        "How is the EMI on a loan calculated?",
        KIND_ANSWERABLE,
        ("kb-02-emi-calculation",),
    ),
    GoldenItem(
        "EQ-02",
        "What is the process to close my savings account?",
        KIND_ANSWERABLE,
        ("kb-06-account-closure",),
        "savings account",
    ),
    GoldenItem(
        "EQ-03",
        "What minimum balance must I keep in my savings account?",
        KIND_ANSWERABLE,
        ("kb-09-minimum-balance",),
        "savings account",
    ),
    GoldenItem(
        "EQ-04",
        "What are the rules for operating a joint account?",
        KIND_ANSWERABLE,
        ("kb-11-joint-account-rules",),
        "joint account",
    ),
    # --- answerable: two relevant documents ---
    GoldenItem(
        "EQ-05",
        "What income do I need to qualify for a personal loan?",
        KIND_ANSWERABLE,
        ("kb-01-loan-eligibility", "kb-13-personal-loan-eligibility"),
        "Personal Loan",
    ),
    GoldenItem(
        "EQ-06",
        "Which documents does the bank need for KYC?",
        KIND_ANSWERABLE,
        ("kb-04-kyc-documents", "kb-16-kyc-reverification"),
    ),
    GoldenItem(
        "EQ-07",
        "What charges apply if I pay my credit-card bill late?",
        KIND_ANSWERABLE,
        ("kb-03-credit-card-fees", "kb-15-card-late-payment-charges"),
    ),
    GoldenItem(
        "EQ-08",
        "Is there a penalty for paying off my loan early?",
        KIND_ANSWERABLE,
        ("kb-08-prepayment-penalty", "kb-17-foreclosure-vs-part-prepayment"),
    ),
    GoldenItem(
        "EQ-09",
        "Can a non-resident Indian open and operate an account here?",
        KIND_ANSWERABLE,
        ("kb-12-nri-account-eligibility", "kb-18-nre-vs-nro-operation"),
    ),
    # --- answerable: three relevant documents ---
    GoldenItem(
        "EQ-10",
        "What interest rate will I get on a home loan and how much can I borrow against the property?",
        KIND_ANSWERABLE,
        ("kb-01-loan-eligibility", "kb-07-interest-rate-slabs", "kb-14-home-loan-ltv"),
        "Home Loan",
    ),
    GoldenItem(
        "EQ-11",
        "How do I report a fraudulent card transaction and will it affect my credit score?",
        KIND_ANSWERABLE,
        ("kb-03-credit-card-fees", "kb-05-fraud-dispute", "kb-10-credit-score-impact"),
    ),
    GoldenItem(
        "EQ-12",
        "What decides how much I can borrow and what the loan will cost me each month?",
        KIND_ANSWERABLE,
        ("kb-01-loan-eligibility", "kb-02-emi-calculation", "kb-07-interest-rate-slabs"),
    ),
    # --- outside_boundary: names a product Meridian does not sell, per D-47 ---
    # These are the items the gate was sized against. Each one names a phrase in
    # rag/scope.KNOWN_ADJACENT, and the correct behaviour is a refusal before
    # any retrieval runs. The product column below is the canonical spelling the
    # refusal carries, not a copy of the item's own wording. Ten items for ten
    # surviving phrases, one each, after fix round 1 retired the three that
    # tested phrases the corpus turned out to answer.
    GoldenItem(
        "OB-01",
        "What is the interest rate on a fixed deposit for 5 years?",
        KIND_OUTSIDE_BOUNDARY,
        (),
        "fixed deposit",
    ),
    GoldenItem(
        "OB-02",
        "How do I open a recurring deposit with a monthly instalment of 5,000 rupees?",
        KIND_OUTSIDE_BOUNDARY,
        (),
        "recurring deposit",
    ),
    GoldenItem(
        "OB-03",
        "Which mutual funds does the bank recommend for a five-year horizon?",
        KIND_OUTSIDE_BOUNDARY,
        (),
        "mutual fund",
    ),
    GoldenItem(
        "OB-04",
        "Can I start a SIP of 2,000 rupees every month?",
        KIND_OUTSIDE_BOUNDARY,
        (),
        "SIP",
    ),
    GoldenItem(
        "OB-05",
        "Does an ELSS investment qualify for a deduction under section 80C?",
        KIND_OUTSIDE_BOUNDARY,
        (),
        "ELSS",
    ),
    GoldenItem(
        "OB-06",
        "How do I open a demat account to hold my securities?",
        KIND_OUTSIDE_BOUNDARY,
        (),
        "demat",
    ),
    GoldenItem(
        "OB-07",
        "What are the stock market timings on a settlement holiday?",
        KIND_OUTSIDE_BOUNDARY,
        (),
        "stock market",
    ),
    GoldenItem(
        "OB-08",
        "Does the bank sell term life insurance cover?",
        KIND_OUTSIDE_BOUNDARY,
        (),
        "insurance",
    ),
    GoldenItem(
        "OB-09",
        "What is today's gold rate per gram?",
        KIND_OUTSIDE_BOUNDARY,
        (),
        "gold",
    ),
    GoldenItem(
        "OB-10",
        "Can I buy cryptocurrency through this bank's mobile app?",
        KIND_OUTSIDE_BOUNDARY,
        (),
        "cryptocurrency",
    ),
    # --- inside_uncovered: inside D-47's boundary, no document covers it ---
    # The residue named in spec 18.1 item 8, verbatim. The first names a product
    # a reader would call in-catalogue but asks about a competitor; the second is
    # squarely inside the boundary with no document behind it. Neither is caught
    # by the gate, and both fall through to the threshold and the support rule.
    GoldenItem(
        "IU-01",
        "Can I get a credit card from another bank with a low limit?",
        KIND_INSIDE_UNCOVERED,
    ),
    GoldenItem(
        "IU-02",
        "How do I transfer money to an account in another country?",
        KIND_INSIDE_UNCOVERED,
    ),
    # --- far_out_of_scope: not banking at all ---
    # Written fresh for this dataset. Acceptance criterion 30 forbids reusing any
    # of the 17 far probes in eval/calibration.py, because a threshold scored on
    # the strings that set it measures nothing. The subjects are deliberately
    # chosen away from the probes' subjects too, not reworded from them.
    GoldenItem("FO-01", "How many moons orbit the planet Neptune?", KIND_FAR_OUT_OF_SCOPE),
    GoldenItem("FO-02", "What is the correct way to castle in a game of chess?", KIND_FAR_OUT_OF_SCOPE),
    GoldenItem("FO-03", "In which year did the Berlin Wall come down?", KIND_FAR_OUT_OF_SCOPE),
    GoldenItem("FO-04", "How often should I water a snake plant indoors?", KIND_FAR_OUT_OF_SCOPE),
    GoldenItem("FO-05", "How do I say thank you in Japanese?", KIND_FAR_OUT_OF_SCOPE),
]

# The answerable subset, kept under its original name so every Task 5 caller
# keeps working unchanged. Derived, never a second copy of the strings.
EVAL_QUERIES: list[GoldenItem] = [
    item for item in GOLDEN_DATASET if item.kind == KIND_ANSWERABLE
]


def items_of_kind(kind: str) -> list[GoldenItem]:
    """Every golden item in one class, in dataset order."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}, expected one of {list(KINDS)}")
    return [item for item in GOLDEN_DATASET if item.kind == kind]
