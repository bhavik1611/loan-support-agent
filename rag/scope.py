"""The product gate. It runs before retrieval, and it decides scope (D-51).

D-46 measured why this file exists. Holding one sentence frame fixed and
swapping only the product noun, "fixed deposit" against "car insurance policy"
scores 0.6805, while the lowest genuine in-scope probe scores 0.3263 against
the document that actually answers it. Cosine similarity encodes the shape of
a question, not the product it names, so no cut on that signal separates a
deposit from an education loan. The similarity threshold is demoted to a floor
against gibberish; topicality is decided here instead.

Two lists and one function, and the matching is deterministic string work only.
D-51 rejected an embedding router over product-name centroids for losing to the
same measurement the threshold lost to, and D-52 rejected a positive catalogue
on its own: "Suggest me a good SIP" names no Meridian product, so a membership
test sees nothing and falls through to the search that answers it wrongly.

The weakness is stated rather than engineered around, and it is item 8 of spec
section 18.1: KNOWN_ADJACENT is curated, so an adjacent product nobody thought
to add still falls through to the threshold.

**KNOWN_ADJACENT names products and instruments Meridian does not sell. It does
not name topics.** That is the whole admission rule, and it is what all four
phrases removed on 2026-09-12 had in common: GST, income tax and tax returns are
topics, "shares" is also ordinary English, and the corpus discusses every one of
them where it touches banking.

The rule is enforced by measurement, not by taxonomy. Spec section 9.4 sets the
standard: a gate that refuses a real question is worse than the failure it was
built to fix. So the test is not "is this a product Meridian sells" but "does
refusing every query containing this phrase refuse a question the knowledge base
answers". All four failed it:

- "GST": kb-03, kb-08 and kb-09 each state Meridian's GST treatment of a charge,
  and "Does GST apply to the late payment fee on my credit card?" answers at
  0.5765 on kb-15 unfiltered. The gate was also inconsistent with itself, since
  the spelled-out "goods and services tax" the documents actually use was never
  in the list and passed the same question through.
- "tax return": kb-01 and kb-04 both name income-tax returns as accepted proof
  of income, and the hyphenated form matches the pattern.
- "shares": the third-person verb collides with it, so "My wife shares the
  account with me, can she operate it?" was refused although kb-11 answers it at
  0.3619. Removal costs nothing: "Which shares should I buy this quarter?" reads
  0.1968 and 0.2198, far under T, so the threshold refuses it anyway.
- "income tax": the same inconsistency as GST, one spelling apart. "Do I need my
  income-tax returns to prove income for a loan?" passed the gate and "Do I need
  my income tax returns to prove income for a loan?" was refused by it.

The argument that kept "income tax" for one round is worth recording because it
is a tempting mistake: every in-scope tax sentence in the corpus does name a
Meridian product beside it, kb-12 and kb-18 both saying "NRE account", so longest
match was expected to resolve to the catalogue. But the gate matches the query,
not the corpus. That query ends "for a loan", and bare "loan" is not a catalogue
product - the list holds "Personal Loan", "Auto Loan", "Education Loan",
"Business Loan" and "Home Loan", never "loan" alone - so there was nothing for
longest match to win against.

Removing it has a measured cost, carried rather than hidden. "How much income tax
will I owe on my salary this year?" now falls through, and under kb_fixed_400_80
it is answered at 0.3200 from kb-13 with two chunks agreeing. Under kb_sentences,
the recommended collection and Part 2's fixed input, it is refused at 0.3466 with
no two chunks agreeing. The removal buys a correct refusal of an answerable
question and costs one false answer on the collection we do not ship.

The same admission rule cost the bare noun "insurance" on 2026-09-12. It names
an industry, not a product Meridian declines to sell, and in a banking question
"insurance" usually names a merchant, a salary deduction or a document: "An
insurance company debited my card twice without my authorisation, how do I
dispute it?" answers from kb-05 at 0.4221 with its chunks agreeing, and the gate
refused it along with seven other probes the corpus answers.

Deleting it outright was measured and rejected. Without any insurance phrase,
"Does the bank sell term life insurance cover?" is answered at 0.3685 out of
kb-06-account-closure, which is a confident wrong answer to a product question.
So the noun is replaced by the four phrases that name the product rather than
the industry - "insurance policy", "life insurance", "insurance cover" and
"term insurance" - and the industry noun on its own now passes through to
retrieval, where the corpus can answer it.
"""

import re
from dataclasses import dataclass

from rag import kb

# Products Meridian Bank does not sell that people ask about anyway. The
# catalogue side of the gate is read from knowledge_base/catalogue.json and is
# deliberately never duplicated here, so the corpus stays the authority.
# Thirteen phrases, every one of them a product or an instrument, each measured
# against the corpus per the docstring above.
KNOWN_ADJACENT: list[str] = [
    "fixed deposit",
    "recurring deposit",
    "mutual fund",
    "SIP",
    "ELSS",
    "demat",
    "stock market",
    "insurance policy",
    "life insurance",
    "insurance cover",
    "term insurance",
    "gold",
    "cryptocurrency",
]


@dataclass(frozen=True)
class ScopeVerdict:
    """What the gate decided, and the product name the refusal has to carry."""
    query: str
    product: str
    in_catalogue: bool
    known_adjacent: bool


def _pattern(phrase: str) -> re.Pattern:
    """Whole-phrase match on the case-folded query, singular or plural.

    The alternation is the canonical phrase and the canonical phrase with a
    trailing "s", not an optional trailing "s" on the canonical itself. Those
    are different on any phrase that already ends in "s": an optional "s" on
    "deposits" would match "deposit" as well, and the point of matching whole
    phrases is that a shorter word inside one is not the phrase.

    No surviving entry ends in "s", because the one that did, "shares", was
    removed for colliding with the verb. The alternation still earns its place
    on "mutual funds", "fixed deposits" and "SIPs", which people do write.
    """
    escaped = re.escape(phrase.casefold())
    return re.compile(rf"\b(?:{escaped}|{escaped}s)\b")


def classify(query: str) -> ScopeVerdict:
    """The three outcomes of spec section 8.5, and only three.

    **The catalogue wins whenever a query names both sides, whatever the
    lengths.** If the query names something Meridian sells, the question is in
    scope, and how many characters the competing phrase happens to run to has
    nothing to do with that. Length only breaks a tie within one side.

    This replaced longest-match-first, which ranked by length across both lists
    and let the catalogue win only an exact character tie. That rule refused
    "Can I open a fixed deposit in my NRE account?", a question kb-18 answers
    outright at 0.6910 with all three chunks agreeing, because "fixed deposit"
    runs two characters longer than "NRE account". Ten phrase pairs in the two
    lists sat within two characters of each other, so five of the surviving
    verdicts were correct by an accident of spelling rather than by the rule.
    """
    folded = query.casefold()
    candidates = []
    for product in kb.catalogue_products():
        if _pattern(product).search(folded):
            candidates.append((len(product), 0, product, True))
    for product in KNOWN_ADJACENT:
        if _pattern(product).search(folded):
            candidates.append((len(product), 1, product, False))

    if not candidates:
        return ScopeVerdict(query=query, product="", in_catalogue=False, known_adjacent=False)

    # Catalogue ahead of adjacent unconditionally, then longest phrase within
    # one side, then alphabetical so the verdict never depends on iteration
    # order. The tuple is (length, side, phrase, in_catalogue), so the key puts
    # side first and negates length.
    _, _, product, in_catalogue = sorted(
        candidates, key=lambda c: (c[1], -c[0], c[2])
    )[0]
    return ScopeVerdict(
        query=query,
        product=product,
        in_catalogue=in_catalogue,
        known_adjacent=not in_catalogue,
    )
