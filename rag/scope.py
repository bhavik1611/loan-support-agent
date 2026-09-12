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
# deliberately never duplicated here, so the corpus stays the authority. So is
# the licence that lets one of these phrases stand beside a Meridian product
# without refusing; see classify.
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


# Every letter that is not a vowel, for the one plural rule that needs to look
# at the letter before the final "y". "salary account" must not become "salary
# accounties", and it does not, because the rule reads the last letter of the
# phrase and that is "t".
_CONSONANTS = frozenset("bcdfghjklmnpqrstvwxz")


def _plural(phrase: str) -> str:
    """The natural English plural of a gate phrase, case-folded.

    Two rules cover all 24 names in the catalogue and KNOWN_ADJACENT. A phrase
    ending in a consonant plus "y" takes "-ies": English writes "insurance
    policies" and "cryptocurrencies", and nobody has ever written "insurance
    policys". Everything else takes "-s".

    The "-y" rule was missing for one round, and the cost was measured rather
    than imagined: "What insurance policies does Meridian offer?" passed the
    gate and was answered at 0.4838 on kb_sentences out of kb-13 and kb-01,
    with an answer about the fixed-obligation-to-income ratio for a Personal
    Loan. That is the exact confident wrong answer the four insurance phrases
    were added to prevent, arriving through the plural of one of them.
    """
    if len(phrase) >= 2 and phrase.endswith("y") and phrase[-2] in _CONSONANTS:
        return phrase[:-1] + "ies"
    return phrase + "s"


def _pattern(phrase: str) -> re.Pattern:
    """Whole-phrase match on the case-folded query, singular or plural.

    The alternation is the canonical phrase and its plural, not an optional
    trailing "s" on the canonical itself. Those are different on any phrase
    that already ends in "s": an optional "s" on "shares" would match the verb
    "share" as well, and the point of matching whole phrases is that a shorter
    word inside one is not the phrase. "shares" is gone for that collision, and
    the alternation is what keeps the guarantee available to whatever replaces
    it.

    Mass nouns and acronyms get a plural nobody writes - "golds", "ELSSs" - and
    that is harmless: a form no customer types matches no query. The rule is
    there for "mutual funds", "fixed deposits", "SIPs", "savings accounts" and
    "insurance policies", which people do write.
    """
    folded = phrase.casefold()
    escaped = re.escape(folded)
    plural = re.escape(_plural(folded))
    return re.compile(rf"\b(?:{escaped}|{plural})\b")


def _longest(phrases: list[str]) -> str:
    """Longest phrase, ties broken alphabetically so iteration order is inert."""
    return sorted(phrases, key=lambda phrase: (-len(phrase), phrase))[0]


def classify(query: str) -> ScopeVerdict:
    """The three outcomes of spec section 8.5, and only three.

    **When a query names both a Meridian product and an adjacent one, the
    adjacent phrase refuses unless the corpus licenses that exact pairing.**
    The licence is data in knowledge_base/catalogue.json, read through
    kb.licensed_pairs, not a branch written here, so the corpus stays the
    authority for the boundary exactly as it is for the document tags.

    Two rules were tried before this one and both were wrong, each on a
    different half of the queries, and both for the same reason: they decided a
    two-product query by a property with nothing to do with whether the
    documents cover the pair.

    - Longest-match-first across both lists refused "Can I open a fixed deposit
      in my NRE account?", which kb-18 answers outright at 0.6910 with all three
      chunks agreeing, because "fixed deposit" runs two characters longer than
      "NRE account". Ten phrase pairs across the two lists sit within two
      characters of each other, so the verdicts it got right were right by an
      accident of spelling.
    - Catalogue-wins-unconditionally fixed that one and opened the answering
      direction, which is worse. "Is my joint account covered by life insurance?"
      resolved to "joint account" and was answered at 0.5615 out of
      kb-11-joint-account-rules, a document about survivorship on death that says
      nothing about insurance. The user asking whether Meridian sells life cover
      got told what happens to the account when a holder dies, with a citation.

    What made the NRE case right was never the lengths: kb-18 line 10 says an
    NRE or NRO account can be opened as a term deposit. That is a fact in the
    corpus, so the corpus is what the rule now asks. On today's documents the
    licence holds two entries, the NRE and NRO halves of that one sentence, and
    nothing else pairs a Meridian product with something Meridian does not sell.
    """
    folded = query.casefold()
    catalogue_hits = [p for p in kb.catalogue_products() if _pattern(p).search(folded)]
    adjacent_hits = [p for p in KNOWN_ADJACENT if _pattern(p).search(folded)]

    # An adjacent phrase is discharged only by a catalogue product the query
    # also names and the corpus licenses it against. Any phrase left standing
    # refuses, because the corpus has nothing that covers the combination.
    licensed = {(pair.product, pair.adjacent) for pair in kb.licensed_pairs()}
    standing = [
        adjacent
        for adjacent in adjacent_hits
        if not any((product, adjacent) in licensed for product in catalogue_hits)
    ]

    if standing:
        return ScopeVerdict(
            query=query,
            product=_longest(standing),
            in_catalogue=False,
            known_adjacent=True,
        )
    if catalogue_hits:
        return ScopeVerdict(
            query=query,
            product=_longest(catalogue_hits),
            in_catalogue=True,
            known_adjacent=False,
        )
    return ScopeVerdict(query=query, product="", in_catalogue=False, known_adjacent=False)
