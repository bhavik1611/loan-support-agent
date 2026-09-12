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
"""

import re
from dataclasses import dataclass

from rag import kb

# Products Meridian Bank does not sell that people ask about anyway. The
# catalogue side of the gate is read from knowledge_base/catalogue.json and is
# deliberately never duplicated here, so the corpus stays the authority.
KNOWN_ADJACENT: list[str] = [
    "fixed deposit",
    "recurring deposit",
    "mutual fund",
    "SIP",
    "ELSS",
    "demat",
    "shares",
    "stock market",
    "insurance",
    "gold",
    "cryptocurrency",
    "income tax",
    "GST",
    "tax return",
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
    are different: an optional "s" on "shares" would also match the verb
    "share", and "Can I share my account?" is not a question about equities.
    """
    escaped = re.escape(phrase.casefold())
    return re.compile(rf"\b(?:{escaped}|{escaped}s)\b")


def classify(query: str) -> ScopeVerdict:
    """The three outcomes of spec section 8.5, and only three.

    Longest match first, so "joint account" beats nothing it contains and a
    query naming both a catalogue product and an adjacent one is decided by the
    more specific phrase. On an exact tie the catalogue wins: Meridian sells it,
    so it is in scope.
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

    # Longest phrase first, catalogue ahead of adjacent on a tie, then
    # alphabetical so the verdict never depends on iteration order.
    _, _, product, in_catalogue = sorted(
        candidates, key=lambda c: (-c[0], c[1], c[2])
    )[0]
    return ScopeVerdict(
        query=query,
        product=product,
        in_catalogue=in_catalogue,
        known_adjacent=not in_catalogue,
    )
