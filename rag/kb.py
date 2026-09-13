"""Load knowledge_base/ into Document objects.

The documents are plain text and carry no metadata of their own. A document is
the prose a support team would actually write; what that prose is *about* lives
beside it in catalogue.json, the way a real document store keeps its index
separate from its content (D-32).

Both sides are parsed strictly. A catalogue entry with no file, a file with no
catalogue entry, or a missing or mistyped field all raise, because a document
that silently loses its doc_id becomes unscoreable in Task 5 and nothing else
would report it.

The catalogue also carries the product boundary of D-47: a top-level `products`
list naming what Meridian Bank sells, and a per-document `products` tag naming
which of them that document covers. The gate in rag/scope.py and the optional
filter in rag/retrieve.py both read it from here, so the catalogue stays the
single authority and nothing duplicates the list (D-52).

The third top-level key, `licensed_pairs`, is the corpus licence the gate needs
when a query names a Meridian product and an adjacent one in the same sentence.
It says which of those pairings the documents actually cover, and it lives here
rather than in rag/scope.py for the same reason the tags do: the corpus is the
authority, and a special case written into the gate would not be.

Each entry names the document and quotes the sentence that licenses the pair,
plus `corpus_phrase`, which is the wording the document uses where the gate's
vocabulary uses another. kb-18 line 10 says an NRE or NRO account "can be opened
as a savings account or as a term deposit"; customers write "fixed deposit". The
two spellings are one instrument, and recording both is what lets a test hold
the licence against the corpus instead of against somebody's memory.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import config

CATALOGUE_NAME = "catalogue.json"
DOCUMENT_SUFFIX = ".txt"

PRODUCTS_KEY = "products"
DOCUMENTS_KEY = "documents"
LICENCES_KEY = "licensed_pairs"

_CATALOGUE_FIELDS = {"title": str, "topic": str, "required": bool, "products": list}
_LICENCE_FIELDS = {
    "product": str,
    "adjacent": str,
    "corpus_phrase": str,
    "document": str,
    "sentence": str,
}


@dataclass(frozen=True)
class LicensedPair:
    """One product pairing the corpus covers, and the sentence that says so.

    `product` is a catalogue product, `adjacent` is the phrase from
    rag/scope.KNOWN_ADJACENT it is licensed against, and `corpus_phrase` is how
    the document words that instrument. `document` and `sentence` are the
    receipt: tests/test_scope.py reads the body and fails if the sentence is not
    in it or does not name both terms, so the licence cannot outlive the prose.
    """
    product: str
    adjacent: str
    corpus_phrase: str
    document: str
    sentence: str


@dataclass(frozen=True)
class Document:
    """A document in the knowledge base."""
    doc_id: str
    title: str
    topic: str
    required: bool
    products: tuple[str, ...]
    body: str
    path: Path


def _load_catalogue(kb_dir: Path) -> tuple[list[str], dict[str, dict], list[LicensedPair]]:
    """The product list, the document entries and the licences, parsed strictly."""
    path = kb_dir / CATALOGUE_NAME
    if not path.exists():
        raise ValueError(f"{path} is missing; the catalogue is not optional")

    catalogue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(catalogue, dict):
        raise ValueError(
            f"{CATALOGUE_NAME}: expected an object with {PRODUCTS_KEY!r} and "
            f"{DOCUMENTS_KEY!r} keys"
        )
    for key in (PRODUCTS_KEY, DOCUMENTS_KEY, LICENCES_KEY):
        if key not in catalogue:
            raise ValueError(f"{CATALOGUE_NAME}: missing the top-level {key!r} key")

    products = catalogue[PRODUCTS_KEY]
    if not isinstance(products, list) or not products:
        raise ValueError(f"{CATALOGUE_NAME}: {PRODUCTS_KEY} must be a non-empty list")
    for name in products:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{CATALOGUE_NAME}: {PRODUCTS_KEY} holds a non-string entry")
    if len(set(products)) != len(products):
        raise ValueError(f"{CATALOGUE_NAME}: {PRODUCTS_KEY} repeats a name")

    entries = catalogue[DOCUMENTS_KEY]
    if not isinstance(entries, dict):
        raise ValueError(f"{CATALOGUE_NAME}: {DOCUMENTS_KEY} must be an object keyed by doc_id")

    known = set(products)
    for doc_id, meta in entries.items():
        if not isinstance(meta, dict):
            raise ValueError(f"{CATALOGUE_NAME}: {doc_id} is not an object")
        for field, expected in _CATALOGUE_FIELDS.items():
            if field not in meta:
                raise ValueError(f"{CATALOGUE_NAME}: {doc_id} is missing {field}")
            if not isinstance(meta[field], expected):
                raise ValueError(
                    f"{CATALOGUE_NAME}: {doc_id}.{field} must be {expected.__name__}, "
                    f"got {type(meta[field]).__name__}"
                )
        unknown = set(meta) - set(_CATALOGUE_FIELDS)
        if unknown:
            raise ValueError(f"{CATALOGUE_NAME}: {doc_id} has unknown fields {sorted(unknown)}")
        # A document covering nothing would be invisible to every filtered query
        # while still being retrievable unfiltered, which is the catalogue
        # disagreeing with itself quietly. tests/test_kb.py already requires the
        # real corpus to have none; the loader now requires it of any corpus.
        if not meta[PRODUCTS_KEY]:
            raise ValueError(
                f"{CATALOGUE_NAME}: {doc_id}.{PRODUCTS_KEY} is empty; every document "
                f"covers at least one product"
            )
        # Acceptance criterion 31. A document tag naming a product the catalogue
        # does not sell would let the gate filter on an id no search can match,
        # and the catalogue would be disagreeing with itself.
        stray = sorted(set(meta[PRODUCTS_KEY]) - known)
        if stray:
            raise ValueError(
                f"{CATALOGUE_NAME}: {doc_id}.{PRODUCTS_KEY} names {stray}, which is not in "
                f"the top-level {PRODUCTS_KEY} list"
            )

    raw_licences = catalogue[LICENCES_KEY]
    if not isinstance(raw_licences, list):
        raise ValueError(f"{CATALOGUE_NAME}: {LICENCES_KEY} must be a list")
    licences = []
    for position, entry in enumerate(raw_licences):
        where = f"{CATALOGUE_NAME}: {LICENCES_KEY}[{position}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} is not an object")
        for field, expected in _LICENCE_FIELDS.items():
            if field not in entry:
                raise ValueError(f"{where} is missing {field}")
            if not isinstance(entry[field], expected):
                raise ValueError(
                    f"{where}.{field} must be {expected.__name__}, "
                    f"got {type(entry[field]).__name__}"
                )
        unknown = set(entry) - set(_LICENCE_FIELDS)
        if unknown:
            raise ValueError(f"{where} has unknown fields {sorted(unknown)}")
        # A licence naming a product the bank does not sell, or a document that
        # does not exist, would widen the gate on the strength of nothing.
        if entry["product"] not in known:
            raise ValueError(
                f"{where}.product names {entry['product']!r}, which is not in the "
                f"top-level {PRODUCTS_KEY} list"
            )
        if entry["document"] not in entries:
            raise ValueError(
                f"{where}.document names {entry['document']!r}, which is not a "
                f"catalogued document"
            )
        licences.append(LicensedPair(**entry))
    pairs = [(licence.product, licence.adjacent) for licence in licences]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{CATALOGUE_NAME}: {LICENCES_KEY} licenses a pair twice")
    return products, entries, licences


def catalogue_products(kb_dir: Path | None = None) -> list[str]:
    """What Meridian Bank sells, in catalogue order. The authority for D-47."""
    kb_dir = config.KB_DIR if kb_dir is None else Path(kb_dir)
    products, _, _ = _load_catalogue(kb_dir)
    return list(products)


def licensed_pairs(kb_dir: Path | None = None) -> list[LicensedPair]:
    """The product pairings the corpus covers, in catalogue order.

    rag/scope.py turns these into the one exception to its refusal rule: an
    adjacent product phrase refuses even when the query also names something
    Meridian sells, unless some document covers that specific pairing.
    """
    kb_dir = config.KB_DIR if kb_dir is None else Path(kb_dir)
    _, _, licences = _load_catalogue(kb_dir)
    return list(licences)


def documents_for_product(product: str, kb_dir: Path | None = None) -> list[str]:
    """The doc_ids the catalogue tags with this product, sorted.

    This is the whole of the D-53 filter: rag/retrieve.py turns the list into a
    `doc_id` `$in` clause. Nothing about the product reaches chunk metadata, so
    neither collection is rebuilt for it.
    """
    kb_dir = config.KB_DIR if kb_dir is None else Path(kb_dir)
    products, entries, _ = _load_catalogue(kb_dir)
    if product not in products:
        raise ValueError(
            f"{product!r} is not in {CATALOGUE_NAME}'s {PRODUCTS_KEY} list; "
            f"known products are {sorted(products)}"
        )
    return sorted(
        doc_id for doc_id, meta in entries.items() if product in meta[PRODUCTS_KEY]
    )


def load_documents(kb_dir: Path | None = None) -> list[Document]:
    """Every document in the knowledge base, sorted by doc_id for determinism.

    The catalogue and the directory must describe exactly the same set. Either
    one drifting is the failure this check exists to catch: a body with no
    entry would embed with no topic, and an entry with no body would leave a
    doc_id that citations can name but retrieval can never return.
    """
    kb_dir = config.KB_DIR if kb_dir is None else Path(kb_dir)
    _, entries, _ = _load_catalogue(kb_dir)

    bodies = {path.stem: path for path in sorted(kb_dir.glob(f"*{DOCUMENT_SUFFIX}"))}
    if not bodies:
        raise ValueError(f"no {DOCUMENT_SUFFIX} documents found in {kb_dir}")

    missing_file = sorted(set(entries) - set(bodies))
    missing_entry = sorted(set(bodies) - set(entries))
    if missing_file or missing_entry:
        raise ValueError(
            f"{CATALOGUE_NAME} and {kb_dir.name}/ disagree: "
            f"catalogued with no file {missing_file}, "
            f"file with no catalogue entry {missing_entry}"
        )

    documents = []
    for doc_id in sorted(entries):
        path = bodies[doc_id]
        body = path.read_text(encoding="utf-8").strip()
        if not body:
            raise ValueError(f"{path.name}: empty body")
        documents.append(
            Document(
                doc_id=doc_id,
                title=entries[doc_id]["title"],
                topic=entries[doc_id]["topic"],
                required=entries[doc_id]["required"],
                products=tuple(sorted(entries[doc_id][PRODUCTS_KEY])),
                body=body,
                path=path,
            )
        )
    return documents


def document_titles() -> dict[str, str]:
    """doc_id to title, for transcripts and evaluation tables."""
    return {d.doc_id: d.title for d in load_documents()}
