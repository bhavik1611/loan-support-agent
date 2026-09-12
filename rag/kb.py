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
"""

import json
from dataclasses import dataclass
from pathlib import Path

import config

CATALOGUE_NAME = "catalogue.json"
DOCUMENT_SUFFIX = ".txt"

PRODUCTS_KEY = "products"
DOCUMENTS_KEY = "documents"

_CATALOGUE_FIELDS = {"title": str, "topic": str, "required": bool, "products": list}


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


def _load_catalogue(kb_dir: Path) -> tuple[list[str], dict[str, dict]]:
    """The product list and the document entries, both parsed strictly."""
    path = kb_dir / CATALOGUE_NAME
    if not path.exists():
        raise ValueError(f"{path} is missing; the catalogue is not optional")

    catalogue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(catalogue, dict):
        raise ValueError(
            f"{CATALOGUE_NAME}: expected an object with {PRODUCTS_KEY!r} and "
            f"{DOCUMENTS_KEY!r} keys"
        )
    for key in (PRODUCTS_KEY, DOCUMENTS_KEY):
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
    return products, entries


def catalogue_products(kb_dir: Path | None = None) -> list[str]:
    """What Meridian Bank sells, in catalogue order. The authority for D-47."""
    kb_dir = config.KB_DIR if kb_dir is None else Path(kb_dir)
    products, _ = _load_catalogue(kb_dir)
    return list(products)


def documents_for_product(product: str, kb_dir: Path | None = None) -> list[str]:
    """The doc_ids the catalogue tags with this product, sorted.

    This is the whole of the D-53 filter: rag/retrieve.py turns the list into a
    `doc_id` `$in` clause. Nothing about the product reaches chunk metadata, so
    neither collection is rebuilt for it.
    """
    kb_dir = config.KB_DIR if kb_dir is None else Path(kb_dir)
    products, entries = _load_catalogue(kb_dir)
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
    _, entries = _load_catalogue(kb_dir)

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
