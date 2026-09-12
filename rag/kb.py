"""Load knowledge_base/ into Document objects.

The documents are plain text and carry no metadata of their own. A document is
the prose a support team would actually write; what that prose is *about* lives
beside it in catalogue.json, the way a real document store keeps its index
separate from its content (D-32).

Both sides are parsed strictly. A catalogue entry with no file, a file with no
catalogue entry, or a missing or mistyped field all raise, because a document
that silently loses its doc_id becomes unscoreable in Task 5 and nothing else
would report it.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import config

CATALOGUE_NAME = "catalogue.json"
DOCUMENT_SUFFIX = ".txt"

_CATALOGUE_FIELDS = {"title": str, "topic": str, "required": bool}


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    topic: str
    required: bool
    body: str
    path: Path


def _load_catalogue(kb_dir: Path) -> dict[str, dict]:
    path = kb_dir / CATALOGUE_NAME
    if not path.exists():
        raise ValueError(f"{path} is missing; the catalogue is not optional")

    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, dict):
        raise ValueError(f"{CATALOGUE_NAME}: expected an object keyed by doc_id")

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
    return entries


def load_documents(kb_dir: Path | None = None) -> list[Document]:
    """Every document in the knowledge base, sorted by doc_id for determinism.

    The catalogue and the directory must describe exactly the same set. Either
    one drifting is the failure this check exists to catch: a body with no
    entry would embed with no topic, and an entry with no body would leave a
    doc_id that citations can name but retrieval can never return.
    """
    kb_dir = config.KB_DIR if kb_dir is None else Path(kb_dir)
    entries = _load_catalogue(kb_dir)

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
                body=body,
                path=path,
            )
        )
    return documents


def document_titles() -> dict[str, str]:
    """doc_id to title, for transcripts and evaluation tables."""
    return {d.doc_id: d.title for d in load_documents()}
