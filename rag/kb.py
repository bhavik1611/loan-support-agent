"""Load knowledge_base/ into Document objects.

Front matter is parsed strictly: a missing or mistyped field raises rather
than defaulting, because a document that silently loses its doc_id becomes
unscoreable in Task 5 and nothing would report it.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

import config

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)

_REQUIRED_FIELDS = {"doc_id": str, "title": str, "topic": str, "required": bool}


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    topic: str
    required: bool
    body: str
    path: Path


def _parse(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(raw)
    if match is None:
        raise ValueError(f"{path.name}: no YAML front matter delimited by ---")

    meta = yaml.safe_load(match.group(1)) or {}
    for field, expected in _REQUIRED_FIELDS.items():
        if field not in meta:
            raise ValueError(f"{path.name}: front matter is missing {field}")
        if not isinstance(meta[field], expected):
            raise ValueError(
                f"{path.name}: {field} must be {expected.__name__}, "
                f"got {type(meta[field]).__name__}"
            )

    if meta["doc_id"] != path.stem:
        raise ValueError(f"{path.name}: doc_id {meta['doc_id']!r} does not match the filename")

    body = match.group(2).strip()
    if not body:
        raise ValueError(f"{path.name}: empty body")

    return Document(
        doc_id=meta["doc_id"],
        title=meta["title"],
        topic=meta["topic"],
        required=meta["required"],
        body=body,
        path=path,
    )


def load_documents(kb_dir: Path | None = None) -> list[Document]:
    """Every document in the knowledge base, sorted by doc_id for determinism."""
    kb_dir = config.KB_DIR if kb_dir is None else Path(kb_dir)
    paths = sorted(kb_dir.glob("*.md"))
    if not paths:
        raise ValueError(f"no markdown documents found in {kb_dir}")
    return sorted((_parse(p) for p in paths), key=lambda d: d.doc_id)


def document_titles() -> dict[str, str]:
    """doc_id to title, for transcripts and evaluation tables."""
    return {d.doc_id: d.title for d in load_documents()}
