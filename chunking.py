"""Two chunking strategies and the knowledge-base loader (Part 1, Task 3).

The brief asks for fixed-size-with-overlap AND sentence-based chunks, each
indexed in its own ChromaDB collection, so that Task 5 can compare them.

The fixed-size parameters are not a tutorial default. They were chosen by
measuring this corpus: run `python chunking.py` to reproduce that measurement.
At 400 characters these documents yield roughly two chunks each against the
sentence strategy's five, so the comparison would measure chunk count rather
than boundary quality. 200/40 lands within 7% of the sentence strategy's chunk
count, which is as close to a fair contest as this corpus allows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).parent / "knowledge_base"

# Chosen from the measurement in `main()`, not from a tutorial.
FIXED_SIZE = 200
FIXED_OVERLAP = 40

# A sentence boundary: terminator, then whitespace, then a capital or digit.
# Guards against "8.5 percent" and "1.5 crore" being read as two sentences.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')


@dataclass(frozen=True)
class Document:
    doc_id: str
    topic: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str          # the parent, so Task 5 can map chunks back to documents
    text: str
    strategy: str


def load_documents() -> list[Document]:
    """Read the knowledge base off disk, in a stable order."""
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        raw = path.read_text()
        _, front, body = raw.split("---", 2)
        meta = dict(
            line.split(":", 1) for line in front.strip().splitlines() if ":" in line
        )
        docs.append(Document(
            doc_id=meta["doc_id"].strip(),
            topic=meta["topic"].strip(),
            title=meta["title"].strip(),
            text=" ".join(body.split()),
        ))
    return docs


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def fixed_size_chunks(doc: Document, size: int = FIXED_SIZE,
                      overlap: int = FIXED_OVERLAP) -> list[Chunk]:
    """Sliding character windows. Boundaries fall wherever they fall.

    This is the strategy that can cut a sentence in half, which is the whole
    point of comparing it against sentence-based splitting.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = doc.text
    chunks, start, i = [], 0, 0
    while start < len(text):
        window = text[start:start + size].strip()
        if window:
            chunks.append(Chunk(f"{doc.doc_id}#fixed-{i}", doc.doc_id, window, "fixed"))
            i += 1
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def sentence_chunks(doc: Document) -> list[Chunk]:
    """One chunk per sentence. Boundaries always land on a full thought."""
    return [
        Chunk(f"{doc.doc_id}#sent-{i}", doc.doc_id, s, "sentence")
        for i, s in enumerate(split_sentences(doc.text))
    ]


def build_chunks(docs: list[Document], strategy: str) -> list[Chunk]:
    if strategy == "fixed":
        return [c for d in docs for c in fixed_size_chunks(d)]
    if strategy == "sentence":
        return [c for d in docs for c in sentence_chunks(d)]
    raise ValueError(f"unknown strategy {strategy!r}")


def main() -> None:
    docs = load_documents()
    lengths = [len(d.text) for d in docs]
    sent_counts = [len(split_sentences(d.text)) for d in docs]

    print(f"Corpus: {len(docs)} documents, "
          f"{sum(lengths)} characters, {sum(sent_counts)} sentences")
    print(f"  per document: {min(lengths)}-{max(lengths)} chars "
          f"(mean {sum(lengths) / len(docs):.0f}), "
          f"{min(sent_counts)}-{max(sent_counts)} sentences")

    print("\nChoosing the fixed-size parameters by measurement, not by default:")
    print(f"{'size/overlap':>14} {'chunks':>8} {'per doc':>9} {'mean chars':>11} "
          f"{'vs sentence':>12}")
    n_sentence = sum(sent_counts)
    for size, overlap in [(150, 30), (200, 40), (250, 50), (300, 60),
                          (400, 80), (500, 100)]:
        chunks = [c for d in docs for c in fixed_size_chunks(d, size, overlap)]
        mean_chars = sum(len(c.text) for c in chunks) / len(chunks)
        print(f"{f'{size}/{overlap}':>14} {len(chunks):>8} "
              f"{len(chunks) / len(docs):>9.1f} {mean_chars:>11.0f} "
              f"{len(chunks) / n_sentence:>11.2f}x")
    print(f"{'sentence':>14} {n_sentence:>8} {n_sentence / len(docs):>9.1f} "
          f"{sum(len(s) for d in docs for s in split_sentences(d.text)) / n_sentence:>11.0f} "
          f"{1.0:>11.2f}x")

    print(f"\nChosen: {FIXED_SIZE}/{FIXED_OVERLAP}.")
    print("A strategy that emits more chunks per document gets more chances to")
    print("place one in a top-3, so comparing strategies with very different")
    print("chunk counts measures the count, not the boundaries. 200/40 is the")
    print("row closest to parity with sentence chunking at 0.93x; 400/80, the")
    print("common tutorial default, would have been 0.49x.")

    fixed = build_chunks(docs, "fixed")
    sentence = build_chunks(docs, "sentence")
    print(f"\nIndexable: {len(fixed)} fixed chunks, {len(sentence)} sentence chunks")
    print("\nA fixed chunk boundary, showing the cut mid-sentence:")
    print(f"  ...{fixed[0].text[-70:]!r}")
    print(f"  {fixed[1].text[:70]!r}...")


if __name__ == "__main__":
    main()
