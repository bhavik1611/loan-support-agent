"""Task 3. The two chunking strategies the brief asks to compare.

Both are pure functions from text to a list of strings. Nothing here knows
about embeddings, ChromaDB or document metadata, which is what keeps the V2
swap to semantic or late chunking a one-file change.
"""

import re

import config

# A boundary is terminal punctuation followed by whitespace. The guards below
# undo the split when the token before the period was not a sentence end.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Lower-cased, without the trailing period. NLTK punkt would handle this
# better and needs a download, which breaks the zero-network-access rule.
_ABBREVIATIONS = {
    "approx",
    "co",
    "dr",
    "e.g",
    "etc",
    "i.e",
    "inc",
    "jr",
    "ltd",
    "mr",
    "mrs",
    "ms",
    "no",
    "pvt",
    "rs",
    "sr",
    "st",
    "vs",
    "govt",
    "dept",
    "fig",
}

_LAST_TOKEN = re.compile(r"([A-Za-z.]+)\.\Z")


def _ends_on_an_abbreviation(fragment: str) -> bool:
    """True when the fragment's final period closes an abbreviation, not a sentence."""
    stripped = fragment.rstrip()
    if not stripped.endswith("."):
        return False
    match = _LAST_TOKEN.search(stripped)
    if match is None:
        # A digit before the period: "10." in "10.5" or a numbered item.
        return bool(re.search(r"\d\.\Z", stripped))
    token = match.group(1).lower().rstrip(".")
    if token in _ABBREVIATIONS:
        return True
    # A single capital letter is an initial, as in "S. Rao".
    return len(match.group(1)) == 1 and match.group(1).isupper()


def chunk_fixed(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Walk the character stream in steps of size minus overlap.

    400 and 80 are the decided values. 600 lost because a twelve-sentence
    document then yields three chunks against twelve sentence chunks, and the
    fixed-size collection loses the granularity the comparison needs. 250 lost
    because chunks cut mid-clause often enough to hurt embedding quality.
    """
    size = config.CHUNK_SIZE if size is None else size
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if overlap >= size:
        raise ValueError(f"overlap {overlap} must be smaller than size {size}")
    if overlap < 0 or size <= 0:
        raise ValueError(f"size {size} and overlap {overlap} must be positive")

    text = text.strip()
    if not text:
        return []

    step = size - overlap
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
        start += step
    return chunks


def chunk_sentences(text: str) -> list[str]:
    """Split on terminal punctuation, rejoining across known abbreviations."""
    text = text.strip()
    if not text:
        return []

    fragments = _BOUNDARY.split(text)
    sentences: list[str] = []
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        if sentences and _ends_on_an_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {fragment}"
        else:
            sentences.append(fragment)
    return sentences


def chunk(text: str, strategy: str) -> list[str]:
    """Dispatch to the named strategy. Used by rag/index.py for both collections."""
    if strategy == config.STRATEGY_FIXED:
        return chunk_fixed(text)
    if strategy == config.STRATEGY_SENTENCES:
        return chunk_sentences(text)
    raise ValueError(
        f"unknown strategy {strategy!r}, expected one of "
        f"{sorted(config.COLLECTION_FOR_STRATEGY)}"
    )
