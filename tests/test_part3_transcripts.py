"""The Part 3 transcripts, and the claims they have to keep."""

import re

import config


def _read(name: str) -> str:
    return (config.TRANSCRIPT_DIR / name).read_text(encoding="utf-8")


PAN = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")


def test_38_the_triad_transcript_reports_fifteen_rows_and_three_averages():
    """Spec section 16 test 38, Part 3 criterion 3.

    Criterion 3 is "all three scores per query and the average of each" - a
    per-row check, not only a per-transcript one. Asserting `item_id in text`
    and that three phrases occur is satisfied by the three "average ...:"
    lines alone: stripping every numeric score from all fifteen rows of
    `part3-triad.txt` still leaves the item ids, the three phrases and one
    "average" occurrence in the text, and this test used to still pass. Each
    row is now matched against its own three numeric scores, so a row with
    its numbers deleted fails here instead of only failing a human reading it.

    D-13's rule applies: this asserts the *shape* (three numeric scores per
    row, in EVERY ROW's own layout), never the values - the values are
    unpinned on purpose and move legitimately when the judge or the chunker
    is retuned.
    """
    import re

    from eval import triad

    text = _read("part3-triad.txt")
    assert "context relevance" in text.lower()
    assert "groundedness" in text.lower()
    assert "answer relevance" in text.lower()
    assert text.lower().count("average") >= 1

    items = triad.triad_items()
    assert len(items) == 15
    for item in items:
        row = re.search(
            rf"^\s*{re.escape(item.item_id)}\s+\S+"
            rf"\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
            text,
            re.MULTILINE,
        )
        assert row, (
            f"no row with three numeric scores found for {item.item_id!r} in "
            f"part3-triad.txt's EVERY ROW table"
        )


def test_the_logging_transcript_shows_a_masked_pan_and_no_raw_one():
    text = _read("part3-logging.txt")
    assert config.PII_PLACEHOLDERS["PAN"] in text
    assert not PAN.search(text)


def test_the_api_transcript_shows_both_endpoints():
    text = _read("part3-api.txt")
    assert "/ask" in text
    assert "/add-document" in text


def test_the_runner_is_idempotent():
    """Two runs, same bytes. The determinism ground rule, applied to Part 3.

    scripts/run_part3.py is run for real below, as a subprocess, and it opens
    its own chromadb PersistentClient on the same config.CHROMA_DIR this
    session's built_index fixture already has open. chromadb's own
    PersistentClient.close() docstring: "This is particularly important for
    PersistentClient to avoid SQLite file locking issues" - two live clients
    on the same path, one per process, is exactly that, and it is what made
    the subprocess's own cleanup (index.build_index inside its
    _clear_own_uploads) crash intermittently before reaching its
    shutil.rmtree, leaving that run's uploads on disk and corrupting this
    session's cached collection handles for every RAG test that ran after.
    Closing the parent's cached client before the subprocess runs, and
    dropping the cache afterwards so the next caller opens a fresh one
    against the subprocess's final on-disk state, removes the contention
    rather than working around its symptom.
    """
    import subprocess
    import sys

    from rag import index

    names = ["part3-api.txt", "part3-logging.txt", "part3-triad.txt"]
    first = {name: _read(name) for name in names}

    index.get_client().close()
    index.get_client.cache_clear()
    try:
        subprocess.run(
            [sys.executable, "scripts/run_part3.py"],
            cwd=config.REPO_ROOT, check=True, capture_output=True,
        )
    finally:
        index.get_client.cache_clear()

    assert {name: _read(name) for name in names} == first
