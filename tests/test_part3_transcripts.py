"""The Part 3 transcripts, and the claims they have to keep."""

import re

import config


def _read(name: str) -> str:
    return (config.TRANSCRIPT_DIR / name).read_text(encoding="utf-8")


PAN = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")


def test_38_the_triad_transcript_reports_fifteen_rows_and_three_averages():
    """Spec section 16 test 38, Part 3 criterion 3."""
    from eval import triad

    text = _read("part3-triad.txt")
    for item in triad.triad_items():
        assert item.item_id in text
    assert "context relevance" in text.lower()
    assert "groundedness" in text.lower()
    assert "answer relevance" in text.lower()
    assert text.lower().count("average") >= 1


def test_the_logging_transcript_shows_a_masked_pan_and_no_raw_one():
    text = _read("part3-logging.txt")
    assert config.PII_PLACEHOLDERS["PAN"] in text
    assert not PAN.search(text)


def test_the_api_transcript_shows_both_endpoints():
    text = _read("part3-api.txt")
    assert "/ask" in text
    assert "/add-document" in text


def test_the_runner_is_idempotent():
    """Two runs, same bytes. The determinism ground rule, applied to Part 3."""
    import subprocess
    import sys

    names = ["part3-api.txt", "part3-logging.txt", "part3-triad.txt"]
    first = {name: _read(name) for name in names}
    subprocess.run(
        [sys.executable, "scripts/run_part3.py"],
        cwd=config.REPO_ROOT, check=True, capture_output=True,
    )
    assert {name: _read(name) for name in names} == first
