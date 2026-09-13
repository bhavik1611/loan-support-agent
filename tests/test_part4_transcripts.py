"""The Part 4 transcripts, and the claims they have to keep."""

import config


def _read(name: str) -> str:
    return (config.TRANSCRIPT_DIR / name).read_text(encoding="utf-8")


def test_the_mcp_transcript_shows_a_real_round_trip_for_two_ids():
    text = _read("part4-mcp.txt")
    assert "LN-1042" in text and "LN-1057" in text
    assert config.MCP_URL in text
    assert "check_loan_application_status" in text


def test_the_checkpoint_transcript_names_what_ran_and_what_was_loaded():
    text = _read("part4-checkpoint.txt")
    assert "run 1" in text.lower() and "run 2" in text.lower()
    assert "loaded from the checkpoint" in text.lower()
    assert "guard_input" in text and "compose" in text


def test_the_resilience_transcript_shows_all_three_demonstrations():
    text = _read("part4-resilience.txt").lower()
    assert "attempt 1" in text and "attempt 3" in text
    assert "nodetimeouterror" in text
    assert "global" in text


def test_the_retry_parameters_are_all_stated():
    text = _read("part4-resilience.txt")
    for parameter in ("max attempts", "initial interval", "max interval", "jitter"):
        assert parameter in text.lower(), f"the brief asks you to state {parameter}"


def test_the_runner_is_idempotent():
    import subprocess
    import sys

    names = ["part4-mcp.txt", "part4-checkpoint.txt", "part4-resilience.txt"]
    first = {name: _read(name) for name in names}
    subprocess.run(
        [sys.executable, "scripts/run_part4.py"],
        cwd=config.REPO_ROOT, check=True, capture_output=True,
    )
    assert {name: _read(name) for name in names} == first
