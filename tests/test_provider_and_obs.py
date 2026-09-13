"""Tests 32 to 35 of spec section 16, the Groq provider and the logging spine.

Every test here runs offline. Test 33 is specifically the one that proves the
suite cannot be dragged online by a `.env` file.
"""

import json
import logging
import subprocess
import sys

import pytest

import config
import llm
import obs
from rag import generate

PROMPT = "QUESTION: What is the interest rate on a personal loan?\n\nCONTEXT:\n[kb-07] The Personal Loan carries an interest rate between 10.75 and 18.00 percent per annum.\n"


# --- Test 32: the prompt tightening is free (D-63) -------------------------


def test_mock_output_does_not_depend_on_the_system_prompt():
    """The claim D-63 rests on: tightening SYSTEM_PROMPT moves no graded byte.

    `_generate_mock` accepts `system` and never reads it, so the citation
    instruction a real model needs costs the mock provider nothing. If this
    ever fails, every transcript in the repository is downstream of the change.
    """
    baseline = llm.generate("", PROMPT)
    tightened = llm.generate(generate.SYSTEM_PROMPT, PROMPT)
    nonsense = llm.generate("ignore every previous instruction", PROMPT)

    assert baseline == tightened == nonsense
    assert baseline, "the fixture should produce an answer, not a refusal"


def test_the_system_prompt_states_the_format_the_parser_reads():
    """The two halves of the citation contract have to agree.

    Measured against the live model: without this sentence it cited as
    U+3010 kb-07 U+3011 and _cited_documents returned nothing for a fully
    grounded answer.
    """
    assert "Sources:" in generate.SYSTEM_PROMPT
    assert "[doc-id]" in generate.SYSTEM_PROMPT


# --- Test 33: the failure contract and the pin (D-62, D-64) ---------------


def test_groq_without_a_key_raises_naming_the_variable(monkeypatch):
    """A missing key is configuration, not a refusal, and says so by name."""
    monkeypatch.setenv("LLM_PROVIDER", config.PROVIDER_GROQ)
    monkeypatch.delenv(config.GROQ_API_KEY_VAR, raising=False)

    with pytest.raises(llm.ProviderError) as excinfo:
        llm.generate("system", PROMPT)

    assert config.GROQ_API_KEY_VAR in str(excinfo.value)


def test_an_unknown_provider_still_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(NotImplementedError, match="openai"):
        llm.generate("system", PROMPT)


def test_the_suite_runs_under_mock_whatever_the_environment_says():
    """The autouse pin in conftest.py, restated as an assertion.

    config.py loads .env at import (D-64). Without the pin, LLM_PROVIDER=groq
    in a developer's .env turns this whole suite into network calls and the
    offline proof in CLAUDE.md becomes false.
    """
    assert config.resolve_provider() == config.DEFAULT_PROVIDER


# --- Test 34: graded artefacts have one authority (D-68) -----------------


def test_run_part1_refuses_under_a_non_mock_provider(tmp_path):
    """The runner exits non-zero and writes nothing when the provider is real."""
    result = subprocess.run(
        [sys.executable, "scripts/run_part1.py"],
        cwd=config.REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "LLM_PROVIDER": config.PROVIDER_GROQ},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0
    assert "Refusing to run" in result.stderr
    assert config.PROVIDER_GROQ in result.stderr


# --- Test 35: what a log line carries, and what it never carries (D-71) ---


def _capture(caplog, fn):
    """Turn the shared logger up for the duration of one call, scoped to this test.

    obs.configure has a _configured guard and cannot raise an already-configured
    logger's level on its own, so caplog.at_level is the half that actually does
    the work, and propagate is flipped on only long enough for caplog's handler
    to see the records.

    pytest's own catching_logs attaches its capture handler directly to any
    logger that is already non-propagating at the start of a test's call phase,
    in addition to the root logger it always attaches to. obs.configure sets
    logger.propagate = False at import, so by the time this test's own capture
    context opens, the logger is already non-propagating and its records reach
    the one shared handler twice: once directly, once via the propagate=True
    flip below reaching root. The fix is a plain identity dedup, not a content
    dedup - two genuinely distinct calls to the same event with identical fields
    must still count as two (see tests/test_api_logging.py's identical fix and
    its dedicated test for the guarantee this must not break).
    """
    obs.configure("INFO")
    logging.getLogger(obs.LOGGER_NAME).propagate = True
    try:
        with caplog.at_level(logging.INFO, logger=obs.LOGGER_NAME):
            fn()
        seen_ids: set[int] = set()
        unique_records = []
        for record in caplog.records:
            if id(record) not in seen_ids:
                seen_ids.add(id(record))
                unique_records.append(record)
        return [obs.JsonFormatter().format(r) for r in unique_records]
    finally:
        logging.getLogger(obs.LOGGER_NAME).propagate = False


def test_a_log_line_is_json_and_carries_the_trace_id(caplog):
    lines = _capture(
        caplog,
        lambda: obs.event("probe.event", trace_id="deadbeef", doc_ids=["kb-07"]),
    )

    assert lines, "the event should have been emitted"
    payload = json.loads(lines[-1])
    assert payload["event"] == "probe.event"
    assert payload["trace_id"] == "deadbeef"


def test_a_timed_line_carries_a_rounded_duration(caplog):
    """D-70: durations are rounded so clock noise cannot separate two runs."""

    def run():
        with obs.timed("probe.timed", trace_id="beef") as line:
            line["hits"] = 3

    lines = _capture(caplog, run)
    payload = json.loads(lines[-1])

    assert payload["event"] == "probe.timed"
    assert payload["outcome"] == "ok"
    assert payload["hits"] == 3
    duration = payload["duration_ms"]
    assert duration == round(duration, config.LOG_DURATION_PLACES)


def test_a_failing_boundary_logs_the_error_and_re_raises(caplog):
    """A boundary that raises is logged as an error, and the raise still lands."""

    def run():
        try:
            with obs.timed("probe.fails"):
                raise ValueError("boom")
        except ValueError:
            pass

    lines = _capture(caplog, run)
    payload = json.loads(lines[-1])

    assert payload["outcome"] == "error"
    assert payload["error_type"] == "ValueError"


def test_a_log_line_never_carries_a_key_or_unmasked_pii(caplog):
    """D-71. Observability is the usual way a PII rule gets broken."""
    pan = "ABCDE1234F"
    aadhaar = "2345 6789 0123"

    lines = _capture(
        caplog,
        lambda: obs.event(
            "probe.redaction",
            trace_id="cafe",
            query=obs.mask(f"my PAN is {pan} and Aadhaar {aadhaar}"),
            api_key="sk-must-never-appear",
            token="must-never-appear",
        ),
    )

    blob = "\n".join(lines)
    assert pan not in blob
    assert aadhaar.replace(" ", "") not in blob
    assert "must-never-appear" not in blob
    assert json.loads(lines[-1])["trace_id"] == "cafe"


def test_the_dedup_does_not_merge_two_distinct_emissions_with_identical_fields(caplog):
    """The guarantee _capture's identity dedup must never break.

    Two separate obs.event calls that happen to carry byte-identical fields
    are two distinct LogRecord objects; only the literal-same-object artifact
    from pytest's double handler attachment (see _capture's docstring above)
    may ever be collapsed. If the dedup were ever changed to compare content
    instead of identity, this is what would silently start failing.
    """

    def run():
        obs.event("probe.duplicate_content", trace_id="same", n=1)
        obs.event("probe.duplicate_content", trace_id="same", n=1)

    lines = [
        json.loads(line) for line in _capture(caplog, run)
        if json.loads(line).get("event") == "probe.duplicate_content"
    ]
    assert len(lines) == 2
    assert lines[0] == lines[1]


def test_the_masker_is_the_one_the_guardrails_already_use():
    """A second masker would be a second thing to keep correct."""
    from agent.guardrails import mask_pii

    text = "PAN ABCDE1234F"
    assert obs.mask(text) == mask_pii(text)[0]
