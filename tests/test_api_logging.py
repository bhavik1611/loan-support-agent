"""Task 12. One structured log line per request, with nothing unmasked in it."""

import json
import logging
import re

from fastapi.testclient import TestClient

import obs
from api.main import app

client = TestClient(app)

PAN = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")


def _capture(caplog, fn):
    """Turn the shared logger up for the duration of one call, scoped to this test.

    The established pattern (tests/test_provider_and_obs.py::_capture): obs.configure
    has a _configured guard and cannot raise an already-configured logger's level on
    its own, so caplog.at_level is the half that actually does the work, and
    propagate is flipped on only long enough for caplog's handler to see the
    records. Nothing here depends on the ambient LOG_LEVEL; production stays quiet
    by default (D-70), this test turns its own logger up and hands it back.

    One addition to the established pattern, measured rather than assumed: pytest's
    own catching_logs attaches its capture handler directly to any logger that is
    *already* non-propagating at the start of a test's call phase (this is how it
    still captures loggers that never propagate at all), in addition to the root
    logger it always attaches to. `api/main.py` module-level configures obs (and,
    in a full-suite run, whichever test happens to run first through the RAG
    pipeline configures it earlier still), so by the time this test's own capture
    context opens, the logger is already non-propagating and its records reach the
    one shared handler twice: once directly, once via the propagate=True flip below
    reaching root. Confirmed by isolating it: a probe test with no other obs caller
    beforehand sees exactly one record; the same test placed after any earlier test
    that already touched obs sees two, always the same record object twice. The fix
    is a plain identity dedup, not a content dedup - two genuinely distinct calls to
    the same event with identical fields (the add-document determinism test sends
    two identical requests on purpose) must still count as two.
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


def _http_request_lines(caplog, fn) -> list[dict]:
    return [
        json.loads(line) for line in _capture(caplog, fn)
        if json.loads(line).get("event") == "http_request"
    ]


def test_37_one_request_writes_exactly_one_line_carrying_the_response_trace_id(caplog):
    """Spec section 16 test 37, Part 3 criterion 2, and D-75."""
    captured = {}

    def fn():
        captured["body"] = client.post(
            "/ask", json={"query": "How is the EMI calculated?"}
        ).json()

    lines = _http_request_lines(caplog, fn)
    assert len(lines) == 1
    assert lines[0]["trace_id"] == captured["body"]["trace_id"]
    assert lines[0]["path"] == "/ask"
    assert lines[0]["status"] == 200
    assert isinstance(lines[0]["duration_ms"], (int, float))


def test_the_logged_query_never_carries_an_unmasked_pan(caplog):
    """D-71. The query is the input side, so the input-side mask applies to it."""
    lines = _http_request_lines(
        caplog,
        lambda: client.post(
            "/ask",
            json={"query": "My PAN is FXZPG5049K, what is the minimum balance?"},
        ),
    )
    assert lines
    assert not PAN.search(lines[0]["query_masked"])
    assert "FXZPG5049K" not in json.dumps(lines[0])


def test_the_dedup_does_not_merge_two_distinct_emissions_with_identical_fields(caplog):
    """The guarantee _capture's identity dedup must never break.

    Two separate obs.event calls that happen to carry byte-identical fields
    are two distinct LogRecord objects; only the literal-same-object artifact
    from pytest's double handler attachment (see _capture's docstring above)
    may ever be collapsed. If the dedup were ever changed to compare content
    instead of identity, this is what would silently start failing.
    """
    def fn():
        obs.event("probe.duplicate_content", trace_id="same", n=1)
        obs.event("probe.duplicate_content", trace_id="same", n=1)

    lines = [
        json.loads(line) for line in _capture(caplog, fn)
        if json.loads(line).get("event") == "probe.duplicate_content"
    ]
    assert len(lines) == 2
    assert lines[0] == lines[1]


def test_the_trace_id_is_deterministic_for_add_document(caplog):
    """D-38's rule, applied to a request that produces no AgentResponse."""
    import shutil

    import config
    from rag import index

    payload = {
        "title": "Business loan collateral",
        "body": "A business loan above twenty lakh rupees requires collateral. " * 3,
        "products": ["Business Loan"],
    }
    try:
        first = _http_request_lines(
            caplog, lambda: client.post("/add-document", json=payload)
        )[0]

        caplog.clear()
        second = _http_request_lines(
            caplog, lambda: client.post("/add-document", json=payload)
        )[0]

        assert first["trace_id"] == second["trace_id"]
    finally:
        # Two identical uploads each land in the shared, persistent kb_sentences
        # collection under a new doc_id (D-72's rollback is a rebuild, not an
        # undo), so this cleans up rather than leaving them for test_index.py's
        # exact-count assertion to trip over.
        index.build_index(rebuild=True)
        if config.UPLOAD_DIR.exists():
            shutil.rmtree(config.UPLOAD_DIR)
