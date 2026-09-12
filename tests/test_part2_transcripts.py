"""D-40. The evidence exists, is generated, and says what it claims."""

import config

EXPECTED = (
    "part2-escalation.txt",
    "part2-routing.txt",
    "part2-graph.txt",
    "part2-memory.txt",
    "part2-memory-fresh.txt",
    "part2-schema.txt",
    "part2-guardrails.txt",
)


def test_every_part2_transcript_exists():
    missing = [name for name in EXPECTED if not (config.TRANSCRIPT_DIR / name).exists()]
    assert not missing, f"run scripts/run_part2.py: {missing}"


def test_every_transcript_declares_how_it_was_made():
    for name in EXPECTED:
        body = (config.TRANSCRIPT_DIR / name).read_text(encoding="utf-8")
        assert "scripts/run_part2.py" in body, name
        assert "MOCK_LLM" in body, name


def test_the_memory_transcripts_show_opposite_outcomes():
    """The brief asks for carried state and, separately, absent state."""
    warm = (config.TRANSCRIPT_DIR / "part2-memory.txt").read_text(encoding="utf-8")
    cold = (config.TRANSCRIPT_DIR / "part2-memory-fresh.txt").read_text(encoding="utf-8")
    assert "LN-" in warm
    assert "clarify" in cold


def test_the_routing_transcript_reports_the_ellipsis_matcher():
    """Spec 18.4. The matcher has a calibration set now, so the evidence shows it."""
    body = (config.TRANSCRIPT_DIR / "part2-routing.txt").read_text(encoding="utf-8")
    assert "THE ELLIPSIS MATCHER" in body
    assert "known uncaught:" in body  # the residue is reported, not hidden


def test_the_guardrail_transcript_names_every_rule():
    from agent import guardrails

    body = (config.TRANSCRIPT_DIR / "part2-guardrails.txt").read_text(encoding="utf-8")
    for name, _ in guardrails.INJECTION_RULES:
        assert name in body, name
    for label in config.PII_PLACEHOLDERS:
        assert label in body, label


def test_the_guardrail_transcript_separates_the_two_out_of_scope_refusals():
    """[D-54] Criteria 24a and 24b are different events and must read as two.

    A transcript that showed only one refusal would let a grader conclude the
    threshold is still deciding scope, which is precisely what D-46 disproved.
    """
    body = (config.TRANSCRIPT_DIR / "part2-guardrails.txt").read_text(encoding="utf-8")
    assert "refused_gate" in body
    assert "refused_threshold" in body
    assert "fixed deposit" in body
    assert body.index("THE PRODUCT GATE") < body.index("OUTPUT SIDE, GROUNDEDNESS")
