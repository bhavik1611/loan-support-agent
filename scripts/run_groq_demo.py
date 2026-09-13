"""The only end-to-end Groq entry point. Spec section 20, D-68.

`scripts/run_part1.py` refuses to run under a real provider because it writes
byte-guarded artefacts. This script exists so the real provider can still be
seen working, and it writes only to a gitignored directory: a Groq run is not
reproducible, and non-reproducible bytes do not belong in a repository whose
second ground rule is that the same input produces the same bytes.

    LLM_PROVIDER=groq .venv/bin/python scripts/run_groq_demo.py

Provenance is stamped in the header rather than in the response envelope,
per D-66: the provider is constant across a run, not a property of a turn.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import llm  # noqa: E402
import obs  # noqa: E402
from rag import generate  # noqa: E402

OUT_DIR = config.REPO_ROOT / "demo"

# One query per outcome the pipeline can reach, so a reader sees the guardrails
# working and not only the happy path.
QUERIES = [
    ("answerable", "What is the interest rate on a personal loan?"),
    ("answerable", "How long does loan disbursal take after approval?"),
    ("answerable", "What documents do I need for address proof?"),
    ("outside_boundary", "What is the interest rate on a fixed deposit?"),
]


def _preflight() -> None:
    provider = config.resolve_provider()
    if provider == config.DEFAULT_PROVIDER:
        sys.exit(
            f"LLM_PROVIDER is {provider!r}, so this would only exercise the mock "
            f"provider that scripts/run_part1.py already covers. Run it as:\n\n"
            f"    LLM_PROVIDER={config.PROVIDER_GROQ} "
            f".venv/bin/python scripts/run_groq_demo.py"
        )
    if provider != config.PROVIDER_GROQ:
        sys.exit(f"LLM_PROVIDER={provider!r} is not a provider this repository implements.")
    if not os.environ.get(config.GROQ_API_KEY_VAR):
        sys.exit(
            f"{config.GROQ_API_KEY_VAR} is not set. Put it in .env "
            f"(see .env.example) or export it."
        )


def main() -> None:
    _preflight()
    obs.configure("INFO")

    lines = [
        "# Groq demo run",
        "",
        f"- provider: `{config.resolve_provider()}`",
        f"- model: `{config.GROQ_MODEL}`",
        f"- collection: `{config.STRATEGY_SENTENCES}`",
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Not reproducible and not committed. The graded evidence is `transcripts/`,",
        "which is produced under `MOCK_LLM` and is byte-identical on every run.",
        "",
    ]

    failures = 0
    for label, query in QUERIES:
        print(f"[{label}] {query}")
        lines += [f"## {query}", "", f"- class: {label}"]
        try:
            answer = generate.answer(query, strategy=config.STRATEGY_SENTENCES)
        except llm.ProviderError as exc:
            # Deliberately not swallowed into a refusal (D-62).
            failures += 1
            print(f"    provider error: {exc}")
            lines += [f"- outcome: **provider error**", "", f"```\n{exc}\n```", ""]
            continue

        print(f"    outcome: {answer.outcome}  citations: {list(answer.citations)}")
        lines += [
            f"- outcome: `{answer.outcome}`",
            f"- top-1 similarity: {answer.top1_similarity}",
            f"- citations: {', '.join(f'`{c}`' for c in answer.citations) or 'none'}",
            "",
            answer.text,
            "",
        ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "groq-run.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out.relative_to(config.REPO_ROOT)} (gitignored)")

    if failures:
        sys.exit(f"{failures} of {len(QUERIES)} queries failed at the provider.")


if __name__ == "__main__":
    main()
