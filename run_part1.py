"""Run every Part 1 task end to end, in order (Tasks 1 to 5).

    python run_part1.py

Needs no API key and no network once the embedding model is cached. Exits
non-zero if any task fails a check, so it works as a gate rather than a demo.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

TASKS = [
    ("Task 1  Dataset design and validation", ["dataset.py"]),
    ("Task 3a Chunk parameter measurement", ["chunking.py"]),
    ("Task 3b Index both collections", ["index_kb.py"]),
    ("Task 4a Threshold calibration", ["rag.py", "--calibrate", "--strategy", "fixed"]),
    ("Task 4b Grounded generation and fallback", ["rag.py", "--strategy", "fixed"]),
    ("Task 5  Precision@3 and Recall@3, both collections", ["evaluate_retrieval.py"]),
]


def run(title: str, argv: list[str]) -> int:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}", flush=True)
    result = subprocess.run([sys.executable, *argv], cwd=HERE)
    if result.returncode != 0:
        print(f"\n!! {title} exited {result.returncode}", file=sys.stderr)
    return result.returncode


def main() -> int:
    print("Cred domain support agent, Part 1.")
    print("Task 2 (the knowledge base) is the 18 documents in knowledge_base/;")
    print("it is data rather than a script, and every task below reads it.")

    failures = [title for title, argv in TASKS if run(title, argv) != 0]

    print("\n" + "=" * 78)
    if failures:
        print("PART 1 INCOMPLETE. Failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PART 1 COMPLETE. All five tasks ran and every check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
