"""Shared fixtures. The index is built once per session because embedding is slow."""

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@pytest.fixture(scope="session", autouse=True)
def _no_stray_uploads():
    """Fail the whole session before it starts if data/uploads/ holds a file.

    Same reasoning as scripts/run_part1.py and run_part2.py's
    require_no_stray_uploads: rag/retrieve.py's product narrowing reads
    config.UPLOAD_DIR live, so a stray file left by a manual /add-document
    session, a crashed test, or an interrupted run would silently change
    what a product-narrowed query returns for the whole session, not just
    for whichever test happens to touch it. Session-scoped and autouse, so
    it runs once, before built_index's own (non-autouse) rebuild.
    """
    import config

    if config.UPLOAD_DIR.exists():
        stray = sorted(p.name for p in config.UPLOAD_DIR.iterdir())
        if stray:
            pytest.fail(
                f"{config.UPLOAD_DIR.relative_to(config.REPO_ROOT)} holds "
                f"{stray} before the suite even started. rag/retrieve.py's "
                f"product narrowing reads this directory live, so a stray "
                f"upload changes what every product-narrowed query returns "
                f"for the whole run. Clear it first: rm -r "
                f"{config.UPLOAD_DIR.relative_to(config.REPO_ROOT)}",
                pytrace=False,
            )


@pytest.fixture(autouse=True)
def pinned_to_mock(monkeypatch):
    """Every test runs under MOCK_LLM, whatever the environment says.

    config.py loads .env at import (D-64), so without this a developer who sets
    LLM_PROVIDER=groq in .env turns the whole offline suite into network calls
    and the offline claim in CLAUDE.md quietly becomes false. Pinning here is
    what makes `HF_HUB_OFFLINE=1 pytest` a proof rather than a habit.
    """
    monkeypatch.setenv("LLM_PROVIDER", "mock")


@pytest.fixture(scope="session")
def built_index():
    from rag import index

    counts = index.build_index(rebuild=True)
    return counts


@pytest.fixture(scope="session")
def built_db(tmp_path_factory):
    """A freshly built relational store.

    Built into a temp directory so the suite never depends on a developer
    having run `python -m db.build` first.
    """
    from db import build

    path = tmp_path_factory.mktemp("db") / "meridian_bank.db"
    build.build_database(path)
    return path


@pytest.fixture(scope="session")
def db_conn(built_db):
    from db import schema

    conn = schema.connect(built_db)
    yield conn
    conn.close()
