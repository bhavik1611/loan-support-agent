"""Shared fixtures. The index is built once per session because embedding is slow."""

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


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
