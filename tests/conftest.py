"""Shared fixtures. The index is built once per session because embedding is slow."""

import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@pytest.fixture(scope="session")
def built_index():
    from rag import index

    counts = index.build_index(rebuild=True)
    return counts
