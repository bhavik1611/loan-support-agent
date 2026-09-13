"""Config is the single source of every path and tunable in the project."""

import subprocess
import sys

import config


def test_paths_are_rooted_in_the_repository():
    assert config.KB_DIR == config.REPO_ROOT / "knowledge_base"
    assert config.DATA_DIR == config.REPO_ROOT / "data"
    assert config.CHROMA_DIR == config.REPO_ROOT / "chroma"
    assert config.TRANSCRIPT_DIR == config.REPO_ROOT / "transcripts"
    assert config.DATASET_SNAPSHOT == config.DATA_DIR / "loan_applications.json"


def test_chunk_parameters_are_the_decided_values():
    assert config.CHUNK_SIZE == 400
    assert config.CHUNK_OVERLAP == 80
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
    assert config.TOP_K == 3
    assert config.SUPPORT_MIN_SHARED == 2


def test_every_strategy_maps_to_its_own_collection():
    mapping = config.COLLECTION_FOR_STRATEGY
    assert mapping[config.STRATEGY_FIXED] == "kb_fixed_400_80"
    assert mapping[config.STRATEGY_SENTENCES] == "kb_sentences"
    assert len(set(mapping.values())) == 2


def test_every_category_has_a_band_low_below_high():
    assert set(config.CATEGORY_BANDS) == set(config.CATEGORIES)
    for category, (low, high) in config.CATEGORY_BANDS.items():
        assert 0 < low < high, category


def test_mock_is_the_default_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert config.resolve_provider() == "mock"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert config.resolve_provider() == "openai"


def test_an_empty_log_level_falls_back_to_the_default():
    """An empty value is not a value, and `.env` routinely carries one.

    `.env.example` declares LOG_LEVEL with no value on the right-hand side,
    and README.md tells the reader to `cp .env.example .env`, so an ordinary
    setup puts LOG_LEVEL="" into the process environment. Two things then read
    it wrong at once: `os.environ.get("LOG_LEVEL", "WARNING")` returns the
    empty string rather than the default, and `os.environ.setdefault` in
    scripts/run_part3.py sees a present key and declines to fill it in. obs.py
    stayed quiet, the runner read back zero log lines, and it died on an
    IndexError three frames from the cause. Measured on 2026-09-13: three
    suite failures that appeared and disappeared with a file that is
    gitignored and therefore invisible to the diff.

    Read out of a subprocess rather than by reloading config in-process,
    because config.LOG_LEVEL is computed at import: a reload here would hand
    every later test in the session a different module object, which is the
    same class of bug this test exists to pin.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import config; print(repr(config.LOG_LEVEL))"],
        cwd=config.REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "LOG_LEVEL": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "'WARNING'"
