"""Config is the single source of every path and tunable in the project."""

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
