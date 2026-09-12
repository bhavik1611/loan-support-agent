"""Test 5 of spec section 16, plus the invariants the index must hold."""

import config
from rag import index, kb

DOCS = kb.load_documents()


def test_every_document_yields_at_least_two_chunks_in_both_strategies():
    """Test 5. Without this a short document is permanently unanswerable under
    the support rule in D-07, and nothing would report it."""
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        counts = {}
        for chunk in index.build_chunks(strategy, DOCS):
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        assert set(counts) == {d.doc_id for d in DOCS}
        for doc_id, count in sorted(counts.items()):
            assert count >= config.MIN_CHUNKS_PER_DOCUMENT, f"{strategy}/{doc_id}: {count}"


def test_chunk_ids_are_unique_within_a_strategy():
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        ids = [c.chunk_id for c in index.build_chunks(strategy, DOCS)]
        assert len(set(ids)) == len(ids)


def test_chunk_metadata_carries_the_parent_document():
    chunk = index.build_chunks(config.STRATEGY_SENTENCES, DOCS)[0]
    assert chunk.doc_id
    assert chunk.title
    assert chunk.topic
    assert isinstance(chunk.required, bool)
    assert chunk.chunk_index == 0
    assert chunk.strategy == config.STRATEGY_SENTENCES


def test_embeddings_are_unit_length_and_384_dimensional():
    vectors = index.embed(["Meridian Bank charges a foreclosure fee."])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    norm = sum(v * v for v in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-5


def test_both_collections_exist_and_hold_every_chunk(built_index):
    assert set(built_index) == {config.COLLECTION_FIXED, config.COLLECTION_SENTENCES}
    for strategy, name in sorted(config.COLLECTION_FOR_STRATEGY.items()):
        collection = index.get_collection(strategy)
        assert collection.count() == built_index[name]
        assert collection.count() == len(index.build_chunks(strategy, DOCS))


def test_both_collections_return_a_sensible_hit_on_a_sample_query(built_index):
    """The brief's wording: both collections must produce sensible retrieval."""
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        collection = index.get_collection(strategy)
        result = collection.query(
            query_embeddings=index.embed(["How is the EMI on a loan calculated?"]),
            n_results=3,
        )
        assert "kb-02-emi-calculation" in [m["doc_id"] for m in result["metadatas"][0]]
