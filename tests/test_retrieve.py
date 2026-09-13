"""Retrieval must be ranked, bounded, and honest about similarity."""

import pytest

import config
from rag import retrieve


def hit(doc_id, similarity, chunk_index=0):
    return retrieve.Hit(
        text=f"chunk of {doc_id}",
        doc_id=doc_id,
        title=doc_id,
        chunk_index=chunk_index,
        similarity=similarity,
    )


def test_similarity_is_bounded_and_descending(built_index):
    hits = retrieve.retrieve("What annual fee does the credit card carry?", config.STRATEGY_FIXED)
    assert len(hits) == config.TOP_K
    assert all(-1.0 <= h.similarity <= 1.0 for h in hits)
    assert [h.similarity for h in hits] == sorted((h.similarity for h in hits), reverse=True)


def test_an_in_scope_query_scores_far_above_an_out_of_scope_one(built_index):
    in_scope = retrieve.retrieve("How is the EMI calculated?", config.STRATEGY_SENTENCES)
    out_of_scope = retrieve.retrieve(
        "What is the best recipe for a chocolate sponge cake?", config.STRATEGY_SENTENCES
    )
    assert retrieve.top1_similarity(in_scope) > retrieve.top1_similarity(out_of_scope) + 0.2


def test_both_strategies_answer_the_same_query(built_index):
    for strategy in sorted(config.COLLECTION_FOR_STRATEGY):
        hits = retrieve.retrieve("How do I close my account?", strategy)
        assert "kb-06-account-closure" in retrieve.parent_documents(hits)


def test_parent_documents_dedups_and_keeps_rank_order():
    hits = [hit("kb-02", 0.9), hit("kb-01", 0.8, 1), hit("kb-02", 0.7, 2)]
    assert retrieve.parent_documents(hits) == ["kb-02", "kb-01"]


def test_top1_similarity_of_nothing_is_zero():
    assert retrieve.top1_similarity([]) == 0.0


def test_support_needs_two_of_the_top_three_to_agree():
    agreeing = [hit("kb-02", 0.9), hit("kb-01", 0.8), hit("kb-02", 0.7)]
    scattered = [hit("kb-02", 0.9), hit("kb-01", 0.8), hit("kb-03", 0.7)]
    assert retrieve.has_same_parent_support(agreeing)
    assert not retrieve.has_same_parent_support(scattered)


def test_the_answer_decision_needs_both_conditions():
    agreeing_high = [hit("kb-02", 0.62), hit("kb-02", 0.55), hit("kb-01", 0.40)]
    agreeing_low = [hit("kb-02", 0.30), hit("kb-02", 0.28), hit("kb-01", 0.20)]
    scattered_high = [hit("kb-02", 0.62), hit("kb-01", 0.55), hit("kb-03", 0.40)]
    assert retrieve.is_supported(agreeing_high, threshold=0.45)
    assert not retrieve.is_supported(agreeing_low, threshold=0.45)
    assert not retrieve.is_supported(scattered_high, threshold=0.45)


def test_an_unknown_strategy_raises():
    with pytest.raises(ValueError):
        retrieve.retrieve("anything", "semantic")


def test_an_orphaned_upload_file_is_silently_ignored_by_the_union(built_index):
    """The gap between api/main.py writing data/uploads/<doc_id>.txt and indexing
    its chunks into Chroma is not atomic: a crash in between leaves a file
    whose doc id `_uploaded_doc_ids()` still lists, but which no chunk carries.
    Chroma's `doc_id` `$in` clause simply matches nothing for that id, so this
    pins the behaviour as a silent no-op rather than leaving it
    correct-by-inspection only - no KeyError, no wrong hit, and retrieval for
    the same query and product is byte-identical with or without the orphan.
    """
    baseline = retrieve.retrieve(
        "How is the EMI calculated?", config.STRATEGY_SENTENCES, product="Personal Loan"
    )

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    orphan = config.UPLOAD_DIR / f"{config.UPLOAD_DOC_PREFIX}99.txt"
    orphan.write_text("An orphaned upload file with no indexed chunks.\n")
    try:
        with_orphan = retrieve.retrieve(
            "How is the EMI calculated?", config.STRATEGY_SENTENCES, product="Personal Loan"
        )
        assert with_orphan == baseline
    finally:
        orphan.unlink()
        if not any(config.UPLOAD_DIR.iterdir()):
            config.UPLOAD_DIR.rmdir()
