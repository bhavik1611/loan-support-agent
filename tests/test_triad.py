"""Task 13. The query set the RAG triad is scored on."""

import eval.queries as queries
from eval import triad


def test_the_triad_set_holds_fifteen_items_in_a_fixed_order():
    items = triad.triad_items()
    assert len(items) == 15
    assert [item.item_id for item in items] == list(triad.TRIAD_ITEM_IDS)


def test_every_required_topic_appears_at_least_once():
    """The brief wants at least one query per required KB topic, kb-01 to kb-12."""
    covered = {
        doc_id
        for item in triad.triad_items()
        for doc_id in item.gold_doc_ids
    }
    required = {
        document.doc_id
        for document in __import__("rag.kb", fromlist=["kb"]).load_documents()
        if document.required
    }
    assert required <= covered, f"uncovered required topics: {sorted(required - covered)}"


def test_the_set_carries_at_least_two_non_answerable_queries():
    """The brief's floor is two. D-73 spends the third on IU-02 deliberately."""
    kinds = [item.kind for item in triad.triad_items()]
    assert sum(1 for kind in kinds if kind != queries.KIND_ANSWERABLE) >= 2
    assert "IU-02" in triad.TRIAD_ITEM_IDS


def test_the_set_is_a_selection_and_never_a_copy():
    """D-73. Selected by item_id, so the two sets cannot drift apart."""
    by_id = {item.item_id: item for item in queries.GOLDEN_DATASET}
    for item in triad.triad_items():
        assert item is by_id[item.item_id]
