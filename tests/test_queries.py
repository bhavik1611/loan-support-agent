"""The label set must match D-11's design and reference only real documents."""

from collections import Counter

from eval import calibration, queries
from rag import kb

DOC_IDS = {d.doc_id for d in kb.load_documents()}


def test_twelve_queries_with_unique_ids():
    assert len(queries.EVAL_QUERIES) == 12
    ids = [q.query_id for q in queries.EVAL_QUERIES]
    assert len(set(ids)) == 12


def test_every_gold_label_names_a_real_document():
    for query in queries.EVAL_QUERIES:
        for doc_id in query.gold_doc_ids:
            assert doc_id in DOC_IDS, f"{query.query_id} references unknown {doc_id}"


def test_gold_sets_are_graded_four_five_three():
    """D-11: 4 queries with 1 relevant document, 5 with 2, 3 with 3."""
    sizes = Counter(len(q.gold_doc_ids) for q in queries.EVAL_QUERIES)
    assert sizes == Counter({1: 4, 2: 5, 3: 3})


def test_gold_sets_have_no_duplicates():
    for query in queries.EVAL_QUERIES:
        assert len(set(query.gold_doc_ids)) == len(query.gold_doc_ids)


def test_every_required_document_is_gold_for_at_least_one_query():
    covered = {d for q in queries.EVAL_QUERIES for d in q.gold_doc_ids}
    required = {d.doc_id for d in kb.load_documents() if d.required}
    missing = sorted(required - covered)
    assert not missing, f"required topics never scored: {missing}"


def test_probe_counts_oversample_the_brief_floors():
    assert len(calibration.IN_SCOPE_PROBES) == 12
    assert len(calibration.FAR_OUT_OF_SCOPE_PROBES) == 17
    assert len(set(calibration.IN_SCOPE_PROBES)) == 12
    assert len(set(calibration.FAR_OUT_OF_SCOPE_PROBES)) == 17


def test_probes_do_not_reuse_the_evaluation_query_strings():
    evaluation = {q.text for q in queries.EVAL_QUERIES}
    assert not evaluation & set(calibration.IN_SCOPE_PROBES)
