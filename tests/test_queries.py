"""The golden dataset must match D-11 and D-55 and reference only real documents."""

from collections import Counter

import pytest

from eval import calibration, queries
from rag import kb

DOC_IDS = {d.doc_id for d in kb.load_documents()}


def test_twelve_queries_with_unique_ids():
    assert len(queries.EVAL_QUERIES) == 12
    ids = [q.item_id for q in queries.EVAL_QUERIES]
    assert len(set(ids)) == 12


def test_every_gold_label_names_a_real_document():
    for query in queries.EVAL_QUERIES:
        for doc_id in query.gold_doc_ids:
            assert doc_id in DOC_IDS, f"{query.item_id} references unknown {doc_id}"


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


def test_the_dataset_has_thirty_two_items_in_four_classes():
    """D-55 and spec 9.1: 12 answerable, 13 outside, 2 uncovered, 5 far out."""
    assert len(queries.GOLDEN_DATASET) == 32
    assert Counter(i.kind for i in queries.GOLDEN_DATASET) == Counter(
        {
            queries.KIND_ANSWERABLE: 12,
            queries.KIND_OUTSIDE_BOUNDARY: 13,
            queries.KIND_INSIDE_UNCOVERED: 2,
            queries.KIND_FAR_OUT_OF_SCOPE: 5,
        }
    )


def test_item_ids_are_unique_and_carry_their_class_prefix():
    ids = [i.item_id for i in queries.GOLDEN_DATASET]
    assert len(set(ids)) == 32
    prefixes = {
        queries.KIND_ANSWERABLE: "EQ-",
        queries.KIND_OUTSIDE_BOUNDARY: "OB-",
        queries.KIND_INSIDE_UNCOVERED: "IU-",
        queries.KIND_FAR_OUT_OF_SCOPE: "FO-",
    }
    for item in queries.GOLDEN_DATASET:
        assert item.item_id.startswith(prefixes[item.kind]), item.item_id


def test_item_texts_are_unique():
    texts = [i.text for i in queries.GOLDEN_DATASET]
    assert len(set(texts)) == 32


def test_only_answerable_items_carry_gold_documents():
    """A gold label on an item that must be refused would be scoring a refusal."""
    for item in queries.GOLDEN_DATASET:
        if item.kind == queries.KIND_ANSWERABLE:
            assert item.gold_doc_ids
        else:
            assert item.gold_doc_ids == (), item.item_id


def test_eval_queries_is_the_answerable_subset_and_not_a_second_copy():
    assert queries.EVAL_QUERIES == queries.items_of_kind(queries.KIND_ANSWERABLE)
    assert [i.item_id for i in queries.EVAL_QUERIES] == [
        f"EQ-{n:02d}" for n in range(1, 13)
    ]


def test_the_far_out_of_scope_items_never_reuse_a_calibration_probe():
    """Acceptance criterion 30, checked in both directions.

    The fitting set derives T and the golden dataset scores it. A string in both
    would mean the threshold was scored on the readings that set it, which
    measures nothing, and it is the property that lets this be called a golden
    dataset at all.
    """
    far_items = {i.text for i in queries.items_of_kind(queries.KIND_FAR_OUT_OF_SCOPE)}
    probes = set(calibration.FAR_OUT_OF_SCOPE_PROBES)
    assert len(far_items) == 5
    assert not far_items & probes
    assert not probes & far_items


def test_no_golden_item_reuses_any_calibration_probe():
    """The same boundary over the whole dataset, not only its far-out class."""
    golden = {i.text for i in queries.GOLDEN_DATASET}
    fitting = set(calibration.IN_SCOPE_PROBES) | set(calibration.FAR_OUT_OF_SCOPE_PROBES)
    assert not golden & fitting


def test_the_inside_uncovered_items_are_the_residue_the_spec_names():
    """Spec 18.1 item 8, verbatim. These are what the gate is known to miss."""
    assert [i.text for i in queries.items_of_kind(queries.KIND_INSIDE_UNCOVERED)] == [
        "Can I get a credit card from another bank with a low limit?",
        "How do I transfer money to an account in another country?",
    ]


def test_every_outside_boundary_item_declares_a_product():
    for item in queries.items_of_kind(queries.KIND_OUTSIDE_BOUNDARY):
        assert item.product, item.item_id


def test_items_of_kind_rejects_an_unknown_class():
    with pytest.raises(ValueError, match="unknown kind"):
        queries.items_of_kind("nearly_in_scope")
