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


def test_the_dataset_has_twenty_nine_items_in_four_classes():
    """D-55 and spec 9.1: 12 answerable, 10 outside, 2 uncovered, 5 far out."""
    assert len(queries.GOLDEN_DATASET) == 29
    assert Counter(i.kind for i in queries.GOLDEN_DATASET) == Counter(
        {
            queries.KIND_ANSWERABLE: 12,
            queries.KIND_OUTSIDE_BOUNDARY: 10,
            queries.KIND_INSIDE_UNCOVERED: 2,
            queries.KIND_FAR_OUT_OF_SCOPE: 5,
        }
    )


def test_item_ids_are_unique_and_carry_their_class_prefix():
    ids = [i.item_id for i in queries.GOLDEN_DATASET]
    assert len(set(ids)) == 29
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
    assert len(set(texts)) == 29


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
    """Acceptance criterion 30: the two sets are disjoint.

    The fitting set derives T and the golden dataset scores it. A string in both
    would mean the threshold was scored on the readings that set it, which
    measures nothing, and it is the property that lets this be called a golden
    dataset at all.

    This asserted the intersection twice and called it "both directions", which
    is not a thing set intersection has: `a & b` and `b & a` are the same
    expression written twice. Stated once here, and the coverage that genuinely
    widens it is the next test, which takes all 29 items against both probe
    lists rather than 5 against one.
    """
    far_items = {i.text for i in queries.items_of_kind(queries.KIND_FAR_OUT_OF_SCOPE)}
    probes = set(calibration.FAR_OUT_OF_SCOPE_PROBES)
    assert len(far_items) == 5
    assert not far_items & probes


def test_no_golden_item_reuses_any_calibration_probe():
    """The same boundary over the whole dataset, not only its far-out class."""
    golden = {i.text for i in queries.GOLDEN_DATASET}
    fitting = set(calibration.IN_SCOPE_PROBES) | set(calibration.FAR_OUT_OF_SCOPE_PROBES)
    assert not golden & fitting


def test_no_golden_item_reuses_a_routing_probe_except_the_named_eq_11_case():
    """The same boundary again, this time against eval/routing.py.

    Nothing previously checked this: the calibration separation above only
    ever compared eval/queries.py against eval/calibration.py, so a routing
    probe set could quietly reuse golden text without any test noticing.

    EQ-11 is the one deliberate exception. eval/routing.py::SELF_CONTAINED_PROBES
    references it by id from eval.queries rather than retyping it, because it
    is the literal regression probe for a named ellipsis-matcher defect (spec
    18.4) and must be the exact query that exposed the defect, not a
    paraphrase of it. One documented exception is a decision; a second one
    found here would be a pattern, so this fails rather than growing a second
    name into the exception if it ever finds one.

    Probe texts are read off the module's own list/tuple collections rather
    than retyped, so a probe set added to eval/routing.py later is covered
    without this test being edited.
    """
    from eval import routing

    probe_texts: set[str] = set()
    for name, value in vars(routing).items():
        if name.startswith("_") or not isinstance(value, (list, tuple)):
            continue
        for item in value:
            if isinstance(item, str):
                probe_texts.add(item)
            elif isinstance(item, tuple) and item and isinstance(item[0], str):
                probe_texts.add(item[0])

    eq_11_text = next(item.text for item in queries.EVAL_QUERIES if item.item_id == "EQ-11")
    golden_texts = {item.text for item in queries.GOLDEN_DATASET}

    assert golden_texts & probe_texts == {eq_11_text}


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
