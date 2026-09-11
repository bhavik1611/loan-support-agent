"""The knowledge-base loader must be complete, sorted and strict."""

import pytest

from rag import kb

DOCS = kb.load_documents()


def test_all_eighteen_documents_load():
    assert len(DOCS) == 18


def test_documents_are_sorted_by_doc_id():
    assert [d.doc_id for d in DOCS] == sorted(d.doc_id for d in DOCS)


def test_twelve_required_and_six_neighbours():
    assert sum(1 for d in DOCS if d.required) == 12
    assert sum(1 for d in DOCS if not d.required) == 6


def test_doc_id_matches_the_filename():
    for doc in DOCS:
        assert doc.doc_id == doc.path.stem


def test_topics_are_unique():
    topics = [d.topic for d in DOCS]
    assert len(set(topics)) == len(topics)


def test_body_excludes_the_front_matter():
    for doc in DOCS:
        assert "doc_id:" not in doc.body
        assert not doc.body.startswith("---")
        assert doc.body == doc.body.strip()
        assert len(doc.body) > 200


def test_a_missing_field_is_an_error_not_a_default(tmp_path):
    (tmp_path / "kb-99-broken.md").write_text(
        "---\ndoc_id: kb-99-broken\ntitle: Broken\n---\n\nBody.\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="topic"):
        kb.load_documents(tmp_path)


def test_titles_map_covers_every_document():
    titles = kb.document_titles()
    assert len(titles) == 18
    assert titles["kb-01-loan-eligibility"] == "Loan eligibility criteria by loan type"
