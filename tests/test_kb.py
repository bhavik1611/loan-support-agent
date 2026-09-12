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


def test_the_body_is_prose_and_carries_no_metadata():
    """D-32: a document is content only. Its metadata lives in the catalogue."""
    for doc in DOCS:
        assert "doc_id:" not in doc.body
        assert "required:" not in doc.body
        assert not doc.body.startswith("---")
        assert doc.body == doc.body.strip()
        assert len(doc.body) > 200


def _seed(tmp_path, catalogue: str, bodies=("kb-99-broken",)):
    (tmp_path / kb.CATALOGUE_NAME).write_text(catalogue, encoding="utf-8")
    for stem in bodies:
        (tmp_path / f"{stem}{kb.DOCUMENT_SUFFIX}").write_text("Body." * 60, encoding="utf-8")
    return tmp_path


def test_a_missing_field_is_an_error_not_a_default(tmp_path):
    _seed(tmp_path, '{"kb-99-broken": {"title": "Broken", "required": true}}')
    with pytest.raises(ValueError, match="topic"):
        kb.load_documents(tmp_path)


def test_a_mistyped_field_is_an_error_not_a_coercion(tmp_path):
    _seed(
        tmp_path,
        '{"kb-99-broken": {"title": "Broken", "topic": "t", "required": "yes"}}',
    )
    with pytest.raises(ValueError, match="required must be bool"):
        kb.load_documents(tmp_path)


def test_a_catalogued_document_with_no_file_is_an_error(tmp_path):
    """The half of the drift a glob alone would never notice."""
    _seed(
        tmp_path,
        '{"kb-99-broken": {"title": "B", "topic": "t", "required": true},'
        ' "kb-98-ghost": {"title": "G", "topic": "u", "required": false}}',
    )
    with pytest.raises(ValueError, match="kb-98-ghost"):
        kb.load_documents(tmp_path)


def test_a_file_with_no_catalogue_entry_is_an_error(tmp_path):
    """The other half: a body that would embed with no topic and no required flag."""
    _seed(
        tmp_path,
        '{"kb-99-broken": {"title": "B", "topic": "t", "required": true}}',
        bodies=("kb-99-broken", "kb-97-orphan"),
    )
    with pytest.raises(ValueError, match="kb-97-orphan"):
        kb.load_documents(tmp_path)


def test_a_missing_catalogue_is_an_error(tmp_path):
    (tmp_path / f"kb-99-broken{kb.DOCUMENT_SUFFIX}").write_text("Body." * 60, encoding="utf-8")
    with pytest.raises(ValueError, match="catalogue"):
        kb.load_documents(tmp_path)


def test_titles_map_covers_every_document():
    titles = kb.document_titles()
    assert len(titles) == 18
    assert titles["kb-01-loan-eligibility"] == "Loan eligibility criteria by loan type"
