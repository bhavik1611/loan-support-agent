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


def _seed(tmp_path, documents: str, products: str = '["Widget Loan"]', bodies=("kb-99-broken",)):
    """A throwaway knowledge base. `documents` is the entries object verbatim."""
    catalogue = f'{{"products": {products}, "documents": {documents}}}'
    (tmp_path / kb.CATALOGUE_NAME).write_text(catalogue, encoding="utf-8")
    for stem in bodies:
        (tmp_path / f"{stem}{kb.DOCUMENT_SUFFIX}").write_text("Body." * 60, encoding="utf-8")
    return tmp_path


def test_a_missing_field_is_an_error_not_a_default(tmp_path):
    _seed(
        tmp_path,
        '{"kb-99-broken": {"title": "Broken", "required": true,'
        ' "products": ["Widget Loan"]}}',
    )
    with pytest.raises(ValueError, match="topic"):
        kb.load_documents(tmp_path)


def test_a_mistyped_field_is_an_error_not_a_coercion(tmp_path):
    _seed(
        tmp_path,
        '{"kb-99-broken": {"title": "Broken", "topic": "t", "required": "yes",'
        ' "products": ["Widget Loan"]}}',
    )
    with pytest.raises(ValueError, match="required must be bool"):
        kb.load_documents(tmp_path)


def test_a_catalogued_document_with_no_file_is_an_error(tmp_path):
    """The half of the drift a glob alone would never notice."""
    _seed(
        tmp_path,
        '{"kb-99-broken": {"title": "B", "topic": "t", "required": true,'
        ' "products": ["Widget Loan"]},'
        ' "kb-98-ghost": {"title": "G", "topic": "u", "required": false,'
        ' "products": ["Widget Loan"]}}',
    )
    with pytest.raises(ValueError, match="kb-98-ghost"):
        kb.load_documents(tmp_path)


def test_a_file_with_no_catalogue_entry_is_an_error(tmp_path):
    """The other half: a body that would embed with no topic and no required flag."""
    _seed(
        tmp_path,
        '{"kb-99-broken": {"title": "B", "topic": "t", "required": true,'
        ' "products": ["Widget Loan"]}}',
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


def test_every_document_carries_a_products_tag():
    """D-52: the catalogue names what each document covers, not just its topic."""
    for doc in DOCS:
        assert isinstance(doc.products, tuple)
        assert doc.products, f"{doc.doc_id} covers no product"
        assert list(doc.products) == sorted(doc.products)


def test_an_empty_product_tag_is_an_error(tmp_path):
    """The loader is as strict as the corpus invariant above, not one step behind.

    A document covering nothing is retrievable unfiltered and invisible to every
    filtered query, which is the catalogue disagreeing with itself quietly.
    """
    _seed(
        tmp_path,
        '{"kb-99-broken": {"title": "B", "topic": "t", "required": true,'
        ' "products": []}}',
    )
    with pytest.raises(ValueError, match="is empty"):
        kb.load_documents(tmp_path)


def test_minimum_balance_covers_every_account_it_applies_to():
    """kb-09 states a requirement for accounts, not for one account type.

    The document is the authority over catalogue.json. It sets an average
    monthly balance by branch tier for any account, charges a shortfall on any
    account, and names salary and basic savings accounts only to exempt them.
    Tagging it with those two alone hid it from a filtered NRE or NRO query,
    which then answered from kb-12 and kb-18 without ever seeing the figures.
    """
    tagged = set(kb.documents_for_product("NRE account"))
    assert "kb-09-minimum-balance" in tagged
    for product in ("NRO account", "joint account", "salary account", "savings account"):
        assert "kb-09-minimum-balance" in kb.documents_for_product(product), product


def test_every_product_tag_names_a_catalogued_product():
    """Acceptance criterion 31, read off the loaded documents."""
    products = set(kb.catalogue_products())
    for doc in DOCS:
        stray = sorted(set(doc.products) - products)
        assert not stray, f"{doc.doc_id} names {stray}"


def test_the_catalogue_sells_eleven_products_and_no_current_account():
    """The list is read out of the corpus (D-47), never invented.

    "Current account" is the one a reader expects and the corpus does not have:
    no document mentions one, so adding it would make catalogue.json disagree
    with the documents it describes, and the documents are the authority.
    """
    products = kb.catalogue_products()
    assert len(products) == 11
    assert len(set(products)) == 11
    assert not [p for p in products if "current" in p.casefold()]


def test_every_catalogued_product_is_tagged_on_some_document():
    """A product no document covers would filter every query to nothing."""
    tagged = {p for doc in DOCS for p in doc.products}
    missing = sorted(set(kb.catalogue_products()) - tagged)
    assert not missing, f"catalogued but covered by no document: {missing}"


def test_a_product_tag_outside_the_catalogue_is_an_error(tmp_path):
    """Criterion 31 the other way: the loader refuses rather than filtering oddly."""
    _seed(
        tmp_path,
        '{"kb-99-broken": {"title": "B", "topic": "t", "required": true,'
        ' "products": ["Widget Loan", "Fixed Deposit"]}}',
    )
    with pytest.raises(ValueError, match="Fixed Deposit"):
        kb.load_documents(tmp_path)


def test_a_catalogue_with_no_products_list_is_an_error(tmp_path):
    (tmp_path / kb.CATALOGUE_NAME).write_text(
        '{"documents": {"kb-99-broken": {"title": "B", "topic": "t",'
        ' "required": true, "products": ["Widget Loan"]}}}',
        encoding="utf-8",
    )
    (tmp_path / f"kb-99-broken{kb.DOCUMENT_SUFFIX}").write_text("Body." * 60, encoding="utf-8")
    with pytest.raises(ValueError, match="products"):
        kb.load_documents(tmp_path)


def test_documents_for_product_reads_the_catalogue_not_the_prose():
    tagged = kb.documents_for_product("Home Loan")
    assert "kb-14-home-loan-ltv" in tagged
    assert "kb-01-loan-eligibility" in tagged
    assert "kb-11-joint-account-rules" not in tagged
    assert tagged == sorted(tagged)
