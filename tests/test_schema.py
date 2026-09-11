"""The schema must create cleanly and actually enforce its foreign keys."""

import sqlite3

import pytest

import config
from db import schema


def test_table_order_covers_exactly_the_create_statements():
    assert set(schema.TABLE_ORDER) == set(schema.CREATE_STATEMENTS)
    assert len(schema.TABLE_ORDER) == 7


def test_every_statement_creates(tmp_path):
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    names = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == set(schema.TABLE_ORDER)


def test_foreign_keys_are_enforced_not_merely_declared(tmp_path):
    """Without PRAGMA foreign_keys=ON this insert would silently succeed."""
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO kyc_documents VALUES ('D1','NO-SUCH-CUSTOMER','passport','identity',5,1)"
        )


def test_every_table_has_a_primary_key(tmp_path):
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    for table in schema.TABLE_ORDER:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        assert any(c["pk"] for c in cols), table


def test_each_table_has_its_own_distinct_stream():
    offsets = config.STREAM_OFFSETS
    assert len(set(offsets.values())) == len(offsets)
    for table in schema.TABLE_ORDER:
        assert table in offsets, table
    assert config.stream("loan_applications") == config.SEED


def test_an_unknown_table_has_no_stream():
    with pytest.raises(ValueError, match="unknown table"):
        config.stream("nope")
