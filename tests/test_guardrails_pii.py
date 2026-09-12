"""Test 22 of spec section 16, and the format collision D-35 exists for."""

import config
from agent import guardrails
from db import generate as dbgen

VALID_AADHAAR = "392847105628"      # fabricated, Verhoeff valid
INVALID_TWELVE = "392847105629"     # same digits, check digit deliberately wrong
ACCOUNT_14 = "88400575668282"
PAN = "FXZPG5049K"


def test_a_valid_aadhaar_is_recognised_as_one():
    assert dbgen.is_valid_aadhaar(VALID_AADHAAR) is True


def test_a_twelve_digit_number_with_a_bad_check_digit_is_not_an_aadhaar():
    assert dbgen.is_valid_aadhaar(INVALID_TWELVE) is False


def test_every_generated_aadhaar_validates(db_conn):
    rows = db_conn.execute("SELECT aadhaar FROM customers").fetchall()
    assert rows
    assert all(dbgen.is_valid_aadhaar(r["aadhaar"]) for r in rows)


def test_pan_is_masked():
    masked, rules = guardrails.mask_pii(f"My PAN is {PAN} please check")
    assert PAN not in masked
    assert config.PII_PLACEHOLDERS["PAN"] in masked
    assert rules == ["PAN"]


def test_aadhaar_is_masked_and_labelled():
    masked, rules = guardrails.mask_pii(f"Aadhaar {VALID_AADHAAR}")
    assert VALID_AADHAAR not in masked
    assert config.PII_PLACEHOLDERS["AADHAAR"] in masked
    assert rules == ["AADHAAR"]


def test_aadhaar_is_masked_when_written_in_groups_of_four():
    masked, rules = guardrails.mask_pii("Aadhaar 3928 4710 5628")
    assert "3928" not in masked
    assert config.PII_PLACEHOLDERS["AADHAAR"] in masked
    assert rules == ["AADHAAR"]


def test_a_fourteen_digit_account_number_is_masked():
    masked, rules = guardrails.mask_pii(f"Account {ACCOUNT_14}")
    assert ACCOUNT_14 not in masked
    assert config.PII_PLACEHOLDERS["ACCOUNT"] in masked
    assert rules == ["ACCOUNT"]


def test_a_twelve_digit_account_number_is_still_masked():
    """D-35. The label may be wrong; the redaction may not."""
    masked, rules = guardrails.mask_pii(f"Account {INVALID_TWELVE}")
    assert INVALID_TWELVE not in masked
    assert rules == ["ACCOUNT"]


def test_every_generated_account_number_is_masked(db_conn):
    """The nine twelve-digit ones are the reason this test iterates all 66."""
    rows = db_conn.execute("SELECT account_number FROM customers").fetchall()
    for row in rows:
        number = row["account_number"]
        masked, rules = guardrails.mask_pii(f"my account is {number}")
        assert number not in masked, number
        assert rules, number


def test_all_three_rules_fire_together():
    masked, rules = guardrails.mask_pii(
        f"PAN {PAN}, Aadhaar {VALID_AADHAAR}, account {ACCOUNT_14}"
    )
    assert rules == ["AADHAAR", "ACCOUNT", "PAN"]
    for raw in (PAN, VALID_AADHAAR, ACCOUNT_14):
        assert raw not in masked


def test_ordinary_text_is_left_alone():
    text = "What is the minimum credit score for a home loan in 2026?"
    masked, rules = guardrails.mask_pii(text)
    assert masked == text
    assert rules == []


def test_a_record_id_is_not_mistaken_for_pii():
    """LN-1042 must survive masking, because the router reads it afterwards."""
    masked, rules = guardrails.mask_pii("What is the status of LN-1042?")
    assert "LN-1042" in masked
    assert rules == []
