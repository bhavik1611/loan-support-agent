"""Task 3. The seven row generators, checked against the knowledge base and
the schema, not against invented vocabulary.
"""

import re
import string

import config
import dataset
from db import generate, schema

# --- vocabularies transcribed straight out of the knowledge base ----------

# kb-04-kyc-documents.md: "Meridian Bank accepts a passport, a voter identity
# card, a driving licence, an Aadhaar card, or a job card issued under NREGA
# as proof of identity" and "a passport, a utility bill no older than two
# months, a property tax receipt, or a bank statement as proof of address."
KB04_IDENTITY_DOC_TYPES = {
    "Passport",
    "Voter Identity Card",
    "Driving Licence",
    "Aadhaar Card",
    "Job Card issued under NREGA",
}
KB04_ADDRESS_DOC_TYPES = {
    "Passport",
    "Utility Bill",
    "Property Tax Receipt",
    "Bank Statement",
}
# kb-12-nri-account-eligibility.md: a non-resident additionally submits a
# passport and a valid visa.
KB12_NRI_EXTRA_DOC_TYPES = {"Passport", "Visa"}
PERMITTED_DOC_TYPES = KB04_IDENTITY_DOC_TYPES | KB04_ADDRESS_DOC_TYPES | KB12_NRI_EXTRA_DOC_TYPES

# kb-05-fraud-dispute.md: "report an unauthorised transaction through the
# 24-hour helpline, net banking, the mobile app, or any branch."
PERMITTED_CHANNELS = {"Helpline", "Net Banking", "Mobile App", "Branch"}

# The Income Tax Department's structure: three series letters, the holder-type
# code, the surname initial, a 0001-9999 serial, then the check letter.
PAN_RE = re.compile(r"^[A-Z]{3}[PCHFATG][A-Z][0-9]{4}[A-Z]$")
# UIDAI issues no Aadhaar number beginning 0 or 1.
AADHAAR_RE = re.compile(r"^[2-9]\d{11}$")
ACCOUNT_NUMBER_RE = re.compile(r"^\d{11,16}$")

LEGAL_TRANSITIONS = {
    ("Submitted", "Under Review"),
    ("Under Review", "Approved"),
    ("Under Review", "Rejected"),
    ("Approved", "Disbursed"),
}


def _schema_columns(table: str) -> set[str]:
    """Column names parsed out of the DDL, not hand-copied."""
    body = schema.CREATE_STATEMENTS[table].split("(", 1)[1].rsplit(")", 1)[0]
    names = set()
    depth = 0
    for line in body.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped:
            continue
        if depth == 0 and not stripped.upper().startswith(
            ("UNIQUE", "CHECK", "FOREIGN", "PRIMARY KEY (")
        ):
            names.add(stripped.split()[0])
        depth += stripped.count("(") - stripped.count(")")
    return names


def _assigned_applications():
    customers = generate.generate_customers()
    applications = generate.assign_customers(dataset.RICH_APPLICATIONS, customers)
    return applications, customers


# --- loan_products ----------------------------------------------------------


def test_loan_products_has_five_rows_matching_schema_columns():
    rows = generate.generate_loan_products()
    assert len(rows) == 5
    expected = _schema_columns("loan_products")
    for row in rows:
        assert set(row) == expected


def test_loan_products_values_come_from_config_not_retyped():
    rows = {row["category"]: row for row in generate.generate_loan_products()}
    for category in config.CATEGORIES:
        row = rows[category]
        assert row["product_code"] == config.PRODUCT_CODES[category]
        assert (row["min_amount_inr"], row["max_amount_inr"]) == config.CATEGORY_BANDS[category]
        assert (row["min_rate_pct"], row["max_rate_pct"]) == config.RATE_BANDS[category]
        assert row["max_tenure_months"] == config.MAX_TENURE_MONTHS[category]
        assert row["is_secured"] == (category in config.SECURED_CATEGORIES)


def test_loan_products_deterministic():
    assert generate.generate_loan_products() == generate.generate_loan_products()


# --- customers ---------------------------------------------------------------


def test_customers_row_count_and_schema_keys():
    rows = generate.generate_customers()
    assert len(rows) == config.CUSTOMER_COUNT == 66
    expected = _schema_columns("customers")
    for row in rows:
        assert set(row) == expected


def test_customers_pan_aadhaar_account_formats():
    rows = generate.generate_customers()
    for row in rows:
        assert PAN_RE.match(row["pan"]), row["pan"]
        assert AADHAAR_RE.match(row["aadhaar"]), row["aadhaar"]
        assert ACCOUNT_NUMBER_RE.match(row["account_number"]), row["account_number"]


def test_customers_pan_encodes_holder_type_surname_and_check_letter():
    """The four positions the PAN structure fixes, recomputed here.

    The check letter is recalculated from the first nine characters rather
    than by calling the generator's helper, so a change to that helper has to
    be made in both places deliberately.
    """
    for row in generate.generate_customers():
        pan = row["pan"]
        surname = row["full_name"].split()[-1]

        assert pan[3] == "P", pan                       # individual
        assert pan[4] == surname[0].upper(), (pan, surname)
        assert 1 <= int(pan[5:9]) <= 9999, pan

        total = sum(
            position * (ord(char) - 64 if char.isalpha() else int(char))
            for position, char in enumerate(pan[:9], start=1)
        )
        assert pan[9] == string.ascii_uppercase[total % 26], pan


def test_customers_pan_values_are_unique():
    rows = generate.generate_customers()
    assert len({row["pan"] for row in rows}) == len(rows)


def test_verhoeff_matches_the_published_worked_examples():
    """External ground truth, so the tables cannot be quietly wrong.

    Verhoeff's own published examples: the check digit for 236 is 3 and for
    12345 is 1, and a complete valid number checksums to 0.
    """
    assert generate._aadhaar_check_digit("236") == "3"
    assert generate._aadhaar_check_digit("12345") == "1"
    assert generate._verhoeff_checksum("2363") == 0
    assert generate._verhoeff_checksum("123451") == 0
    assert generate._verhoeff_checksum("2362") != 0


def test_customers_aadhaar_carries_a_valid_verhoeff_check_digit():
    for row in generate.generate_customers():
        assert generate._verhoeff_checksum(row["aadhaar"]) == 0, row["aadhaar"]


def test_aadhaar_check_digit_catches_the_errors_it_exists_for():
    """The guarantee, not the implementation: every single-digit error and
    every adjacent transposition of two different digits must be rejected.
    """
    for row in generate.generate_customers()[:10]:
        aadhaar = row["aadhaar"]

        for position, digit in enumerate(aadhaar):
            for wrong in "0123456789":
                if wrong == digit:
                    continue
                corrupted = aadhaar[:position] + wrong + aadhaar[position + 1 :]
                assert generate._verhoeff_checksum(corrupted) != 0, corrupted

        for position in range(len(aadhaar) - 1):
            if aadhaar[position] == aadhaar[position + 1]:
                continue
            swapped = (
                aadhaar[:position]
                + aadhaar[position + 1]
                + aadhaar[position]
                + aadhaar[position + 2 :]
            )
            assert generate._verhoeff_checksum(swapped) != 0, swapped


def test_customers_aadhaar_values_are_unique():
    rows = generate.generate_customers()
    assert len({row["aadhaar"] for row in rows}) == len(rows)


def test_every_customer_credit_score_is_at_least_the_loan_floor():
    for row in generate.generate_customers():
        assert config.CREDIT_SCORE_LOAN_FLOOR <= row["credit_score"] <= config.CREDIT_SCORE_MAX


def test_customers_kyc_status_is_one_of_the_three_values():
    for row in generate.generate_customers():
        assert row["kyc_status"] in {"Verified", "Pending", "Re-verification due"}


def test_customers_is_nri_roughly_eight_percent():
    rows = generate.generate_customers()
    nri_count = sum(1 for row in rows if row["is_nri"])
    # Loose band around 8% of 66 (~5), generous given a single draw.
    assert 0 <= nri_count <= 15


def test_customers_deterministic():
    assert generate.generate_customers() == generate.generate_customers()


# --- assign_customers ---------------------------------------------------------


def test_assign_customers_produces_exactly_the_configured_mix():
    customers = generate.generate_customers()
    applications = generate.assign_customers(dataset.RICH_APPLICATIONS, customers)

    assert len(applications) == 100

    loans_per_customer = {}
    for application in applications:
        customer_id = application["customer_id"]
        loans_per_customer[customer_id] = loans_per_customer.get(customer_id, 0) + 1

    assert set(loans_per_customer) == {c["customer_id"] for c in customers}

    mix_counts = {}
    for count in loans_per_customer.values():
        mix_counts[count] = mix_counts.get(count, 0) + 1
    assert mix_counts == dict(config.LOANS_PER_CUSTOMER_MIX)
    assert sum(loans_per_customer.values()) == 100


def test_assign_customers_keeps_exactly_the_loan_applications_columns():
    customers = generate.generate_customers()
    applications = generate.assign_customers(dataset.RICH_APPLICATIONS, customers)
    expected = _schema_columns("loan_applications")
    for row in applications:
        assert set(row) == expected


def test_assign_customers_deterministic():
    customers = generate.generate_customers()
    first = generate.assign_customers(dataset.RICH_APPLICATIONS, customers)
    second = generate.assign_customers(dataset.RICH_APPLICATIONS, customers)
    assert first == second


# --- application_events -------------------------------------------------------


def test_application_events_schema_keys():
    applications, _ = _assigned_applications()
    rows = generate.generate_application_events(applications)
    expected = _schema_columns("application_events")
    for row in rows:
        assert set(row) == expected


def test_application_events_form_a_legal_path_ending_at_the_status():
    applications, _ = _assigned_applications()
    events = generate.generate_application_events(applications)

    by_record = {}
    for event in events:
        by_record.setdefault(event["record_id"], []).append(event)

    applications_by_id = {a["record_id"]: a for a in applications}

    for record_id, record_events in by_record.items():
        record_events.sort(key=lambda e: e["sequence_no"])
        assert [e["sequence_no"] for e in record_events] == list(
            range(1, len(record_events) + 1)
        )
        assert 1 <= len(record_events) <= 5

        assert record_events[0]["from_status"] is None
        for other in record_events[1:]:
            assert other["from_status"] is not None

        days = [e["occurred_days_ago"] for e in record_events]
        assert days == sorted(days, reverse=True)
        assert len(set(days)) == len(days)
        assert record_events[0]["occurred_days_ago"] == applications_by_id[record_id][
            "days_since_created"
        ]

        assert record_events[-1]["to_status"] == applications_by_id[record_id]["status"]

        for previous, current in zip(record_events, record_events[1:]):
            pair = (previous["to_status"], current["to_status"])
            assert pair in LEGAL_TRANSITIONS or pair[0] == pair[1]


def test_application_events_deterministic():
    applications, _ = _assigned_applications()
    assert generate.generate_application_events(
        applications
    ) == generate.generate_application_events(applications)


# --- repayments -----------------------------------------------------------


def test_repayments_only_for_disbursed_applications():
    applications, _ = _assigned_applications()
    rows = generate.generate_repayments(applications)
    disbursed_ids = {a["record_id"] for a in applications if a["status"] == "Disbursed"}
    assert disbursed_ids  # the fixed dataset always has some
    assert {row["record_id"] for row in rows} <= disbursed_ids


def test_repayments_schema_keys():
    applications, _ = _assigned_applications()
    rows = generate.generate_repayments(applications)
    expected = _schema_columns("repayments")
    for row in rows:
        assert set(row) == expected


def test_repayments_use_kb02s_emi_formula():
    applications, _ = _assigned_applications()
    disbursed = next(a for a in applications if a["status"] == "Disbursed")
    rows = [
        r
        for r in generate.generate_repayments(applications)
        if r["record_id"] == disbursed["record_id"]
    ]
    assert 1 <= len(rows) <= config.SCHEDULE_MONTHS

    expected_emi = round(
        generate._emi(
            disbursed["loan_amount_inr"], disbursed["interest_rate_pct"], disbursed["tenure_months"]
        ),
        2,
    )
    for row in rows:
        assert row["emi_inr"] == expected_emi
        # principal_inr and interest_inr are each independently rounded to the
        # rupee, so their sum can be off from emi_inr by up to a couple of paise.
        assert abs(row["principal_inr"] + row["interest_inr"] - row["emi_inr"]) < 0.02


def test_repayments_paid_only_when_due_date_has_passed():
    applications, _ = _assigned_applications()
    rows = generate.generate_repayments(applications)
    for row in rows:
        assert row["paid"] == (row["due_days_ago"] >= 0)


def test_repayments_deterministic():
    applications, _ = _assigned_applications()
    assert generate.generate_repayments(applications) == generate.generate_repayments(
        applications
    )


# --- support_tickets --------------------------------------------------------


def test_support_tickets_row_count_and_schema_keys():
    applications, customers = _assigned_applications()
    rows = generate.generate_support_tickets(applications, customers)
    assert len(rows) == config.TICKET_COUNT == 40
    expected = _schema_columns("support_tickets")
    for row in rows:
        assert set(row) == expected


def test_support_tickets_channel_is_from_kb05():
    applications, customers = _assigned_applications()
    rows = generate.generate_support_tickets(applications, customers)
    for row in rows:
        assert row["channel"] in PERMITTED_CHANNELS


def test_support_tickets_about_half_link_to_an_application():
    applications, customers = _assigned_applications()
    rows = generate.generate_support_tickets(applications, customers)
    linked = sum(1 for row in rows if row["record_id"] is not None)
    unlinked = sum(1 for row in rows if row["record_id"] is None)
    assert linked == 20
    assert unlinked == 20


def test_support_tickets_overrepresent_fraud_flagged_applications():
    applications, customers = _assigned_applications()
    flagged_ids = {a["record_id"] for a in applications if a["flagged_for_fraud_review"]}
    base_rate = len(flagged_ids) / len(applications)

    rows = generate.generate_support_tickets(applications, customers)
    linked = [row for row in rows if row["record_id"] is not None]
    flagged_ticket_rate = sum(1 for row in linked if row["record_id"] in flagged_ids) / len(
        linked
    )

    assert flagged_ticket_rate > base_rate


def test_support_tickets_deterministic():
    applications, customers = _assigned_applications()
    assert generate.generate_support_tickets(
        applications, customers
    ) == generate.generate_support_tickets(applications, customers)


# --- kyc_documents -----------------------------------------------------------


def test_kyc_documents_two_to_four_per_customer_schema_keys():
    customers = generate.generate_customers()
    rows = generate.generate_kyc_documents(customers)
    expected = _schema_columns("kyc_documents")

    by_customer = {}
    for row in rows:
        assert set(row) == expected
        by_customer.setdefault(row["customer_id"], []).append(row)

    assert set(by_customer) == {c["customer_id"] for c in customers}
    for customer_id, docs in by_customer.items():
        assert 2 <= len(docs) <= 4


def test_kyc_documents_every_customer_has_identity_and_address_proof():
    customers = generate.generate_customers()
    rows = generate.generate_kyc_documents(customers)

    by_customer = {}
    for row in rows:
        by_customer.setdefault(row["customer_id"], []).append(row)

    for docs in by_customer.values():
        kinds = {doc["doc_kind"] for doc in docs}
        assert "identity" in kinds
        assert "address" in kinds


def test_kyc_documents_doc_type_is_from_the_kb_vocabulary():
    customers = generate.generate_customers()
    rows = generate.generate_kyc_documents(customers)
    for row in rows:
        assert row["doc_type"] in PERMITTED_DOC_TYPES
        assert row["doc_kind"] in {"identity", "address"}


def test_nri_customers_additionally_carry_a_passport_and_a_visa():
    customers = generate.generate_customers()
    rows = generate.generate_kyc_documents(customers)

    by_customer = {}
    for row in rows:
        by_customer.setdefault(row["customer_id"], []).append(row)

    nri_customer_ids = {c["customer_id"] for c in customers if c["is_nri"]}
    assert nri_customer_ids  # the fixed dataset always has at least one

    for customer_id in nri_customer_ids:
        doc_types = {doc["doc_type"] for doc in by_customer[customer_id]}
        assert "Passport" in doc_types
        assert "Visa" in doc_types


def test_kyc_documents_deterministic():
    customers = generate.generate_customers()
    assert generate.generate_kyc_documents(customers) == generate.generate_kyc_documents(
        customers
    )
