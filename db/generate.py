"""Task 3. The seven row generators that feed db/build.py.

Every function opens its own seeded stream with random.Random(config.stream(<table>))
and never shares that Random instance with another generator. Two calls to the
same function draw from a freshly seeded stream each time, so the output is
deterministic and calling one generator twice cannot perturb another's draws.

Every controlled vocabulary here is transcribed from the knowledge base, not
invented:
- kyc_documents.doc_type: identity and address proofs from kb-04-kyc-documents.md,
  plus the passport-and-visa pair kb-12-nri-account-eligibility.md requires of a
  non-resident.
- support_tickets.channel: the reporting channels kb-05-fraud-dispute.md names.
- The EMI arithmetic in _emi()/_schedule() is kb-02-emi-calculation.md's formula
  and nothing else in this repository computes an EMI.
"""

import random
import string

import config

# ---------------------------------------------------------------------------
# loan_products
# ---------------------------------------------------------------------------


def generate_loan_products() -> list[dict]:
    """One row per category, every number sourced from config, none retyped."""
    products = []
    for category in config.CATEGORIES:
        min_amount_inr, max_amount_inr = config.CATEGORY_BANDS[category]
        min_rate_pct, max_rate_pct = config.RATE_BANDS[category]
        products.append(
            {
                "product_code": config.PRODUCT_CODES[category],
                "category": category,
                "min_amount_inr": min_amount_inr,
                "max_amount_inr": max_amount_inr,
                "min_rate_pct": min_rate_pct,
                "max_rate_pct": max_rate_pct,
                "max_tenure_months": config.MAX_TENURE_MONTHS[category],
                "is_secured": category in config.SECURED_CATEGORIES,
            }
        )
    return products


# ---------------------------------------------------------------------------
# customers
# ---------------------------------------------------------------------------

# Fabricated Indian first and last names. No real person's data.
FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Krishna",
    "Ishaan", "Rohan", "Kabir", "Ananya", "Diya", "Saanvi", "Aadhya", "Isha",
    "Kavya", "Meera", "Priya", "Riya", "Sneha", "Neha", "Pooja", "Anjali",
    "Rahul", "Amit", "Vikram", "Sanjay", "Manish", "Deepak", "Suresh",
    "Lakshmi", "Divya", "Nisha", "Rajesh", "Ramesh", "Naveen",
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Kumar", "Singh", "Patel", "Reddy", "Nair",
    "Iyer", "Rao", "Menon", "Chatterjee", "Mukherjee", "Banerjee", "Joshi",
    "Desai", "Kulkarni", "Pillai", "Agarwal", "Bhatt", "Malhotra", "Kapoor",
    "Chauhan", "Yadav", "Mehta", "Shah",
]

# (city, state) pairs, Indian metros and second-tier cities.
CITY_STATE_PAIRS = [
    ("Mumbai", "Maharashtra"),
    ("Pune", "Maharashtra"),
    ("Nagpur", "Maharashtra"),
    ("Delhi", "Delhi"),
    ("Bengaluru", "Karnataka"),
    ("Chennai", "Tamil Nadu"),
    ("Coimbatore", "Tamil Nadu"),
    ("Hyderabad", "Telangana"),
    ("Kolkata", "West Bengal"),
    ("Ahmedabad", "Gujarat"),
    ("Surat", "Gujarat"),
    ("Jaipur", "Rajasthan"),
    ("Lucknow", "Uttar Pradesh"),
    ("Chandigarh", "Punjab"),
    ("Bhopal", "Madhya Pradesh"),
    ("Indore", "Madhya Pradesh"),
    ("Patna", "Bihar"),
    ("Kochi", "Kerala"),
]

EMPLOYMENT_TYPES = ["Salaried", "Self-employed", "Business Owner", "Retired"]
EMPLOYMENT_WEIGHTS = [0.55, 0.25, 0.15, 0.05]

KYC_STATUSES = ["Verified", "Pending", "Re-verification due"]
KYC_STATUS_WEIGHTS = [0.75, 0.15, 0.10]

IS_NRI_PROBABILITY = 0.08


# The Income Tax Department's PAN structure, AAAAA9999A: three alphabetic
# series characters, a holder-type code, the surname initial, a four-digit
# serial and a check letter. Every customer here is an individual, so the
# holder-type code is always P.
PAN_HOLDER_TYPE = "P"


def _pan_check_letter(first_nine: str) -> str:
    """The tenth character, derived from the other nine.

    The real check character comes from a formula the Income Tax Department
    does not publish, so this is a fabricated stand-in: letters score A=1 to
    Z=26, digits score face value, each is weighted by its position, and the
    total modulo 26 picks the letter. Deterministic, and recomputable by
    anyone holding the first nine characters.
    """
    total = sum(
        position * (ord(char) - 64 if char.isalpha() else int(char))
        for position, char in enumerate(first_nine, start=1)
    )
    return string.ascii_uppercase[total % 26]


def _unique_pan(rng: random.Random, last_name: str, seen: set[str]) -> str:
    """A structurally valid fabricated PAN for one individual.

    Only the three series characters and the four-digit serial are drawn; the
    holder type is fixed and the surname initial and check letter are derived,
    so a PAN cannot contradict the name it sits beside.
    """
    while True:
        series = "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
        serial = rng.randint(1, 9999)
        first_nine = f"{series}{PAN_HOLDER_TYPE}{last_name[0].upper()}{serial:04d}"
        pan = first_nine + _pan_check_letter(first_nine)
        if pan not in seen:
            seen.add(pan)
            return pan


def _unique_aadhaar(rng: random.Random, seen: set[str]) -> str:
    while True:
        first = str(rng.randint(2, 9))
        rest = "".join(str(rng.randint(0, 9)) for _ in range(11))
        aadhaar = first + rest
        if aadhaar not in seen:
            seen.add(aadhaar)
            return aadhaar


def _unique_account_number(rng: random.Random, seen: set[str]) -> str:
    while True:
        length = rng.randint(11, 16)
        first = str(rng.randint(1, 9))
        rest = "".join(str(rng.randint(0, 9)) for _ in range(length - 1))
        account_number = first + rest
        if account_number not in seen:
            seen.add(account_number)
            return account_number


def _phone(rng: random.Random) -> str:
    first = rng.choice("6789")
    rest = "".join(str(rng.randint(0, 9)) for _ in range(9))
    return first + rest


def generate_customers() -> list[dict]:
    """config.CUSTOMER_COUNT (66) fabricated customers on their own stream.

    credit_score is drawn across the full config.CREDIT_SCORE_MIN..MAX band,
    skewed toward the high end with a triangular distribution, then floored at
    config.CREDIT_SCORE_LOAN_FLOOR (700). kb-01 sets 700 as the minimum score
    for any loan product, and every customer generated here ends up holding at
    least one loan through assign_customers, so no customer may fall below it.
    """
    rng = random.Random(config.stream("customers"))
    seen_pan: set[str] = set()
    seen_aadhaar: set[str] = set()
    seen_account: set[str] = set()

    customers = []
    for i in range(1, config.CUSTOMER_COUNT + 1):
        first_name = rng.choice(FIRST_NAMES)
        last_name = rng.choice(LAST_NAMES)
        city, state = rng.choice(CITY_STATE_PAIRS)

        raw_score = rng.triangular(
            config.CREDIT_SCORE_MIN, config.CREDIT_SCORE_MAX, config.CREDIT_SCORE_MAX
        )
        credit_score = max(config.CREDIT_SCORE_LOAN_FLOOR, round(raw_score))

        customers.append(
            {
                "customer_id": f"CUST-{i:04d}",
                "full_name": f"{first_name} {last_name}",
                "city": city,
                "state": state,
                "pan": _unique_pan(rng, last_name, seen_pan),
                "aadhaar": _unique_aadhaar(rng, seen_aadhaar),
                "account_number": _unique_account_number(rng, seen_account),
                "email": f"{first_name.lower()}.{last_name.lower()}{i}@example.com",
                "phone": _phone(rng),
                "employment_type": rng.choices(
                    EMPLOYMENT_TYPES, weights=EMPLOYMENT_WEIGHTS, k=1
                )[0],
                "annual_income_inr": int(round(rng.uniform(300_000, 30_00_000), -3)),
                "credit_score": credit_score,
                "kyc_status": rng.choices(
                    KYC_STATUSES, weights=KYC_STATUS_WEIGHTS, k=1
                )[0],
                "is_nri": rng.random() < IS_NRI_PROBABILITY,
            }
        )
    return customers


# ---------------------------------------------------------------------------
# assign_customers
# ---------------------------------------------------------------------------


def assign_customers(applications: list[dict], customers: list[dict]) -> list[dict]:
    """Implement D-20: shuffle the applications, deal them out to the mix.

    Uses its own Random instance seeded from the customers stream, per the
    plan ("shuffle the 100 applications on the customers stream"). This is a
    fresh Random object, not the one generate_customers() used, so no state is
    shared between the two calls even though they share a seed.

    The first LOANS_PER_CUSTOMER_MIX[0][1] customers (in the order customers
    was generated) each get one loan, the next batch get two, and so on, so
    every customer ends up with at least one and the totals match the mix
    exactly.
    """
    rng = random.Random(config.stream("customers"))
    shuffled_applications = list(applications)
    rng.shuffle(shuffled_applications)

    customer_ids_per_loan = []
    index = 0
    for loans_per_customer, customer_count in config.LOANS_PER_CUSTOMER_MIX:
        for _ in range(customer_count):
            customer_ids_per_loan.extend(
                [customers[index]["customer_id"]] * loans_per_customer
            )
            index += 1

    return [
        {**application, "customer_id": customer_id}
        for application, customer_id in zip(shuffled_applications, customer_ids_per_loan)
    ]


# ---------------------------------------------------------------------------
# application_events
# ---------------------------------------------------------------------------

# Legal status chain leading to each terminal status. Submitted -> Under
# Review -> (Approved | Rejected), Approved -> Disbursed.
STATUS_CHAINS = {
    "Submitted": ["Submitted"],
    "Under Review": ["Submitted", "Under Review"],
    "Approved": ["Submitted", "Under Review", "Approved"],
    "Rejected": ["Submitted", "Under Review", "Rejected"],
    "Disbursed": ["Submitted", "Under Review", "Approved", "Disbursed"],
}

ARRIVAL_NOTES = {
    "Submitted": "Application submitted through the online channel",
    "Under Review": "Documents received, application under credit review",
    "Approved": "Loan sanctioned, offer letter issued to the applicant",
    "Rejected": "Application rejected after credit review",
    "Disbursed": "Sanctioned amount disbursed to the linked account",
}

# Extra, same-status logging events used to pad a short chain up toward the
# 2-to-5 range without inventing an illegal transition.
FOLLOW_UP_NOTES = [
    "Follow-up call logged with the applicant",
    "Additional document requested from the applicant",
    "Status confirmed to the applicant on request",
    "Internal checklist updated, no status change",
]

MIN_EVENTS = 2
MAX_EVENTS = 5


def generate_application_events(applications: list[dict]) -> list[dict]:
    """2 to 5 events per application, a legal path ending at its status.

    occurred_days_ago strictly decreases along the sequence and the first
    event's occurred_days_ago equals the application's days_since_created, so
    the number of events an application can carry is bounded by how many days
    it has been alive: at most days_since_created + 1 distinct non-negative
    day values exist.

    Applications younger than the full legal chain (there is exactly one in
    the committed dataset: an Approved application 0 days old) cannot show
    every intermediate stage in that budget, so the trail is compressed to the
    most recent stages that do fit rather than fabricating days that have not
    happened yet. The first event of a compressed trail still has
    from_status = NULL; it anchors the trail at whichever stage the day
    budget allows rather than always at "Submitted".
    """
    rng = random.Random(config.stream("application_events"))
    events = []
    event_id = 1

    for application in applications:
        status = application["status"]
        days_since_created = application["days_since_created"]
        chain = STATUS_CHAINS[status]

        max_events_by_days = days_since_created + 1
        if max_events_by_days < len(chain):
            chain = chain[-max_events_by_days:]

        min_count = max(MIN_EVENTS, len(chain))
        max_count = min(MAX_EVENTS, max_events_by_days)
        if max_count >= min_count:
            total = rng.randint(min_count, max_count)
        else:
            # Day budget too tight even for the compressed chain (the 0-day
            # case): fall back to the minimal legal chain, accepting a single
            # event rather than fabricating a day that has not occurred.
            total = len(chain)

        extra_days_needed = total - 1
        sampled_days = (
            rng.sample(range(0, days_since_created), extra_days_needed)
            if extra_days_needed > 0
            else []
        )
        days_sequence = [days_since_created] + sorted(sampled_days, reverse=True)

        stage_sequence = list(chain) + [chain[-1]] * (total - len(chain))

        previous_status = None
        for position, to_status in enumerate(stage_sequence):
            note = (
                ARRIVAL_NOTES[to_status]
                if position < len(chain)
                else rng.choice(FOLLOW_UP_NOTES)
            )
            events.append(
                {
                    "event_id": event_id,
                    "record_id": application["record_id"],
                    "sequence_no": position + 1,
                    "from_status": previous_status,
                    "to_status": to_status,
                    "occurred_days_ago": days_sequence[position],
                    "note": note,
                }
            )
            event_id += 1
            previous_status = to_status

    return events


# ---------------------------------------------------------------------------
# repayments
# ---------------------------------------------------------------------------


def _emi(principal: int, annual_rate_pct: float, tenure_months: int) -> float:
    """kb-02's formula, and nothing else may compute an EMI in this repository."""
    r = annual_rate_pct / 12.0 / 100.0
    growth = (1.0 + r) ** tenure_months
    return principal * r * growth / (growth - 1.0)


def _schedule(principal: int, annual_rate_pct: float, tenure_months: int, months: int):
    """Standard amortisation: interest on the running balance, principal is the rest."""
    r = annual_rate_pct / 12.0 / 100.0
    emi = _emi(principal, annual_rate_pct, tenure_months)
    balance = float(principal)
    for instalment_no in range(1, min(months, tenure_months) + 1):
        interest = balance * r
        principal_part = emi - interest
        balance -= principal_part
        yield {
            "instalment_no": instalment_no,
            "emi_inr": round(emi, 2),
            "interest_inr": round(interest, 2),
            "principal_inr": round(principal_part, 2),
            "balance_inr": round(max(balance, 0.0), 2),
        }


# First instalment falls due 30 days after disbursal, then every 30 days.
DAYS_BETWEEN_INSTALMENTS = 30


def generate_repayments(applications: list[dict]) -> list[dict]:
    """The first config.SCHEDULE_MONTHS instalments, only for Disbursed loans.

    Loans in this dataset are 0 to 30 days old (days_since_created stands in
    for days since disbursal, since a disbursed application was created very
    recently relative to today). The first instalment is not due until
    DAYS_BETWEEN_INSTALMENTS days after disbursal, so with a loan book this
    young at most one instalment's due date can have passed, and in the
    generated data every disbursed loan is younger than that, so `paid` is
    False throughout. The comment stands regardless of the exact mix: no
    repayment history is fabricated beyond what the loan's age can support.
    """
    repayments = []
    repayment_id = 1

    for application in applications:
        if application["status"] != "Disbursed":
            continue

        days_since_created = application["days_since_created"]
        for instalment in _schedule(
            application["loan_amount_inr"],
            application["interest_rate_pct"],
            application["tenure_months"],
            config.SCHEDULE_MONTHS,
        ):
            due_days_ago = days_since_created - (
                DAYS_BETWEEN_INSTALMENTS * instalment["instalment_no"]
            )
            repayments.append(
                {
                    "repayment_id": repayment_id,
                    "record_id": application["record_id"],
                    "instalment_no": instalment["instalment_no"],
                    "due_days_ago": due_days_ago,
                    "emi_inr": instalment["emi_inr"],
                    "principal_inr": instalment["principal_inr"],
                    "interest_inr": instalment["interest_inr"],
                    "balance_inr": instalment["balance_inr"],
                    "paid": due_days_ago >= 0,
                }
            )
            repayment_id += 1

    return repayments


# ---------------------------------------------------------------------------
# support_tickets
# ---------------------------------------------------------------------------

# kb-05: "helpline, net banking, the mobile app, or any branch."
TICKET_CHANNELS = ["Helpline", "Net Banking", "Mobile App", "Branch"]

TICKET_CATEGORIES = [
    "Fraud Dispute",
    "Loan Enquiry",
    "EMI or Repayment",
    "KYC Update",
    "Account Service",
    "Card Complaint",
]
NON_FRAUD_CATEGORIES = [c for c in TICKET_CATEGORIES if c != "Fraud Dispute"]

TICKET_STATUSES = ["Open", "In Progress", "Resolved", "Escalated", "Closed"]

# Application-linked tickets drawn from fraud-flagged applications with this
# probability, well above their base rate in the population, so Part 2 gets a
# signal from the fraud flag.
FRAUD_TICKET_OVERWEIGHT = 0.65


def generate_support_tickets(applications: list[dict], customers: list[dict]) -> list[dict]:
    """config.TICKET_COUNT (40) tickets, about half linked to an application.

    Fraud-flagged applications are over-represented among the linked half.
    """
    rng = random.Random(config.stream("support_tickets"))

    customer_ids = [customer["customer_id"] for customer in customers]
    flagged_applications = [
        a for a in applications if a.get("flagged_for_fraud_review")
    ]
    other_applications = [
        a for a in applications if not a.get("flagged_for_fraud_review")
    ]

    linked_count = config.TICKET_COUNT // 2
    account_level_count = config.TICKET_COUNT - linked_count

    tickets = []
    ticket_no = 1

    for _ in range(linked_count):
        use_flagged = bool(flagged_applications) and (
            rng.random() < FRAUD_TICKET_OVERWEIGHT or not other_applications
        )
        pool = flagged_applications if use_flagged else other_applications
        application = rng.choice(pool)
        is_fraud_related = bool(application.get("flagged_for_fraud_review"))
        category = (
            "Fraud Dispute" if is_fraud_related else rng.choice(NON_FRAUD_CATEGORIES)
        )
        days_since_created = application["days_since_created"]
        opened_days_ago = (
            rng.randint(0, days_since_created) if days_since_created > 0 else 0
        )
        channel = rng.choice(TICKET_CHANNELS)
        tickets.append(
            {
                "ticket_id": f"TKT-{ticket_no:04d}",
                "customer_id": application["customer_id"],
                "record_id": application["record_id"],
                "channel": channel,
                "category": category,
                "opened_days_ago": opened_days_ago,
                "status": rng.choice(TICKET_STATUSES),
                "summary": f"{category} reported via {channel.lower()}",
            }
        )
        ticket_no += 1

    for _ in range(account_level_count):
        category = rng.choice(NON_FRAUD_CATEGORIES)
        channel = rng.choice(TICKET_CHANNELS)
        tickets.append(
            {
                "ticket_id": f"TKT-{ticket_no:04d}",
                "customer_id": rng.choice(customer_ids),
                "record_id": None,
                "channel": channel,
                "category": category,
                "opened_days_ago": rng.randint(0, 90),
                "status": rng.choice(TICKET_STATUSES),
                "summary": f"{category} query logged via {channel.lower()}",
            }
        )
        ticket_no += 1

    return tickets


# ---------------------------------------------------------------------------
# kyc_documents
# ---------------------------------------------------------------------------

# kb-04: identity proofs Meridian Bank accepts.
IDENTITY_DOC_TYPES = [
    "Passport",
    "Voter Identity Card",
    "Driving Licence",
    "Aadhaar Card",
    "Job Card issued under NREGA",
]

# kb-04: address proofs Meridian Bank accepts.
ADDRESS_DOC_TYPES = [
    "Passport",
    "Utility Bill",
    "Property Tax Receipt",
    "Bank Statement",
]

MIN_DOCS_PER_CUSTOMER = 2
MAX_DOCS_PER_CUSTOMER = 4
EXTRA_DOC_VERIFIED_PROBABILITY = 0.85


def generate_kyc_documents(customers: list[dict]) -> list[dict]:
    """2 to 4 documents per customer: at least one identity, one address.

    kb-12 additionally requires a non-resident to submit a passport and a
    valid visa, so an NRI customer carries a second passport entry plus a
    visa on top of the base identity and address proof, landing at exactly 4.
    """
    rng = random.Random(config.stream("kyc_documents"))
    documents = []
    document_no = 1

    for customer in customers:
        identity_type = rng.choice(IDENTITY_DOC_TYPES)
        address_type = rng.choice(ADDRESS_DOC_TYPES)
        entries = [(identity_type, "identity"), (address_type, "address")]

        if customer["is_nri"]:
            entries.append(("Passport", "identity"))
            entries.append(("Visa", "identity"))
        else:
            extra_pool = [
                (doc_type, "identity")
                for doc_type in IDENTITY_DOC_TYPES
                if doc_type != identity_type
            ] + [
                (doc_type, "address")
                for doc_type in ADDRESS_DOC_TYPES
                if doc_type != address_type
            ]
            extra_count = rng.randint(0, MAX_DOCS_PER_CUSTOMER - MIN_DOCS_PER_CUSTOMER)
            entries.extend(rng.sample(extra_pool, extra_count))

        for doc_type, doc_kind in entries:
            documents.append(
                {
                    "document_id": f"KYC-{document_no:04d}",
                    "customer_id": customer["customer_id"],
                    "doc_type": doc_type,
                    "doc_kind": doc_kind,
                    "submitted_days_ago": rng.randint(1, 60),
                    "verified": rng.random() < EXTRA_DOC_VERIFIED_PROBABILITY,
                }
            )
            document_no += 1

    return documents
