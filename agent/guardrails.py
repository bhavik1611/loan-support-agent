"""Task 10. Input-side masking and detection, output-side groundedness.

mask_pii is a contract rather than a helper: Part 3 Task 12 runs the same
function over what it writes to disk, because the brief requires that a
fixed-format PII field never reach a log in the clear.

Aadhaar and account numbers overlap by format. An Aadhaar is twelve digits;
an account number is eleven to sixteen, so nine of the sixty-six generated
customers collide exactly. The mask therefore keys on format alone and is
fail-safe, and the Verhoeff check digit only chooses which label to print.
One of those nine also passes Verhoeff, so the label is wrong once in
sixty-six and the redaction is never wrong. That is the trade D-35 makes.
"""

import re

import config
from db.generate import is_valid_aadhaar

_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

# Grouped form first, because the bare-digits pattern would otherwise match
# only the first group of four and leave the rest of the number in the clear.
_GROUPED_12 = re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b")
_BARE_DIGITS = re.compile(
    rf"\b\d{{{config.ACCOUNT_DIGITS_MIN},{config.ACCOUNT_DIGITS_MAX}}}\b"
)


def _label_for(digits: str) -> str:
    """AADHAAR when the check digit agrees, ACCOUNT otherwise. Both are masked."""
    return "AADHAAR" if is_valid_aadhaar(digits) else "ACCOUNT"


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Replace every fixed-format PII value, and name the rules that fired.

    Returns the masked text and the sorted distinct rule names, so a caller
    can report which guardrail fired without re-running the patterns.
    """
    fired: set[str] = set()

    def _pan(match: re.Match) -> str:
        fired.add("PAN")
        return config.PII_PLACEHOLDERS["PAN"]

    def _digits(match: re.Match) -> str:
        bare = re.sub(r"[ -]", "", match.group(0))
        label = _label_for(bare)
        fired.add(label)
        return config.PII_PLACEHOLDERS[label]

    masked = _PAN.sub(_pan, text)
    masked = _GROUPED_12.sub(_digits, masked)
    masked = _BARE_DIGITS.sub(_digits, masked)
    return masked, sorted(fired)
