"""
guardrails/validator.py
-----------------------
Validates user input before passing it to the LangGraph pipeline.
Prevents prompt injection, invalid inputs, and harmful queries.
"""

import re
from dataclasses import dataclass
from typing import Optional


# ── Injection / harmful patterns ─────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|prior)\s+instructions?",
    r"forget\s+(everything|all|prior|previous)",
    r"(system\s*prompt|base\s*prompt)",
    r"(delete|clear|reset)\s+(agent|memory|context|history)",
    r"(jailbreak|dan\s+mode|act\s+as\s+(an?\s+)?(ai|llm|gpt))",
    r"(sudo|root|admin)\s+(mode|access|override)",
    r"(reveal|show|print|display)\s+(your\s+)?(instructions|prompt|system)",
    r"</?(system|instruction|prompt)>",
    r"\bexecute\s+(code|shell|command|script)\b",
    r"(bomb|weapon|explosive|poison|drug\s+synthesis)",
]

VALID_DESTINATION_MIN_LEN = 2
MAX_BUDGET = 100_000_000  # ₹10 Crore cap
MIN_BUDGET = 1_000        # ₹1,000 minimum
MAX_DAYS = 30
MIN_DAYS = 1
MAX_TRAVELERS = 50


@dataclass
class ValidationResult:
    valid: bool
    error_message: Optional[str] = None
    sanitized_query: Optional[str] = None


def _check_injection(text: str) -> Optional[str]:
    """Return an error string if injection / harmful pattern is detected."""
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return (
                "⚠️ Your query contains restricted content. "
                "Please enter a genuine travel request."
            )
    return None


def validate_inputs(
    destination: str,
    budget: float,
    days: int,
    travelers: int,
    preferences: list,
    raw_query: str = "",
) -> ValidationResult:
    """
    Full validation pipeline for user travel inputs.

    Parameters
    ----------
    destination : str
    budget      : float  – total budget in INR
    days        : int    – trip duration
    travelers   : int    – number of people
    preferences : list   – selected travel styles
    raw_query   : str    – free-text query (checked for injections)

    Returns
    -------
    ValidationResult
    """

    # ── 1. Destination ────────────────────────────────────────────────────────
    destination = destination.strip()
    if len(destination) < VALID_DESTINATION_MIN_LEN:
        return ValidationResult(
            valid=False,
            error_message="🌍 Please enter a valid destination (city or country).",
        )

    injection_err = _check_injection(destination)
    if injection_err:
        return ValidationResult(valid=False, error_message=injection_err)

    # ── 2. Budget ─────────────────────────────────────────────────────────────
    if budget < MIN_BUDGET:
        return ValidationResult(
            valid=False,
            error_message=f"💰 Budget must be at least ₹{MIN_BUDGET:,}.",
        )
    if budget > MAX_BUDGET:
        return ValidationResult(
            valid=False,
            error_message=f"💰 Budget cannot exceed ₹{MAX_BUDGET:,}.",
        )

    # ── 3. Days ───────────────────────────────────────────────────────────────
    if days < MIN_DAYS:
        return ValidationResult(
            valid=False, error_message="📅 Trip must be at least 1 day."
        )
    if days > MAX_DAYS:
        return ValidationResult(
            valid=False,
            error_message=f"📅 Trip cannot exceed {MAX_DAYS} days.",
        )

    # ── 4. Travelers ──────────────────────────────────────────────────────────
    if travelers < 1:
        return ValidationResult(
            valid=False, error_message="👥 At least 1 traveler is required."
        )
    if travelers > MAX_TRAVELERS:
        return ValidationResult(
            valid=False,
            error_message=f"👥 Maximum {MAX_TRAVELERS} travelers supported.",
        )

    # ── 5. Free-text injection check ─────────────────────────────────────────
    if raw_query:
        injection_err = _check_injection(raw_query)
        if injection_err:
            return ValidationResult(valid=False, error_message=injection_err)

    # ── 6. Sanitize destination ───────────────────────────────────────────────
    safe_destination = re.sub(r"[<>{}\[\]|\\^`]", "", destination)

    return ValidationResult(
        valid=True,
        sanitized_query=safe_destination,
    )
