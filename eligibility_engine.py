
from dataclasses import dataclass, field

UNKNOWN_VALUE = "Not specified"
ALL_VALUE = "All"

ELIGIBLE = "ELIGIBLE"
NOT_ELIGIBLE = "NOT_ELIGIBLE"
UNKNOWN = "UNKNOWN"

PASS = "PASS"
FAIL = "FAIL"
DIM_UNKNOWN = "UNKNOWN"
NOT_APPLICABLE = "NOT_APPLICABLE"  # scheme has no requirement on this dimension


@dataclass
class DimensionCheck:
    dimension: str # what are we checking ?
    result: str  # PASS / FAIL / UNKNOWN / NOT_APPLICABLE
    reason: str # why ?


@dataclass
class EligibilityResult:
    status: str  # ELIGIBLE / NOT_ELIGIBLE / UNKNOWN
    checks: list = field(default_factory=list)  # list[DimensionCheck]
    missing_fields: list = field(default_factory=list)  # user fields needed to resolve UNKNOWN
    summary: str = ""


def _has_value(x) -> bool:
    """True if a scheme field carries an actual requirement (not a sentinel)."""
    return x is not None and str(x).strip() not in (UNKNOWN_VALUE, ALL_VALUE, "", "nan")


def _to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _check_age(user, scheme) -> DimensionCheck:
    min_age, max_age = scheme.get("min_age"), scheme.get("max_age")
    if not _has_value(min_age) and not _has_value(max_age):
        return DimensionCheck("age", NOT_APPLICABLE, "This scheme has no stated age restriction.")
    user_age = _to_int(user.get("age"))
    if user_age is None:
        return DimensionCheck("age", DIM_UNKNOWN, "Need the user's age to check this scheme's age requirement.")
    min_a = _to_int(min_age) if _has_value(min_age) else None
    max_a = _to_int(max_age) if _has_value(max_age) else None
    if min_a is not None and user_age < min_a:
        return DimensionCheck("age", FAIL, f"Age {user_age} is below the minimum required age of {min_a}.")
    if max_a is not None and user_age > max_a:
        return DimensionCheck("age", FAIL, f"Age {user_age} is above the maximum allowed age of {max_a}.")
    parts = []
    if min_a is not None:
        parts.append(f"at least {min_a}")
    if max_a is not None:
        parts.append(f"at most {max_a}")
    return DimensionCheck("age", PASS, f"Age {user_age} satisfies the required age ({' and '.join(parts)}).")


def _check_income(user, scheme) -> DimensionCheck:
    min_i, max_i = scheme.get("min_income"), scheme.get("max_income")
    if not _has_value(min_i) and not _has_value(max_i):
        return DimensionCheck("income", NOT_APPLICABLE, "This scheme has no stated income limit.")
    user_income = _to_int(user.get("income"))
    if user_income is None:
        return DimensionCheck("income", DIM_UNKNOWN, "Need the user's annual family income to check this scheme's income limit.")
    min_v = _to_int(min_i) if _has_value(min_i) else None
    max_v = _to_int(max_i) if _has_value(max_i) else None
    if max_v is not None and user_income > max_v:
        return DimensionCheck("income", FAIL, f"Income {user_income} exceeds the stated limit of {max_v} by {user_income - max_v}.")
    if min_v is not None and user_income < min_v:
        return DimensionCheck("income", FAIL, f"Income {user_income} is below the stated minimum of {min_v}.")
    return DimensionCheck("income", PASS, "Income is within the scheme's stated limit.")


def _check_categorical(user_key, scheme_key, label, user, scheme, match_fn=None) -> DimensionCheck:
    scheme_val = scheme.get(scheme_key)
    if not _has_value(scheme_val):
        return DimensionCheck(label, NOT_APPLICABLE, f"This scheme has no stated {label} restriction.")
    user_val = user.get(user_key)
    if not user_val:
        return DimensionCheck(label, DIM_UNKNOWN, f"Need the user's {label} to check this scheme's {label} requirement.")
    matched = match_fn(user_val, scheme_val) if match_fn else str(user_val).strip().lower() == str(scheme_val).strip().lower()
    if matched:
        return DimensionCheck(label, PASS, f"{label.capitalize()} '{user_val}' matches the requirement '{scheme_val}'.")
    return DimensionCheck(label, FAIL, f"{label.capitalize()} '{user_val}' does not match the requirement '{scheme_val}'.")


def _social_category_match(user_val, scheme_val) -> bool:
    allowed = {v.strip().lower() for v in str(scheme_val).split("/")}
    return str(user_val).strip().lower() in allowed


def _occupation_match(user_val, scheme_val) -> bool:
    allowed = {v.strip().lower() for v in str(scheme_val).split(",")}
    user_val = str(user_val).strip().lower()
    return any(user_val in a or a in user_val for a in allowed)


def check_eligibility(user_profile: dict, scheme: dict) -> EligibilityResult:
    
    checks = [
        _check_age(user_profile, scheme),
        _check_income(user_profile, scheme),
        _check_categorical("gender", "gender", "gender", user_profile, scheme),
        _check_categorical("state", "state", "state", user_profile, scheme),
        _check_categorical("rural_urban", "rural_urban", "rural/urban", user_profile, scheme),
        _check_categorical("social_category", "social_category", "social category", user_profile, scheme, _social_category_match),
        _check_categorical("occupation", "occupation", "occupation", user_profile, scheme, _occupation_match),
        _check_categorical("education", "education", "education level", user_profile, scheme),
    ]

    failed = [c for c in checks if c.result == FAIL]
    unknown = [c for c in checks if c.result == DIM_UNKNOWN]
    applicable = [c for c in checks if c.result != NOT_APPLICABLE]

    field_for_dimension = {
        "age": "age", "income": "income", "gender": "gender", "state": "state",
        "rural/urban": "rural_urban", "social category": "social_category",
        "occupation": "occupation", "education level": "education",
    }

    if failed:
        summary = "; ".join(c.reason for c in failed)
        return EligibilityResult(NOT_ELIGIBLE, checks, [], summary)

    if unknown:
        missing = [field_for_dimension[c.dimension] for c in unknown]
        summary = "Need more information: " + ", ".join(missing)
        return EligibilityResult(UNKNOWN, checks, missing, summary)

    if not applicable:
        return EligibilityResult(
            UNKNOWN, checks, [],
            "No structured eligibility criteria available for this scheme in our dataset.",
        )

    passed = [c.reason for c in applicable if c.result == PASS]
    return EligibilityResult(ELIGIBLE, checks, [], "; ".join(passed))


def gap_summary(result: EligibilityResult) -> list:
    symbol = {PASS: "\u2713", FAIL: "\u2717", DIM_UNKNOWN: "?", NOT_APPLICABLE: "-"}
    return [(c.dimension, symbol[c.result], c.reason) for c in result.checks if c.result != NOT_APPLICABLE]
