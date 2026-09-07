
import json
import re
from typing import Optional

INTENT_REQUIRED_FIELDS = {
    "education": ["age", "state", "income", "education"],
    "health": ["age", "income", "state"],
    "agriculture": ["occupation", "state"],
    "housing": ["income", "rural_urban", "state"],
    "pension": ["age", "occupation"],
    "business": ["occupation", "state"],
    "employment_skill": ["age", "education"],
    "women_empowerment": ["gender", "state"],
    "food_security": ["income", "rural_urban"],
    "general": ["state"],
}

INTENT_KEYWORDS = {
    "education": ["scholarship", "education", "student", "study", "college", "school fees", "tuition"],
    "health": ["health", "hospital", "medical", "insurance", "treatment", "illness"],
    "agriculture": ["farm", "farmer", "crop", "agriculture", "irrigation", "kisan"],
    "housing": ["house", "housing", "home loan", "pucca house", "shelter"],
    "pension": ["pension", "old age", "senior citizen", "retirement"],
    "business": ["loan", "business", "startup", "enterprise", "shop", "self-employ"],
    "employment_skill": ["job", "employment", "skill", "training", "internship", "apprenticeship"],
    "women_empowerment": ["women", "woman", "girl child", "self-help group"],
    "food_security": ["ration", "food security", "food grain", "pds"],
}

INDIAN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "delhi", "jammu and kashmir",
    "ladakh", "puducherry", "chandigarh",
]

FIELD_QUESTIONS = {
    "age": "your age",
    "state": "which state you're from",
    "income": "your approximate annual family income",
    "education": "your current education level",
    "occupation": "your occupation",
    "gender": "your gender",
    "rural_urban": "whether you live in a rural or urban area",
    "social_category": "your social category (General/OBC/SC/ST), if you'd like schemes tailored to that",
}


def detect_intent(message: str) -> str:
    text = message.lower()
    scores = {
        intent: sum(1 for kw in kws if kw in text)
        for intent, kws in INTENT_KEYWORDS.items()
    }
    best_intent = max(scores, key=scores.get)
    return best_intent if scores[best_intent] > 0 else "general"


def extract_profile_fallback(message: str, existing_profile: Optional[dict] = None) -> dict:
    """Dependency-free NL profile extractor. Merges newly-found fields into
    existing_profile (new info never overwrites an already-known field with
    nothing -- it only adds/updates fields it actually finds)."""
    profile = dict(existing_profile or {})
    text = message.lower()

    age_match = re.search(r"\b(\d{1,3})\s*[-]?\s*year[s]?[-]?\s*old\b", text) or re.search(r"\bage\s*(?:is|:)?\s*(\d{1,3})\b", text) or re.search(r"\bi'?m\s*(\d{1,3})\b", text)
    if age_match:
        profile["age"] = int(age_match.group(1))

    if re.search(r"\b(female|woman|girl)\b", text):
        profile["gender"] = "female"
    elif re.search(r"\b(male|man|boy)\b", text):
        profile["gender"] = "male"

    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*lakh", text)
    crore_match = re.search(r"(\d+(?:\.\d+)?)\s*crore", text)
    plain_income_match = re.search(r"income[^\d]{0,15}(\d{4,9})", text)
    if lakh_match:
        profile["income"] = int(float(lakh_match.group(1)) * 100_000)
    elif crore_match:
        profile["income"] = int(float(crore_match.group(1)) * 10_000_000)
    elif plain_income_match:
        profile["income"] = int(plain_income_match.group(1))

    for state in INDIAN_STATES:
        if re.search(rf"\b{re.escape(state)}\b", text):
            profile["state"] = state.title()
            break

    occupation_map = [
        ("student", r"\bstudent\b"),
        ("farmer", r"\bfarmer\b"),
        ("entrepreneur", r"\b(entrepreneur|startup founder|business owner|businessman|businesswoman)\b"),
        ("unemployed", r"\bunemployed\b"),
        ("self-employed", r"\bself[- ]employed\b"),
        ("government employee", r"\bgovernment employee\b"),
        ("unorganised sector worker", r"\bunorgani[sz]ed (sector )?worker\b"),
    ]
    for label, pattern in occupation_map:
        if re.search(pattern, text):
            profile["occupation"] = label
            break

    education_map = [
        ("higher education", r"\b(graduate|college|university|higher education|postgraduate)\b"),
        ("school education", r"\b(school|10th|12th|class \d)\b"),
    ]
    for label, pattern in education_map:
        if re.search(pattern, text):
            profile["education"] = label
            break

    if re.search(r"\brural\b", text):
        profile["rural_urban"] = "rural"
    elif re.search(r"\burban\b", text):
        profile["rural_urban"] = "urban"

    social_category_map = [
        ("SC", r"\bsc category\b|\bscheduled caste\b"),
        ("ST", r"\bst category\b|\bscheduled tribe\b"),
        ("OBC", r"\bobc\b"),
        ("Minority", r"\bminorit"),
        ("BPL", r"\bbpl\b|below poverty line|poor family|poor household"),
        ("General", r"\bgeneral category\b"),
    ]
    for label, pattern in social_category_map:
        if re.search(pattern, text):
            profile["social_category"] = label
            break

    intent = detect_intent(message)
    if intent != "general" or "intent" not in profile:
        profile["intent"] = intent

    return profile


def extract_profile_gemini(message: str, existing_profile: dict, gemini_model) -> dict:
    prompt = f"""Extract a user profile from the message below as to help
match them to Indian government welfare schemes.

EXISTING KNOWN PROFILE (do not drop fields already known unless the new
message clearly updates them):
{json.dumps(existing_profile or {})}

USER MESSAGE:
{message}

Return ONLY a JSON object (no markdown, no commentary) with any of these
keys you can confidently fill in from the message -- omit a key entirely if
it is not mentioned or not clearly inferable, do not guess:
age (integer, years), gender (string: "male"/"female"/other as stated),
income (integer, annual family income in INR), state (string, Indian
state/UT name), occupation (string), education (string), rural_urban
(string: "rural" or "urban"), social_category (string), intent (short
string describing what kind of help they want, e.g. "education",
"health", "agriculture", "housing", "pension", "business",
"employment_skill", "women_empowerment", "food_security", or "general").
"""
    try:
        response = gemini_model.generate_content(prompt)
        cleaned = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
        extracted = json.loads(cleaned)
        profile = dict(existing_profile or {})
        profile.update({k: v for k, v in extracted.items() if v not in (None, "")})
        return profile
    except Exception:
        return extract_profile_fallback(message, existing_profile)


def missing_fields_for_intent(profile: dict) -> list:
    """UPGRADE 4: which relevant fields are still missing for this user's
    apparent intent. Never returns fields irrelevant to the intent."""
    intent = profile.get("intent", "general")
    required = INTENT_REQUIRED_FIELDS.get(intent, INTENT_REQUIRED_FIELDS["general"])
    return [f for f in required if not profile.get(f)]


def build_followup_question(missing_fields: list) -> str:
    if not missing_fields:
        return ""
    phrases = [FIELD_QUESTIONS.get(f, f) for f in missing_fields]
    if len(phrases) == 1:
        joined = phrases[0]
    else:
        joined = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
    return f"Could you tell me {joined}? That'll help me find schemes you're likely eligible for."


if __name__ == "__main__":
    # Quick local sanity checks for the dependency-free fallback extractor.
    msg = ("I'm a 22-year-old male student from Punjab. My family income is "
           "around 2.5 lakh per year and I'm looking for education support.")
    profile = extract_profile_fallback(msg)
    print(profile)
    assert profile["age"] == 22
    assert profile["gender"] == "male"
    assert profile["occupation"] == "student"
    assert profile["state"] == "Punjab"
    assert profile["income"] == 250_000
    assert profile["intent"] == "education"

    # Turn 2: partial follow-up should not erase what's already known
    profile2 = extract_profile_fallback("I need a scholarship", {})
    print(profile2)
    missing = missing_fields_for_intent(profile2)
    print("missing:", missing)
    print(build_followup_question(missing))

    profile3 = extract_profile_fallback("I'm 22 and from Punjab.", profile2)
    print(profile3)
    missing3 = missing_fields_for_intent(profile3)
    print("still missing:", missing3)
    assert "age" not in missing3 and "state" not in missing3

    print("All fallback extractor checks passed.")
