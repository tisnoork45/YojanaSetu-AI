
import re
import pandas as pd

INPUT_FILE = "data/Final_Govt_Schemes_Dataset.csv"
OUTPUT_FILE = "data/Final_Govt_Schemes_Dataset_Enriched.csv"

UNKNOWN = "Not specified"
ALL = "All"

MINISTRY_MAP = {
    "meity": "Ministry of Electronics and Information Technology",
    "mnre": "Ministry of New and Renewable Energy",
    "moa": "Ministry of Agriculture and Farmers Welfare",
    "moafw": "Ministry of Agriculture and Farmers Welfare",
    "moca": "Ministry of Civil Aviation",
    "mocafpd": "Ministry of Consumer Affairs, Food and Public Distribution",
    "mocf": "Ministry of Chemicals and Fertilizers",
    "moci": "Ministry of Commerce and Industry",
    "moe": "Ministry of Education",
    "mof": "Ministry of Finance",
    "mohfw": "Ministry of Health and Family Welfare",
    "mohrd": "Ministry of Human Resource Development (now Ministry of Education)",
    "mohua": "Ministry of Housing and Urban Affairs",
    "mojs": "Ministry of Jal Shakti",
    "momsde": "Ministry of Skill Development and Entrepreneurship",
    "momsme": "Ministry of Micro, Small and Medium Enterprises",
    "mop": "Ministry of Power",
    "mopng": "Ministry of Petroleum and Natural Gas",
    "mopr": "Ministry of Panchayati Raj",
    "mor": "Ministry of Railways",
    "mord": "Ministry of Rural Development",
    "mota": "Ministry of Tribal Affairs",
    "mowcd": "Ministry of Women and Child Development",
    "mowr": "Ministry of Water Resources, River Development and Ganga Rejuvenation",
}
SCHEME_TYPE_MAP = {"cs": "Central Sector Scheme", "css": "Centrally Sponsored Scheme"}
MINISTRY_TOKEN_PATTERN = re.compile(r"\b(css|cs)\s+([a-z]{2,10})\s+((?:19|20)\d{2})\b")

AGE_PATTERNS = [
    # "after age 60"  -> min_age = 60
    (re.compile(r"after age (\d{1,3})"), lambda m: {"min_age": int(m.group(1))}),
    # "age-group 0-6 years" -> min_age = 0, max_age = 6
    (
        re.compile(r"age-?group (\d{1,3})[- ]?(\d{1,3}) years"),
        lambda m: {"min_age": int(m.group(1)), "max_age": int(m.group(2))},
    ),
    # "birth to 18 years" -> min_age = 0, max_age = 18
    (
        re.compile(r"birth to (\d{1,3}) years"),
        lambda m: {"min_age": 0, "max_age": int(m.group(1))},
    ),
    # "attaining 18 years" (paired with "minors") -> max_age = 18
    (re.compile(r"attaining (\d{1,3}) years"), lambda m: {"max_age": int(m.group(1))}),
]

MINOR_KEYWORD = re.compile(r"\bminors?\b")

FEMALE_KEYWORDS = re.compile(
    r"\b(women|woman|girls?|female|widow|pregnant women|mahila)\b"
)
FEMALE_EXCLUSIONS = [
    re.compile(r"children and pregnant women"),  # mixed audience, e.g. immunization drives
    re.compile(r"\byouth\b"),  # mixed audience, e.g. "youth and women"
]
OR_ELIGIBILITY_QUOTA_PATTERN = re.compile(r"at least one")

SOCIAL_CATEGORY_PATTERNS = [
    ("SC/ST", re.compile(r"\bsc st\b")),
    ("OBC", re.compile(r"\bobc\b")),
    ("Minority", re.compile(r"\bminorit")),
    ("BPL/EWS", re.compile(r"\bbpl\b")),
]

OCCUPATION_PATTERNS = [
    ("Farmer", re.compile(r"\bfarmers?\b")),
    ("Student", re.compile(r"\bstudents?\b")),
    ("Street Vendor", re.compile(r"street vendors?")),
    ("Artisan/Craftsperson", re.compile(r"\bartisans?\b|craftspeople")),
    ("Self-Help Group member", re.compile(r"self-help groups?")),
    ("Unorganised Sector Worker", re.compile(r"unorgani[sz]ed sector")),
    ("Government Employee", re.compile(r"government employees?")),
]

EDUCATION_PATTERNS = [
    ("Higher Education", re.compile(r"higher education|\bcollege\b|\buniversity\b")),
    ("School Education", re.compile(r"\bschool")),
]

RURAL_PATTERN = re.compile(r"\brural\b")
URBAN_PATTERN = re.compile(r"\burban\b")

REGION_OVERRIDES = {
    "north east": "North Eastern States",
}


def extract_age(desc: str) -> dict:
    result = {}
    for pattern, handler in AGE_PATTERNS:
        m = pattern.search(desc)
        if m:
            result.update(handler(m))
    if "max_age" not in result and MINOR_KEYWORD.search(desc):
        result["max_age"] = 18
    return result


def extract_gender(desc: str) -> str:
    if OR_ELIGIBILITY_QUOTA_PATTERN.search(desc):
        return ALL
    if FEMALE_KEYWORDS.search(desc):
        if any(pat.search(desc) for pat in FEMALE_EXCLUSIONS):
            return ALL
        return "Female"
    return ALL


def extract_social_category(desc: str) -> str:
    if OR_ELIGIBILITY_QUOTA_PATTERN.search(desc):
        return ALL
    hits = [label for label, pat in SOCIAL_CATEGORY_PATTERNS if pat.search(desc)]
    return "/".join(dict.fromkeys(hits)) if hits else ALL


def extract_occupation(desc: str) -> str:
    hits = [label for label, pat in OCCUPATION_PATTERNS if pat.search(desc)]
    return ", ".join(hits) if hits else ALL


def extract_education(desc: str) -> str:
    for label, pat in EDUCATION_PATTERNS:
        if pat.search(desc):
            return label
    return UNKNOWN


def extract_rural_urban(desc: str) -> str:
    is_rural = bool(RURAL_PATTERN.search(desc))
    is_urban = bool(URBAN_PATTERN.search(desc))
    if is_rural and not is_urban:
        return "Rural"
    if is_urban and not is_rural:
        return "Urban"
    return ALL  


def extract_state(desc: str) -> str:
    for keyword, label in REGION_OVERRIDES.items():
        if keyword in desc:
            return label
    return ALL


def extract_ministry_fields(desc: str) -> dict:
    m = MINISTRY_TOKEN_PATTERN.search(desc)
    if not m:
        return {"ministry": UNKNOWN, "scheme_type": UNKNOWN, "launch_year": UNKNOWN}
    scheme_type_abbrev, ministry_abbrev, year = m.group(1), m.group(2), m.group(3)
    return {
        "ministry": MINISTRY_MAP.get(ministry_abbrev, UNKNOWN),
        "scheme_type": SCHEME_TYPE_MAP.get(scheme_type_abbrev, UNKNOWN),
        "launch_year": int(year),
    }


def enrich_row(desc: str) -> dict:
    row = {
        "min_age": UNKNOWN,
        "max_age": UNKNOWN,
        "gender": extract_gender(desc),
        "min_income": UNKNOWN,  
        "max_income": UNKNOWN,
        "state": extract_state(desc),
        "rural_urban": extract_rural_urban(desc),
        "social_category": extract_social_category(desc),
        "occupation": extract_occupation(desc),
        "education": extract_education(desc),
        "source_url": UNKNOWN, 
        "last_updated": UNKNOWN, 
    }
    row.update(extract_age(desc))
    row.update(extract_ministry_fields(desc))
    return row


def main():
    print("Loading cleaned dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} schemes")

    print("Extracting structured eligibility fields...")
    enriched_rows = [enrich_row(desc) for desc in df["description"].astype(str)]
    enriched_df = pd.DataFrame(enriched_rows)


    column_order = [
        "scheme_name",
        "category",
        "description",
        "min_age",
        "max_age",
        "gender",
        "min_income",
        "max_income",
        "state",
        "rural_urban",
        "social_category",
        "occupation",
        "education",
        "ministry",
        "source_url",
        "last_updated",
        "scheme_type",
        "launch_year",
    ]
    result = pd.concat([df.reset_index(drop=True), enriched_df], axis=1)[column_order]

    assert len(result) == len(df), "Row count changed during enrichment!"
    assert list(result["scheme_name"]) == list(df["scheme_name"]), "Row order changed!"

    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved enriched dataset -> {OUTPUT_FILE}")

    print("\nField coverage (non-default values found):")
    sentinel_by_col = {
        "min_age": UNKNOWN, "max_age": UNKNOWN, "min_income": UNKNOWN,
        "max_income": UNKNOWN, "source_url": UNKNOWN, "last_updated": UNKNOWN,
        "ministry": UNKNOWN, "scheme_type": UNKNOWN, "launch_year": UNKNOWN,
        "education": UNKNOWN, "gender": ALL, "state": ALL, "rural_urban": ALL,
        "social_category": ALL, "occupation": ALL,
    }
    for col, sentinel in sentinel_by_col.items():
        non_default = (result[col].astype(str) != str(sentinel)).sum()
        print(f"  {col:16s}: {non_default:3d} / {len(result)} rows")


if __name__ == "__main__":
    main()
