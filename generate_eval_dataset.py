
import json
import pandas as pd

DATA_PATH = "data/Final_Govt_Schemes_Dataset_Enriched.csv"
OUT_PATH = "data/eval_dataset.json"

VALID_ELIGIBILITY = {"ELIGIBLE", "NOT_ELIGIBLE", "UNKNOWN"}
VALID_BEHAVIOR = {"RETRIEVE", "ASK_FOR_INFORMATION", "NO_MATCH"}

df = pd.read_csv(DATA_PATH)
known_scheme_names = set(df["scheme_name"])

cases = []
_id = 0


def add_case(category, query, profile=None, expected_scheme=None, expected_behavior="RETRIEVE",
             expected_eligibility=None, notes=""):
    global _id
    _id += 1
    profile = profile or {}
    case = {
        "id": f"case_{_id:03d}",
        "category": category,
        "query": query,
        "profile": profile,
        "expected_scheme": expected_scheme,
        "expected_behavior": expected_behavior,
        "expected_eligibility": expected_eligibility,
        "relevant_eligibility_conditions": notes if (expected_scheme and expected_behavior == "RETRIEVE") else "",
        "notes": notes,
    }
    cases.append(case)

# ---------------------------------------------------------------------------
# 1. General scheme queries (category-level, no profile)
# Every case here has an empty profile, so no user fact can ever satisfy a
# scheme's requirement even if one exists -- by the rules above that always
# resolves to UNKNOWN (either "no criteria in our data" or "criteria exist
# but the needed fact is missing"), regardless of which scheme it is.
# ---------------------------------------------------------------------------
general_queries = [
    ("What government schemes exist for agriculture?", "Pradhan Mantri Fasal Bima Yojana (PMFBY)"),
    ("Tell me about health schemes available in India.", "Ayushman Bharat (PM-JAY)"),
    ("What schemes are available for rural development?", "Pradhan Mantri Gram Sadak Yojana(PMGSY)"),
    ("Are there any schemes related to education?", "PM-SHRI Scheme"),
    ("What skill development schemes does the government offer?", "Pradhan Mantri Kaushal Vikas Yojana(PMKVY)"),
    ("Tell me about pension schemes for Indian citizens.", "Atal Pension Yojana (APY)"),
    ("What housing schemes are available?", "Pradhan Mantri Awas Yojana(PMAY"),
    ("What schemes exist for water and sanitation?", "Jal Jeevan Mission(JJM"),
]
for q, scheme in general_queries:
    add_case("general", q, expected_scheme=scheme, expected_eligibility="UNKNOWN",
              notes="Category-level browse query, no personalization -> empty profile means no scheme "
                    "requirement can ever be confirmed satisfied, so this is always UNKNOWN regardless "
                    "of the scheme's own criteria.")

# ---------------------------------------------------------------------------
# 2. Personalized queries (profile + open-ended ask, complete enough info)
# ---------------------------------------------------------------------------
add_case("personalized", "I'm a farmer looking for crop insurance.",
          {"occupation": "farmer", "state": "Maharashtra", "intent": "agriculture"},
          expected_scheme="Pradhan Mantri Fasal Bima Yojana (PMFBY)",
          expected_eligibility="ELIGIBLE",
          notes="PMFBY's only structured requirement is occupation=Farmer; profile occupation=farmer matches -> eligible.")
add_case("personalized", "I run a small shop and need a business loan.",
          {"occupation": "entrepreneur", "state": "Gujarat", "intent": "business"},
          expected_scheme="Pradhan Mantri Mudra Yojana (PMMY)",
          expected_eligibility="UNKNOWN",
          notes="PMMY has no structured eligibility criteria at all in our data -> UNKNOWN regardless of profile.")
add_case("personalized", "I'm a street vendor and need working capital.",
          {"occupation": "street vendor", "state": "Delhi", "intent": "business"},
          expected_scheme="PM SVANidhi",
          expected_eligibility="ELIGIBLE",
          notes="PM SVANidhi's only requirement is occupation=Street Vendor; profile matches -> eligible.")
add_case("personalized", "I want free skill training to get a job.",
          {"age": 24, "education": "school education", "intent": "employment_skill"},
          expected_scheme="Pradhan Mantri Kaushal Vikas Yojana(PMKVY)",
          expected_eligibility="UNKNOWN",
          notes="PMKVY has no structured eligibility criteria at all in our data -> UNKNOWN regardless of profile.")
add_case("personalized", "I'm part of a self-help group and want to increase our income.",
          {"occupation": "self-help group member", "state": "Bihar", "intent": "business"},
          expected_scheme="Lakhpati Didi Scheme",
          expected_eligibility="UNKNOWN",
          notes="Lakhpati Didi requires gender=Female, rural_urban=Rural, and occupation=Self-Help Group "
                "member. Profile only supplies occupation (matches) -- gender and rural_urban are unknown, "
                "so overall status is UNKNOWN (needs more info), not ELIGIBLE.")
add_case("personalized", "I'm an unorganised sector worker worried about my pension.",
          {"occupation": "unorganised sector worker", "age": 30, "intent": "pension"},
          expected_scheme="Atal Pension Yojana (APY)",
          expected_eligibility="ELIGIBLE",
          notes="APY's only requirement is occupation=Unorganised Sector Worker; profile matches -> eligible "
                "(this scheme has no age requirement in our data, so age is not a factor).")
add_case("personalized", "I'm a government employee planning for retirement.",
          {"occupation": "government employee", "intent": "pension"},
          expected_scheme="Unified Pension Scheme (UPS)",
          expected_eligibility="ELIGIBLE",
          notes="UPS's only requirement is occupation=Government Employee; profile matches -> eligible.")
add_case("personalized", "I'm an artisan and want support for my craft business.",
          {"occupation": "artisan", "intent": "business"},
          expected_scheme="PM Vishwakarma Scheme",
          expected_eligibility="ELIGIBLE",
          notes="PM Vishwakarma's only requirement is occupation=Artisan/Craftsperson; profile 'artisan' "
                "matches -> eligible.")

# ---------------------------------------------------------------------------
# 3. Age-specific queries
# ---------------------------------------------------------------------------
add_case("age_specific", "I'm 65 and a small farmer. Is there a pension scheme for me?",
          {"age": 65, "occupation": "farmer"},
          expected_scheme="Pradhan Mantri Kisan Maandhan Yojana (PM-KMY)",
          expected_eligibility="ELIGIBLE",
          notes="age 65 >= min_age 60 AND occupation=farmer matches Farmer -> both requirements satisfied -> eligible")
add_case("age_specific", "I'm 25 and work as a small farmer, can I get the farmer pension now?",
          {"age": 25, "occupation": "farmer"},
          expected_scheme="Pradhan Mantri Kisan Maandhan Yojana (PM-KMY)",
          expected_eligibility="NOT_ELIGIBLE",
          notes="age 25 < min_age 60 -> not eligible on age (occupation matching doesn't save it)")
add_case("age_specific", "My newborn needs the government child development program, what's available?",
          {"age": 2},
          expected_scheme="Integrated Child Development Services(ICDS",
          expected_eligibility="ELIGIBLE",
          notes="age 2 within the scheme's 0-6 age band, no other criteria -> eligible")
add_case("age_specific", "Is the anganwadi child development scheme relevant for a 10 year old?",
          {"age": 10},
          expected_scheme="Integrated Child Development Services(ICDS",
          expected_eligibility="NOT_ELIGIBLE",
          notes="age 10 > max_age 6 -> not eligible")
add_case("age_specific", "My 15 year old needs the child health screening program, can they use it?",
          {"age": 15},
          expected_scheme="Rashtriya Bal Swasthya Karyakram(RBSK)",
          expected_eligibility="ELIGIBLE",
          notes="age 15 within 0-18, no other criteria -> eligible")
add_case("age_specific", "Can a 20 year old still use the child health screening program?",
          {"age": 20},
          expected_scheme="Rashtriya Bal Swasthya Karyakram(RBSK)",
          expected_eligibility="NOT_ELIGIBLE",
          notes="age 20 > max_age 18 -> not eligible")
add_case("age_specific", "My child is 10, can they open an NPS Vatsalya pension account?",
          {"age": 10},
          expected_scheme="NPS Vatsalya",
          expected_eligibility="ELIGIBLE",
          notes="age 10 <= max_age 18, no other criteria -> eligible")
add_case("age_specific", "I'm 40, can I open an NPS Vatsalya account meant for my kid?",
          {"age": 40},
          expected_scheme="NPS Vatsalya",
          expected_eligibility="NOT_ELIGIBLE",
          notes="age 40 > max_age 18 -> not eligible")

# ---------------------------------------------------------------------------
# 4. Gender-specific queries
# Each scheme's expected label under a gender-only profile depends on
# whether gender is its ONLY requirement (-> ELIGIBLE when it matches) or
# one of several requirements (-> UNKNOWN when it matches, since the other
# requirements are still unresolved; a mismatch is NOT_ELIGIBLE either way,
# since a single failed dimension is enough regardless of the rest).
# ---------------------------------------------------------------------------
gender_cases = [
    # scheme_name, query, female_label, male_label, female_note, male_note
    ("NAMO DRONE DIDI SCHEME",
     "I'm part of a women's self-help group interested in the drone scheme.",
     "UNKNOWN", "NOT_ELIGIBLE",
     "gender=Female matches, but occupation=Self-Help Group member is also required and not provided -> UNKNOWN",
     "gender=male mismatches the Female requirement -> not eligible"),
    ("Janani Suraksha Yojana(JSY",
     "I'm a pregnant woman looking for institutional delivery support.",
     "ELIGIBLE", "NOT_ELIGIBLE",
     "gender=Female is the scheme's only requirement and matches -> eligible",
     "gender=male mismatches the Female-only requirement -> not eligible"),
    ("SUMAN (Surakshit Matritva Aashwasan)",
     "I'm expecting and want free maternal healthcare at a government hospital.",
     "ELIGIBLE", "NOT_ELIGIBLE",
     "gender=Female is the scheme's only requirement and matches -> eligible",
     "gender=male mismatches the Female-only requirement -> not eligible"),
    ("Mission Shakti (Mission for Empowerment and Protection of Wo",
     "Is there a scheme focused on women's safety and empowerment?",
     "ELIGIBLE", "NOT_ELIGIBLE",
     "gender=Female is the scheme's only requirement and matches -> eligible",
     "gender=male mismatches the Female-only requirement -> not eligible"),
    ("Beti Bachao Beti Padhao(BBBP",
     "Is there a scheme to support the girl child's welfare?",
     "ELIGIBLE", "NOT_ELIGIBLE",
     "gender=Female is the scheme's only requirement and matches -> eligible",
     "gender=male mismatches the Female-only requirement -> not eligible"),
    ("Sukanya Samriddhi Yojana(SSY",
     "I want to open a small savings account for my daughter.",
     "ELIGIBLE", "NOT_ELIGIBLE",
     "gender=Female is the scheme's only requirement and matches -> eligible",
     "gender=male mismatches the Female-only requirement -> not eligible"),
    ("Lakhpati Didi Scheme",
     "I'm a rural woman in a self-help group wanting to grow our income.",
     "UNKNOWN", "NOT_ELIGIBLE",
     "gender=Female matches, but rural_urban=Rural and occupation=Self-Help Group member are also "
     "required and not provided -> UNKNOWN",
     "gender=male mismatches the Female requirement -> not eligible"),
    ("Kasturba Gandhi Balika Vidyalaya",
     "Is there a residential school scheme for girls from poor families?",
     "UNKNOWN", "NOT_ELIGIBLE",
     "gender=Female matches, but social_category (SC/ST/OBC/Minority/BPL/EWS) and education=School "
     "Education are also required and not provided -> UNKNOWN",
     "gender=male mismatches the Female requirement -> not eligible"),
]
for scheme, query, female_label, male_label, female_note, male_note in gender_cases:
    add_case("gender_specific", query, {"gender": "female"}, expected_scheme=scheme,
              expected_eligibility=female_label, notes=female_note)
    male_query = query.replace("I'm", "My friend is").replace("I want", "My friend wants")
    add_case("gender_specific", male_query, {"gender": "male"}, expected_scheme=scheme,
              expected_eligibility=male_label, notes=male_note)

# ---------------------------------------------------------------------------
# 5. Income-specific queries
# No scheme in this dataset has a *numeric* income eligibility limit (see
# enrich_eligibility.py docstring) -- the only income-related structured
# signal we could reliably extract is the BPL/EWS social-category flag.
# These cases test that: a self-identified BPL/EWS user against BPL/EWS
# -flagged schemes, and confirm income alone is correctly treated as
# NOT_APPLICABLE when a scheme has no stated income limit.
# ---------------------------------------------------------------------------
add_case("income_specific", "My family income is very low, around 80,000 a year. Is there an LPG subsidy for us?",
          {"income": 80000, "social_category": "bpl"},
          expected_scheme="Pradhan Mantri Ujjwala Yojana(PMUY)",
          expected_eligibility="ELIGIBLE",
          notes="social_category 'bpl' is in the scheme's allowed set BPL/EWS -> eligible; "
                "income itself is NOT_APPLICABLE (no numeric limit in data)")
add_case("income_specific", "We're a general-category family with a comfortable income. Can we get the LPG subsidy scheme meant for poor households?",
          {"income": 1200000, "social_category": "general"},
          expected_scheme="Pradhan Mantri Ujjwala Yojana(PMUY)",
          expected_eligibility="NOT_ELIGIBLE",
          notes="social_category mismatch (General vs BPL/EWS) -> not eligible")
add_case("income_specific", "My rural family earns about 1.5 lakh a year, are we eligible for the rural skills program targeted at poor households?",
          {"income": 150000, "social_category": "bpl", "rural_urban": "rural"},
          expected_scheme="Deendayal Upadhyaya Grameen Kaushalya Yojana(DDU)",
          expected_eligibility="ELIGIBLE",
          notes="social_category 'bpl' matches BPL/EWS (part of scheme's SC/ST/BPL/EWS set) and "
                "rural_urban 'rural' matches Rural -> eligible; income NOT_APPLICABLE")
add_case("income_specific", "I have a high income and want to know if the girls' residential school scheme for poor families applies to us.",
          {"income": 2000000, "social_category": "general"},
          expected_scheme="Kasturba Gandhi Balika Vidyalaya",
          expected_eligibility="NOT_ELIGIBLE",
          notes="social_category mismatch (General not in SC/ST/OBC/Minority/BPL/EWS) -> not eligible")
add_case("income_specific", "My family's annual income is about 3 lakh, can I get support under PM-KISAN?",
          {"income": 300000, "occupation": "farmer"},
          expected_scheme="PM KISAN SAMMAN NIDHI (PM-KISAN)",
          expected_eligibility="ELIGIBLE",
          notes="PM-KISAN's only requirement is occupation=Farmer; profile matches -> eligible; "
                "income NOT_APPLICABLE (no numeric limit in data)")

# ---------------------------------------------------------------------------
# 6. State-specific queries
# ---------------------------------------------------------------------------
add_case("state_specific", "I'm a young person from the North East looking for livelihood support.",
          {"state": "North Eastern States", "intent": "employment_skill"},
          expected_scheme="PM-DEVINE",
          expected_eligibility="ELIGIBLE",
          notes="state matches the scheme's only requirement, North Eastern States -> eligible")
add_case("state_specific", "I live in Kerala, is the North-East livelihood scheme relevant to me?",
          {"state": "Kerala"},
          expected_scheme="PM-DEVINE",
          expected_eligibility="NOT_ELIGIBLE",
          notes="state 'Kerala' mismatches the required North Eastern States -> not eligible")
add_case("state_specific", "I'm from Assam and want to know about industrialisation support in the North East.",
          {"state": "North Eastern States"},
          expected_scheme="UTTAR POORVA TRANSFORMATIVE INDUSTRIALIZATION \nSCHEME (UNNATI)",
          expected_eligibility="ELIGIBLE",
          notes="state matches the scheme's only requirement, North Eastern States -> eligible")
add_case("state_specific", "Does the PM-KISAN scheme have any specific state restriction?",
          {"state": "Tamil Nadu", "occupation": "farmer"},
          expected_scheme="PM KISAN SAMMAN NIDHI (PM-KISAN)",
          expected_eligibility="ELIGIBLE",
          notes="PM-KISAN has no state restriction (state=All, NOT_APPLICABLE), but its occupation=Farmer "
                "requirement IS present and the profile's occupation=farmer matches it -> eligible")

# ---------------------------------------------------------------------------
# 7. Social-category queries
# ---------------------------------------------------------------------------
add_case("social_category", "I belong to the SC community and my family is below the poverty line, is there a rural skills scheme for us?",
          {"social_category": "sc", "rural_urban": "rural"},
          expected_scheme="Deendayal Upadhyaya Grameen Kaushalya Yojana(DDU)",
          expected_eligibility="ELIGIBLE",
          notes="social_category 'sc' is in the allowed set SC/ST/BPL/EWS and rural_urban matches Rural -> eligible")
add_case("social_category", "I'm from a general category, well-off family. Is the rural skills scheme for BPL/SC/ST youth open to us too?",
          {"social_category": "general"},
          expected_scheme="Deendayal Upadhyaya Grameen Kaushalya Yojana(DDU)",
          expected_eligibility="NOT_ELIGIBLE",
          notes="General is not in the allowed set SC/ST/BPL/EWS -> not eligible")
add_case("social_category", "My daughter is from an OBC family below the poverty line, can she join the girls' residential school scheme?",
          {"social_category": "obc", "gender": "female", "education": "school education"},
          expected_scheme="Kasturba Gandhi Balika Vidyalaya",
          expected_eligibility="ELIGIBLE",
          notes="social_category 'obc' is in the allowed set, gender=female matches Female, and "
                "education matches School Education -> all three requirements satisfied -> eligible")
add_case("social_category", "We're a minority community family, can we get the LPG connection scheme meant for BPL households?",
          {"social_category": "minority"},
          expected_scheme="Pradhan Mantri Ujjwala Yojana(PMUY)",
          expected_eligibility="NOT_ELIGIBLE",
          notes="scheme's allowed set is BPL/EWS only -> Minority alone does not match -> not eligible")

# ---------------------------------------------------------------------------
# 8. Education / occupation queries
# ---------------------------------------------------------------------------
add_case("education_occupation", "I've finished 12th grade and want a loan for a top university abroad.",
          {"education": "higher education", "occupation": "student"},
          expected_scheme="PM Vidyalaxmi Scheme",
          expected_eligibility="ELIGIBLE",
          notes="PM Vidyalaxmi requires occupation=Student AND education=Higher Education; profile "
                "matches both -> eligible")
add_case("education_occupation", "I'm a working professional, not a student, can I still get the education loan scheme meant for students?",
          {"occupation": "government employee"},
          expected_scheme="PM Vidyalaxmi Scheme",
          expected_eligibility="NOT_ELIGIBLE",
          notes="occupation mismatch (government employee vs required Student) -> not eligible")
add_case("education_occupation", "I'm in class 10 and interested in the leadership program for classes 9-12.",
          {"age": 15},
          expected_scheme="Prerana",
          expected_eligibility="UNKNOWN",
          notes="Prerana requires occupation=Student, which the profile doesn't provide (only age is "
                "given, and this scheme has no age requirement) -> UNKNOWN")
add_case("education_occupation", "I'm a farmer wanting to check my soil quality through a government scheme.",
          {"occupation": "farmer"},
          expected_scheme="Soil Health Card Scheme",
          expected_eligibility="ELIGIBLE",
          notes="occupation matches the scheme's only requirement, Farmer -> eligible")
add_case("education_occupation", "I'm a student, not a farmer -- is the soil health card scheme still useful for me directly?",
          {"occupation": "student"},
          expected_scheme="Soil Health Card Scheme",
          expected_eligibility="NOT_ELIGIBLE",
          notes="occupation mismatch (student vs required Farmer) -> not eligible")

# ---------------------------------------------------------------------------
# 9. Clearly eligible cases (extra, beyond the ones embedded above)
# ---------------------------------------------------------------------------
add_case("eligible", "I'm an artisan making handicrafts and want government recognition and support.",
          {"occupation": "artisan"}, expected_scheme="PM Vishwakarma Scheme",
          expected_eligibility="ELIGIBLE",
          notes="occupation matches the scheme's only requirement, Artisan/Craftsperson -> eligible")
add_case("eligible", "I'm a woman running a small self-help group business, want to grow to a lakh a year.",
          {"occupation": "self-help group member", "gender": "female"}, expected_scheme="NAMO DRONE DIDI SCHEME",
          expected_eligibility="ELIGIBLE",
          notes="both of the scheme's requirements are satisfied: occupation matches Self-Help Group "
                "member, gender matches Female -> eligible")
add_case("eligible", "I'm 62 years old and a small farmer, can I join the pension scheme?",
          {"age": 62, "occupation": "farmer"}, expected_scheme="Pradhan Mantri Kisan Maandhan Yojana (PM-KMY)",
          expected_eligibility="ELIGIBLE",
          notes="both age (62 >= 60) and occupation (farmer) match -> eligible")

# ---------------------------------------------------------------------------
# 10. Clearly ineligible cases (extra, beyond the ones embedded above)
# ---------------------------------------------------------------------------
add_case("ineligible", "I'm a 19 year old wanting to join the small farmer pension scheme.",
          {"age": 19, "occupation": "farmer"}, expected_scheme="Pradhan Mantri Kisan Maandhan Yojana (PM-KMY)",
          expected_eligibility="NOT_ELIGIBLE",
          notes="age fails (19 < 60) -> not eligible even though occupation matches")
add_case("ineligible", "I'm a 30 year old man, can I benefit from the girl child savings scheme directly?",
          {"age": 30, "gender": "male"}, expected_scheme="Sukanya Samriddhi Yojana(SSY",
          expected_eligibility="NOT_ELIGIBLE",
          notes="gender fails (male vs required Female) -> not eligible (this scheme has no age "
                "requirement in our data, so age plays no role)")
add_case("ineligible", "My child is 8 years old, can they still join the 0-6 age group nutrition scheme?",
          {"age": 8}, expected_scheme="Integrated Child Development Services(ICDS",
          expected_eligibility="NOT_ELIGIBLE",
          notes="age fails (8 > max_age 6) -> not eligible")

# ---------------------------------------------------------------------------
# 11. Missing-information cases (ASK_FOR_INFORMATION)
# expected_behavior is authored directly based on what each case is
# designed to test -- never computed via missing_fields_for_intent().
# ---------------------------------------------------------------------------
missing_info_queries = [
    "I need a scholarship.",
    "I want health insurance for my family.",
    "I'm looking for a pension scheme.",
    "I need a loan to start a business.",
    "Can you help me find a housing scheme?",
    "I want to know about food security schemes for my family.",
    "Is there a scheme to support women's self-help groups?",
    "I need financial help, what schemes can I get?",
]
for q in missing_info_queries:
    add_case("missing_information", q, {}, expected_behavior="ASK_FOR_INFORMATION",
              notes="Intent-relevant profile fields are all missing -> system should ask a targeted follow-up, not assume values.")

# Partial information: some fields known, some still missing
add_case("missing_information", "I'm 22 and from Punjab, and I need a scholarship.",
          {"age": 22, "state": "Punjab", "intent": "education"}, expected_behavior="ASK_FOR_INFORMATION",
          notes="age/state known but income & education still missing for the 'education' intent -> should ask only for those, not re-ask age/state.")
add_case("missing_information", "I'm a farmer from Maharashtra needing crop insurance.",
          {"occupation": "farmer", "state": "Maharashtra", "intent": "agriculture"}, expected_behavior="RETRIEVE",
          expected_scheme="Pradhan Mantri Fasal Bima Yojana (PMFBY)",
          expected_eligibility="ELIGIBLE",
          notes="occupation & state (the only two fields 'agriculture' intent needs) are both known -> "
                "should proceed to retrieval, not ask again; occupation=farmer also matches PMFBY's only "
                "requirement (Farmer) -> eligible")

# ---------------------------------------------------------------------------
# 12. Ambiguous queries
# ---------------------------------------------------------------------------
ambiguous_queries = [
    "I need help.",
    "What can you do for me?",
    "Tell me something useful.",
    "I'm not sure what I'm looking for.",
]
for q in ambiguous_queries:
    add_case("ambiguous", q, {}, expected_behavior="ASK_FOR_INFORMATION",
              notes="Too vague to identify a specific intent -> falls back to 'general' intent, which still asks for at least a state/area of interest.")

# ---------------------------------------------------------------------------
# 13. Low-relevance / no-match queries
# ---------------------------------------------------------------------------
no_match_queries = [
    "Tell me a joke.",
    "What's the weather like today?",
    "Who won the cricket match yesterday?",
    "Can you write a poem about the ocean?",
    "What's the capital of France?",
    "Recommend me a movie to watch tonight.",
]
for q in no_match_queries:
    add_case("no_match", q, {"state": "Delhi", "age": 30, "occupation": "student", "income": 500000, "education": "higher education"},
              expected_behavior="NO_MATCH",
              notes="Profile fully specified (so it won't trip ASK_FOR_INFORMATION) and query is unrelated to any government scheme -> every FAISS candidate should fall below the similarity threshold.")

# ---------------------------------------------------------------------------
# Extra cases (additional anchor schemes for more coverage / diversity)
# ---------------------------------------------------------------------------

# More general/category-browse queries (empty profile -> always UNKNOWN)
extra_general = [
    ("What schemes support urban infrastructure development?", "Smart Cities Mission"),
    ("Are there any schemes for road connectivity in villages?", "Rashtriya Gram Swaraj Abhiyan(RGSA )"),
    ("What food security schemes exist for poor households?", "Antyodaya Anna Yojana(AAY)"),
    ("Tell me about digital connectivity schemes.", "PM-WANI"),
]
for q, scheme in extra_general:
    add_case("general", q, expected_scheme=scheme, expected_eligibility="UNKNOWN",
              notes="Category-level browse query, no personalization -> empty profile means UNKNOWN regardless of the scheme's own criteria.")

# More rural/urban eligible + ineligible pairs
add_case("rural_urban", "I live in a city and want to know about the urban transformation scheme for better services.",
          {"rural_urban": "urban"}, expected_scheme="Atal Mission for Rejuvenation and Urban Transformation(AMRUT)",
          expected_eligibility="ELIGIBLE",
          notes="rural_urban matches the scheme's only requirement, Urban -> eligible")
add_case("rural_urban", "I live in a village, is the urban transformation scheme relevant to me?",
          {"rural_urban": "rural"}, expected_scheme="Atal Mission for Rejuvenation and Urban Transformation(AMRUT)",
          expected_eligibility="NOT_ELIGIBLE",
          notes="rural_urban mismatch (rural vs required Urban) -> not eligible")
add_case("rural_urban", "I'm from a rural household and want guaranteed wage employment support.",
          {"rural_urban": "rural"}, expected_scheme="Mahatma Gandhi National Rural Employment Guarantee Act(MGNREGA",
          expected_eligibility="ELIGIBLE",
          notes="rural_urban matches the scheme's only requirement, Rural -> eligible")
add_case("rural_urban", "I live in a city, can I get the rural employment guarantee scheme meant for villages?",
          {"rural_urban": "urban"}, expected_scheme="Mahatma Gandhi National Rural Employment Guarantee Act(MGNREGA",
          expected_eligibility="NOT_ELIGIBLE",
          notes="rural_urban mismatch (urban vs required Rural) -> not eligible")

# More occupation eligible + ineligible pairs
add_case("education_occupation", "I'm a government employee and want to know about the National Pension System.",
          {"occupation": "government employee"}, expected_scheme="National Pension System 2004",
          expected_eligibility="ELIGIBLE",
          notes="occupation matches the scheme's only requirement, Government Employee -> eligible")
add_case("education_occupation", "I'm self-employed, does the government-employee pension system apply to me?",
          {"occupation": "self-employed"}, expected_scheme="National Pension System 2004",
          expected_eligibility="NOT_ELIGIBLE",
          notes="occupation mismatch (self-employed vs required Government Employee) -> not eligible")
add_case("education_occupation", "I work in the unorganised sector and want a low-cost pension plan.",
          {"occupation": "unorganised sector worker"}, expected_scheme="Atal Pension Yojana(APY)",
          expected_eligibility="ELIGIBLE",
          notes="occupation matches the scheme's only requirement, Unorganised Sector Worker -> eligible")
add_case("education_occupation", "I'm a salaried government employee, is the unorganised-sector pension scheme meant for me?",
          {"occupation": "government employee"}, expected_scheme="Atal Pension Yojana(APY)",
          expected_eligibility="NOT_ELIGIBLE",
          notes="occupation mismatch (government employee vs required Unorganised Sector Worker) -> not eligible")

# More personalized queries
add_case("personalized", "I stay in a village and need help with clean drinking water access.",
          {"rural_urban": "rural", "intent": "food_security"}, expected_scheme="Jal Jeevan Mission(JJM",
          expected_eligibility="ELIGIBLE",
          notes="rural_urban matches the scheme's only requirement, Rural -> eligible")
add_case("personalized", "I'm from a well-off urban family, is the rural piped water mission for us too?",
          {"rural_urban": "urban"}, expected_scheme="Jal Jeevan Mission(JJM",
          expected_eligibility="NOT_ELIGIBLE",
          notes="rural_urban mismatch (urban vs required Rural) -> not eligible")

# More clearly-eligible / clearly-ineligible cases
add_case("eligible", "I'm a government employee with 25+ years of service planning retirement.",
          {"occupation": "government employee"}, expected_scheme="Unified Pension Scheme (UPS)",
          expected_eligibility="ELIGIBLE",
          notes="occupation matches the scheme's only requirement, Government Employee -> eligible")
add_case("eligible", "I'm 12 years old and my parents want to open a pension account in my name.",
          {"age": 12}, expected_scheme="NPS Vatsalya",
          expected_eligibility="ELIGIBLE",
          notes="age 12 <= max_age 18, no other criteria -> eligible")
add_case("ineligible", "I'm a farmer, can I access the government-employee-only pension system?",
          {"occupation": "farmer"}, expected_scheme="National Pension System 2004",
          expected_eligibility="NOT_ELIGIBLE",
          notes="occupation mismatch (farmer vs required Government Employee) -> not eligible")
add_case("ineligible", "I'm 45 years old, can I still open a minor's NPS Vatsalya account for myself?",
          {"age": 45}, expected_scheme="NPS Vatsalya",
          expected_eligibility="NOT_ELIGIBLE",
          notes="age 45 exceeds max_age 18 -> not eligible")

# More ambiguous queries
for q in ["Can you help?", "I don't know where to start."]:
    add_case("ambiguous", q, {}, expected_behavior="ASK_FOR_INFORMATION",
              notes="Too vague to identify a specific intent -> falls back to 'general' intent.")

# More low-relevance / no-match queries
for q in ["Write me a short story about a dragon.", "What's the square root of 144?"]:
    add_case("no_match", q, {"state": "Delhi", "age": 30, "occupation": "student", "income": 500000, "education": "higher education"},
              expected_behavior="NO_MATCH",
              notes="Unrelated to any government scheme -> every FAISS candidate should fall below the similarity threshold.")

print(f"Generated {len(cases)} test cases")
by_category = {}
for c in cases:
    by_category.setdefault(c["category"], 0)
    by_category[c["category"]] += 1
for cat, n in by_category.items():
    print(f"  {cat:22s}: {n}")

for c in cases:
    assert c["expected_behavior"] in VALID_BEHAVIOR, \
        f"Invalid expected_behavior for {c['id']}: {c['expected_behavior']!r}"
    if c["expected_behavior"] == "RETRIEVE" and c["expected_scheme"]:
        assert c["expected_eligibility"] in VALID_ELIGIBILITY, \
            f"Invalid/missing eligibility label for {c['id']}: {c['expected_eligibility']!r}"
        assert c["expected_scheme"] in known_scheme_names, \
            f"expected_scheme for {c['id']} is not a real scheme name: {c['expected_scheme']!r}"
print(f"\nValidation passed: every RETRIEVE case has an explicit label in {VALID_ELIGIBILITY}, "
      f"every case has expected_behavior in {VALID_BEHAVIOR}, every expected_scheme is a real scheme name.")

with open(OUT_PATH, "w") as f:
    json.dump(cases, f, indent=2)
print(f"\nSaved -> {OUT_PATH}")
