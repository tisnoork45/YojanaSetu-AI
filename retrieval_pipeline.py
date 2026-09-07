
import pandas as pd

from eligibility_engine import check_eligibility, gap_summary, ELIGIBLE, NOT_ELIGIBLE, UNKNOWN

INDEX_PATH = "faiss_store/schemes_faiss.index"
DATA_PATH = "faiss_store/schemes_data.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 10
SIMILARITY_THRESHOLD = 0.30
MAX_RECOMMENDATIONS = 5

NO_MATCH_MESSAGE = (
    "I couldn't find a sufficiently relevant government scheme for your "
    "request. Please describe your requirement in a little more detail "
    "(for example, the kind of support you're looking for, your "
    "occupation, or your state)."
)


def load_resources():
    import faiss
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(INDEX_PATH)
    df = pd.read_pickle(DATA_PATH)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return index, df, embedding_model


def retrieve_candidates(query, embedding_model, index, df, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    query_vector = embedding_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vector, top_k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < threshold:
            continue
        row = df.iloc[idx].to_dict()
        row["_similarity"] = float(score)
        candidates.append(row)
    return candidates


def attach_eligibility(candidates, user_profile):
    for c in candidates:
        c["_eligibility"] = check_eligibility(user_profile, c)
    return candidates


def rank_candidates(candidates, max_results=MAX_RECOMMENDATIONS):
    priority = {ELIGIBLE: 0, UNKNOWN: 1, NOT_ELIGIBLE: 2}
    ranked = sorted(candidates, key=lambda c: (priority[c["_eligibility"].status], -c["_similarity"]))
    recommended = [c for c in ranked if c["_eligibility"].status != NOT_ELIGIBLE][:max_results]
    not_eligible = [c for c in ranked if c["_eligibility"].status == NOT_ELIGIBLE][:max_results]
    return recommended, not_eligible


def _format_scheme_block(candidate):
    row = candidate
    elig = candidate["_eligibility"]
    gaps = gap_summary(elig)
    gap_lines = "\n".join(f"  {sym} {dim}: {reason}" for dim, sym, reason in gaps) or "  (no structured eligibility data available)"
    return f"""
Scheme Name: {row.get('scheme_name')}
Category: {row.get('category')}
Description: {row.get('description')}
Ministry: {row.get('ministry', 'Not specified')}
Source URL: {row.get('source_url', 'Not specified')}
Similarity score: {row.get('_similarity'):.2f}
Deterministic eligibility status: {elig.status}
Eligibility checks:
{gap_lines}
"""


def build_gemini_prompt(user_message, user_profile, recommended, not_eligible):
    recommended_block = "\n---\n".join(_format_scheme_block(c) for c in recommended) or "(none)"
    not_eligible_block = "\n---\n".join(_format_scheme_block(c) for c in not_eligible) or "(none)"

    return f"""You are YojanaSetu AI, an assistant that helps people discover Indian
government welfare schemes they may be eligible for.

KNOWN USER PROFILE (may be partial):
{user_profile}

USER MESSAGE:
{user_message}

RECOMMENDED SCHEMES (relevant, and ELIGIBLE or UNKNOWN eligibility):
{recommended_block}

RELEVANT BUT NOT ELIGIBLE (only mention if it helps the user):
{not_eligible_block}

RULES (follow all of these):
1. Only use the scheme information given above. Never invent a scheme,
   benefit, eligibility condition, income limit, age limit, ministry, or
   source URL that isn't stated above.
2. For each recommended scheme, briefly state: why it's relevant, why the
   user appears eligible (cite the "Eligibility checks" lines above -- use
   check marks style like the eligibility checks shown), key benefits from
   the description, and the source/ministry if given.
3. If a scheme's deterministic eligibility status is UNKNOWN, say plainly
   that eligibility could not be fully determined and name which
   information (from the eligibility checks above) would resolve it. Do
   NOT say the user is eligible or not eligible for an UNKNOWN scheme.
4. For any "relevant but not eligible" scheme you choose to mention,
   explain briefly why (cite the failed check reason above) -- this is a
   gap explanation, not a rejection of the user.
5. Recommend at most 5 schemes. Do not list every candidate just because
   it was retrieved.
6. Keep the tone simple and friendly. If a scheme's eligibility wording is
   technical, you may add one short plain-language restatement of it
   without changing its meaning.
7. Do not claim to know about any scheme not listed above.
"""


def build_eli5_prompt(previous_answer, follow_up_message):
    return f"""A user previously received this explanation from you:

{previous_answer}

They now asked: "{follow_up_message}"

Rewrite the eligibility conditions from your previous answer in clear,
simple, everyday language (like explaining it to a friend). Do not change
what the conditions actually mean, do not add new conditions, and do not
invent numbers that weren't already stated.
"""
