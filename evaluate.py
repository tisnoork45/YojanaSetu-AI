import json
import re
import time

import retrieval_pipeline as rp
from eligibility_engine import check_eligibility

from dotenv import load_dotenv
load_dotenv()

EVAL_DATASET_PATH = "data/eval_dataset.json"

def load_eval_dataset():
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# A. Retrieval Recall@10
# ---------------------------------------------------------------------------

def evaluate_retrieval(cases, embedding_model, index, df):
    hits, total = 0, 0
    misses = []
    for case in cases:
        if case["expected_behavior"] != "RETRIEVE" or not case["expected_scheme"]:
            continue
        total += 1
        query_vector = embedding_model.encode([case["query"]], normalize_embeddings=True).astype("float32")
        _, indices = index.search(query_vector, 10)
        retrieved_names = {df.iloc[i]["scheme_name"] for i in indices[0] if i != -1}
        if case["expected_scheme"] in retrieved_names:
            hits += 1
        else:
            misses.append(case["id"])
    recall_at_10 = hits / total if total else 0.0
    return recall_at_10, total, misses


# ---------------------------------------------------------------------------
# B. Eligibility Accuracy (no Gemini, no FAISS -- pure deterministic check)
# ---------------------------------------------------------------------------

def evaluate_eligibility(cases, df):
    by_name = {row["scheme_name"]: row.to_dict() for _, row in df.iterrows()}
    correct, total = 0, 0
    mismatches = []
    for case in cases:
        if case["expected_behavior"] != "RETRIEVE" or not case["expected_scheme"]:
            continue
        scheme = by_name.get(case["expected_scheme"])
        if scheme is None:
            continue
        total += 1
        result = check_eligibility(case["profile"], scheme)
        if result.status == case["expected_eligibility"]:
            correct += 1
        else:
            mismatches.append((case["id"], case["expected_eligibility"], result.status))
    accuracy = correct / total if total else 0.0
    return accuracy, total, mismatches


# ---------------------------------------------------------------------------
# B2. Conversational behavior: ASK_FOR_INFORMATION vs RETRIEVE
# ---------------------------------------------------------------------------

def evaluate_conversational_behavior(cases):
    from profile_extraction import missing_fields_for_intent

    correct, total = 0, 0
    mismatches = []
    for case in cases:
        if case["expected_behavior"] not in ("ASK_FOR_INFORMATION", "RETRIEVE"):
            continue
        if case["category"] not in ("missing_information", "ambiguous"):
            continue
        total += 1
        missing = missing_fields_for_intent(case["profile"])
        predicted = "ASK_FOR_INFORMATION" if missing else "RETRIEVE"
        if predicted == case["expected_behavior"]:
            correct += 1
        else:
            mismatches.append((case["id"], case["expected_behavior"], predicted))
    accuracy = correct / total if total else 0.0
    return accuracy, total, mismatches


# ---------------------------------------------------------------------------
# C. Answer Quality / Groundedness via Gemini-as-a-judge
# ---------------------------------------------------------------------------

def safe_generate_content(gemini_model, prompt, retries=3):
    for attempt in range(retries):
        try:
            return gemini_model.generate_content(prompt).text
        except Exception as e:
            if attempt == retries - 1:
                return f"Error: {e}"
            time.sleep(20)
    return "Failed"


JUDGE_PROMPT_TEMPLATE = """You are evaluating a government-scheme chatbot's answer.

USER QUERY:
{query}

SCHEME DATA GIVEN TO THE CHATBOT:
{context}

CHATBOT'S ANSWER:
{answer}

Rate the answer on four criteria, each 1-5:
- relevance: does it address the user's query?
- groundedness: does it stick to the scheme data given (no invented facts)?
- eligibility_correctness: does it represent eligibility the way the data supports (not overclaiming eligible/not eligible)?
- overall_quality: is it clear and useful?

Respond ONLY as JSON: {{"relevance": X, "groundedness": X, "eligibility_correctness": X, "overall_quality": X}}
"""


def evaluate_generation(cases, embedding_model, index, df, gemini_model, sample_size=15):
    """Runs the full pipeline for a sample of RETRIEVE cases and scores the
    final answer with Gemini-as-a-judge."""
    sampled = [
        c for c in cases
        if c["expected_behavior"] == "RETRIEVE" and c["expected_scheme"]
    ][:sample_size]

    scores = []

    for case in sampled:
        candidates = rp.retrieve_candidates(
            case["query"],
            embedding_model,
            index,
            df
        )

        if not candidates:
            print(f"  [warn] no candidates for {case['id']}")
            continue

        rp.attach_eligibility(candidates, case["profile"])
        recommended, not_eligible = rp.rank_candidates(candidates)

        prompt = rp.build_gemini_prompt(
            case["query"],
            case["profile"],
            recommended,
            not_eligible
        )

        answer = safe_generate_content(gemini_model, prompt)

        context = "\n".join(
            c["scheme_name"]
            for c in recommended + not_eligible
        )

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=case["query"],
            context=context,
            answer=answer
        )

        judge_output = safe_generate_content(
            gemini_model,
            judge_prompt
        ).strip()
        match = re.search(r"\{.*\}", judge_output, re.DOTALL)

        if not match:
            print(
                f"  [warn] could not find JSON for {case['id']}: "
                f"{judge_output[:200]}"
            )
            continue

        cleaned = match.group(0)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(
                f"  [warn] invalid JSON for {case['id']}: "
                f"{e}. Response: {judge_output[:200]}"
            )
            continue

        keys = [
            "relevance",
            "groundedness",
            "eligibility_correctness",
            "overall_quality"
        ]

        # Make sure all four scores exist and are valid 1-5 values.
        valid = True

        for key in keys:
            value = parsed.get(key)

            if not isinstance(value, (int, float)) or not 1 <= value <= 5:
                valid = False
                break

        if not valid:
            print(
                f"  [warn] invalid score format for {case['id']}: "
                f"{parsed}"
            )
            continue

        scores.append(parsed)

    if not scores:
        print("\n[ERROR] No valid Gemini judge scores were collected.")
        print("The Gemini API responded, but none of the judge responses")
        print("could be parsed into valid 1-5 scores.")
        return {}

    keys = [
        "relevance",
        "groundedness",
        "eligibility_correctness",
        "overall_quality"
    ]

    return {
        key: sum(score[key] for score in scores) / len(scores)
        for key in keys
    }

def main():
    cases = load_eval_dataset()
    print(f"Loaded {len(cases)} evaluation cases\n")

    print("=" * 60)
    print("B. ELIGIBILITY ACCURACY (deterministic, no Gemini/FAISS needed)")
    print("=" * 60)
    df_for_eligibility = __import__("pandas").read_csv("data/Final_Govt_Schemes_Dataset_Enriched.csv")
    elig_acc, elig_total, elig_mismatches = evaluate_eligibility(cases, df_for_eligibility)
    print(f"Eligibility Accuracy: {elig_acc:.2%} ({elig_total} cases)")
    if elig_mismatches:
        print("Mismatches:", elig_mismatches)

    print()
    print("=" * 60)
    print("B2. CONVERSATIONAL BEHAVIOR (ask-for-info vs retrieve)")
    print("=" * 60)
    conv_acc, conv_total, conv_mismatches = evaluate_conversational_behavior(cases)
    print(f"Conversational Behavior Accuracy: {conv_acc:.2%} ({conv_total} cases)")
    if conv_mismatches:
        print("Mismatches:", conv_mismatches)

    print()
    print("=" * 60)
    print("A. RETRIEVAL RECALL@10 and C. ANSWER QUALITY (need FAISS + Gemini)")
    print("=" * 60)
    try:
        index, df, embedding_model = rp.load_resources()
    except Exception as e:
        print(f"Could not load the FAISS store ({e}).")
        print("Run: python data_cleaning.py && python enrich_eligibility.py && python build_index.py")
        print("(and make sure requirements.txt is installed). Skipping retrieval and generation metrics.")
        return

    recall, recall_total, misses = evaluate_retrieval(cases, embedding_model, index, df)
    print(f"Retrieval Recall@10: {recall:.2%} ({recall_total} cases)")
    if misses:
        print("Missed cases:", misses)

    import os
    import google.generativeai as genai
    if not os.getenv("GOOGLE_API_KEY"):
        print("\nGOOGLE_API_KEY not set -- skipping Gemini-as-judge generation metric.")
        return
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    print("\nRunning Gemini-as-a-judge on a sample (this uses API quota)...")
    quality = evaluate_generation(cases, embedding_model, index, df, gemini_model)
    print("\nAverage Answer Quality / Groundedness (LLM-as-a-judge, 1-5 scale):")
    for k, v in quality.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
