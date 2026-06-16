import os
import re
import time
import faiss
import pandas as pd
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
from google.api_core import exceptions

# ==========================
# GEMINI CONFIG
# ==========================

API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=API_KEY)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ==========================
# LOAD DATABASE
# ==========================

index = faiss.read_index(
    "faiss_store/schemes_faiss.index"
)

df = pd.read_pickle(
    "faiss_store/schemes_data.pkl"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================
# SEARCH FUNCTION
# ==========================

def search_schemes(user_query, top_k=3):

    query_vector = model.encode(
        [user_query],
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for idx in indices[0]:

        row = df.iloc[idx]

        results.append(
            f"""
Scheme Name: {row['scheme_name']}
Category: {row['category']}
Description: {row['description']}
"""
        )

    return results


# ==========================
# SAFE GEMINI CALL
# ==========================

def safe_generate_content(prompt, retries=3):

    for attempt in range(retries):

        try:
            response = gemini_model.generate_content(
                prompt
            )

            return response.text

        except exceptions.TooManyRequests:

            print(
                "Quota hit. Waiting..."
            )

            time.sleep(60)

        except Exception as e:

            return f"Error: {e}"

    return "Failed"


# ==========================
# TEST DATASET
# ==========================

test_dataset = [

    {
        "category": "Housing",
        "question": "I do not have a pukka house and I want financial help to build one.",
        "expected_fact": "subsidy"
    },

    {
        "category": "Education",
        "question": "I am a meritorious student from a poor family looking for a scholarship.",
        "expected_fact": "financial assistance"
    },

    {
        "category": "Pension",
        "question": "I am an unorganized worker concerned about income after old age.",
        "expected_fact": "pension"
    },

    {
        "category": "Business",
        "question": "I need a small loan of 50000 rupees to start a shop.",
        "expected_fact": "loan"
    },

    {
        "category": "Health",
        "question": "My family needs insurance for hospitalization expenses up to 5 lakh.",
        "expected_fact": "5 lakh"
    }

]


# ==========================
# RUN EVALUATION
# ==========================

results = []

print("\nRunning Evaluation...\n")

for test in test_dataset:

    question = test["question"]
    expected = test["expected_fact"]

    schemes = search_schemes(
        question,
        top_k=3
    )

    context = "\n".join(schemes)

    bot_prompt = f"""
You are a Government Scheme Consultant.

SCHEME DATA:
{context}

USER QUESTION:
{question}

Answer using the provided data only.
"""

    bot_answer = safe_generate_content(
        bot_prompt
    )

    judge_prompt = f"""
You are an evaluator.

QUESTION:
{question}

EXPECTED FACT:
{expected}

BOT RESPONSE:
{bot_answer}

Give score from 1-5.

Format:

Score: X
Reason: ...
"""

    judge_output = safe_generate_content(
        judge_prompt
    )

    match = re.search(
        r"Score:\s*([1-5])",
        judge_output
    )

    score = (
        int(match.group(1))
        if match
        else 0
    )

    results.append(score)

    print("=" * 40)
    print(question)
    print(f"Score: {score}")
    print(judge_output)

# ==========================
# FINAL SCORE
# ==========================

if results:

    avg = sum(results) / len(results)

    print("\n")
    print("=" * 50)
    print(f"Average Score: {avg:.2f}/5")
    print("=" * 50)