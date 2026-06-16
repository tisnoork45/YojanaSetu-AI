import os
import faiss
import pickle
import pandas as pd
import gradio as gr
import google.generativeai as genai

from sentence_transformers import SentenceTransformer

# ==========================
# GEMINI CONFIG
# ==========================


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# ==========================
# LOAD FAISS DATABASE
# ==========================

index = faiss.read_index(
    "faiss_store/schemes_faiss.index"
)

df = pd.read_pickle(
    "faiss_store/schemes_data.pkl"
)

with open(
    "faiss_store/scheme_texts.pkl",
    "rb"
) as f:
    scheme_texts = pickle.load(f)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Database Loaded Successfully")

# ==========================
# RETRIEVAL FUNCTION
# ==========================

def search_schemes(user_query, top_k=3):

    query_vector = embedding_model.encode(
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
# CHATBOT LOGIC
# ==========================

def scheme_chat_logic(message, history):

    try:

        schemes = search_schemes(
            message,
            top_k=3
        )

        context = "\n\n".join(schemes)

        prompt = f"""
You are YojanaSetu AI.

Use ONLY the provided scheme information.

SCHEME DATA:
{context}

USER QUESTION:
{message}

RULES:
- Do not invent schemes.
- Mention exact scheme names.
- Mention benefits if available.
- Keep answers simple and friendly.
"""

        response = gemini_model.generate_content(
            prompt
        )

        return response.text

    except Exception as e:

        return f"Error: {str(e)}"


# ==========================
# GRADIO UI
# ==========================

demo = gr.ChatInterface(
    fn=scheme_chat_logic,
    title="YojanaSetu AI",
    description="Government Scheme Recommendation Assistant",
    examples=[
        "I am a farmer with 2 acres of land. Can I get financial help?",
        "I want to start a small business.",
        "Are there scholarships for students?",
        "My family needs health insurance."
    ]
)

demo.launch(
    share=True,
    inbrowser=True
)