import os

import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

import retrieval_pipeline as rp
from profile_extraction import (
    extract_profile_gemini,
    extract_profile_fallback,
    missing_fields_for_intent,
    build_followup_question,
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

try:
    index, df, embedding_model = rp.load_resources()
    print(f"Database Loaded Successfully ({len(df)} schemes)")
except Exception as e:
    raise SystemExit(
        "Could not load the FAISS store. Before running app.py, build it "
        "with:\n"
        "    python data_cleaning.py\n"
        "    python enrich_eligibility.py\n"
        "    python build_index.py\n"
        f"(original error: {e})"
    )

INTENT_LABELS = {
    "education": "education/scholarship schemes",
    "health": "health schemes",
    "agriculture": "agriculture schemes",
    "housing": "housing schemes",
    "pension": "pension schemes",
    "business": "business/loan schemes",
    "employment_skill": "employment and skill-training schemes",
    "women_empowerment": "women-empowerment schemes",
    "food_security": "food-security schemes",
    "general": "government schemes",
}

ELI5_PATTERN_WORDS = ("explain", "simply", "simple terms", "simple language", "eli5", "in short")

def history_to_text(history):
    lines = []
    for item in history or []:
        if isinstance(item, dict):
            lines.append(str(item.get("content", "")))
        elif isinstance(item, (list, tuple)):
            lines.extend(str(part) for part in item if part)
    return "\n".join(lines)


def last_assistant_message(history):
    for item in reversed(history or []):
        if isinstance(item, dict) and item.get("role") == "assistant":
            return item.get("content", "")
        if isinstance(item, (list, tuple)) and len(item) >= 2 and item[1]:
            return item[1]
    return None


def build_profile(history, message):
    conversation_text = (history_to_text(history) + "\n" + message).strip()
    profile = {}
    if GOOGLE_API_KEY:
        profile = extract_profile_gemini(conversation_text, profile, gemini_model)
    else:
        profile = extract_profile_fallback(conversation_text, profile)
    return profile

def scheme_chat_logic(message, history):
    try:
        if any(w in message.lower() for w in ELI5_PATTERN_WORDS):
            previous = last_assistant_message(history)
            if previous:
                prompt = rp.build_eli5_prompt(previous, message)
                return gemini_model.generate_content(prompt).text

        profile = build_profile(history, message)

        missing = missing_fields_for_intent(profile)
        if missing:
            label = INTENT_LABELS.get(profile.get("intent", "general"), "government schemes")
            return f"I can help you find suitable {label}. " + build_followup_question(missing)

        candidates = rp.retrieve_candidates(message, embedding_model, index, df)
        if not candidates:
            return rp.NO_MATCH_MESSAGE

        rp.attach_eligibility(candidates, profile)
        recommended, not_eligible = rp.rank_candidates(candidates)
        prompt = rp.build_gemini_prompt(message, profile, recommended, not_eligible)
        response = gemini_model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error: {str(e)}"

SUBTITLE = "Eligibility-Aware Government Scheme Recommendation Assistant"

CUSTOM_CSS = """
.gradio-container {
    max-width: 880px !important;
    margin: 0 auto !important;
}
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

demo = gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="YojanaSetu AI")
with demo:
    gr.ChatInterface(
        fn=scheme_chat_logic,
        title="YojanaSetu AI",
        description=(
            f"**{SUBTITLE}**\n\n"
            "Describe your situation in plain language -- age, state, income, "
            "occupation -- and what kind of support you're looking for. If "
            "something important is still missing, the assistant will ask "
            "before recommending schemes."
        ),
        textbox=gr.Textbox(
            placeholder=(
                "e.g. \"I'm a 22-year-old student from Punjab looking for "
                "financial assistance for education.\""
            )
        ),
        examples=[
            "I'm a 22-year-old student from Punjab looking for financial assistance for education.",
            "I am a farmer with 2 acres of land. Can I get financial help?",
            "I want to start a small business.",
            "Are there scholarships for students?",
            "My family needs health insurance.",
        ],
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
