# YojanaSetu AI

An AI-powered Government Scheme Recommendation System built using Retrieval-Augmented Generation (RAG).

## Features

- Semantic Search using SBERT
- Fast Retrieval using FAISS
- Response Generation using Gemini
- Interactive Gradio Chatbot

## Tech Stack

- Python
- Sentence Transformers
- FAISS
- Google Gemini API
- Gradio

## Project Workflow

User Query
↓
SBERT Embedding
↓
FAISS Retrieval
↓
Relevant Schemes
↓
Gemini Response Generation
↓
Final Recommendation

## Files

- app.py → Main chatbot application
- build_index.py → Creates FAISS vector database
- data_cleaning.py → Dataset preprocessing
- test_retrieval.py → Retrieval testing
- evaluate.py → Evaluation framework

## Example Query

"I am a farmer with 2 acres of land. Can I get financial help?"

## Author

Tisnoor Kaur