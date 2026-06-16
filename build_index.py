import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/Final_Govt_Schemes_Dataset.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

texts = (
    df["scheme_name"].astype(str)
    + " "
    + df["category"].astype(str)
    + " "
    + df["description"].astype(str)
).tolist()

print(f"Loaded {len(texts)} schemes")

print("Loading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = embeddings.astype("float32")

dimension = embeddings.shape[1]

print("Creating FAISS index...")
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

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
    pickle.dump(texts, f)

print("FAISS index created successfully!")