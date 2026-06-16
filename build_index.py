import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = r"C:\Users\dell\Documents\Final_Govt_Schemes_Dataset.csv"

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

faiss.write_index(
    index,
    r"C:\Users\dell\Documents\schemes_faiss.index"
)

df.to_pickle(
    r"C:\Users\dell\Documents\schemes_data.pkl"
)

with open(
    r"C:\Users\dell\Documents\scheme_texts.pkl",
    "wb"
) as f:
    pickle.dump(texts, f)

print("FAISS index created successfully!")