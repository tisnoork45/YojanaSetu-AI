
import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/Final_Govt_Schemes_Dataset_Enriched.csv"
INDEX_OUT = "faiss_store/schemes_faiss.index"
DATA_OUT = "faiss_store/schemes_data.pkl"
TEXTS_OUT = "faiss_store/scheme_texts.pkl"

print("Loading enriched dataset...")
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

assert index.ntotal == len(df), "FAISS index size and dataframe row count do not match!"

print("Saving FAISS index...")
faiss.write_index(index, INDEX_OUT)

print("Saving scheme dataframe (with eligibility columns)...")
df.to_pickle(DATA_OUT)

print("Saving scheme texts...")
with open(TEXTS_OUT, "wb") as f:
    pickle.dump(texts, f)

print(f"FAISS index created successfully! ({index.ntotal} schemes indexed)")
