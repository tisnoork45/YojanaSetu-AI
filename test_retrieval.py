import faiss
import pandas as pd
import pickle

from sentence_transformers import SentenceTransformer

index = faiss.read_index(
    "faiss_store/schemes_faiss.index"
)

df = pd.read_pickle(
    "faiss_store/schemes_data.pkl"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def search_schemes(user_query, top_k=3):

    query_vector = model.encode(
        [user_query],
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(
        query_vector,
        top_k
    )

    for idx in indices[0]:

        row = df.iloc[idx]

        print("\n================")
        print(row["scheme_name"])
        print(row["category"])
        print("================")

search_schemes(
    "I need money for my crops"
)