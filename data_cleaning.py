import pandas as pd
import re

INPUT_FILE = r"C:\Users\dell\Documents\ConvoProject_CustomMadeDataset.csv"
OUTPUT_FILE = r"C:\Users\dell\Documents\Final_Govt_Schemes_Dataset.csv"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\n|\t", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

df = pd.read_csv(INPUT_FILE)

df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

if "unnamed:_3" in df.columns:
    df.drop(columns=["unnamed:_3"], inplace=True)

df["description"] = df["description"].apply(clean_text)
df["category"] = df["category"].str.lower()
df["scheme_name"] = df["scheme_name"].str.strip()

df.to_csv(OUTPUT_FILE, index=False)

print("Dataset cleaned successfully")