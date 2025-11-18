# build_vectorstore.py
import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# build_vectorstore.py
# ============ CONFIG ============
DATA_PATH = "data/policies.csv"  # <-- Change this line
INDEX_DIR = "policy_index"       # Folder to save FAISS index
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# ================================

print("🔍 Loading dataset...")
df = pd.read_csv(DATA_PATH)

required = {"policy_id", "title", "full_text", "state", "year", "category", "status", "region"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["combined_text"] = df.apply(
    lambda x: f"Policy ID: {x['policy_id']}\nTitle: {x['title']}\nState: {x['state']}\nYear: {x['year']}\nCategory: {x['category']}\nStatus: {x['status']}\nRegion: {x['region']}\n\n{x['full_text']}",
    axis=1,
)

print("📖 Splitting and embedding...")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
docs = splitter.create_documents(df["combined_text"].tolist())

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

vectorstore = FAISS.from_documents(docs, embeddings)

os.makedirs(INDEX_DIR, exist_ok=True)
vectorstore.save_local(INDEX_DIR)
print(f"✅ FAISS index saved at: {INDEX_DIR}")
