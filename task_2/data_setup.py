# data_setup.py
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------------------
# 1. File Paths
# -------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "csv", "updated_data.csv")

VECTOR_PATH = os.path.join(DATA_DIR, "policy_vectorizer.pkl")
MATRIX_PATH = os.path.join(DATA_DIR, "policy_tfidf_matrix.pkl")

# -------------------------------------------------------
# 2. Load Dataset
# -------------------------------------------------------
print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)

# Basic cleanup: remove missing summaries/text
text_columns = [c for c in df.columns if df[c].dtype == "object"]
for col in text_columns:
    df[col] = df[col].fillna("")

# -------------------------------------------------------
# 3. Text Preparation
# -------------------------------------------------------
# Combine key text fields for TF-IDF input
if {"title", "summary"}.issubset(df.columns):
    df["full_text"] = df["title"] + " " + df["summary"]
else:
    df["full_text"] = df.apply(lambda x: " ".join(str(v) for v in x if isinstance(v, str)), axis=1)

print(f"✅ Loaded {len(df)} entries. Building TF-IDF matrix...")

# -------------------------------------------------------
# 4. Build TF-IDF
# -------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["full_text"])

# -------------------------------------------------------
# 5. Save Models
# -------------------------------------------------------
print("💾 Saving vectorizer and matrix files...")
joblib.dump(vectorizer, VECTOR_PATH)
joblib.dump({"matrix": tfidf_matrix, "df": df}, MATRIX_PATH)

print(f"✅ Done! Files saved to {DATA_DIR}")
print(f"  - Vectorizer: {VECTOR_PATH}")
print(f"  - TF-IDF Matrix: {MATRIX_PATH}")
print("All set 🚀")
