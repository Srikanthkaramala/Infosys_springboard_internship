import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Paths
DATA_PATH = os.path.join("data", "updated_data.csv")
VECTORIZER_PATH = "policy_vectorizer.pkl"
MATRIX_PATH = "policy_tfidf_matrix.pkl"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Build missing columns with reasonable defaults
df["policy_id"] = df.index + 1
df["title"] = df["scheme_name"]
df["region"] = "India"
df["year"] = 2025
df["status"] = "Active"

# Create a unified text column
df["full_text"] = (
    df["details"].fillna("") + " " +
    df["benefits"].fillna("") + " " +
    df["eligibility"].fillna("") + " " +
    df["application"].fillna("") + " " +
    df["documents"].fillna("") + " " +
    df["schemeCategory"].fillna("") + " " +
    df["tags"].fillna("")
)

# Drop duplicates or empties
df = df[df["full_text"].str.strip() != ""]
print(f"Dataset loaded with {len(df)} valid entries.")

# TF-IDF setup
print("Generating TF-IDF matrix...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["full_text"])

# Save vectorizer
print("Saving vectorizer to:", VECTORIZER_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

# Save TF-IDF matrix and DataFrame
print("Saving TF-IDF matrix and DataFrame to:", MATRIX_PATH)
data_to_save = {"matrix": tfidf_matrix, "df": df}
joblib.dump(data_to_save, MATRIX_PATH)

print("✅ Done! Files generated successfully.")
