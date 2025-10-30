# add_missing_columns.py
import pandas as pd
import os

CSV_PATH = "data/csv/updated_data.csv"
df = pd.read_csv(CSV_PATH)

# Ensure the required columns exist
required = ["title", "region", "year", "status"]
for col in required:
    if col not in df.columns:
        df[col] = "N/A"

# Optional: if your dataset has a name or text column, use it as the title
# Example: df["title"] = df["Scheme Name"] or df["Policy"] if present
# Replace "Scheme Name" below with the actual column name if it exists
if "Scheme Name" in df.columns:
    df["title"] = df["Scheme Name"]

df.to_csv(CSV_PATH, index=False)
print(f"✅ File updated with {len(required)} standard columns.")
