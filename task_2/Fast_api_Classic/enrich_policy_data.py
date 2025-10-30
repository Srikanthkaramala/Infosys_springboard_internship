# enrich_policy_data.py (fixed)
import pandas as pd
import re

CSV_PATH = "data/csv/updated_data.csv"
df = pd.read_csv(CSV_PATH)

# Find a column that looks like the main text
possible_text_cols = [c for c in df.columns if c.lower() in ["summary", "description", "text", "content", "details"]]
text_col = possible_text_cols[0] if possible_text_cols else df.columns[0]
print(f"Using '{text_col}' as text source.")

# Ensure columns exist
for col in ["title", "region", "year", "status"]:
    if col not in df.columns:
        df[col] = "N/A"

# Use first few words of text as title if blank
df["title"] = df["title"].fillna("")
df.loc[df["title"].str.strip() == "", "title"] = df[text_col].astype(str).str.split().str[:6].str.join(" ")

# Extract year
def extract_year(text):
    match = re.search(r"\b(19[9]\d|20[0-2]\d|2030)\b", str(text))
    return match.group(0) if match else "N/A"

df["year"] = df[text_col].apply(extract_year)

# Detect region/state names
states = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

def find_region(text):
    for state in states:
        if state.lower() in str(text).lower():
            return state
    return "N/A"

df["region"] = df[text_col].apply(find_region)

# Infer status
def find_status(text):
    text = str(text).lower()
    if "implemented" in text or "launched" in text:
        return "Implemented"
    if "proposed" in text or "draft" in text:
        return "Proposed"
    if "ongoing" in text:
        return "Ongoing"
    return "N/A"

df["status"] = df[text_col].apply(find_status)

# Save back
df.to_csv(CSV_PATH, index=False)
print("✅ Enriched dataset saved with inferred title, region, year, and status.")
