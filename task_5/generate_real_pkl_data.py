import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Define paths for input and output files
DATA_FILES = [
    'ind_poverty.csv',
    # We skip this file since it was not found previously
    # 'National_Health_Policy_2002_-_Goals_to_be_achieved.xls - National Health Policy 2002.csv',
    'updated_data.csv'
]
MODEL_PATH = "policy_vectorizer.pkl"
MATRIX_PATH = "policy_tfidf_matrix.pkl"

# Required columns for the app
REQUIRED_COLUMNS = {
    'policy_id': str,
    'title': str,
    'full_text': str, 
    'region': str,
    'year': int,
    'status': str,
    'domain': str 
}

all_data = []

print("--- Starting Data Consolidation and Cleaning ---")

# --- 1. Load, Clean, and Standardize Data ---
for file_name in DATA_FILES:
    if not os.path.exists(file_name):
        print(f"⚠️ Warning: File not found: {file_name}. Skipping.")
        continue

    try:
        df = pd.read_csv(file_name, encoding='utf-8', on_bad_lines='skip')
        print(f"\n   > Loaded {file_name} with {len(df)} rows.")

        
        # --- File-Specific Mapping Logic ---
        if file_name == 'updated_data.csv':
            print("   > Applying mapping for updated_data.csv...")
            
            # Map scheme_name to title
            df = df.rename(columns={'scheme_name': 'title'})
            # Map schemeCategory to domain
            df = df.rename(columns={'schemeCategory': 'domain'})
            # Map level to region
            df = df.rename(columns={'level': 'region'})
            
            # CRITICAL FIX: Combine details and benefits into the required 'full_text' column
            df['details'] = df['details'].astype(str).fillna('')
            df['benefits'] = df['benefits'].astype(str).fillna('')
            df['full_text'] = df['details'] + " " + df['benefits']
            
            # Add placeholders for missing columns
            if 'status' not in df.columns:
                df['status'] = 'Active'
            if 'year' not in df.columns:
                df['year'] = 2023 # Default year
            if 'policy_id' not in df.columns:
                df['policy_id'] = range(1, len(df) + 1)
                
            df = df.drop(columns=['details', 'benefits'], errors='ignore') # Drop originals

        elif file_name == 'ind_poverty.csv':
            print("   > Applying mapping for ind_poverty.csv...")
            
            # This file is purely statistical and has no policy text (full_text)
            # We will use the 'State' column for 'region' and hardcode the rest
            df = df.rename(columns={'State': 'region'})
            
            # Add placeholders for all missing text columns
            df['full_text'] = 'Statistical data on poverty expenditure and ratios.'
            df['title'] = 'Poverty Statistics'
            df['domain'] = 'poverty'
            df['status'] = 'Active'
            df['year'] = 2011 
            df['policy_id'] = range(len(df) + 1, len(df) + len(df) + 1)


        # Ensure all required columns exist and handle missing ones
        for col, dtype in REQUIRED_COLUMNS.items():
            if col not in df.columns:
                df[col] = 'Unknown' if dtype == str else 0
            
            # Data Cleaning
            if dtype == str:
                df[col] = df[col].astype(str).fillna('Unknown')
                if col == 'domain':
                    df[col] = df[col].str.lower()
            elif dtype == int:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        df = df[list(REQUIRED_COLUMNS.keys())].copy()
        all_data.append(df)
        
    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")

if not all_data:
    print("❌ Critical Error: No data could be loaded. Cannot proceed.")
    exit()

# Combine all DataFrames
df_combined = pd.concat(all_data, ignore_index=True)
print(f"--- Data Consolidation Complete. Total Policies: {len(df_combined)} ---")

# --- 2. Filter Policies with Empty/Insufficient Text ---
df_combined['text_length'] = df_combined['full_text'].str.len()
MIN_TEXT_LENGTH = 30 
df_final = df_combined[
    (df_combined['full_text'].str.strip() != '') &
    (df_combined['full_text'].str.strip().str.lower() != 'unknown') & 
    (df_combined['text_length'] >= MIN_TEXT_LENGTH)
].reset_index(drop=True)

print(f"   > Removed {len(df_combined) - len(df_final)} policies with short/empty full_text.")
print(f"   > Final policies remaining for vectorization: {len(df_final)}")


# --- 3. Create TF-IDF Model and Matrix ---
print("Creating TF-IDF Vectorizer and Matrix...")

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

try:
    if len(df_final) == 0:
        raise ValueError("Cannot fit TFIDF model on 0 documents.")

    text_data = df_final['full_text'].str.lower()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    
    if tfidf_matrix.shape[1] == 0:
        raise ValueError("empty vocabulary; perhaps the documents only contain stop words")

    # --- 4. Save Model and Data ---
    joblib.dump(vectorizer, MODEL_PATH)
    joblib.dump({"matrix": tfidf_matrix, "df": df_final}, MATRIX_PATH)

    print(f"✅ Success: Updated model data and saved {len(df_final)} policies to {MODEL_PATH} and {MATRIX_PATH}.")

except Exception as e:
    print(f"❌ An error occurred during vectorization/saving: {e}")