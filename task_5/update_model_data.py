import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Define the required columns based on your README.md and app.py
REQUIRED_COLUMNS = ['policy_id', 'title', 'full_text', 'region', 'year', 'category', 'status', 'domain']

# List all new data files
data_files = [
    'ind_poverty.csv',
    'National_Health_Policy_2002_-_Goals_to_be_achieved.xls - National Health Policy 2002.csv',
    'updated_data.csv'
]

all_data = []

# --- 1. Load and Standardize Data ---
for file_name in data_files:
    if not os.path.exists(file_name):
        print(f"Warning: File not found: {file_name}. Skipping.")
        continue

    try:
        # Assuming most are CSVs; adjusting for potential Excel formatting in file names
        df = pd.read_csv(file_name, encoding='utf-8', on_bad_lines='skip')
        
        # Standardize column names (IMPORTANT: You may need to adjust these mappings based on your actual file contents)
        # Example mappings if your files have different headers:
        
        # 1. Rename columns to match the application's expected schema (app.py)
        # Assuming 'category' maps to 'domain' for filtering
        if 'category' in df.columns and 'domain' not in df.columns:
             df = df.rename(columns={'category': 'domain'})
        
        # 2. Add missing columns with default values if necessary, based on REQUIRED_COLUMNS
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = 'Unknown'
        
        # 3. Ensure essential text columns are strings and fill NA
        df['full_text'] = df['full_text'].astype(str).fillna('')
        df['title'] = df['title'].astype(str).fillna('Untitled Policy')
        
        # 4. Filter for only required columns and append
        df = df[REQUIRED_COLUMNS].copy()
        all_data.append(df)
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

if not all_data:
    print("No data loaded. Cannot proceed.")
    exit()

# Combine all DataFrames
df_combined = pd.concat(all_data, ignore_index=True)

# Ensure 'domain' is lower case for filtering in search_policies
df_combined['domain'] = df_combined['domain'].str.lower()
print(f"Total policies loaded: {len(df_combined)}")

# --- 2. Create TF-IDF Model and Matrix ---
print("Creating TF-IDF Vectorizer and Matrix...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

try:
    # Use 'full_text' column for vectorization
    tfidf_matrix = vectorizer.fit_transform(df_combined['full_text'].str.lower())

    # --- 3. Save Model and Data ---
    MODEL_PATH = "policy_vectorizer.pkl"
    MATRIX_PATH = "policy_tfidf_matrix.pkl"

    joblib.dump(vectorizer, MODEL_PATH)
    joblib.dump({"matrix": tfidf_matrix, "df": df_combined}, MATRIX_PATH)

    print(f"Successfully updated model data and saved to {MODEL_PATH} and {MATRIX_PATH}.")

except ValueError as e:
    print(f"Error during TF-IDF fitting (likely empty text data): {e}")
except Exception as e:
    print(f"An unexpected error occurred during saving: {e}")