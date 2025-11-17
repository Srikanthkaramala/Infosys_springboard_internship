import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pennylane as qml

# Model file paths
VECTORIZER_FILE = "models/vectorizer.pkl"
TFIDF_MATRIX_FILE = "models/tfidf_matrix.pkl"
POLICIES_FILE = "models/policies.pkl"

os.makedirs("models", exist_ok=True)

def load_or_train_models(csv_path: str = "policies.csv"):
    """Load trained models if available, else train from CSV."""
    if all(os.path.exists(f) for f in [VECTORIZER_FILE, TFIDF_MATRIX_FILE, POLICIES_FILE]):
        vectorizer = joblib.load(VECTORIZER_FILE)
        tfidf_matrix = joblib.load(TFIDF_MATRIX_FILE)
        policies = joblib.load(POLICIES_FILE)
        print("✅ Loaded existing models from disk.")
        return vectorizer, tfidf_matrix, policies

    print("⚙️ Training new models from policies.csv...")
    policies = pd.read_csv(csv_path)
    text_data = policies["full_text"].astype(str).tolist()

    # Apply L2 normalization during TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english", norm='l2')
    tfidf_matrix = vectorizer.fit_transform(text_data)

    os.makedirs(os.path.dirname(VECTORIZER_FILE), exist_ok=True)
    
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_FILE)
    joblib.dump(policies, POLICIES_FILE)
    print("✅ Models trained and saved.")

    return vectorizer, tfidf_matrix, policies


def quantum_similarity(vec1, vec2):
    """Compute similarity using a small PennyLane quantum circuit.

    The 4-component vectors are locally L2-normalized and used as RY rotation angles.
    The result is clamped to the intuitive [0, 1] similarity range.
    """
    N_QUBITS = 4

    # 1. Local L2-normalization (Crucial for consistent quantum encoding)
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)
    
    v1_norm = vec1 / norm_v1 if norm_v1 > 1e-6 else np.zeros(N_QUBITS)
    v2_norm = vec2 / norm_v2 if norm_v2 > 1e-6 else np.zeros(N_QUBITS)

    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(v1_n, v2_n):
        # State Preparation (Query Vector)
        for i in range(N_QUBITS):
            qml.RY(v1_n[i] * np.pi, wires=i) 
        qml.Barrier()
        # Rotation to measure overlap difference (Policy Vector)
        for i in range(N_QUBITS):
            qml.RY(-v2_n[i] * np.pi, wires=i)
        
        # Mean Pauli-Z expectation value is a proxy for overlap
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    raw_result = np.mean(circuit(v1_norm, v2_norm))
    
    # Map the [-1, 1] circuit output to the intuitive [0, 1] similarity score
    clamped_raw_result = np.clip(raw_result, -1.0, 1.0)
    similarity_score = (clamped_raw_result + 1.0) / 2.0 # Maps -1 to 0 and 1 to 1

    return float(similarity_score)


def answer_query(query: str, top_k: int = 5):
    """Return top-k most similar policy results using quantum similarity."""
    vectorizer, tfidf_matrix, policies = load_or_train_models()

    # Use classical cosine similarity for fast initial ranking (Top 50 candidates)
    query_vec_sparse = vectorizer.transform([query])
    similarities_classic = cosine_similarity(query_vec_sparse, tfidf_matrix).flatten()

    # Get indices of top candidates (use a larger pool for better quantum reranking)
    initial_top_indices = similarities_classic.argsort()[-50:][::-1]
    results = []
    
    N_QUBITS = 4 

    # Convert query vector to dense once
    query_vec_full = query_vec_sparse.toarray()[0]
    
    for i in initial_top_indices:
        policy_vec_sparse = tfidf_matrix[i]
        policy_vec_full = policy_vec_sparse.toarray()[0]

        # --- QUANTUM SIMILARITY INPUT PREPARATION ---
        
        non_zero_indices = np.where((query_vec_full != 0) | (policy_vec_full != 0))[0]
        
        if len(non_zero_indices) == 0:
             quantum_score = 0.0
        else:
            # Select the top N_QUBITS most important features
            magnitudes = query_vec_full[non_zero_indices] + policy_vec_full[non_zero_indices]
            sorted_indices_of_non_zero = non_zero_indices[np.argsort(magnitudes)[::-1]]
            selected_indices = sorted_indices_of_non_zero[:N_QUBITS]
            
            # Create N_QUBITS-length dense vectors (padding with zeros)
            q_input = np.zeros(N_QUBITS)
            p_input = np.zeros(N_QUBITS)

            q_input[:len(selected_indices)] = query_vec_full[selected_indices]
            p_input[:len(selected_indices)] = policy_vec_full[selected_indices]

            # Compute quantum similarity
            quantum_score = quantum_similarity(q_input, p_input)

        # Append to results
        policy_data = policies.iloc[i].to_dict()
        
        results.append({
            "policy_id": policy_data["policy_id"],
            "title": policy_data["title"],
            "region": policy_data["region"],
            # Ensure year is treated as an integer for cleaner display
            "year": int(policy_data["year"]) if pd.notna(policy_data["year"]) else 'N/A', 
            "status": policy_data["status"],
            "summary": policy_data["full_text"][:200] + "...",
            # Store ONLY the quantum score
            "similarity": quantum_score, 
        })
    
    # 5. Final re-sort and truncation by the Quantum score
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results[:top_k]