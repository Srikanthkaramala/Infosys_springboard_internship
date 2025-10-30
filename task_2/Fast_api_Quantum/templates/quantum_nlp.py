# Quantum NLP version of TF-IDF pipeline using Qiskit

# ----------------------------------------------------

from qiskit import QuantumCircuit

from qiskit_algorithms.kernels import FidelityQuantumKernel

from qiskit_machine_learning.kernels import QuantumKernel

from qiskit.primitives import Sampler

from qiskit.utils import algorithm_globals

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

import numpy as np

import joblib
 
# Paths

MODEL_PATH = "quantum_policy_kernel.pkl"

MATRIX_PATH = "quantum_policy_matrix.pkl"
 
# Load datasets

train_df = pd.read_csv("train_policies.csv")

full_df = pd.read_csv("education_policies.csv")
 
# --- Preprocess text ---

def preprocess(df):

    df = df.copy()

    df["text_for_nlp"] = (df["title"].astype(str) + ". " +

                          df["full_text"].astype(str) + ". Stakeholders: " +

                          df["stakeholders"].astype(str)).str.lower()

    return df
 
train_df = preprocess(train_df)

full_df = preprocess(full_df)
 
# --- Classical vectorization ---

vectorizer = TfidfVectorizer(max_features=8)   # small feature size for quantum encoding

X_train_tfidf = vectorizer.fit_transform(train_df["text_for_nlp"]).toarray()

X_full_tfidf = vectorizer.transform(full_df["text_for_nlp"]).toarray()
 
# Normalize to [0, π] for angle encoding

X_train_norm = np.pi * (X_train_tfidf / np.max(X_train_tfidf))

X_full_norm = np.pi * (X_full_tfidf / np.max(X_full_tfidf))
 
# --- Define quantum feature map (angle encoding) ---

def feature_map(x):

    qc = QuantumCircuit(len(x))

    for i, val in enumerate(x):

        qc.ry(val, i)

    qc.barrier()

    for i in range(len(x) - 1):

        qc.cx(i, i + 1)

    return qc
 
# --- Quantum kernel ---

sampler = Sampler()

quantum_kernel = FidelityQuantumKernel(

    feature_map=feature_map,

    fidelity=Sampler()

)
 
# Compute Quantum Kernel Matrix (similarity)

kernel_matrix = quantum_kernel.evaluate(X_full_norm, X_full_norm)
 
# Save results

joblib.dump(quantum_kernel, MODEL_PATH)

joblib.dump({"kernel_matrix": kernel_matrix, "df": full_df}, MATRIX_PATH)
 
print(f"✅ Quantum kernel model saved to {MODEL_PATH} and {MATRIX_PATH}")

print("Quantum similarity matrix shape:", kernel_matrix.shape)

 