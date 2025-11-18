# quantum_model.py

import numpy as np

def predict_score(policy_vector, base_score=0.0):
    """
    Simulates a more complex quantum model prediction that is influenced 
    by the base TF-IDF relevance and introduces realistic variation.
    """
    if policy_vector.size == 0:
        return 0.0
        
    # 1. Calculate a base metric from the policy vector itself (like policy complexity)
    vector_metric = np.linalg.norm(policy_vector)
    
    # 2. Combine base metric with query relevance (base_score)
    # Add a small random element (noise) to ensure scores are not identical
    noise = np.random.uniform(-0.05, 0.05)
    
    # Score calculation: (Relevance * 50%) + (Policy Metric * 40%) + Baseline + Noise
    quantum_score = (base_score * 0.5) + (vector_metric * 0.4) + 0.1 + noise 

    # Cap score between 0 and 1
    return np.clip(quantum_score, 0.0, 1.0)