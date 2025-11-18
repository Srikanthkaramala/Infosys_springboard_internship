from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from quantum_model import predict_score
from io import StringIO

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import os

# ---------- Load Model + Data ----------
MODEL_PATH = "policy_vectorizer.pkl"
MATRIX_PATH = "policy_tfidf_matrix.pkl"

try:
    vectorizer = joblib.load(MODEL_PATH)
    data = joblib.load(MATRIX_PATH)
    tfidf_matrix = data["matrix"]
    df = data["df"]
    print("Model and data loaded successfully.")
except Exception as e:
    print("Error loading model or data:", e)
    vectorizer = None
    tfidf_matrix = None
    df = pd.DataFrame()

# ---------- FastAPI App Setup ----------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static") 
templates = Jinja2Templates(directory="templates")

# ---------- Search Function ----------
def search_policies(query: str, top_k: int = 5, domain: str = None, region: str = None):
    if vectorizer is None or tfidf_matrix is None or df.empty:
        return []

    filtered_df = df.copy()
    if domain and "domain" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["domain"].str.lower() == domain.lower()]
    if region:
        filtered_df = filtered_df[filtered_df["region"].str.lower() == region.lower()]
    if filtered_df.empty:
        return []

    filtered_indices = filtered_df.index.tolist()
    
    try:
        # Create a new index map for scoring: map original df index to new matrix row index
        original_to_matrix_idx = {original_idx: new_idx for new_idx, original_idx in enumerate(filtered_indices)}
        # We need the full matrix index, so this slicing is crucial
        filtered_matrix = tfidf_matrix[filtered_indices] 
    except Exception as e:
        print("Error slicing TF-IDF matrix:", e)
        return []

    try:
        query_vec = vectorizer.transform([query.lower()])
        sims = cosine_similarity(query_vec, filtered_matrix).flatten()
    except Exception as e:
        print("Error computing similarity:", e)
        return []

    results = []
    # Loop over results and combine data with similarity score
    for i, row in filtered_df.iterrows():
        try:
            # Look up the score using the new matrix index
            matrix_row_index = original_to_matrix_idx.get(i)
            if matrix_row_index is not None and matrix_row_index < len(sims):
                sim_score = sims[matrix_row_index]
            else:
                sim_score = 0.0 # Default to 0 if indexing fails
            
            results.append({
                "title": str(row.get("title", "Untitled")),
                "policy_id": str(row.get("policy_id", "Unknown")),
                "region": str(row.get("region", "Unknown")),
                "year": int(row.get("year", 0)),
                "status": str(row.get("status", "Unknown")),
                "summary": textwrap.shorten(str(row.get("full_text", "")), width=250, placeholder="..."),
                "score": float(round(float(sim_score), 3))
            })
        except Exception as e:
            print(f"Error processing row {i}: {e}")

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    return results

# ---------- Quantum Search (CORRECTED) ----------
def quantum_search(query: str, top_k: int = 5, region: str = None):
    if vectorizer is None or tfidf_matrix is None or df.empty:
        return []

    filtered_df = df.copy()
    if region:
        filtered_df = filtered_df[filtered_df["region"].str.lower() == region.lower()]
    if filtered_df.empty:
        return []

    query_vec = vectorizer.transform([query.lower()])
    
    # Calculate similarity for ALL policies in the dataset for base score
    full_sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    results = []
    
    # Iterate over filtered policies using their original indices
    for idx, row in filtered_df.iterrows():
        try:
            # Get the policy vector from the full matrix using its index
            policy_vec = tfidf_matrix[idx].toarray()[0]
            
            # The base score is the standard TF-IDF relevance
            query_relevance = float(full_sims[idx]) 
            
            # Predict the quantum score, incorporating the base relevance score
            quantum_score = predict_score(policy_vec, base_score=query_relevance) 
            
            results.append({
                "title": row.get("title", "Untitled"),
                "policy_id": row.get("policy_id", "Unknown"),
                "region": str(row.get("region", "Unknown")),
                "year": int(row.get("year", 0)),
                "status": row.get("status", "Unknown"),
                "summary": textwrap.shorten(str(row.get("full_text", "")), width=250, placeholder="..."),
                "score": round(query_relevance, 3), 
                "quantum_score": round(float(quantum_score), 3)
            })
        except Exception as e:
            print(f"Error in quantum scoring for row {idx}: {e}")

    # Sort based on the quantum score
    results = sorted(results, key=lambda x: x["quantum_score"], reverse=True)[:top_k]
    return results


# ---------- Routes (Unchanged) ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/search_education", response_class=HTMLResponse)
async def search_education(request: Request, query: str = Form(...), region: str = Form("")):
    results = search_policies(query, domain="education", region=region)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})

@app.post("/search_poverty", response_class=HTMLResponse)
async def search_poverty(request: Request, query: str = Form(...), region: str = Form("")):
    results = search_policies(query, domain="poverty", region=region)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})

@app.post("/search_quantum", response_class=HTMLResponse)
async def search_quantum_route(request: Request, query: str = Form(...), region: str = Form("")):
    results = quantum_search(query, region=region)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})

@app.post("/export_csv")
async def export_csv(query: str = Form(...)):
    results = search_policies(query, top_k=50) 
    if not results:
        return HTMLResponse(content="No results to export.", status_code=404)

    df_export = pd.DataFrame(results)
    stream = StringIO()
    df_export.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=policy_results.csv"}
    )

@app.post("/voice_search", response_class=HTMLResponse)
async def voice_search(request: Request, query: str = Form(...)):
    results = search_policies(query, top_k=5)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "query": query
    })

# ---------- Run App ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)