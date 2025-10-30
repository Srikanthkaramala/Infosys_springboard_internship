# main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import textwrap
import os

# ---------- Load Model + Data ----------
MODEL_PATH = "data/policy_vectorizer.pkl"
MATRIX_PATH = "data/policy_tfidf_matrix.pkl"

print("🔹 Loading vectorizer and TF-IDF matrix...")
vectorizer = joblib.load(MODEL_PATH)
data = joblib.load(MATRIX_PATH)

# Make sure we got the expected structure
if isinstance(data, dict) and "matrix" in data and "df" in data:
    tfidf_matrix = data["matrix"]
    df = data["df"]
else:
    raise ValueError("❌ Invalid TF-IDF matrix file format.")

# Fill in missing columns with defaults
for col in ["title", "policy_id", "region", "year", "status", "full_text"]:
    if col not in df.columns:
        df[col] = "N/A"

print(f"✅ Data loaded: {len(df)} records")

# ---------- FastAPI App Setup ----------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------- Search Function ----------
def search_policies(query: str, top_k: int = 3):
    print(f"\n🔍 Searching for: {query}")
    query_vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Print top 5 scores for debugging
    top_scores = sorted(sims, reverse=True)[:5]
    print(f"Top 5 similarity scores: {top_scores}")

    top_idx = sims.argsort()[::-1][:top_k]
    results = []

    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "title": row.get("title", "Untitled Policy"),
            "policy_id": row.get("policy_id", "N/A"),
            "region": row.get("region", "Unknown Region"),
            "year": row.get("year", "N/A"),
            "status": row.get("status", "N/A"),
            "summary": textwrap.shorten(str(row.get("full_text", "")), width=250, placeholder="..."),
            "score": round(sims[idx], 3)
        })
    print(f"✅ Returned {len(results)} results\n")
    return results


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    results = search_policies(query)
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})

# ---------- Dashboard API ----------
@app.get("/api/policy_stats", response_class=JSONResponse)
async def get_policy_stats():
    """Provides aggregated data for dashboard charts."""
    if "year" not in df.columns or "region" not in df.columns:
        return {"error": "Missing columns for stats."}

    year_counts = df["year"].value_counts().sort_index()
    region_counts = df["region"].value_counts().sort_index()

    return {
        "years": {"labels": year_counts.index.tolist(), "data": year_counts.values.tolist()},
        "regions": {"labels": region_counts.index.tolist(), "data": region_counts.values.tolist()}
    }

print("🚀 App ready. Open http://127.0.0.1:8000 in your browser.")
