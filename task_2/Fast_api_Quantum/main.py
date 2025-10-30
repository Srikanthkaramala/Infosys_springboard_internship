from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
import os 
from fastapi.staticfiles import StaticFiles # We need this import if we use it

# ---------- Load Model + Data ----------
# This section assumes you have run data_setup.py successfully 
MODEL_PATH = "policy_vectorizer.pkl"
MATRIX_PATH = "policy_tfidf_matrix.pkl"

# Use os.path.exists for a safer check, though joblib.load is strict.
try:
    vectorizer = joblib.load(MODEL_PATH)
    data = joblib.load(MATRIX_PATH)
    tfidf_matrix = data["matrix"]
    df = data["df"]
except FileNotFoundError as e:
    print(f"ERROR: Could not find required data file: {e}. Please run data_setup.py first.")
    # Exit or raise error if crucial files are missing
    raise

# ---------- FastAPI App Setup ----------
app = FastAPI()

# FIX: Since you kept getting the "static directory does not exist" error,
# we are adding a check to ensure the directory exists before mounting it.
STATIC_DIR = "static"
if os.path.isdir(STATIC_DIR):
    app.mount(f"/{STATIC_DIR}", StaticFiles(directory=STATIC_DIR), name=STATIC_DIR)
else:
    # If the static folder is missing, we log a note but allow the app to run.
    print(f"NOTE: Directory '{STATIC_DIR}' not found. Skipping static file mounting.")
    # We still need the StaticFiles import if this block were enabled,
    # but since the error was happening *on import*, we only rely on the try/except on the path.

templates = Jinja2Templates(directory="templates")


def search_policies(query: str, top_k: int = 5): # Increased top_k for better search visibility
    """Performs TF-IDF based cosine similarity search on the policy data."""
    query_vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        # Only include results with a score > 0 (meaning some relevance)
        if sims[idx] > 0:
            results.append({
                "title": row["title"],
                "policy_id": row["policy_id"],
                "region": row["region"],
                "year": row["year"],
                "status": row["status"],
                # Ensure we use the original full_text column for the summary
                "summary": textwrap.shorten(row["full_text"], width=250, placeholder="..."),
                "score": round(sims[idx], 3)
            })
    return results


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the main page with no initial search results."""
    return templates.TemplateResponse("index.html", {"request": request, "results": None, "query": ""})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    """Handles the form submission and performs the policy search."""
    results = search_policies(query)
    # Renders the same index.html template, but passes the search results and the original query back
    return templates.TemplateResponse("index.html", {"request": request, "results": results, "query": query})
