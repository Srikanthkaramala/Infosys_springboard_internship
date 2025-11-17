# 1. Corrected main.py

from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
# --- FIX APPLIED HERE ---
# Original: from .processor import load_or_train_models, answer_query
from processor import load_or_train_models, answer_query # <--- Corrected absolute import

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: UPLOAD_PATH now correctly points to the parent directory to simulate where FastAPI saves the file for model training
UPLOAD_PATH = os.path.join(BASE_DIR, "..", "policies.csv") 

# Configure static files and templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, message: str | None = None):
    """Render main interface"""
    # Assuming 'templates' is configured to look in a 'templates' folder relative to BASE_DIR
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": None, "message": message},
    )


@app.post("/upload-csv")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    """Upload CSV file and trigger quantum model training"""
    save_path = os.path.join(BASE_DIR, "policies.csv") # Save within the app directory for easier access
    
    # Ensure the model directory exists for processor.py
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    
    with open(save_path, "wb") as buffer:
        # Move the cursor to the start of the file before copying
        file.file.seek(0)
        shutil.copyfileobj(file.file, buffer)

    # Train or reload the models. Pass the correct path to the CSV.
    load_or_train_models(save_path)

    return RedirectResponse(url="/?message=✅+Policies+uploaded+and+quantum+model+built+successfully!", status_code=303)


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str | None = Form(None),
    query: str | None = Form(None),
    top_k: int = Form(5),
):
    """Search top-k similar policies"""
    user_query = query if query else q # q is the input name from index.html

    if not user_query:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": None,
                "message": "❌ Please enter a query before searching.",
            },
        )

    results = answer_query(user_query, top_k)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": results, "query": user_query},
    )