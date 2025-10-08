from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Initialize the FastAPI app
app = FastAPI()

# Set up the templates directory
templates = Jinja2Templates(directory="templates")

# Define a route for the homepage
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})