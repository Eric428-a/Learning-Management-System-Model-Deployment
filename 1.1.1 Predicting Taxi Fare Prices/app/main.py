"""
app/main.py
Entry point for the FastAPI application.
"""
import os
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.predict import router as predict_router

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- FastAPI app ---
app = FastAPI(title="NYC Taxi Fare Prediction", version="1.0")

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent  # → .../app
templates_dir = BASE_DIR / "templates"
static_dir = BASE_DIR / "static"

# Mount static + templates
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# === Routers ===
# Mount the prediction API under /api
app.include_router(predict_router, prefix="/api")

# === Pages ===
@app.get("/", include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "project_title": "NYC Taxi Fare Prediction"}
    )

@app.get("/demo", include_in_schema=False)
async def demo(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})

@app.get("/about", include_in_schema=False)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/dataset", include_in_schema=False)
async def dataset(request: Request):
    return templates.TemplateResponse("dataset.html", {"request": request})

@app.get("/gallery", include_in_schema=False)
async def gallery(request: Request):
    return templates.TemplateResponse("gallery.html", {"request": request})

@app.get("/tutorial", include_in_schema=False)
async def tutorial(request: Request):
    return templates.TemplateResponse("tutorial.html", {"request": request})

@app.get("/notebooks", include_in_schema=False)
async def notebooks(request: Request):
    return templates.TemplateResponse("notebooks.html", {"request": request})

@app.get("/contact", include_in_schema=False)
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

# --- Run locally ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting app on port {port}")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
