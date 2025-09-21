"""
app/api/predict.py
API routes for NYC Taxi Fare prediction.
"""

import io
import logging
from typing import Optional

import pandas as pd
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Request,
    HTTPException,
    Form,
)
from fastapi.responses import JSONResponse

from app import model as model_module
from app.model import compute_features  # <-- import feature computation

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch prediction (CSV / JSON)
# ---------------------------------------------------------------------------
@router.post("/predict")
async def api_predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    """
    Accept input as:
      • A CSV/JSON file via multipart upload (param `file`)
      • Raw JSON body: {"rows": [{...}, {...}]} or just a list of row dicts
    Returns predictions as JSON.
    """
    try:
        if file is not None:
            content = await file.read()
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception:
                try:
                    df = pd.read_json(io.BytesIO(content))
                except Exception:
                    raise HTTPException(
                        status_code=400,
                        detail="Uploaded file is not valid CSV or JSON.",
                    )
        else:
            body = await request.json()
            if isinstance(body, dict) and "rows" in body:
                df = pd.DataFrame(body["rows"])
            elif isinstance(body, list):
                df = pd.DataFrame(body)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="JSON body must be {'rows': [...]} or a list of objects.",
                )
    except Exception as e:
        logger.exception("Failed to parse request body")
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    try:
        # Compute derived features automatically
        df = compute_features(df)

        result = model_module.predict_from_dataframe(df)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Manual prediction (Form data)
# ---------------------------------------------------------------------------
@router.post("/predict/manual")
async def predict_manual(
    pickup_datetime: str = Form(...),
    pickup_longitude: float = Form(...),
    pickup_latitude: float = Form(...),
    dropoff_longitude: float = Form(...),
    dropoff_latitude: float = Form(...),
    passenger_count: int = Form(...),
):
    """
    Handle manual entry from the demo page.
    Returns one prediction as JSON.
    """
    try:
        # Build a single-row DataFrame safely
        row = pd.DataFrame([{
            "pickup_datetime": pickup_datetime,
            "pickup_longitude": pickup_longitude,
            "pickup_latitude": pickup_latitude,
            "dropoff_longitude": dropoff_longitude,
            "dropoff_latitude": dropoff_latitude,
            "passenger_count": passenger_count,
        }])

        # Compute derived features automatically
        row = compute_features(row)

        result = model_module.predict_from_dataframe(row)

    except Exception as e:
        logger.exception("Manual prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Manual prediction failed: {e}",
        )

    return JSONResponse(content=result)
