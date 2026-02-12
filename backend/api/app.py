"""
app.py - FastAPI Server for Digit Recognition

This module provides the REST API endpoints for the digit recognizer:
    - POST /predict: Submit pixel data and receive predictions
    - GET /health: Server health check
    - GET /: Serve the frontend application

The server also serves static files from the frontend directory.
"""

import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.interface.predictor import get_predictor


# Initialize FastAPI app
app = FastAPI(
    title="Digit Recognizer API",
    description="A machine learning API for recognizing handwritten digits",
    version="1.0.0"
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BACKEND_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")


# --- Pydantic Models ---

class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    pixels: List[float] = Field(
        ...,
        description="Array of 784 pixel values (28x28 image) with values 0-255",
        min_length=784,
        max_length=784
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "pixels": [0.0] * 784  # Black image
            }
        }


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    success: bool
    digit: int = Field(..., ge=-1, le=9)
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: List[float] = Field(..., min_length=10, max_length=10)
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    model_loaded: bool
    version: str


# --- API Endpoints ---

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """
    Serve the main frontend HTML page.
    """
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check server health and model status.
    """
    predictor = get_predictor()
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_digit(request: PredictRequest):
    """
    Predict a digit from pixel data.
    
    Accepts a 784-element array representing a 28x28 grayscale image.
    Pixel values should be in the range 0-255.
    
    Returns the predicted digit (0-9), confidence score, and 
    probability distribution across all digits.
    """
    predictor = get_predictor()
    
    if not predictor.is_loaded:
        return PredictResponse(
            success=False,
            digit=-1,
            confidence=0.0,
            probabilities=[0.0] * 10,
            error="Model not loaded. Please train the model first."
        )
    
    # Get prediction
    result = predictor.predict(request.pixels)
    
    return PredictResponse(
        success=result['success'],
        digit=result['digit'],
        confidence=result['confidence'],
        probabilities=result['probabilities'],
        error=result.get('error')
    )


@app.post("/predict/top-k")
async def predict_top_k(request: PredictRequest, k: int = 3):
    """
    Get top-k predictions for the given pixel data.
    
    Args:
        request: Pixel data
        k: Number of top predictions to return (1-10)
    """
    predictor = get_predictor()
    
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    k = max(1, min(k, 10))  # Clamp k to valid range
    result = predictor.get_top_k(request.pixels, k=k)
    
    return {
        "success": result['success'],
        "predictions": [
            {"digit": d, "probability": p}
            for d, p in result['predictions']
        ],
        "all_probabilities": result['probabilities']
    }


# --- Static Files ---

# Mount static files for frontend assets (CSS, JS)
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    
    # Also serve files directly from frontend root
    @app.get("/{filename:path}")
    async def serve_static(filename: str):
        """Serve static files from frontend directory."""
        file_path = os.path.join(FRONTEND_DIR, filename)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        raise HTTPException(status_code=404, detail="File not found")


# --- Startup Event ---

@app.on_event("startup")
async def startup_event():
    """Initialize resources on server startup."""
    print("\n" + "=" * 50)
    print("  Digit Recognizer API Starting...")
    print("=" * 50)
    
    # Pre-load the predictor
    predictor = get_predictor()
    
    if predictor.is_loaded:
        print("  ✓ Model loaded successfully")
    else:
        print("  ⚠ Model not loaded - train it first!")
        print("    Run: python -m backend.train.train")
    
    print("\n  Endpoints:")
    print("    GET  /         - Frontend UI")
    print("    GET  /health   - Health check")
    print("    POST /predict  - Digit prediction")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)