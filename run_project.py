#!/usr/bin/env python3
"""
run_backend.py - Launch the Digit Recognizer FastAPI server

This script starts the backend server using Uvicorn.
The server serves both the API endpoints and the frontend static files.

Usage:
    python run_backend.py
    
Then open http://localhost:8000 in your browser.
"""

import uvicorn
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

if __name__ == "__main__":
    print("=" * 50)
    print("  Digit Recognizer - Starting Server")
    print("=" * 50)
    print("\n  Open http://localhost:8000 in your browser\n")
    print("  Press Ctrl+C to stop the server")
    print("=" * 50 + "\n")
    
    uvicorn.run(
        "backend.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )