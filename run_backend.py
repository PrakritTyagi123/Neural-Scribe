"""
run_backend.py - Launch the Digit AI server
"""
import uvicorn

if __name__ == '__main__':
    print("=" * 50)
    print("  Digit AI - Live Neural Network Dashboard")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 50)
    uvicorn.run(
        "backend.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        ws_max_size=16 * 1024 * 1024,
    )