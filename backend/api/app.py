"""
app.py - FastAPI Server with Real-Time WebSocket
Handles live inference, training triggers, and broadcasts updates.

Fixes over original:
- Safer CUDA threading for training
- Consistent error handling
"""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.interface.predictor import Predictor


# === Globals ===
predictor: Predictor = None
connected_clients: set[WebSocket] = set()
training_in_progress = False
training_history = {'train_loss': [], 'test_loss': [], 'accuracy': []}
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    global predictor, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    predictor = Predictor(device=device, use_tta=True)
    yield
    print("Shutting down...")


app = FastAPI(title="Digit AI - Live Neural Network", lifespan=lifespan)


# === Broadcast Helper ===
async def broadcast(message: dict):
    """Send message to all connected WebSocket clients."""
    dead = set()
    payload = json.dumps(message)
    for ws in connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    connected_clients.difference_update(dead)


# === WebSocket Endpoint ===
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    print(f"Client connected. Total: {len(connected_clients)}")

    # Send initial state
    init_history = predictor.saved_history if predictor.model_loaded else training_history
    await ws.send_text(json.dumps({
        'type': 'init',
        'data': {
            'model_loaded': predictor.model_loaded,
            'device': str(device),
            'training_history': init_history,
        }
    }))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg['type'] == 'predict':
                # Real-time inference
                pixels = msg['data']['pixels']
                result = predictor.predict(pixels)
                await ws.send_text(json.dumps({
                    'type': 'prediction',
                    'data': result
                }))

            elif msg['type'] == 'ping':
                await ws.send_text(json.dumps({'type': 'pong'}))

            elif msg['type'] == 'train':
                # Start training in background
                if not training_in_progress:
                    epochs = msg.get('data', {}).get('epochs', 15)
                    asyncio.create_task(run_training(epochs))

            elif msg['type'] == 'reset_model':
                await reset_model()

            elif msg['type'] == 'shutdown':
                await broadcast({'type': 'shutdown_ack', 'data': {}})
                import os, signal
                os.kill(os.getpid(), signal.SIGTERM)

    except WebSocketDisconnect:
        connected_clients.discard(ws)
        print(f"Client disconnected. Total: {len(connected_clients)}")
    except Exception as e:
        connected_clients.discard(ws)
        print(f"WebSocket error: {e}")


# === Training Runner ===
async def run_training(epochs=15):
    global training_in_progress, predictor, training_history
    training_in_progress = True

    await broadcast({
        'type': 'training_started',
        'data': {'total_epochs': epochs}
    })

    try:
        import queue as queue_module
        from backend.train.train import train_model

        progress_queue = queue_module.Queue()
        loop = asyncio.get_event_loop()

        def train_sync():
            # Device is created inside train_model for thread safety
            # but we pass it as a hint
            return train_model(
                epochs=epochs,
                save_path='backend/models/digit_model.pt',
                progress_queue=progress_queue,
                device=device
            )

        # Start training in thread pool
        train_future = loop.run_in_executor(None, train_sync)

        # Poll the queue every 0.5s and broadcast updates
        while not train_future.done():
            await asyncio.sleep(0.5)
            while True:
                try:
                    update = progress_queue.get_nowait()
                    await broadcast({
                        'type': 'training_update',
                        'data': update
                    })
                except queue_module.Empty:
                    break

        # Get final result
        model, history = train_future.result()

        # Drain remaining updates
        while True:
            try:
                update = progress_queue.get_nowait()
                await broadcast({
                    'type': 'training_update',
                    'data': update
                })
            except queue_module.Empty:
                break

        # Update training history
        training_history['train_loss'] = history['train_loss']
        training_history['test_loss'] = history['test_loss']
        training_history['accuracy'] = history['accuracy']

        # Reload model
        predictor.load_model('backend/models/digit_model.pt')

        # Broadcast completion
        await broadcast({
            'type': 'training_complete',
            'data': {
                'accuracy': round(history['accuracy'][-1], 2),
                'epochs': epochs,
                'history': {
                    'train_loss': [round(x, 4) for x in history['train_loss']],
                    'test_loss': [round(x, 4) for x in history['test_loss']],
                    'accuracy': [round(x, 2) for x in history['accuracy']],
                }
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        await broadcast({
            'type': 'training_error',
            'data': {'error': str(e)}
        })
    finally:
        training_in_progress = False


async def reset_model():
    """Delete model and reset predictor."""
    global predictor, training_history
    model_path = Path('backend/models/digit_model.pt')
    if model_path.exists():
        model_path.unlink()

    predictor = Predictor(device=device, use_tta=True)
    training_history = {'train_loss': [], 'test_loss': [], 'accuracy': []}

    await broadcast({
        'type': 'model_reset',
        'data': {'message': 'Model reset. Train a new model to begin.'}
    })


# === REST Endpoints ===
class PredictRequest(BaseModel):
    pixels: list[float]


@app.post("/api/predict")
async def predict_rest(req: PredictRequest):
    """REST fallback for prediction."""
    result = predictor.predict(req.pixels)
    return result


@app.get("/api/status")
async def status():
    return {
        'model_loaded': predictor.model_loaded,
        'device': str(device),
        'training_in_progress': training_in_progress,
        'connected_clients': len(connected_clients),
        'avg_inference_ms': predictor.get_avg_inference_ms(),
        'fps': predictor.fps,
    }


@app.post("/api/shutdown")
async def shutdown():
    """Shut down the server from UI."""
    import os, signal
    await broadcast({'type': 'shutdown_ack', 'data': {}})
    os.kill(os.getpid(), signal.SIGTERM)
    return {'status': 'shutting_down'}


# === Serve Frontend ===
frontend_dir = Path(__file__).parent.parent.parent / 'frontend'
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(str(frontend_dir / 'index.html'))