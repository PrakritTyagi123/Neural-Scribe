"""
app.py - FastAPI Server with Context Mode Support

New features:
- Context mode ('all', 'text', 'math') sent with predict requests
- Training dataset selection (EMNIST only vs combined)
- Mode state tracked per connection
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
from typing import Optional

from backend.interface.predictor import Predictor
from backend.train.dataset import NUM_CLASSES, ALL_LABELS


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
    print(f"Total classes: {NUM_CLASSES}")
    predictor = Predictor(device=device, use_tta=True)
    yield
    print("Shutting down...")


app = FastAPI(title="NeuralScribe - Handwriting + Math Recognition", lifespan=lifespan)


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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    print(f"Client connected. Total: {len(connected_clients)}")

    # Send initial state with label info
    init_history = predictor.saved_history if predictor.model_loaded else training_history
    await ws.send_text(json.dumps({
        'type': 'init',
        'data': {
            'model_loaded': predictor.model_loaded,
            'device': str(device),
            'training_history': init_history,
            'num_classes': NUM_CLASSES,
            'labels': ALL_LABELS,
        }
    }))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg['type'] == 'predict':
                pixels = msg['data']['pixels']
                mode = msg['data'].get('mode', 'all')
                result = predictor.predict(pixels, mode=mode)
                await ws.send_text(json.dumps({
                    'type': 'prediction',
                    'data': result
                }))

            elif msg['type'] == 'ping':
                await ws.send_text(json.dumps({'type': 'pong'}))

            elif msg['type'] == 'train':
                if not training_in_progress:
                    epochs = msg.get('data', {}).get('epochs', 35)
                    include_symbols = msg.get('data', {}).get('include_symbols', True)
                    asyncio.create_task(run_training(epochs, include_symbols))

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


async def run_training(epochs=35, include_symbols=True):
    global training_in_progress, predictor, training_history
    training_in_progress = True

    dataset_name = "EMNIST + Symbols" if include_symbols else "EMNIST Only"
    await broadcast({
        'type': 'training_started',
        'data': {
            'total_epochs': epochs,
            'include_symbols': include_symbols,
            'dataset': dataset_name,
        }
    })

    try:
        import queue as queue_module
        from backend.train.train import train_model

        progress_queue = queue_module.Queue()
        loop = asyncio.get_event_loop()

        def train_sync():
            return train_model(
                epochs=epochs,
                save_path='backend/models/digit_model.pt',
                progress_queue=progress_queue,
                device=device,
                include_symbols=include_symbols,
            )

        train_future = loop.run_in_executor(None, train_sync)

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

        model, history = train_future.result()

        while True:
            try:
                update = progress_queue.get_nowait()
                await broadcast({
                    'type': 'training_update',
                    'data': update
                })
            except queue_module.Empty:
                break

        training_history['train_loss'] = history['train_loss']
        training_history['test_loss'] = history['test_loss']
        training_history['accuracy'] = history['accuracy']

        predictor.load_model('backend/models/digit_model.pt')

        await broadcast({
            'type': 'training_complete',
            'data': {
                'accuracy': round(history['accuracy'][-1], 2),
                'epochs': epochs,
                'include_symbols': include_symbols,
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
    mode: Optional[str] = 'all'


@app.post("/api/predict")
async def predict_rest(req: PredictRequest):
    result = predictor.predict(req.pixels, mode=req.mode)
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
        'num_classes': NUM_CLASSES,
    }


@app.post("/api/shutdown")
async def shutdown():
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
