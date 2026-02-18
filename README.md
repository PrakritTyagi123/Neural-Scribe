# NeuralScribe

**Real-time handwritten character recognition with live neural network visualization.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22d3ee?style=flat-square)](LICENSE)

<!-- 
📸 HERO IMAGE — This is the most important visual. Take a screenshot showing:
   - Dark theme active
   - A character drawn on the canvas (try "R" or "7" — something with clear network activation)
   - The network visualization lighting up with green predicted path
   - Confidence bars showing the prediction
   - Accuracy chart with training data visible
   
   Save as: assets/hero.png (recommended: 1920×1080 or wider)
   Tip: Use browser DevTools → Ctrl+Shift+P → "Capture full size screenshot" for clean capture
-->
![NeuralScribe Dashboard](assets/hero.png)

---

## What is this?

NeuralScribe is an interactive dashboard for training and running a CNN on the EMNIST dataset — 47 classes covering digits (0–9), uppercase letters (A–Z), and 11 lowercase letters. Draw a character, watch the neural network think in real time.

<!-- 
🎬 DEMO GIF — Record a ~10 second GIF showing:
   1. Drawing a character on the canvas
   2. The prediction updating LIVE as you draw (the confidence bars moving)
   3. The neural network visualization lighting up
   
   How to record:
   - Windows: ShareX (free) → Screen Recording → GIF
   - Mac: Gifox or Kap
   - Any OS: ScreenToGif (free)
   
   Settings: 720p, 15fps, crop to just the dashboard
   Save as: assets/demo.gif
-->
![Live Demo](assets/demo.gif)

### Why it's interesting

| Feature | What happens |
|---------|-------------|
| **Live inference** | Predictions update mid-stroke via WebSocket — no submit button |
| **Neural network visualization** | Watch 6 CNN layers activate. Green = predicted path, red = competing signals |
| **Both themes** | Dark (research lab) and light (clean paper) — one click toggle |
| **Train from the UI** | Hit Train, set epochs (up to 100), watch accuracy climb in real time |
| **Full probability view** | See confidence across all 47 classes — digits in cyan, letters in violet |

---

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/neuralscribe.git
cd neuralscribe

# Environment
python -m venv .venv
.venv\Scripts\activate
python.exe -m pip install --upgrade pip

# Dependencies
pip install -r requirements.txt

# GPU support (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Launch
python run_backend.py
```

Open **http://localhost:8000** → Click **Train** → Set 35 epochs → Start → Draw.

---

## Screenshots

<!-- 
📸 DARK THEME SCREENSHOT — Full dashboard, dark theme, with a prediction active
   Save as: assets/dark-theme.png
-->
### Dark Theme
![Dark Theme](assets/dark-theme.png)

<!-- 
📸 LIGHT THEME SCREENSHOT — Same state but light theme toggled
   Save as: assets/light-theme.png
-->
### Light Theme
![Light Theme](assets/light-theme.png)

<!-- 
📸 TRAINING SCREENSHOT — Capture while training is in progress showing:
   - Progress bar with ETA
   - Accuracy chart building up
   - Loss chart decreasing
   - Status pill showing "TRAINING"
   Save as: assets/training.png
-->
### Live Training
![Training](assets/training.png)

<!-- 
🎬 NETWORK VIZ GIF — Record a ~5 second GIF showing:
   - Draw one character, then clear, draw another
   - Focus on the neural network panel — the connections shifting between predictions
   - Crop to just the network visualization panel
   Save as: assets/network-viz.gif
-->
### Neural Network Visualization
![Network Visualization](assets/network-viz.gif)

The visualization shows 6 layers of the CNN in real time:

```
Input → Conv1 → Conv2 → Dense1 → Dense2 → Output (47 classes)
```

- **Green connections** — the predicted path (strongest signal to the winning class)
- **Red connections** — competing activations (what the network considered but rejected)  
- **Node brightness** — activation strength at each neuron
- **Output labels** — top predicted classes with confidence percentages

---

## Architecture

### Frontend (Modular ES Modules)

```
frontend/
├── index.html              # Page structure
├── style.css               # Design system (dark + light themes)
├── main.js                 # Entry point — WebSocket, DOM bindings
├── state/
│   └── appState.js         # Reactive state store (pub/sub)
└── ui/
    ├── theme.js            # Dark/light toggle
    ├── canvas.js           # Drawing, undo, pixel extraction
    ├── networkViz.js       # 6-layer CNN visualization
    └── charts.js           # Accuracy + Loss charts
```

**State flows one direction:**

```
User action → module → appState.update() → subscribers re-render
```

No module touches another module's DOM. Everything goes through state.

### Backend (PyTorch + FastAPI)

```
backend/
├── api/
│   └── app.py              # FastAPI server + WebSocket
├── interface/
│   ├── preprocess.py       # Canvas → EMNIST tensor pipeline
│   └── predictor.py        # Inference engine with TTA
├── train/
│   ├── model.py            # CNN architecture (ResBlocks + SE attention)
│   ├── dataset.py          # EMNIST data loader with augmentation
│   └── train.py            # Training loop (mixed precision, warm restarts)
└── models/
    └── digit_model.pt      # Trained weights (generated after training)
```

### Model

| Property | Value |
|----------|-------|
| Architecture | Stem → 4× ResBlock (with SE attention) → GAP → FC |
| Parameters | ~650K |
| Input | 28×28 grayscale |
| Output | 47 classes (EMNIST ByMerge) |
| Training | AdamW, CosineAnnealingWarmRestarts, Focal Loss, Mixup |
| Inference | Test-Time Augmentation (5 variants averaged) |

### Communication

```
┌──────────┐    WebSocket     ┌──────────┐
│ Frontend │ ◄──────────────► │ Backend  │
│          │   JSON messages   │ FastAPI  │
│ canvas → │ ── predict ─────► │ → model  │
│ ← bars   │ ◄─ prediction ── │ ← output │
│ ← chart  │ ◄─ train_update ─│ (async)  │
└──────────┘                  └──────────┘
```

Real-time via WebSocket. REST fallback at `/api/predict` and `/api/status`.

---

## Training Details

### Recommended Settings

| Setting | Value | Why |
|---------|-------|-----|
| Epochs | 30–50 | Warm restarts need multiple LR cycles |
| Default LR | 0.003 | BatchNorm allows higher rates |
| Batch size | 128 | GPU utilization vs generalization balance |

### Accuracy Improvements

The model includes 8 techniques stacked for maximum accuracy:

1. **Data augmentation** — rotation, translation, perspective, blur, erasing
2. **Preprocessing alignment** — center-of-mass centering + EMNIST transpose
3. **Warm restarts** — CosineAnnealingWarmRestarts escapes local minima
4. **Test-time augmentation** — 5 inference variants averaged
5. **Model capacity** — wider channels (64→160→320→256) with stochastic depth
6. **Mixup regularization** — blends training pairs for smoother boundaries
7. **Focal loss** — focuses learning on hard/confusable characters
8. **SE attention blocks** — learns which feature channels matter most

Expected accuracy: **95–97%+** on EMNIST ByMerge (SOTA baseline is ~91–92%).

<!-- 
📸 ACCURACY CHART SCREENSHOT — After training completes, capture:
   - The accuracy chart showing the full training curve
   - Final accuracy visible in the stats panel
   Save as: assets/accuracy.png
-->
![Training Accuracy](assets/accuracy.png)

---

## Design

### Typography
- **Syne** — geometric display font for the brand name "Neural"
- **Caveat** — handwriting font for "Scribe" — the neural + handwriting theme
- **Outfit** — clean body text
- **Geist Mono** — monospaced data, stats, labels

### Color System

| Token | Dark | Light | Used for |
|-------|------|-------|----------|
| `--accent` | `#22d3ee` | `#0891b2` | Primary accent, canvas border, predictions |
| `--green` | `#34d399` | `#059669` | Network viz, accuracy, success states |
| `--amber` | `#fbbf24` | `#d97706` | Loss charts, warning states |
| `--violet` | `#a78bfa` | `#7c3aed` | Letter confidence bars, progress gradient |
| `--red` | `#f87171` | `#dc2626` | Danger actions, competing network paths |

### Performance

- **Wireframe caching** — static network connections drawn once, stored as ImageData
- **requestAnimationFrame gating** — no redundant draws
- **Pixel queue** — predictions fire continuously while drawing, never blocked by pending responses
- **Throttled canvas reads** — 50ms minimum between pixel extractions

---

## Configuration

### Environment

| Variable | Default | Description |
|----------|---------|-------------|
| GPU | Auto-detected | Uses CUDA if available, falls back to CPU |
| Port | 8000 | Set in `run_backend.py` |
| Model path | `backend/models/digit_model.pt` | Auto-created on first training |
| Data path | `data/raw/emnist/` | EMNIST auto-downloads (~500 MB) |

### Training from CLI

```bash
# Train directly without the UI
python -m backend.train.train --epochs 50
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No Model" after launch | Click Train → set epochs → Start |
| CUDA out of memory | Reduce batch size in `train.py` or use CPU |
| Slow inference (~400ms) | Disable TTA in `predictor.py` (`use_tta=False`) for ~5x speedup |
| Wrong predictions | Delete `backend/models/digit_model.pt` and retrain with 35+ epochs |
| WebSocket disconnects | Check firewall, ensure port 8000 is open |
| Workers crash on Windows | `dataset.py` handles this — uses `persistent_workers=False` on Windows |

---

## Tech Stack

- **Runtime**: Python 3.10+
- **ML**: PyTorch 2.0+ (mixed precision, AMP)
- **Server**: FastAPI + Uvicorn (WebSocket + REST)
- **Frontend**: Vanilla ES Modules (zero dependencies, no build step)
- **Fonts**: Google Fonts (Syne, Caveat, Outfit, Geist Mono)

---

## License

MIT

---

<p align="center">
  <b>Neural</b><i>Scribe</i> — watch a neural network learn to read your handwriting.
</p>