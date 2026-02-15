# Digit AI — Live Neural Network Dashboard

Real-time handwritten digit recognition with live neural network visualization, 
training feedback, and a professional AI research dashboard UI.

## Features

- **Real-time inference** via WebSocket — no button clicks needed
- **Light & Dark themes** — toggle with one click, preferences saved
- **Live neural network visualization** — watch neurons activate as you draw
- **CNN model** — ~99.2% accuracy on MNIST
- **Live training metrics** — accuracy and loss charts update in real-time
- **Confidence bars** — see probability distribution across all 10 digits
- **GPU support** — auto-detects CUDA for fast training

## Quick Start

### Prerequisites
- Python 3.10+  
- NVIDIA GPU with CUDA (recommended, CPU works too)

### Setup

```bash
# 1. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac  
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install PyTorch with CUDA for GPU support
# Visit https://pytorch.org/get-started/locally/ for your specific setup
# Example for CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Start the server
python run_backend.py

# 5. Open http://localhost:8000 in your browser
```

### First Run

1. Open the app in your browser
2. Click **Train Model** → set epochs (15 recommended) → Start
3. Wait ~3-5 minutes (with GPU) for training to complete
4. Draw a digit on the canvas — prediction happens instantly!

## Project Structure

```
digit-ai/
├── backend/
│   ├── api/
│   │   └── app.py            # FastAPI + WebSocket server
│   ├── interface/
│   │   ├── preprocess.py      # Canvas → MNIST tensor
│   │   └── predictor.py       # Model inference engine
│   ├── models/
│   │   └── digit_model.pt     # Trained weights (generated)
│   └── train/
│       ├── model.py           # CNN architecture
│       ├── dataset.py         # MNIST data loader
│       └── train.py           # Training loop
├── frontend/
│   └── index.html             # Complete dashboard (single file)
├── data/raw/mnist/             # MNIST data (auto-downloaded)
├── run_backend.py              # Server launcher
├── requirements.txt
└── README.md
```

## Tech Stack

- **Backend**: Python, PyTorch, FastAPI, WebSocket
- **Frontend**: Vanilla HTML/CSS/JS (zero dependencies)
- **Model**: CNN (Conv2d → Conv2d → FC → FC)
- **Communication**: WebSocket for real-time, REST as fallback
