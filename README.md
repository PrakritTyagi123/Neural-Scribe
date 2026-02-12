# Digit Recognizer

A full-stack handwritten digit recognition web application combining a Python machine learning backend with a browser-based frontend.

## Project Structure

```
digit-recognizer/
├── backend/
│   ├── api/
│   │   └── app.py              # FastAPI server with /predict endpoint
│   ├── interface/
│   │   ├── predictor.py        # Model inference
│   │   └── preprocess.py       # Image preprocessing
│   ├── train/
│   │   ├── dataset.py          # MNIST data loading
│   │   ├── model.py            # Neural network definition
│   │   ├── save_model.py       # Model saving utilities
│   │   └── train.py            # Training loop
│   └── models/
│       └── digit_model.pt      # Trained model weights (generated)
├── data/
│   ├── raw/
│   │   ├── mnist/              # MNIST dataset files
│   │   └── augmented/          # Augmented training data
│   └── processed/
│       └── training_cache/     # Cached processed data
├── frontend/
│   ├── index.html              # Main HTML page
│   ├── style.css               # Styling
│   ├── canvas.js               # Canvas drawing logic
│   ├── api.js                  # API communication
│   └── ui.js                   # UI updates and display
├── requirements.txt            # Python dependencies
├── run_backend.py              # Server launch script
└── README.md                   # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python -m backend.train.train
```

This downloads MNIST (if needed), trains the neural network, and saves weights to `backend/models/digit_model.pt`.

### 3. Start the Server

```bash
python run_backend.py
```

### 4. Open the App

Navigate to http://localhost:8000 in your browser.

## How It Works

### Training Pipeline
1. `dataset.py` loads MNIST images (28×28 grayscale digits)
2. `model.py` defines a fully connected network (784→128→64→10)
3. `train.py` optimizes the network using backpropagation
4. Trained weights are saved to `digit_model.pt`

### Inference Pipeline
1. User draws a digit on the canvas
2. Frontend downscales to 28×28 and sends pixel data
3. `preprocess.py` normalizes the input
4. `predictor.py` runs forward pass through the model
5. Predicted digit and confidence scores are returned

## API Endpoints

- `POST /predict` - Submit pixel data, receive prediction
- `GET /health` - Server health check
- `GET /` - Serve frontend

## Model Architecture

```
Input (784) → Linear → ReLU → Linear → ReLU → Linear → Softmax
              (128)            (64)            (10)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- FastAPI
- Modern web browser with Canvas support