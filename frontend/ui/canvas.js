/**
 * canvas.js — Drawing Canvas with Undo
 *
 * Handles all drawing interaction, stroke tracking, undo,
 * and pixel extraction (28x28 with center-of-mass).
 * Publishes hasContent to state. Calls onDraw callback for predictions.
 */

import { update } from '../state/appState.js';

let dc, dx;
let cvsSz = 200;
let isDr = false, lx, ly;
let strokes = [];
let currentStroke = [];
let onDrawCallback = null;
let hint = null;
let lastReqTime = 0;

function init(canvasId, hintId, onDraw) {
  dc = document.getElementById(canvasId);
  dx = dc.getContext('2d');
  hint = document.getElementById(hintId);
  onDrawCallback = onDraw;

  resize();

  // Mouse events
  dc.addEventListener('mousedown', startDraw);
  dc.addEventListener('mousemove', moveDraw);
  dc.addEventListener('mouseup', endDraw);
  dc.addEventListener('mouseleave', endDraw);

  // Touch events
  dc.addEventListener('touchstart', startDraw, { passive: false });
  dc.addEventListener('touchmove', moveDraw, { passive: false });
  dc.addEventListener('touchend', endDraw);

  window.addEventListener('resize', resize);
}

function resize() {
  const r = dc.parentElement.getBoundingClientRect();
  cvsSz = Math.round(Math.min(r.width, r.height));
  dc.width = cvsSz;
  dc.height = cvsSz;
  dx.lineCap = 'round';
  dx.lineJoin = 'round';
  redrawAll();
}

function clear() {
  strokes = [];
  currentStroke = [];
  redrawAll();
  update({ hasContent: false, prediction: null });
  if (hint) hint.classList.remove('hidden');
}

function undo() {
  if (strokes.length === 0) return;
  strokes.pop();
  redrawAll();
  update({ hasContent: strokes.length > 0 });
  if (strokes.length === 0) {
    update({ prediction: null });
    if (hint) hint.classList.remove('hidden');
  } else {
    requestPrediction();
  }
}

function redrawAll() {
  dx.fillStyle = '#000';
  dx.fillRect(0, 0, cvsSz, cvsSz);
  dx.strokeStyle = '#fff';
  dx.lineWidth = cvsSz / 14;
  dx.lineCap = 'round';
  dx.lineJoin = 'round';
  for (const s of strokes) {
    if (s.length < 2) continue;
    dx.beginPath();
    dx.moveTo(s[0].x, s[0].y);
    for (let i = 1; i < s.length; i++) dx.lineTo(s[i].x, s[i].y);
    dx.stroke();
  }
}

function getPos(e) {
  const r = dc.getBoundingClientRect();
  const t = e.touches ? e.touches[0] : e;
  return {
    x: (t.clientX - r.left) * (cvsSz / r.width),
    y: (t.clientY - r.top) * (cvsSz / r.height),
  };
}

function startDraw(e) {
  e.preventDefault();
  isDr = true;
  const p = getPos(e);
  lx = p.x;
  ly = p.y;
  currentStroke = [{ x: lx, y: ly }];
  if (hint) hint.classList.add('hidden');
}

function moveDraw(e) {
  if (!isDr) return;
  e.preventDefault();
  const p = getPos(e);
  dx.strokeStyle = '#fff';
  dx.lineWidth = cvsSz / 14;
  dx.beginPath();
  dx.moveTo(lx, ly);
  dx.lineTo(p.x, p.y);
  dx.stroke();
  lx = p.x;
  ly = p.y;
  currentStroke.push({ x: p.x, y: p.y });
  update({ hasContent: true });
  requestPrediction();
}

function endDraw() {
  if (isDr && currentStroke.length > 0) {
    strokes.push(currentStroke);
    currentStroke = [];
  }
  isDr = false;
  if (strokes.length > 0) requestPrediction();
}

function requestPrediction() {
  if (Date.now() - lastReqTime < 50) return; // throttle 50ms
  lastReqTime = Date.now();
  // Use hasContent flag — canvas already has current stroke rendered on it
  // even before it's pushed to the strokes array on mouseup
  if (onDrawCallback && (strokes.length > 0 || currentStroke.length > 1)) {
    onDrawCallback(getPixels());
  }
}

/**
 * Extract 28x28 pixel array — crop + scale, NO centering.
 *
 * We crop to bounding box and scale to fill 28x28.
 * The backend preprocess.py then handles:
 * - Center-of-mass alignment (matches EMNIST training data)
 * - Gaussian smoothing (matches EMNIST stroke style)
 * - Transpose (matches EMNIST orientation)
 * - EMNIST normalization
 *
 * We do NOT center into a 20x20 sub-area here because the backend
 * does its own center-of-mass centering — double centering shifts
 * characters off-center and kills accuracy.
 *
 * Returns 784-length float array (0-255 grayscale).
 */
function getPixels() {
  const sd = dx.getImageData(0, 0, cvsSz, cvsSz);
  let x0 = cvsSz, y0 = cvsSz, x1 = 0, y1 = 0, found = false;

  for (let y = 0; y < cvsSz; y++) {
    for (let x = 0; x < cvsSz; x++) {
      const i = (y * cvsSz + x) * 4;
      const b = 0.299 * sd.data[i] + 0.587 * sd.data[i + 1] + 0.114 * sd.data[i + 2];
      if (b > 30) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        found = true;
      }
    }
  }

  if (!found) return Array(784).fill(0);

  // Add small padding around bounding box
  const pad = Math.max(2, Math.round(cvsSz * 0.03));
  x0 = Math.max(0, x0 - pad);
  y0 = Math.max(0, y0 - pad);
  x1 = Math.min(cvsSz - 1, x1 + pad);
  y1 = Math.min(cvsSz - 1, y1 + pad);

  const cw = x1 - x0 + 1;
  const ch = y1 - y0 + 1;

  // Scale bounding box content to fill 28x28 (maintain aspect ratio)
  const t = document.createElement('canvas');
  t.width = 28;
  t.height = 28;
  const tc = t.getContext('2d');
  tc.fillStyle = '#000';
  tc.fillRect(0, 0, 28, 28);

  // Fit into 28x28 maintaining aspect ratio, centered
  const scale = Math.min(28 / cw, 28 / ch);
  const dw = cw * scale;
  const dh = ch * scale;
  tc.imageSmoothingEnabled = true;
  tc.imageSmoothingQuality = 'high';
  tc.drawImage(dc, x0, y0, cw, ch, (28 - dw) / 2, (28 - dh) / 2, dw, dh);

  const id = tc.getImageData(0, 0, 28, 28);
  const raw = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    raw[i] = 0.299 * id.data[i * 4] + 0.587 * id.data[i * 4 + 1] + 0.114 * id.data[i * 4 + 2];
  }

  // Rotate 90° counter-clockwise before sending.
  const rotated = new Float32Array(784);
  for (let r = 0; r < 28; r++) {
    for (let c = 0; c < 28; c++) {
      // 90° CCW: new[27-c][r] = old[r][c]
      rotated[(27 - c) * 28 + r] = raw[r * 28 + c];
    }
  }

  // Flip vertically (mirror top↔bottom)
  const flipped = new Float32Array(784);
  for (let r = 0; r < 28; r++) {
    for (let c = 0; c < 28; c++) {
      flipped[(27 - r) * 28 + c] = rotated[r * 28 + c];
    }
  }

  return Array.from(flipped);
}

export { init, clear, undo, resize };