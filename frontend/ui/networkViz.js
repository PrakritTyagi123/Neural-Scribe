/**
 * networkViz.js — Neural Network Visualization (Performance Optimized)
 *
 * Renders 6-layer CNN visualization with:
 * - Layer caching: static wireframe drawn once, cached as ImageData
 * - Partial redraw: only active connections + neurons redrawn per frame
 * - requestAnimationFrame: controlled animation loop, no redundant draws
 * - Theme-aware: reads CSS vars on theme change, caches them
 *
 * 6 Layers: Input → Conv1 → Conv2 → Dense1 → Dense2 → Output(47)
 */

import { state, subscribe } from '../state/appState.js';

const LAYERS = [
  { n: 12 },  // Input (sampled)
  { n: 10 },  // Conv1
  { n: 12 },  // Conv2
  { n: 12 },  // Dense1
  { n: 10 },  // Dense2
  { n: 10 },  // Output (top-N)
];

const LABELS = [
  '0','1','2','3','4','5','6','7','8','9',
  'A','B','C','D','E','F','G','H','I','J',
  'K','L','M','N','O','P','Q','R','S','T',
  'U','V','W','X','Y','Z',
  'a','b','d','e','f','g','h','n','q','r','t'
];

let nc, nx;
const dpr = window.devicePixelRatio || 1;

// Cached state
let cachedW = 0, cachedH = 0;
let wireframeCache = null; // ImageData of static wireframe
let cachedColors = {};
let positions = [];
let pendingDraw = false;
let lastActivations = null;
let lastProbs = null;
let lastPred = 0;

function init(canvasId) {
  nc = document.getElementById(canvasId);
  nx = nc.getContext('2d');

  cacheColors();
  drawIdle();

  // Subscribe to relevant state changes
  subscribe('prediction', onPrediction);
  subscribe('theme', onThemeChange);

  window.addEventListener('resize', () => {
    invalidateCache();
    if (lastActivations) {
      scheduleDraw();
    } else {
      drawIdle();
    }
  });
}

function cacheColors() {
  const cs = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  cachedColors = {
    idleDot: cs('--net-idle-dot'),
    idleLine: cs('--net-idle-line'),
    green: cs('--green'),
    red: cs('--red'),
    accent: cs('--accent'),
    t3: cs('--t3'),
    t4: cs('--t4'),
  };
}

function invalidateCache() {
  wireframeCache = null;
  cachedW = 0;
  cachedH = 0;
}

function onThemeChange() {
  cacheColors();
  invalidateCache();
  if (lastActivations) {
    scheduleDraw();
  } else {
    drawIdle();
  }
}

function onPrediction(pred) {
  if (!pred || !pred.activations || !Object.keys(pred.activations).length) {
    lastActivations = null;
    lastProbs = null;
    drawIdle();
    return;
  }
  lastActivations = pred.activations;
  lastProbs = pred.probabilities;
  lastPred = pred.class_index != null ? pred.class_index : 0;
  scheduleDraw();
}

function scheduleDraw() {
  if (pendingDraw) return;
  pendingDraw = true;
  requestAnimationFrame(() => {
    pendingDraw = false;
    drawActive(lastActivations, lastProbs, lastPred);
  });
}

function sizeCanvas() {
  const rect = nc.parentElement.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;
  if (w === cachedW && h === cachedH) return { w, h, changed: false };

  nc.width = Math.round(w * dpr);
  nc.height = Math.round(h * dpr);
  nx.setTransform(dpr, 0, 0, dpr, 0, 0);
  nc.style.width = w + 'px';
  nc.style.height = h + 'px';

  cachedW = w;
  cachedH = h;
  wireframeCache = null;

  // Compute node positions
  const px = 44;
  const gx = (w - px * 2) / (LAYERS.length - 1);
  positions = [];
  for (let l = 0; l < LAYERS.length; l++) {
    const x = px + l * gx;
    const n = LAYERS[l].n;
    const gap = Math.min(26, (h - 20) / (n + 1));
    const sy = (h - (n - 1) * gap) / 2;
    const nodes = [];
    for (let i = 0; i < n; i++) nodes.push({ x, y: sy + i * gap });
    positions.push(nodes);
  }

  return { w, h, changed: true };
}

/**
 * Draw static wireframe and cache it.
 * This is the expensive part — only done once per resize.
 */
function buildWireframe(w, h) {
  nx.clearRect(0, 0, w, h);

  // Sparse connections
  for (let l = 0; l < positions.length - 1; l++) {
    const step = Math.max(1, Math.floor(positions[l].length * positions[l + 1].length / 80));
    let c = 0;
    for (let i = 0; i < positions[l].length; i++) {
      for (let j = 0; j < positions[l + 1].length; j++) {
        if (c++ % step) continue;
        nx.beginPath();
        nx.moveTo(positions[l][i].x, positions[l][i].y);
        nx.lineTo(positions[l + 1][j].x, positions[l + 1][j].y);
        nx.strokeStyle = cachedColors.idleLine;
        nx.lineWidth = 0.4;
        nx.stroke();
      }
    }
  }

  // Idle dots
  for (const L of positions) {
    for (const n of L) {
      nx.beginPath();
      nx.arc(n.x, n.y, 5, 0, Math.PI * 2);
      nx.fillStyle = cachedColors.idleDot;
      nx.fill();
    }
  }

  // Cache as ImageData
  wireframeCache = nx.getImageData(0, 0, nc.width, nc.height);
}

function drawIdle() {
  const { w, h } = sizeCanvas();
  if (!wireframeCache) buildWireframe(w, h);
  else nx.putImageData(wireframeCache, 0, 0);
}

function drawActive(acts, probs, pred) {
  const { w, h } = sizeCanvas();

  // Start from wireframe cache (fast blit instead of redrawing all connections)
  if (!wireframeCache) buildWireframe(w, h);
  nx.putImageData(wireframeCache, 0, 0);

  const GREEN = cachedColors.green;
  const RED = cachedColors.red;
  const ACCENT = cachedColors.accent;

  // Build activation arrays for 6 layers
  const outN = LAYERS[5].n;
  const probArr = (probs || []).map((p, i) => ({ p: p / 100, i, lbl: LABELS[i] || '?' }));
  const sorted = [...probArr].sort((a, b) => b.p - a.p);
  const topSorted = sorted.slice(0, outN);
  const mid = Math.floor(outN / 2);
  const predIdx = topSorted.findIndex(o => o.i === pred);

  const topOut = new Array(outN);
  const predItem = predIdx >= 0 ? topSorted[predIdx] : topSorted[0];
  const others = topSorted.filter((_, i) => i !== predIdx);
  let oi = 0;
  for (let i = 0; i < outN; i++) {
    if (i === mid) topOut[i] = predItem;
    else topOut[i] = others[oi++] || { p: 0, i: -1, lbl: '?' };
  }
  const predSlot = mid;

  // Build activation arrays for 6 layers.
  // Backend sends: conv1, conv2, fc1 (and optionally fc2).
  // If fc2 is missing, split fc1 activations across Dense1 and Dense2
  // so both layers light up properly.
  const conv1Acts = acts.conv1 || [];
  const conv2Acts = acts.conv2 || [];
  const fc1Acts = acts.fc1 || [];
  const fc2Acts = acts.fc2 || [];

  let dense1, dense2;
  if (fc2Acts.length > 0) {
    dense1 = fc1Acts.slice(0, LAYERS[3].n);
    dense2 = fc2Acts.slice(0, LAYERS[4].n);
  } else {
    // Split fc1 across both dense layers
    const half = Math.ceil(fc1Acts.length / 2);
    dense1 = fc1Acts.slice(0, half).slice(0, LAYERS[3].n);
    dense2 = fc1Acts.slice(half).slice(0, LAYERS[4].n);
    // If not enough for dense2, reuse strongest from fc1
    if (dense2.length < LAYERS[4].n) {
      const sorted3 = [...fc1Acts].sort((a, b) => b - a);
      dense2 = sorted3.slice(0, LAYERS[4].n);
    }
  }

  const A = [
    Array(LAYERS[0].n).fill(0.3),
    conv1Acts.slice(0, LAYERS[1].n),
    conv2Acts.slice(0, LAYERS[2].n),
    dense1,
    dense2,
    topOut.map(o => o.p),
  ];
  for (let l = 0; l < A.length; l++) {
    while (A[l].length < LAYERS[l].n) A[l].push(0);
  }

  // Find predicted path neurons (top 3 per layer)
  // Low threshold (0.05) so connections always appear
  const pathN = LAYERS.map(() => new Set());
  pathN[LAYERS.length - 1].add(predSlot);
  for (let l = LAYERS.length - 2; l >= 0; l--) {
    const sorted2 = A[l].map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
    for (let k = 0; k < Math.min(3, sorted2.length); k++) {
      if (sorted2[k].v > 0.05) pathN[l].add(sorted2[k].i);
    }
  }

  // Red connections (non-predicted active paths)
  for (let l = 0; l < positions.length - 1; l++) {
    for (let i = 0; i < positions[l].length; i++) {
      for (let j = 0; j < positions[l + 1].length; j++) {
        const str = (A[l][i] + A[l + 1][j]) / 2;
        if (str < 0.05) continue;
        if (pathN[l].has(i) && pathN[l + 1].has(j)) continue;
        nx.beginPath();
        nx.moveTo(positions[l][i].x, positions[l][i].y);
        nx.lineTo(positions[l + 1][j].x, positions[l + 1][j].y);
        nx.strokeStyle = RED;
        nx.globalAlpha = 0.04 + str * 0.12;
        nx.lineWidth = 0.3 + str * 0.5;
        nx.stroke();
        nx.globalAlpha = 1;
      }
    }
  }

  // Green connections (predicted path)
  for (let l = 0; l < positions.length - 1; l++) {
    for (let i = 0; i < positions[l].length; i++) {
      for (let j = 0; j < positions[l + 1].length; j++) {
        const str = (A[l][i] + A[l + 1][j]) / 2;
        if (str < 0.05) continue;
        if (!pathN[l].has(i) || !pathN[l + 1].has(j)) continue;
        nx.beginPath();
        nx.moveTo(positions[l][i].x, positions[l][i].y);
        nx.lineTo(positions[l + 1][j].x, positions[l + 1][j].y);
        // Glow
        nx.strokeStyle = GREEN;
        nx.globalAlpha = 0.06 + str * 0.06;
        nx.lineWidth = 4 + str * 3;
        nx.stroke();
        // Crisp line
        nx.globalAlpha = 0.35 + str * 0.5;
        nx.lineWidth = 0.8 + str * 1.2;
        nx.stroke();
        nx.globalAlpha = 1;
      }
    }
  }

  // Neurons
  for (let l = 0; l < positions.length; l++) {
    for (let i = 0; i < positions[l].length; i++) {
      const nd = positions[l][i];
      const a = A[l][i] || 0;
      const isOnPath = pathN[l].has(i);
      const r = isOnPath ? (4.5 + a * 2.5) : (3.5 + a * 1.5);

      // Path glow
      if (isOnPath) {
        nx.beginPath();
        nx.arc(nd.x, nd.y, r + 3, 0, Math.PI * 2);
        nx.fillStyle = GREEN;
        nx.globalAlpha = 0.08;
        nx.fill();
        nx.globalAlpha = 1;
      }

      // Main dot
      nx.beginPath();
      nx.arc(nd.x, nd.y, r, 0, Math.PI * 2);
      nx.fillStyle = isOnPath ? GREEN : ACCENT;
      nx.globalAlpha = isOnPath ? 0.85 : (0.15 + a * 0.35);
      nx.fill();
      nx.globalAlpha = 1;

      // Output labels
      if (l === positions.length - 1 && i < topOut.length) {
        const isP = i === predSlot;
        const lbl = topOut[i].lbl;
        const pct = (topOut[i].p * 100).toFixed(1);

        if (isP) {
          nx.beginPath();
          nx.arc(nd.x, nd.y, r + 3.5, 0, Math.PI * 2);
          nx.strokeStyle = GREEN;
          nx.lineWidth = 1.5;
          nx.stroke();

          nx.font = '700 13px "Geist Mono"';
          nx.fillStyle = GREEN;
          nx.textAlign = 'left';
          nx.fillText(lbl, nd.x + r + 7, nd.y - 1);
          nx.font = '600 10px "Geist Mono"';
          nx.globalAlpha = 0.7;
          nx.fillText(pct + '%', nd.x + r + 7, nd.y + 12);
          nx.globalAlpha = 1;
        } else {
          nx.font = '500 9px "Geist Mono"';
          nx.fillStyle = cachedColors.t3;
          nx.textAlign = 'left';
          nx.fillText(lbl, nd.x + r + 5, nd.y + 3);
        }
      }
    }
  }
}

/** Reset to idle state */
function reset() {
  lastActivations = null;
  lastProbs = null;
  invalidateCache();
  drawIdle();
}

export { init, reset, drawIdle };