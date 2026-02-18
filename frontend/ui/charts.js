/**
 * charts.js — Training Charts (Accuracy & Loss)
 *
 * Renders line charts for accuracy and loss history.
 * Subscribes to state.history changes and re-renders.
 * Theme-aware via CSS variable reads.
 */

import { subscribe } from '../state/appState.js';

const dpr = window.devicePixelRatio || 1;
let accCanvas, lossCanvas;

function init(accId, lossId) {
  accCanvas = document.getElementById(accId);
  lossCanvas = document.getElementById(lossId);

  subscribe('history', render);
  subscribe('theme', render);

  window.addEventListener('resize', render);
  render();
}

function render() {
  const st = window.__NS_STATE__;
  if (!st) return;
  const cs = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  renderChart(accCanvas, st.history.accuracy || [], {
    color: cs('--green'),
    minY: 60,
    maxY: 100,
    sfx: '%',
  });

  renderChart(lossCanvas, st.history.train_loss || [], {
    color: cs('--amber'),
    minY: 0,
    maxY: null,
    sfx: '',
  });
}

function renderChart(cv, data, o) {
  if (!cv) return;
  const ct = cv.getContext('2d');
  const rect = cv.parentElement.getBoundingClientRect();
  const sz = Math.min(rect.width, rect.height);
  if (sz < 10) return;

  cv.width = Math.round(sz * dpr);
  cv.height = Math.round(sz * dpr);
  ct.setTransform(dpr, 0, 0, dpr, 0, 0);
  cv.style.width = sz + 'px';
  cv.style.height = sz + 'px';

  const cs = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const gridColor = cs('--chart-grid');
  const lblColor = cs('--chart-lbl');

  const w = sz, h = sz;
  const pd = { t: 14, r: 14, b: 24, l: 48 };
  const cW = w - pd.l - pd.r;
  const cH = h - pd.t - pd.b;

  ct.clearRect(0, 0, w, h);

  if (!data.length) {
    ct.font = '500 12px "Outfit"';
    ct.fillStyle = lblColor;
    ct.textAlign = 'center';
    ct.fillText('Train to see data', w / 2, h / 2);
    return;
  }

  const mn = o.minY ?? Math.min(...data) * 0.95;
  const mx = o.maxY ?? Math.max(...data) * 1.05;
  const rn = mx - mn || 1;

  // Grid lines
  ct.strokeStyle = gridColor;
  ct.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pd.t + (cH / 4) * i;
    ct.beginPath();
    ct.moveTo(pd.l, y);
    ct.lineTo(w - pd.r, y);
    ct.stroke();
    const v = mx - (rn / 4) * i;
    ct.font = '500 10px "Geist Mono"';
    ct.fillStyle = lblColor;
    ct.textAlign = 'right';
    ct.fillText(v.toFixed(v > 10 ? 1 : 3) + o.sfx, pd.l - 6, y + 3.5);
  }

  // Line path
  const sx = data.length > 1 ? cW / (data.length - 1) : 0;
  ct.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = pd.l + i * sx;
    const y = pd.t + cH - ((data[i] - mn) / rn) * cH;
    i === 0 ? ct.moveTo(x, y) : ct.lineTo(x, y);
  }
  ct.strokeStyle = o.color;
  ct.lineWidth = 2;
  ct.lineJoin = 'round';
  ct.stroke();

  // Glow
  ct.globalAlpha = 0.12;
  ct.lineWidth = 7;
  ct.stroke();
  ct.globalAlpha = 1;

  // Fill gradient
  ct.lineTo(pd.l + (data.length - 1) * sx, pd.t + cH);
  ct.lineTo(pd.l, pd.t + cH);
  ct.closePath();
  const gd = ct.createLinearGradient(0, pd.t, 0, pd.t + cH);
  gd.addColorStop(0, o.color + '18');
  gd.addColorStop(1, o.color + '02');
  ct.fillStyle = gd;
  ct.fill();

  // Dots
  for (let i = 0; i < data.length; i++) {
    const x = pd.l + i * sx;
    const y = pd.t + cH - ((data[i] - mn) / rn) * cH;
    ct.beginPath();
    ct.arc(x, y, 2.5, 0, Math.PI * 2);
    ct.fillStyle = o.color;
    ct.fill();
    ct.beginPath();
    ct.arc(x, y, 6, 0, Math.PI * 2);
    ct.fillStyle = o.color;
    ct.globalAlpha = 0.08;
    ct.fill();
    ct.globalAlpha = 1;
  }
}

export { init, render };