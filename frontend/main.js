/**
 * main.js — NeuralScribe Application Entry Point
 *
 * Wires together all modules:
 * - State manager (appState.js)
 * - Theme (theme.js)
 * - Canvas (canvas.js)
 * - Network visualization (networkViz.js)
 * - Charts (charts.js)
 * - WebSocket connection
 * - DOM bindings (subscribe state → update DOM)
 *
 * No module directly mutates DOM of another module.
 * All communication goes through state.
 */

import { state, subscribe, update, pushHistory, setHistory, trackInference } from './state/appState.js';
import * as theme from './ui/theme.js';
import * as canvas from './ui/canvas.js';
import * as netViz from './ui/networkViz.js';
import * as charts from './ui/charts.js';

// Expose state globally for charts module (avoids circular import)
window.__NS_STATE__ = state;

// ============ DOM REFERENCES ============
const $ = id => document.getElementById(id);

// ============ WEBSOCKET ============
let ws = null;
let wsOk = false;
let pendingPred = false;

function connectWS() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.onopen = () => {
    wsOk = true;
    update({ connected: true, status: 'live' });
  };

  ws.onmessage = (e) => handleMessage(JSON.parse(e.data));

  ws.onclose = () => {
    wsOk = false;
    update({ connected: false, status: 'offline' });
    setTimeout(connectWS, 2000);
  };

  ws.onerror = () => {
    wsOk = false;
  };
}

function sendWS(msg) {
  if (wsOk && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

// ============ MESSAGE HANDLER ============
function handleMessage(msg) {
  switch (msg.type) {
    case 'init':
      update({
        modelLoaded: msg.data.model_loaded,
        device: msg.data.device || '—',
      });
      if (msg.data.training_history && msg.data.training_history.accuracy && msg.data.training_history.accuracy.length > 0) {
        setHistory(msg.data.training_history);
        const h = msg.data.training_history;
        update({
          accuracy: h.accuracy[h.accuracy.length - 1],
          trainLoss: h.train_loss.length ? h.train_loss[h.train_loss.length - 1] : 0,
        });
      }
      break;

    case 'prediction':
      pendingPred = false;
      if (!msg.data.error) {
        update({ prediction: msg.data });
        trackInference(msg.data.inference_ms);
      }
      // If new pixels were queued while waiting, send them now
      if (queuedPixels) {
        const px = queuedPixels;
        queuedPixels = null;
        pendingPred = true;
        sendWS({ type: 'predict', data: { pixels: px } });
      }
      break;

    case 'training_started':
      update({
        training: true,
        status: 'training',
        totalEpochs: msg.data.total_epochs,
        epoch: 0,
      });
      setHistory({ train_loss: [], test_loss: [], accuracy: [] });
      break;

    case 'training_update':
      update({
        epoch: msg.data.epoch,
        totalEpochs: msg.data.total_epochs,
        accuracy: msg.data.accuracy,
        trainLoss: msg.data.train_loss,
        testLoss: msg.data.test_loss,
        bestAccuracy: msg.data.best_accuracy || msg.data.accuracy,
        epochTime: msg.data.epoch_time || 0,
        lr: msg.data.lr || 0,
      });
      pushHistory(msg.data.train_loss, msg.data.test_loss, msg.data.accuracy);
      break;

    case 'training_complete':
      update({
        training: false,
        status: 'live',
        modelLoaded: true,
        accuracy: msg.data.accuracy,
      });
      if (msg.data.history) setHistory(msg.data.history);
      break;

    case 'training_error':
      update({
        training: false,
        status: 'live',
        error: msg.data.error || 'Training failed',
      });
      break;

    case 'model_reset':
      update({
        modelLoaded: false,
        prediction: null,
        accuracy: 0,
        trainLoss: 0,
        testLoss: 0,
      });
      setHistory({ train_loss: [], test_loss: [], accuracy: [] });
      canvas.clear();
      netViz.reset();
      break;

    case 'shutdown_ack':
      document.body.innerHTML = `<div style="display:flex;height:100vh;align-items:center;justify-content:center;font-family:'Syne',sans-serif;color:var(--accent);font-size:1.3rem;flex-direction:column;gap:14px"><span style="font-size:2.5rem">⏻</span>SYSTEM OFFLINE<br><span style="font-size:0.8rem;color:var(--t3)">Close this tab</span></div>`;
      break;

    case 'pong':
      break;
  }
}

// ============ PREDICTION REQUEST ============
let queuedPixels = null; // Latest pixels waiting to be sent

function requestPrediction(pixels) {
  if (!wsOk || !state.modelLoaded) return;

  if (pendingPred) {
    // Already waiting for a response — queue the latest pixels.
    // When response arrives, we'll send these immediately.
    queuedPixels = pixels;
    return;
  }

  pendingPred = true;
  queuedPixels = null;
  sendWS({ type: 'predict', data: { pixels } });
}

// ============ DOM SUBSCRIPTIONS ============
// These connect state changes to DOM updates. Each subscription is focused.

function initDOMBindings() {
  // Status pill
  subscribe('status', (status) => {
    const pill = $('sPill');
    const txt = $('sTxt');
    const fst = $('fSt');

    switch (status) {
      case 'live':
        pill.className = 'pill pill-ok';
        txt.textContent = 'LIVE';
        fst.textContent = 'LIVE INFERENCE ACTIVE';
        break;
      case 'training':
        pill.className = 'pill pill-tr';
        txt.textContent = 'TRAINING';
        fst.textContent = 'TRAINING IN PROGRESS';
        break;
      case 'offline':
        pill.className = 'pill pill-off';
        txt.textContent = 'OFFLINE';
        fst.textContent = 'RECONNECTING...';
        break;
      default:
        pill.className = 'pill pill-off';
        txt.textContent = 'INIT';
        fst.textContent = 'INITIALIZING...';
    }
  });

  // Model status tag
  subscribe('modelLoaded', (loaded) => {
    $('mSt').textContent = loaded ? 'Ready' : 'No Model';
  });

  // Device
  subscribe('device', (dev) => {
    $('sDev').textContent = dev;
  });

  // Prediction display
  subscribe('prediction', (pred) => {
    const dig = $('pDig');
    const conf = $('pConf');
    const ms = $('pMs');

    if (!pred) {
      dig.textContent = '—';
      dig.classList.remove('active');
      conf.textContent = '0.0%';
      ms.textContent = '—';
      clearBars();
      return;
    }

    dig.textContent = pred.label || pred.digit;
    dig.classList.add('active');
    conf.textContent = pred.confidence + '%';
    ms.textContent = pred.inference_ms + ' ms';

    renderBars(pred);
  });

  // FPS display
  subscribe(['avgFps', 'avgInferenceMs'], () => {
    $('fFps').textContent = state.avgFps + ' FPS';
    $('sInf').textContent = state.avgInferenceMs + ' ms';
  });

  // Training progress
  subscribe(['epoch', 'totalEpochs', 'accuracy', 'trainLoss', 'testLoss', 'epochTime'], () => {
    if (!state.training) return;

    const pct = state.totalEpochs > 0 ? (state.epoch / state.totalEpochs * 100) : 0;
    $('prgFl').style.width = pct + '%';
    $('prgEp').textContent = `Epoch ${state.epoch}/${state.totalEpochs}`;
    $('prgAc').textContent = state.accuracy + '%';

    // ETA calculation
    if (state.epoch > 0 && state.epochTime > 0) {
      const remaining = (state.totalEpochs - state.epoch) * state.epochTime;
      $('prgEta').textContent = '~' + formatTime(remaining) + ' left';
    }

    $('sAcc').textContent = parseFloat(state.accuracy).toFixed(2) + '%';
    $('sLoss').textContent = state.testLoss.toFixed(4);
    $('epTag').textContent = state.epoch + ' ep';
  });

  // Training state → progress bar visibility
  subscribe('training', (training) => {
    $('prg').classList.toggle('on', training);
  });

  // Accuracy/Loss display (outside training)
  subscribe('accuracy', (acc) => {
    if (!state.training && acc > 0) {
      $('sAcc').textContent = parseFloat(acc).toFixed(2) + '%';
    }
  });

  subscribe('history', () => {
    const h = state.history;
    if (h.accuracy.length > 0) {
      $('epTag').textContent = h.accuracy.length + ' ep';
    }
  });

  // Error toast
  subscribe('error', (err) => {
    if (!err) return;
    showToast(err);
    update({ error: null }); // Clear after showing
  });
}

// ============ CONFIDENCE BARS ============
const LOWER_LABELS = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'];

function buildBars() {
  const digC = $('digBars');
  const letC = $('letBars');
  digC.innerHTML = '';
  letC.innerHTML = '';

  for (let i = 0; i < 10; i++) {
    digC.innerHTML += `<div class="vbar"><div class="vbar-track"><div class="vbar-fill dig" id="db${i}"></div></div><div class="vbar-lbl dig" id="dl${i}">${i}</div></div>`;
  }
  for (let i = 0; i < 26; i++) {
    letC.innerHTML += `<div class="vbar"><div class="vbar-track"><div class="vbar-fill let" id="lb${i}"></div></div><div class="vbar-lbl let" id="ll${i}">${String.fromCharCode(97 + i)}</div></div>`;
  }
}

function clearBars() {
  for (let i = 0; i < 10; i++) {
    $('db' + i).style.height = '0%';
    $('db' + i).className = 'vbar-fill dig';
    $('dl' + i).className = 'vbar-lbl dig';
  }
  for (let i = 0; i < 26; i++) {
    $('lb' + i).style.height = '0%';
    $('lb' + i).className = 'vbar-fill let';
    $('ll' + i).className = 'vbar-lbl let';
    $('ll' + i).textContent = String.fromCharCode(97 + i);
  }
}

function renderBars(d) {
  clearBars();
  const probs = d.probabilities || [];
  const ci = d.class_index != null ? d.class_index : -1;

  // Digit bars
  for (let i = 0; i < 10 && i < probs.length; i++) {
    $('db' + i).style.height = Math.min(probs[i], 100) + '%';
    if (ci === i) {
      $('db' + i).className = 'vbar-fill dig hit';
      $('dl' + i).className = 'vbar-lbl dig hit';
    }
  }

  // Letter bars
  for (let i = 0; i < 26; i++) {
    const up = (10 + i < probs.length) ? probs[10 + i] : 0;
    const ch = String.fromCharCode(97 + i);
    const lIdx = LOWER_LABELS.indexOf(ch);
    const lo = (lIdx >= 0 && 36 + lIdx < probs.length) ? probs[36 + lIdx] : 0;
    $('lb' + i).style.height = Math.min(up + lo, 100) + '%';

    const isUpHit = (ci >= 10 && ci <= 35 && ci - 10 === i);
    const isLoHit = (ci >= 36 && ci <= 46 && LOWER_LABELS[ci - 36] === ch);
    if (isUpHit || isLoHit) {
      $('lb' + i).className = 'vbar-fill let hit';
      $('ll' + i).className = 'vbar-lbl let hit';
      $('ll' + i).textContent = isLoHit ? ch : String.fromCharCode(65 + i);
    }
  }
}

// ============ TOAST ============
function showToast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 5000);
}

// ============ HELPERS ============
function formatTime(seconds) {
  if (seconds < 60) return Math.round(seconds) + 's';
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m + 'm ' + s + 's';
}

// ============ BUTTON HANDLERS ============
function initButtons() {
  $('btnCl').addEventListener('click', () => {
    canvas.clear();
    netViz.reset();
  });

  $('btnUndo').addEventListener('click', () => canvas.undo());

  $('btnTr').addEventListener('click', () => $('mTrain').classList.add('on'));
  $('mCancel').addEventListener('click', () => $('mTrain').classList.remove('on'));

  $('epSl').addEventListener('input', (e) => {
    $('epCnt').textContent = e.target.value;
  });

  $('mStart').addEventListener('click', () => {
    const ep = +$('epSl').value;
    $('mTrain').classList.remove('on');
    setHistory({ train_loss: [], test_loss: [], accuracy: [] });
    sendWS({ type: 'train', data: { epochs: ep } });
  });

  $('btnRs').addEventListener('click', () => {
    if (confirm('Reset model? This deletes trained weights.')) {
      sendWS({ type: 'reset_model' });
    }
  });

  $('btnOff').addEventListener('click', () => $('mShut').classList.add('on'));
  $('mShutNo').addEventListener('click', () => $('mShut').classList.remove('on'));
  $('mShutYes').addEventListener('click', () => {
    $('mShut').classList.remove('on');
    sendWS({ type: 'shutdown' });
    fetch('/api/shutdown', { method: 'POST' }).catch(() => {});
  });
}

// ============ KEEPALIVE ============
setInterval(() => {
  if (wsOk) sendWS({ type: 'ping' });
}, 15000);

// ============ INIT ============
function boot() {
  theme.init();
  canvas.init('drawCanvas', 'drawHint', requestPrediction);
  netViz.init('netCvs');
  charts.init('chAcc', 'chLoss');
  buildBars();
  initDOMBindings();
  initButtons();
  connectWS();
}

// Boot when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}