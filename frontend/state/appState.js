/**
 * appState.js — Reactive State Manager
 *
 * Updated: Added 'mode' state for context mode switching (all/text/math)
 */

const state = {
  // Connection
  connected: false,
  status: 'init',

  // Model
  modelLoaded: false,
  device: '—',

  // Prediction
  prediction: null,

  // Context Mode: 'all' | 'text' | 'math'
  mode: 'all',

  // Training
  training: false,
  epoch: 0,
  totalEpochs: 0,
  trainLoss: 0,
  testLoss: 0,
  accuracy: 0,
  bestAccuracy: 0,
  epochTime: 0,
  lr: 0,
  includeSymbols: true,

  // History (for charts)
  history: {
    train_loss: [],
    test_loss: [],
    accuracy: [],
  },

  // Canvas
  hasContent: false,

  // FPS tracking
  inferenceMs: 0,
  fpsHistory: [],
  avgFps: 0,
  avgInferenceMs: 0,

  // UI
  theme: 'dark',

  // Errors
  error: null,
};

const subscribers = new Map();

function subscribe(keys, callback) {
  const keyList = Array.isArray(keys) ? keys : [keys];
  for (const key of keyList) {
    if (!subscribers.has(key)) subscribers.set(key, new Set());
    subscribers.get(key).add(callback);
  }
  return () => {
    for (const key of keyList) {
      const subs = subscribers.get(key);
      if (subs) subs.delete(callback);
    }
  };
}

function update(patch) {
  const changed = [];

  for (const [key, value] of Object.entries(patch)) {
    if (state[key] !== value) {
      state[key] = value;
      changed.push(key);
    }
  }

  if (changed.length === 0) return;

  for (const key of changed) {
    const subs = subscribers.get(key);
    if (subs) {
      for (const cb of subs) {
        try { cb(state[key], key, state); }
        catch (e) { console.error(`State subscriber error [${key}]:`, e); }
      }
    }
  }

  const wildcard = subscribers.get('*');
  if (wildcard) {
    for (const cb of wildcard) {
      try { cb(state, changed); }
      catch (e) { console.error('State wildcard subscriber error:', e); }
    }
  }
}

function pushHistory(trainLoss, testLoss, accuracy) {
  state.history.train_loss.push(trainLoss);
  state.history.test_loss.push(testLoss);
  state.history.accuracy.push(accuracy);

  const subs = subscribers.get('history');
  if (subs) {
    for (const cb of subs) {
      try { cb(state.history, 'history', state); }
      catch (e) { console.error('History subscriber error:', e); }
    }
  }
}

function setHistory(history) {
  state.history = {
    train_loss: history.train_loss || [],
    test_loss: history.test_loss || [],
    accuracy: history.accuracy || [],
  };

  const subs = subscribers.get('history');
  if (subs) {
    for (const cb of subs) {
      try { cb(state.history, 'history', state); }
      catch (e) { console.error('History subscriber error:', e); }
    }
  }
}

function trackInference(ms) {
  state.fpsHistory.push(ms);
  if (state.fpsHistory.length > 30) state.fpsHistory.shift();
  const avg = state.fpsHistory.reduce((s, v) => s + v, 0) / state.fpsHistory.length;
  update({
    inferenceMs: ms,
    avgInferenceMs: Math.round(avg * 10) / 10,
    avgFps: Math.round(1000 / avg),
  });
}

export { state, subscribe, update, pushHistory, setHistory, trackInference };
