/**
 * appState.js — Reactive State Manager
 *
 * Central state store with pub/sub. All UI modules subscribe to state
 * changes and re-render only what changed. No direct DOM mutation from
 * event handlers — they update state, state notifies subscribers.
 *
 * Usage:
 *   import { state, subscribe, update } from './state/appState.js';
 *   subscribe('prediction', (pred) => renderPrediction(pred));
 *   update({ prediction: { label: '7', confidence: 94.2 } });
 */

const state = {
  // Connection
  connected: false,
  status: 'init', // 'init' | 'live' | 'training' | 'offline'

  // Model
  modelLoaded: false,
  device: '—',

  // Prediction
  prediction: null,
  // { label, class_index, confidence, probabilities, activations, inference_ms, is_digit, is_upper, is_lower }

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

// Subscribers: key → Set of callbacks
// Special key '*' gets all changes
const subscribers = new Map();

/**
 * Subscribe to state changes.
 * @param {string|string[]} keys - State key(s) to watch, or '*' for all
 * @param {function} callback - Called with (newValue, key, fullState)
 * @returns {function} Unsubscribe function
 */
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

/**
 * Update state and notify subscribers.
 * Only notifies if value actually changed (shallow comparison).
 * @param {object} patch - Partial state to merge
 */
function update(patch) {
  const changed = [];

  for (const [key, value] of Object.entries(patch)) {
    if (state[key] !== value) {
      state[key] = value;
      changed.push(key);
    }
  }

  if (changed.length === 0) return;

  // Notify specific key subscribers
  for (const key of changed) {
    const subs = subscribers.get(key);
    if (subs) {
      for (const cb of subs) {
        try { cb(state[key], key, state); }
        catch (e) { console.error(`State subscriber error [${key}]:`, e); }
      }
    }
  }

  // Notify wildcard subscribers
  const wildcard = subscribers.get('*');
  if (wildcard) {
    for (const cb of wildcard) {
      try { cb(state, changed); }
      catch (e) { console.error('State wildcard subscriber error:', e); }
    }
  }
}

/**
 * Batch update for history — avoids triggering per-element.
 * Pushes to history arrays and notifies once.
 */
function pushHistory(trainLoss, testLoss, accuracy) {
  state.history.train_loss.push(trainLoss);
  state.history.test_loss.push(testLoss);
  state.history.accuracy.push(accuracy);

  // Notify history subscribers
  const subs = subscribers.get('history');
  if (subs) {
    for (const cb of subs) {
      try { cb(state.history, 'history', state); }
      catch (e) { console.error('History subscriber error:', e); }
    }
  }
}

/**
 * Replace entire history (on init or training complete).
 */
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

/**
 * Update FPS tracking.
 */
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