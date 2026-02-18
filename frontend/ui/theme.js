/**
 * theme.js — Theme Manager
 *
 * Handles dark/light toggle with localStorage persistence.
 * Updates CSS data-theme attribute and notifies state.
 */

import { state, update } from '../state/appState.js';

let themeBtn = null;

function init() {
  const saved = localStorage.getItem('ns-theme') || 'dark';
  applyTheme(saved);

  themeBtn = document.getElementById('thBtn');
  if (themeBtn) {
    themeBtn.addEventListener('click', toggle);
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  update({ theme });
  updateButton(theme);
}

function toggle() {
  const next = state.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('ns-theme', next);
  applyTheme(next);
}

function updateButton(theme) {
  if (!themeBtn) return;
  themeBtn.innerHTML = theme === 'dark'
    ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>Theme'
    : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>Theme';
}

export { init, toggle };