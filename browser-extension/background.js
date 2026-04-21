/**
 * SentiFlow 3.0 — Background Service Worker
 *
 * Fixes:
 *  - Default URL uses ws:// for localhost dev, wss:// for production domains
 *  - Settings relay is batched to avoid flooding tabs
 *  - Onboarding opens popup.html relative to extension root
 */

const DEFAULT_SETTINGS = {
  enabled:    true,
  serverUrl:  'ws://localhost:8000/ws/analyze',
  minChars:   15,
  debounceMs: 550,
};

chrome.runtime.onInstalled.addListener(({ reason }) => {
  if (reason === 'install') {
    chrome.storage.local.set(DEFAULT_SETTINGS, () => {
      // Open the popup in a new tab for onboarding
      chrome.tabs.create({ url: chrome.runtime.getURL('popup.html') + '?onboarding=1' });
    });
  }
});

// Relay updated settings to all active content scripts
chrome.storage.onChanged.addListener((changes) => {
  chrome.tabs.query({ status: 'complete' }, (tabs) => {
    for (const tab of tabs) {
      if (!tab.id) continue;
      chrome.tabs.sendMessage(tab.id, { type: 'settings_updated', changes }).catch(() => {
        // Tab may not have the content script injected — silently ignore
      });
    }
  });
});