// popup.js - External script file (CSP-compliant)

document.addEventListener('DOMContentLoaded', () => {
  // Load settings
  chrome.storage.local.get(['enabled', 'serverUrl', 'minChars'], (data) => {
    document.getElementById('enabled').checked = data.enabled !== false;
    document.getElementById('serverUrl').value = data.serverUrl || 'ws://localhost:8000/ws/analyze';
    document.getElementById('minChars').value = data.minChars || 15;
    checkHealth(document.getElementById('serverUrl').value);
  });

  // Save button
  document.getElementById('save').addEventListener('click', () => {
    const btn = document.getElementById('save');
    const serverUrl = document.getElementById('serverUrl').value.trim();
    
    chrome.storage.local.set({
      enabled: document.getElementById('enabled').checked,
      serverUrl: serverUrl,
      minChars: parseInt(document.getElementById('minChars').value, 10) || 15,
    }, () => {
      btn.textContent = 'Saved ✓';
      btn.classList.add('saved');
      checkHealth(serverUrl);
      setTimeout(() => { 
        btn.textContent = 'Save settings'; 
        btn.classList.remove('saved'); 
      }, 1800);
    });
  });

  // Clear data
  document.getElementById('clear-data').addEventListener('click', (e) => {
    e.preventDefault();
    chrome.storage.local.clear(() => window.close());
  });
});

function checkHealth(wsUrl) {
  // Derive HTTP health endpoint from WebSocket URL
  const httpUrl = wsUrl
    .replace(/^wss:\/\//, 'https://')
    .replace(/^ws:\/\//, 'http://')
    .replace(/\/ws\/analyze$/, '/health');

  fetch(httpUrl, { signal: AbortSignal.timeout(4000) })
    .then(r => r.ok ? 'ok' : 'error')
    .catch(() => 'offline')
    .then(state => {
      const dot = document.getElementById('status-dot');
      const txt = document.getElementById('status-text');
      dot.className = 'status-dot' + (state === 'ok' ? ' ok' : ' err');
      txt.textContent = state === 'ok' ? 'Connected' : state === 'offline' ? 'Backend offline' : 'Error';
    });
}