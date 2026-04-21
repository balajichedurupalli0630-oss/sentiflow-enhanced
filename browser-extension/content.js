/**
 * SentiFlow 3.0 — Universal Content Script
 *
 * FIXES APPLIED:
 *  - Changed all #sf-root references to #sentiflow-root (matches CSS)
 *  - Consistent element ID usage throughout
 */

(function () {
  'use strict';

  // Prevent double-injection
  if (window.__sentiflowLoaded) return;
  window.__sentiflowLoaded = true;

  // ── Configuration ──────────────────────────────────────────────────────────

  const CFG = {
    wsUrl: 'ws://localhost:8000/ws/analyze',
    debounceMs: 550,
    minChars: 15,
    maxChars: 2000,
    hideDelayMs: 2200,
  };

  // ── Emotion display config ─────────────────────────────────────────────────

  const EMOTIONS = [
    { key: 'joy',          label: 'Joy',          color: '#d97706' },
    { key: 'trust',        label: 'Trust',        color: '#22c55e' },
    { key: 'anticipation', label: 'Anticipation', color: '#f97316' },
    { key: 'surprise',     label: 'Surprise',     color: '#db2777' },
    { key: 'sadness',      label: 'Sadness',      color: '#3b82f6' },
    { key: 'anger',        label: 'Anger',        color: '#ef4444' },
    { key: 'fear',         label: 'Fear',         color: '#a855f7' },
    { key: 'disgust',      label: 'Disgust',      color: '#84cc16' },
  ];

  // ── State ──────────────────────────────────────────────────────────────────

  let ws = null;
  let reconnectTimer = null;
  let debounceTimer = null;
  let hideTimer = null;
  let sentAt = 0;
  let currentEl = null;
  let panelVisible = false;
  let panelPinned = false;
  const attached = new WeakSet();

  // ── Panel Build ────────────────────────────────────────────────────────────

  function buildPanel() {
    if (document.getElementById('sentiflow-root')) return;

    const emotionRows = EMOTIONS.map(e => `
      <div class="sf-emotion-row" data-key="${e.key}">
        <span class="sf-emotion-name">${e.label}</span>
        <div class="sf-bar-track">
          <div class="sf-bar-fill" data-color="${e.color}" style="background:${e.color};width:0%"></div>
        </div>
        <span class="sf-emotion-pct">—</span>
      </div>`).join('');

    const html = `
      <div class="sf-panel" id="sf-panel">
        <div class="sf-header" id="sf-header">
          <div class="sf-brand">
            <div class="sf-brand-dot sf-connecting" id="sf-dot"></div>
            <span class="sf-brand-name">SentiFlow</span>
          </div>
          <div class="sf-header-actions">
            <div class="sf-btn" id="sf-pin" title="Pin panel">⊕</div>
            <div class="sf-btn" id="sf-close" title="Close">✕</div>
          </div>
        </div>

        <div class="sf-primary">
          <div class="sf-primary-eyebrow">Primary emotion</div>
          <div class="sf-primary-name sf-placeholder" id="sf-ename">Waiting…</div>
          <div class="sf-primary-row">
            <span class="sf-confidence-pill" id="sf-conf"></span>
            <span class="sf-sentiment-badge" id="sf-badge"></span>
          </div>
        </div>

        <div class="sf-emotions" id="sf-ebar">${emotionRows}</div>

        <div class="sf-metrics">
          <div class="sf-metric">
            <div class="sf-metric-label sf-estimated">Formality</div>
            <div class="sf-metric-value" id="sf-formality">—</div>
          </div>
          <div class="sf-metric">
            <div class="sf-metric-label sf-estimated">Clarity</div>
            <div class="sf-metric-value" id="sf-clarity">—</div>
          </div>
          <div class="sf-metric">
            <div class="sf-metric-label">Words</div>
            <div class="sf-metric-value" id="sf-words">—</div>
          </div>
        </div>

        <div class="sf-suggestions" id="sf-sug" style="display:none"></div>

        <div class="sf-footer">
          <span class="sf-status-text" id="sf-status">Start typing to analyze</span>
          <span class="sf-latency-text" id="sf-latency"></span>
        </div>
      </div>`;

    const root = document.createElement('div');
    root.id = 'sentiflow-root';
    root.innerHTML = html;
    document.documentElement.appendChild(root);

    document.getElementById('sf-close').addEventListener('click', () => {
      panelPinned = false;
      hidePanel(true);
    });

    document.getElementById('sf-pin').addEventListener('click', () => {
      panelPinned = !panelPinned;
      document.getElementById('sf-pin').style.color = panelPinned ? '#22c55e' : '';
    });
  }

  // ── Panel Show/Hide ────────────────────────────────────────────────────────

  function showPanel() {
    clearTimeout(hideTimer);
    const panel = document.getElementById('sf-panel');
    if (!panel || panelVisible) return;
    panel.classList.add('sf-visible');
    panelVisible = true;
  }

  function hidePanel(immediate = false) {
    if (panelPinned) return;
    clearTimeout(hideTimer);
    const doHide = () => {
      const panel = document.getElementById('sf-panel');
      if (panel) panel.classList.remove('sf-visible');
      panelVisible = false;
    };
    if (immediate) { doHide(); return; }
    hideTimer = setTimeout(doHide, CFG.hideDelayMs);
  }

  // ── Panel Update ───────────────────────────────────────────────────────────

  function setAnalyzing() {
    const panel = document.getElementById('sf-panel');
    if (!panel) return;
    document.getElementById('sf-status').textContent = 'Analyzing…';
    panel.querySelectorAll('.sf-bar-fill').forEach(b => b.classList.add('sf-shimmer'));
  }

  function updatePanel(data) {
    const panel = document.getElementById('sf-panel');
    if (!panel) return;

    panel.querySelectorAll('.sf-bar-fill').forEach(b => b.classList.remove('sf-shimmer'));

    const ename = document.getElementById('sf-ename');
    ename.textContent = capitalize(data.primary_emotion);
    ename.classList.remove('sf-placeholder');

    const emotionCfg = EMOTIONS.find(e => e.key === data.primary_emotion);
    if (emotionCfg) ename.style.color = emotionCfg.color;

    const conf = document.getElementById('sf-conf');
    conf.textContent = `${data.emotion_score}% confidence`;

    const badge = document.getElementById('sf-badge');
    badge.textContent = (data.sentiment || '').toUpperCase();
    badge.className = `sf-sentiment-badge ${data.sentiment || ''}`;

    const scores = data.emotion_scores || {};
    const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    const top3 = new Set(sorted.slice(0, 3).map(e => e[0]));

    EMOTIONS.forEach(({ key }) => {
      const row = panel.querySelector(`[data-key="${key}"]`);
      if (!row) return;
      const score = scores[key] ?? 0;
      const isTop = top3.has(key);

      row.querySelector('.sf-bar-fill').style.width = `${score}%`;
      const pctEl = row.querySelector('.sf-emotion-pct');
      pctEl.textContent = `${Math.round(score)}%`;
      pctEl.classList.toggle('sf-top', isTop);
      row.querySelector('.sf-emotion-name').classList.toggle('sf-top', isTop);
    });

    const fmt = (v) => v === null || v === undefined ? '—' : v;
    document.getElementById('sf-formality').innerHTML =
      `${fmt(data.formality_score)}<span class="sf-unit">%</span>`;
    document.getElementById('sf-clarity').innerHTML =
      `${fmt(data.clarity_score)}<span class="sf-unit">%</span>`;
    document.getElementById('sf-words').textContent = data.word_count ?? '—';

    const sugDiv = document.getElementById('sf-sug');
    sugDiv.innerHTML = '';
    const suggestions = (data.suggestions || []).filter(Boolean);
    if (suggestions.length > 0) {
      suggestions.slice(0, 3).forEach(s => {
        const d = document.createElement('div');
        d.className = 'sf-suggestion';
        d.textContent = s;
        sugDiv.appendChild(d);
      });
      sugDiv.style.display = 'flex';
    } else {
      sugDiv.style.display = 'none';
    }

    const latency = sentAt ? Date.now() - sentAt : 0;
    document.getElementById('sf-status').textContent = 'Analysis complete';
    document.getElementById('sf-latency').textContent = latency ? `${latency}ms` : '';

    showPanel();
  }

  // ── WebSocket ──────────────────────────────────────────────────────────────

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

    try {
      ws = new WebSocket(CFG.wsUrl);
      setDot('connecting');

      ws.onopen = () => {
        setDot('connected');
        clearTimeout(reconnectTimer);
        updateStatus('Connected');
      };

      ws.onclose = () => {
        setDot('disconnected');
        reconnectTimer = setTimeout(connect, 4000);
      };

      ws.onerror = () => ws.close();

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          // Ignore server keepalive pings — they carry no analysis data
          if (data.type === 'ping') return;
          if (data.error) {
            updateStatus(data.message || 'Analysis error');
            return;
          }
          // Discard out-of-order responses from rapid typing
          if (data.requestId && data.requestId !== currentRequestId) return;
          updatePanel(data);
        } catch (_) {
          updateStatus('Response parse error');
        }
      };

    } catch (_) {
      reconnectTimer = setTimeout(connect, 4000);
    }
  }

  function setDot(state) {
    const dot = document.getElementById('sf-dot');
    if (!dot) return;
    dot.className = 'sf-brand-dot';
    if (state === 'connecting') dot.classList.add('sf-connecting');
    else if (state === 'disconnected') dot.classList.add('sf-disconnected');
  }

  function updateStatus(text) {
    const el = document.getElementById('sf-status');
    if (el) el.textContent = text;
  }

  let currentRequestId = 0;

  function sendText(text) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    currentRequestId = Date.now();
    sentAt = currentRequestId;
    ws.send(JSON.stringify({ text: text.slice(0, CFG.maxChars), requestId: currentRequestId }));
    setAnalyzing();
    showPanel();
  }

  // ── Text Extraction ────────────────────────────────────────────────────────

  function extractText(el) {
    if (!el) return '';
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') return el.value || '';
    return (el.innerText || el.textContent || '').trim();
  }

  // ── Sensitive Field Guard ──────────────────────────────────────────────────

  const BLOCKED_TYPES = new Set(['password', 'hidden', 'file', 'submit', 'button', 'reset', 'checkbox', 'radio', 'range', 'color']);
  const BLOCKED_NAMES = ['password', 'passwd', 'card', 'cardnum', 'cvv', 'cvc', 'ssn', 'pin', 'credit', 'debit', 'otp', 'token', 'secret'];

  function isSensitive(el) {
    if (BLOCKED_TYPES.has((el.type || '').toLowerCase())) return true;
    const attrs = ((el.name || '') + (el.id || '') + (el.placeholder || '') + (el.autocomplete || '')).toLowerCase();
    return BLOCKED_NAMES.some(n => attrs.includes(n));
  }

  // ── Eligible Field Check ───────────────────────────────────────────────────

  function isTextField(el) {
    if (!el || el.nodeType !== 1) return false;
    const tag = el.tagName;
    if (tag === 'TEXTAREA') return true;
    if (tag === 'INPUT') {
      const t = (el.type || 'text').toLowerCase();
      return ['text', 'email', 'search', 'url', 'tel', ''].includes(t);
    }
    if (el.isContentEditable) return true;
    const role = (el.getAttribute('role') || '').toLowerCase();
    if (role === 'textbox' || role === 'combobox') return true;
    if (el.getAttribute('aria-multiline') === 'true') return true;
    return false;
  }

  // ── Debounced Analysis ─────────────────────────────────────────────────────

  function scheduleAnalysis(el) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const text = extractText(el);
      if (text.length < CFG.minChars) {
        if (text.length === 0) hidePanel(true);
        return;
      }
      sendText(text);
    }, CFG.debounceMs);
  }

  // ── Element Attachment ─────────────────────────────────────────────────────

  function attach(el) {
    if (attached.has(el)) return;
    if (!isTextField(el)) return;
    if (isSensitive(el)) return;
    attached.add(el);

    const onInput = () => {
      currentEl = el;
      scheduleAnalysis(el);
    };

    el.addEventListener('input', onInput, { passive: true });
    el.addEventListener('keyup', onInput, { passive: true });
    el.addEventListener('compositionend', onInput, { passive: true });

    el.addEventListener('focus', () => {
      clearTimeout(hideTimer);
      currentEl = el;
      const text = extractText(el);
      if (text.length >= CFG.minChars) {
        scheduleAnalysis(el);
      }
    }, { passive: true });

    el.addEventListener('blur', () => {
      if (currentEl === el) hidePanel();
    }, { passive: true });
  }

  // ── DOM Scan (including shadow DOM) ───────────────────────────────────────

  function scan(root) {
    if (!root) return;
    try {
      root.querySelectorAll('input, textarea, [contenteditable], [role="textbox"], [role="combobox"], [aria-multiline]')
        .forEach(attach);
    } catch (_) {}

    try {
      root.querySelectorAll('*').forEach(el => {
        if (el.shadowRoot) scan(el.shadowRoot);
      });
    } catch (_) {}
  }

  // ── MutationObserver ──────────────────────────────────────────────────────

  const observer = new MutationObserver(mutations => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType !== 1) continue;
        attach(node);
        scan(node);
      }
    }
  });

  // ── SPA Navigation Support ────────────────────────────────────────────────

  let lastUrl = location.href;
  const navObserver = new MutationObserver(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      setTimeout(() => scan(document.body), 500);
    }
  });

  // ── Init ──────────────────────────────────────────────────────────────────

  // ── Panel Draggable ─────────────────────────────────────────────────────────

  function makeDraggable() {
    const header = document.getElementById('sf-header');
    const root = document.getElementById('sentiflow-root');
    if (!header || !root) return;

    // Restore previously saved position
    chrome.storage.local.get(['sfPanelLeft', 'sfPanelTop'], (pos) => {
      if (pos.sfPanelLeft !== undefined && pos.sfPanelTop !== undefined) {
        root.style.setProperty('left',   pos.sfPanelLeft + 'px', 'important');
        root.style.setProperty('top',    pos.sfPanelTop  + 'px', 'important');
        root.style.setProperty('right',  'auto', 'important');
        root.style.setProperty('bottom', 'auto', 'important');
      }
    });

    let dragging = false;
    let startX, startY, initX, initY;

    header.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      dragging = true;
      const rect = root.getBoundingClientRect();
      initX = rect.left;
      initY = rect.top;
      startX = e.clientX;
      startY = e.clientY;
      root.style.setProperty('left',   initX + 'px', 'important');
      root.style.setProperty('top',    initY + 'px', 'important');
      root.style.setProperty('right',  'auto', 'important');
      root.style.setProperty('bottom', 'auto', 'important');
    });

    document.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      root.style.setProperty('left', (initX + e.clientX - startX) + 'px', 'important');
      root.style.setProperty('top',  (initY + e.clientY - startY) + 'px', 'important');
    });

    document.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      // Persist the final position so it survives page navigation
      const rect = root.getBoundingClientRect();
      chrome.storage.local.set({ sfPanelLeft: Math.round(rect.left), sfPanelTop: Math.round(rect.top) });
    });
  }

  function init() {
    buildPanel();
    makeDraggable();
    connect();
    scan(document.body || document.documentElement);

    observer.observe(document.body || document.documentElement, {
      childList: true,
      subtree: true,
    });

    if (document.head) {
      navObserver.observe(document.head, { childList: true });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  function capitalize(s) {
    if (!s) return '';
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

})();