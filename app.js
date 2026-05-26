/**
 * app.js — Socra's Twin Frontend
 *
 * Features:
 *  - Session persistence (localStorage)
 *  - Exponential backoff retry on 429 errors
 *  - Auto-resizing textarea
 *  - Accessible live-region updates
 *  - Suggestion chips after each AI response
 *  - XP / character progression system
 *  - Character customisation modal
 *  - Australian-friendly UI copy
 */

// ── API config ────────────────────────────────────────────────────────────

const API_BASE = window.location.origin.includes("localhost")
  ? "http://localhost:8000"
  : "";

const ENDPOINTS = {
  chat:   `${API_BASE}/api/chat`,
  reset:  `${API_BASE}/api/reset`,
  health: `${API_BASE}/api/health`,
};

const MAX_CLIENT_RETRIES  = 2;
const BASE_RETRY_DELAY_MS = 5000;

// ── XP / Levels / Avatars ─────────────────────────────────────────────────

const LEVELS = [
  { level: 1, name: "Learner",      xpRequired: 0},
  { level: 2, name: "Explorer",     xpRequired: 100},
  { level: 3, name: "Deep Thinker", xpRequired: 250},
  { level: 4, name: "Seeker",       xpRequired: 500},
  { level: 5, name: "Sage",         xpRequired: 1000},
  { level: 6, name: "Philosopher",  xpRequired: 2000},
];

const SOCRA_AVATAR = '<img src="soc.png" alt="Socra" />';

const XP_PER_MESSAGE    = 15;  // earned each time you ask something
const XP_PER_SUGGESTION = 5;   // bonus for clicking a follow-up suggestion

// ── App state ─────────────────────────────────────────────────────────────

const state = {
  sessionId: localStorage.getItem("socra_session_id") || null,
  turnCount: 0,
  isLoading: false,
};

// Character — persisted to localStorage
const character = {
  get xp()    { return parseInt(localStorage.getItem("socra_xp")  || "0"); },
  get avatar() { return SOCRA_AVATAR; },                                     // always 🦉
  get name()  { return localStorage.getItem("socra_name")          || "Explorer"; },

  addXP(n) {
    const val = this.xp + n;
    localStorage.setItem("socra_xp", String(val));
  },
  setName(n) { localStorage.setItem("socra_name", n); },
};

// ── DOM refs ──────────────────────────────────────────────────────────────

const chat         = document.getElementById("chat");
const input        = document.getElementById("input");
const sendBtn      = document.getElementById("sendBtn");
const resetBtn     = document.getElementById("resetBtn");
const charCount    = document.getElementById("charCount");
const banner       = document.getElementById("banner");
const bannerText   = document.getElementById("bannerText");
const bannerIcon   = document.getElementById("bannerIcon");
const bannerClose  = document.getElementById("bannerClose");
const charPanel    = document.getElementById("charPanel");
const charName     = document.getElementById("charName");
const charLevel    = document.getElementById("charLevel");
const charXP       = document.getElementById("charXP");
const xpBarFill    = document.getElementById("xpBarFill");
const xpFloat      = document.getElementById("xpFloat");
const levelupToast = document.getElementById("levelupToast");
const charModal    = document.getElementById("charModal");
const closeModal   = document.getElementById("closeModal");
const nameInput    = document.getElementById("nameInput");
const saveCharacter= document.getElementById("saveCharacter");
const welcomeAvatar= document.getElementById("welcomeAvatar");
const welcomeName  = document.getElementById("welcomeName");

// ── Level helpers ─────────────────────────────────────────────────────────

function getCurrentLevel(xp = character.xp) {
  let current = LEVELS[0];
  for (const lvl of LEVELS) {
    if (xp >= lvl.xpRequired) current = lvl;
    else break;
  }
  return current;
}

function getNextLevel(xp = character.xp) {
  const current = getCurrentLevel(xp);
  const idx = LEVELS.findIndex(l => l.level === current.level);
  return idx < LEVELS.length - 1 ? LEVELS[idx + 1] : null;
}

function xpProgress(xp = character.xp) {
  const current  = getCurrentLevel(xp);
  const next     = getNextLevel(xp);
  if (!next) return 100;
  return ((xp - current.xpRequired) / (next.xpRequired - current.xpRequired)) * 100;
}

// ── Character display ─────────────────────────────────────────────────────

function updateCharacterDisplay() {
  const xp      = character.xp;
  const level   = getCurrentLevel(xp);
  const next    = getNextLevel(xp);
  const pct     = xpProgress(xp);

  // Header panel
  charName.textContent   = character.name;
  charLevel.textContent  = level.name;
  charXP.textContent     = `${xp} XP`;
  xpBarFill.style.width  = `${pct}%`;

  // Welcome screen (if still visible)
  if (welcomeAvatar) welcomeAvatar.innerHTML = SOCRA_AVATAR;
  if (welcomeName)   welcomeName.textContent   = character.name;

  // Modal (if open)
  const modalAvatar   = document.getElementById("modalAvatarDisplay");
  const modalLevel    = document.getElementById("modalLevelName");
  const modalXpCount  = document.getElementById("modalXpCount");
  const modalFill     = document.getElementById("modalProgressFill");
  const modalLabel    = document.getElementById("modalProgressLabel");

  if (modalAvatar) modalAvatar.innerHTML = character.avatar;
  if (modalLevel)  modalLevel.textContent  = level.name;
  if (modalXpCount) modalXpCount.textContent = `${xp} XP total`;
  if (modalFill)   modalFill.style.width   = `${pct}%`;
  if (modalLabel) {
    modalLabel.textContent = next
      ? `${next.xpRequired - xp} XP to ${next.name}`
      : "🎉 Max level — you're a legend!";
  }
}

// ── XP gain ───────────────────────────────────────────────────────────────

function addXP(amount, label = null) {
  const oldLevel = getCurrentLevel(character.xp);
  character.addXP(amount);
  const newLevel = getCurrentLevel(character.xp);

  updateCharacterDisplay();
  showXPFloat(label || `+${amount} XP`);

  if (newLevel.level > oldLevel.level) {
    showLevelUp(newLevel);
  }
}

function showXPFloat(text) {
  xpFloat.textContent = text;
  xpFloat.classList.remove("xp-float--show");
  // Force reflow so the animation restarts
  void xpFloat.offsetWidth;
  xpFloat.classList.add("xp-float--show");
  setTimeout(() => xpFloat.classList.remove("xp-float--show"), 1900);
}

function showLevelUp(level) {
  document.getElementById("levelupEmoji").textContent = level.badge;
  document.getElementById("levelupSub").textContent   = `You're now a ${level.name}! Ripper! 🎉`;
  levelupToast.hidden = false;
  // Next frame so the transition fires after display kicks in
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      levelupToast.classList.add("levelup-toast--show");
    });
  });
  setTimeout(() => {
    levelupToast.classList.remove("levelup-toast--show");
    setTimeout(() => { levelupToast.hidden = true; }, 450);
  }, 3800);
}

// ── Character modal ───────────────────────────────────────────────────────

function openModal() {
  nameInput.value = character.name;
  updateCharacterDisplay();
  charModal.hidden = false;
  nameInput.focus();
  document.body.style.overflow = "hidden";
}

function closeModalFn() {
  charModal.hidden = true;
  document.body.style.overflow = "";
}

// ── Init ──────────────────────────────────────────────────────────────────

async function init() {
  updateCharacterDisplay();
  bindEvents();
  autoResizeTextarea();
  await checkHealth();
}

// ── Events ────────────────────────────────────────────────────────────────

function bindEvents() {
  sendBtn.addEventListener("click", handleSend);

  input.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  });

  input.addEventListener("input", () => {
    autoResizeTextarea();
    updateCharCount();
    sendBtn.disabled = input.value.trim().length === 0 || state.isLoading;
  });

  resetBtn.addEventListener("click", handleReset);
  bannerClose.addEventListener("click", hideBanner);

  // Character modal
  charPanel.addEventListener("click", openModal);
  closeModal.addEventListener("click", closeModalFn);
  charModal.addEventListener("click", e => { if (e.target === charModal) closeModalFn(); });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && !charModal.hidden) closeModalFn(); });

  saveCharacter.addEventListener("click", () => {
    const newName = nameInput.value.trim() || "Explorer";
    character.setName(newName);
    updateCharacterDisplay();
    closeModalFn();
  });

  // Starter prompts
  bindStarters();
}

function bindStarters() {
  document.querySelectorAll(".starter").forEach(btn => {
    btn.addEventListener("click", () => {
      input.value = btn.dataset.text;
      autoResizeTextarea();
      updateCharCount();
      sendBtn.disabled = false;
      input.focus();
      handleSend();
    });
  });
}

// ── Health check ──────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res  = await fetch(ENDPOINTS.health, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (!data.gemini_ready) {
      showBanner("warning", "⚙️", "Backend is up, but the Gemini API key isn't set yet.");
    }
  } catch { /* non-fatal */ }
}

// ── Send ──────────────────────────────────────────────────────────────────

async function handleSend() {
  const text = input.value.trim();
  if (!text || state.isLoading) return;

  // Dismiss welcome screen
  const welcome = document.getElementById("welcome");
  if (welcome) {
    welcome.style.opacity = "0";
    welcome.style.transform = "translateY(-10px)";
    welcome.style.transition = "opacity 0.2s ease, transform 0.2s ease";
    setTimeout(() => welcome.remove(), 220);
  }

  hideBanner();
  appendUserMessage(text);
  input.value = "";
  autoResizeTextarea();
  updateCharCount();
  sendBtn.disabled = true;

  const thinking = appendThinking();
  setLoading(true);

  try {
    const response = await sendWithRetry(text);
    thinking.remove();
    appendAIMessage(response.response, response.suggestions || []);

    // Award XP for asking a question
    addXP(XP_PER_MESSAGE, `+${XP_PER_MESSAGE} XP 🎉`);

    state.sessionId = response.session_id;
    localStorage.setItem("socra_session_id", state.sessionId);
    state.turnCount = response.turn_count;
  } catch (err) {
    thinking.remove();
    handleSendError(err);
  } finally {
    setLoading(false);
    sendBtn.disabled = input.value.trim().length === 0;
    scrollToBottom();
  }
}

// ── API with retry ────────────────────────────────────────────────────────

async function sendWithRetry(text, attempt = 0) {
  let res;
  try {
    res = await fetch(ENDPOINTS.chat, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text, session_id: state.sessionId }),
      signal:  AbortSignal.timeout(45000),
    });
  } catch (err) {
    if (err.name === "TimeoutError") throw { type: "timeout" };
    throw { type: "network", message: "Couldn't reach the server. Check your internet!" };
  }

  if (res.ok) return res.json();

  const data   = await res.json().catch(() => ({}));
  const detail = data.detail || {};

  if (res.status === 429) {
    if (attempt < MAX_CLIENT_RETRIES) {
      const waitMs = Math.min(BASE_RETRY_DELAY_MS * Math.pow(2, attempt), 60000);
      showBanner("warning", `Hang tight — retrying in ${Math.round(waitMs / 1000)}s… (attempt ${attempt + 1}/${MAX_CLIENT_RETRIES})`);
      await sleep(waitMs);
      hideBanner();
      return sendWithRetry(text, attempt + 1);
    }
    throw { type: "rate_limit", message: detail.error || "Rate limit hit. Give it a sec and try again!", retry_after: detail.retry_after || 30 };
  }
  if (res.status === 400)  throw { type: "safety",     message: detail.error || "That topic hit a safety filter. Maybe try asking it differently?" };
  if (res.status === 422)  throw { type: "validation", message: "Please type a valid message." };
  if (res.status === 503)  throw { type: "config",     message: "Server isn't configured yet — check that GEMINI_API_KEY is set in .env." };
  throw { type: "api_error", message: detail.error || `Server error (${res.status}).` };
}

// ── Error handling ────────────────────────────────────────────────────────

function handleSendError(err) {
  console.error("Chat error:", err);
  const msgs = {
    rate_limit:  err.message || "Too many requests — wait a moment and try again.",
    safety:      err.message || "That message was blocked by safety filters.",
    validation:  "Please type a valid message.",
    timeout:     "Request timed out — the server might be waking up. Try again!",
    network:     "Can't reach the server. Check your internet connection.",
    config:      "Backend not configured. Set GEMINI_API_KEY in your .env file.",
    api_error:   err.message || "Something went wrong. Give it another go!",
  };
  const type    = err.type || "api_error";
  const icon    = type === "rate_limit" ? "⏳" : "⚠️";
  const variant = type === "rate_limit" ? "warning" : "error";
  showBanner(variant, icon, msgs[type] || msgs.api_error);
}

// ── DOM helpers ───────────────────────────────────────────────────────────

function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message message--user";
  div.setAttribute("role", "article");
  div.setAttribute("aria-label", "You");
  div.innerHTML = `
    <div class="message__avatar" aria-hidden="true" style="color:#fff;font-size:15px;font-weight:800;">${getInitial()}</div>
    <div>
      <div class="message__bubble">${escapeHtml(text)}</div>
      <div class="message__meta">${formatTime()}</div>
    </div>
  `;
  chat.appendChild(div);
  scrollToBottom();
}

function appendAIMessage(text, suggestions = []) {
  // AI bubble
  const div = document.createElement("div");
  div.className = "message message--ai";
  div.setAttribute("role", "article");
  div.setAttribute("aria-label", "Socra");
  div.innerHTML = `
    <div class="message__avatar" aria-hidden="true">${character.avatar}</div>
    <div>
      <div class="message__bubble">${formatAIText(text)}</div>
      <div class="message__meta">${formatTime()}</div>
    </div>
  `;
  chat.appendChild(div);

  // Suggestions
  if (suggestions.length > 0) {
    appendSuggestions(suggestions);
  }

  scrollToBottom();
}

function appendSuggestions(suggestions) {
  const div = document.createElement("div");
  div.className = "suggestions";
  div.setAttribute("aria-label", "Follow-up suggestions");
  div.innerHTML = `
    <p class="suggestions__label">Keep wondering…</p>
    <div class="suggestions__chips">
      ${suggestions.map(s => `<button class="suggestion-chip" data-text="${escapeAttr(s)}">${escapeHtml(s)}</button>`).join("")}
    </div>
  `;

  div.querySelectorAll(".suggestion-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      input.value = btn.dataset.text;
      autoResizeTextarea();
      updateCharCount();
      sendBtn.disabled = false;
      input.focus();
      // Small curiosity bonus
      addXP(XP_PER_SUGGESTION, `+${XP_PER_SUGGESTION} XP — curious! 🔍`);
      handleSend();
    });
  });

  chat.appendChild(div);
}

function appendThinking() {
  const div = document.createElement("div");
  div.className = "thinking";
  div.setAttribute("aria-label", "Socra is thinking");
  div.innerHTML = `
    <div class="thinking__avatar" aria-hidden="true">${character.avatar}</div>
    <div class="thinking__bubble" aria-hidden="true">
      <div class="thinking__dot"></div>
      <div class="thinking__dot"></div>
      <div class="thinking__dot"></div>
    </div>
  `;
  chat.appendChild(div);
  scrollToBottom();
  return div;
}

// ── Banner ────────────────────────────────────────────────────────────────

function showBanner(variant, icon, text) {
  banner.className   = `banner banner--${variant}`;
  bannerIcon.textContent = icon;
  bannerText.textContent = text;
}

function hideBanner() {
  banner.className = "banner banner--hidden";
}

// ── Reset ─────────────────────────────────────────────────────────────────

async function handleReset() {
  if (state.isLoading) return;

  if (state.sessionId) {
    try {
      await fetch(ENDPOINTS.reset, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ session_id: state.sessionId }),
      });
    } catch { /* non-fatal */ }
  }

  state.sessionId = null;
  state.turnCount = 0;
  localStorage.removeItem("socra_session_id");
  hideBanner();

  // Re-render welcome screen
  chat.innerHTML = `
    <div class="welcome" id="welcome">
      <div class="welcome__avatar" aria-hidden="true" id="welcomeAvatar">${character.avatar}</div>
      <h2 class="welcome__heading">Back again, <span id="welcomeName">${escapeHtml(character.name)}</span>!</h2>
      <p class="welcome__sub">Love the curiosity! What shall we explore this time?</p>
      <div class="welcome__starters">
        <button class="starter" data-text="Why is the sky blue?">Why is the sky blue?</button>
        <button class="starter" data-text="Is time travel actually possible?">Is time travel possible?</button>
        <button class="starter" data-text="What makes something fair or unfair?">What makes things fair?</button>
        <button class="starter" data-text="Why do we dream when we sleep?">Why do we dream?</button>
        <button class="starter" data-text="How do planes actually stay up in the air?">How do planes fly?</button>
        <button class="starter" data-text="Does the universe go on forever?">Does space go on forever?</button>
      </div>
      <p class="welcome__xp-hint">You've earned <strong>${character.xp} XP</strong> so far!</p>
    </div>
  `;

  bindStarters();
  input.focus();
}

// ── Utilities ─────────────────────────────────────────────────────────────

function setLoading(val) {
  state.isLoading = val;
  sendBtn.disabled = val;
  input.readOnly = val;
  document.body.style.cursor = val ? "wait" : "";
}

function autoResizeTextarea() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

function updateCharCount() {
  const n = input.value.length;
  charCount.textContent = n > 1500 ? `${n} / 2000` : "";
  charCount.className   = n > 1800 ? "char-count char-count--warn" : "char-count";
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function formatTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.appendChild(document.createTextNode(text));
  return d.innerHTML;
}

function escapeAttr(text) {
  return text.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function formatAIText(text) {
  return escapeHtml(text).replace(/\n/g, "<br>");
}

function getInitial() {
  return escapeHtml(character.name.charAt(0).toUpperCase());
}

// ── Boot ──────────────────────────────────────────────────────────────────

init();
