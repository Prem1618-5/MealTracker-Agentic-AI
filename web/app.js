const $ = (id) => document.getElementById(id);
const chatForm = $("chatForm"), chatInput = $("chatInput"), sendButton = $("sendButton");
const chatMessages = $("chatMessages"), mealList = $("mealList"), traceList = $("traceList");
const budgetValue = $("budgetValue"), consumedValue = $("consumedValue"), remainingValue = $("remainingValue");
const progressBar = $("progressBar"), progressRing = $("progressRing"), progressPercent = $("progressPercent");
const mealCount = $("mealCount"), manualMealButton = $("manualMealButton"), resetButton = $("resetButton");
const helpDialog = $("helpDialog");

let sessionId = localStorage.getItem("meal_helper_session");
if (!sessionId) { sessionId = crypto.randomUUID(); localStorage.setItem("meal_helper_session", sessionId); }

$("todayLabel").textContent = new Intl.DateTimeFormat(undefined, { weekday: "short", month: "short", day: "numeric" }).format(new Date());

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}
function formatText(text) {
  return escapeHtml(text).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/^#{1,3}\s+(.*)$/gm, "<strong>$1</strong>").replace(/\n/g, "<br>");
}
function scrollChat() {
  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}
function addMessage(role, text) {
  const el = document.createElement("article");
  const isUser = role === "user";
  el.className = `message ${isUser ? "user-message" : "agent-message"}`;
  const time = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(new Date());
  el.innerHTML = `<div class="avatar">${isUser ? "YOU" : "AI"}</div><div class="bubble"><div class="name">${isUser ? "YOU" : "MEALTRACKER"}</div><div class="message-text">${formatText(text)}</div><div class="message-time">${time}</div></div>`;
  chatMessages.append(el); requestAnimationFrame(scrollChat);
}
function showTyping() {
  const el = document.createElement("article"); el.id = "typingMessage"; el.className = "message agent-message typing";
  el.innerHTML = '<div class="avatar">AI</div><div class="bubble"><div class="name">MEALTRACKER</div><div class="message-text">Thinking through that…</div></div>';
  chatMessages.append(el); scrollChat();
}
function hideTyping() { $("typingMessage")?.remove(); }

function updateState(state) {
  if (!state) return;
  const budget = state.budget == null ? null : Number(state.budget);
  const consumed = Number(state.consumed || 0);
  const remaining = state.remaining == null ? null : Number(state.remaining);
  budgetValue.textContent = Number.isFinite(budget) ? Math.round(budget) : "—";
  consumedValue.textContent = Math.round(consumed);
  remainingValue.textContent = Number.isFinite(remaining) ? Math.round(remaining) : "—";
  const percent = Number.isFinite(budget) && budget > 0 ? Math.max(0, (consumed / budget) * 100) : 0;
  const shownPercent = Math.round(percent);
  progressBar.style.width = `${Math.min(100, percent)}%`;
  progressPercent.textContent = Number.isFinite(budget) && budget > 0 ? `${shownPercent}%` : "—";
  progressRing.style.strokeDashoffset = `${326.73 * (1 - Math.min(100, percent) / 100)}`;
  progressRing.style.stroke = percent > 100 ? "var(--orange)" : "";
  renderMeals(state.meals || []);
}
function renderMeals(meals) {
  mealCount.textContent = meals.length;
  if (!meals.length) { mealList.innerHTML = '<div class="empty-state"><span class="empty-icon" aria-hidden="true">+</span><strong>Your meal log is clear</strong><p>Meals you add will appear here.</p></div>'; return; }
  mealList.innerHTML = meals.map(meal => {
    const food = escapeHtml(meal.food || "Meal");
    const calories = Math.round(Number(meal.calories) || 0);
    return `<article class="meal-item"><span class="meal-icon" aria-hidden="true">${food.charAt(0).toUpperCase()}</span><div class="meal-info"><div class="meal-name">${food}${meal.estimated ? '<span class="estimated">est.</span>' : ""}</div><div class="meal-calories">${calories} kcal</div></div><span class="meal-time">TODAY</span></article>`;
  }).join("");
}
const traceLabels = { read_memory_context: "Reviewing today’s log", agent_iteration: "Considering your request", tool_call: "Checking nutrition details", tool_result: "Nutrition details received", tool_arguments_error: "Clarifying a detail", scope_check: "Checking your daily balance", llm_error: "Connection issue", final_answer: "Response ready", add_meal: "Meal added", lookup_calories: "Looking up calories" };
function traceDetail(item) {
  const bits = [];
  if (item.tool) bits.push(item.tool);
  if (item.arguments) bits.push(typeof item.arguments === "string" ? item.arguments : JSON.stringify(item.arguments));
  if (item.result) bits.push(typeof item.result === "string" ? item.result : JSON.stringify(item.result));
  return bits.join(" · ");
}
function renderTrace(trace) {
  if (!Array.isArray(trace) || !trace.length) return;
  traceList.innerHTML = trace.slice(-10).map(item => `<div class="trace-item" tabindex="0"><div class="trace-title">${escapeHtml(traceLabels[item.step] || item.step || "Working")}</div>${traceDetail(item) ? `<div class="trace-detail">${escapeHtml(traceDetail(item))}</div>` : ""}</div>`).join("");
  traceList.scrollTop = traceList.scrollHeight;
}
async function sendMessage(message) {
  const clean = message.trim(); if (!clean || sendButton.disabled) return;
  addMessage("user", clean); chatInput.value = ""; sendButton.disabled = true; showTyping();
  try {
    const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: clean, session_id: sessionId }) });
    if (!response.ok) throw new Error(`The service returned ${response.status}.`);
    const data = await response.json(); hideTyping(); addMessage("agent", data.reply || "I couldn’t generate a response."); updateState(data.state); renderTrace(data.trace);
  } catch (error) { hideTyping(); addMessage("agent", `I couldn’t reach MealTracker. ${error.message} Please try again.`); }
  finally { sendButton.disabled = false; chatInput.focus(); }
}
async function resetSession() {
  if (!confirm("Reset your goal and all meals for this session?")) return;
  try {
    const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: "Reset my session", session_id: sessionId }) });
    if (!response.ok) throw new Error(`The service returned ${response.status}.`);
    const data = await response.json(); updateState(data.state); traceList.innerHTML = '<div class="empty-trace">I’ll show the helpful steps I take here.</div>';
    chatMessages.innerHTML = '<article class="welcome-card"><span class="welcome-icon" aria-hidden="true">✓</span><div><h2>A fresh start.</h2><p>Set a calorie goal, log a meal, or ask a question whenever you’re ready.</p></div></article>';
  } catch (error) { addMessage("agent", `I couldn’t reset this session. ${error.message}`); }
}
async function loadState() { try { const response = await fetch(`/api/state?session_id=${encodeURIComponent(sessionId)}`); if (response.ok) updateState(await response.json()); } catch (_) { /* The chat remains usable when state is unavailable. */ } }
chatForm.addEventListener("submit", event => { event.preventDefault(); sendMessage(chatInput.value); });
document.querySelectorAll(".quick-btn").forEach(button => button.addEventListener("click", () => sendMessage(button.dataset.message)));
manualMealButton.addEventListener("click", () => { chatInput.value = "I ate "; chatInput.focus(); });
resetButton.addEventListener("click", resetSession);
$("helpButton").addEventListener("click", () => helpDialog.showModal());
$("closeHelp").addEventListener("click", () => helpDialog.close());
helpDialog.addEventListener("click", event => { if (event.target === helpDialog) helpDialog.close(); });
loadState(); chatInput.focus();
$("themeButton").addEventListener("click", () => { const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = next; try { localStorage.setItem("theme", next); } catch (_) {} });
