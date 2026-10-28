# Meal Calorie Helper — Guide & Viva Prep

## 1. Quick start (Windows, ~2 min)

```powershell
# 1. Create + activate a virtual env
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure keys
copy .env.example .env
#    edit .env -> set GROQ_API_KEY and/or GEMINI_API_KEY
#    optional: LLM_PROVIDER=groq|gemini|auto, LLM_MODEL=<override>

# 4. Run the server
python -m uvicorn api_server:app --port 8000

# 5. Open the app
#    http://localhost:8000      (health check: /api/health)
```

Run the tests (no key or network needed; the LLM client is mocked):

```powershell
python -m pytest
```

CLI alternative: `python main.py` (text loop that prints the agent trace).

---

## 2. Architecture

```
Browser (web/index.html + app.js + styles.css)
   │  fetch /api/chat , /api/state , /api/budget   (SSE: /api/chat/stream)
   ▼
FastAPI  api_server.py ──► MealCalorieAgent (app/agent.py)
                                │  plan-act loop (≤ MAX_AGENT_STEPS)
                                ├─► LLM  (OpenAI-compatible client: Groq or Gemini)
                                └─► tools (app/tools.py) ──► MemoryStore (app/memory.py)
```

**LLM layer (after tuning).** `app/config.py` selects the provider via
`LLM_PROVIDER` (`groq` | `gemini` | `auto`) and an optional `LLM_MODEL`
override. A single `openai` SDK client is used for both providers because both
expose OpenAI-compatible chat/tool-calling endpoints:

- Groq: `https://api.groq.com/openai/v1`
- Gemini: `https://generativelanguage.googleapis.com/v1beta/openai/`

Exact default models and `auto` selection order: see `app/config.py`.

### File-by-file

| File | Role |
|---|---|
| `app/config.py` | Loads `.env`; provider, API key, model, `MAX_AGENT_STEPS` (default 8). Fails fast if no key. |
| `app/agent.py` | The agent. `TOOLS` (JSON schemas), `SYSTEM_PROMPT`, `looks_irrelevant()` scope guard, `_build_cards()` deterministic UI cards, `MealCalorieAgent.run_stream()` (generator of trace events) and `run()` (wrapper returning the final payload). `execute_tool()` dispatches model tool calls to Python. |
| `app/tools.py` | `CALORIE_DATABASE` (~45 foods), `lookup_calories`, `add_meal`, `remove_meal`, `set_calorie_budget`. Pure functions returning dicts with `success`/`found`. |
| `app/memory.py` | `MemoryStore`: per-`session_id` `SessionMemory` (budget, list of `Meal`, day). Thread-safe via `Lock`; auto-resets on a new day; `summary()` yields budget/consumed/remaining/meals. |
| `api_server.py` | FastAPI: `/api/chat`, `/api/chat/stream` (SSE `trace`/`done`/`error` events), `/api/state`, `/api/budget`, `/api/meals` (POST/DELETE), `/api/reset`, `/api/health`; serves `web/` at `/static` and `index.html` at `/`. |
| `web/` | Vanilla HTML/CSS/JS single-page UI: chat, meal list, budget dashboard, trace panel. |
| `main.py` | Terminal REPL around `MealCalorieAgent`. |
| `tests/` | pytest suites for tools, memory, agent loop, API — LLM mocked. |
| `vercel.json` | Vercel deployment config (serverless → memory is non-persistent there). |

### One agent turn (`run_stream`)

1. Empty message → canned reply. 2. `looks_irrelevant()` regex scope check →
polite refusal, **no LLM call**. 3. Read memory (`memory_store.summary`) →
`read_memory_context` trace step. 4. Build messages: system prompt + one
`CURRENT MEMORY` system message + user message. 5. Loop up to
`MAX_AGENT_STEPS`: call LLM with `tools=TOOLS, tool_choice="auto"`; if no tool
calls → final answer; else append the assistant tool-call message, execute each
tool, append `role: tool` results, **refresh the memory snapshot in place**,
repeat. 6. Build cards from tool results + memory (not from model prose) and
return `answer, trace, cards, state`. Hitting the cap returns a `SYSTEM_LIMIT`
message.

---

## 3. Viva Q&A

### A. Agent concepts
1. **What makes this an agent rather than a chatbot?** The LLM decides *which tool to call and when*, sees results, and decides the next action in a loop (`run_stream`). The app only executes what the model requests; state is changed by tools, not by text.
2. **Which tools does it have?** `lookup_calories`, `add_meal` (required), plus extensions `remove_meal`, `set_calorie_budget`. Defined as JSON schemas in `TOOLS`, implemented in `app/tools.py`.
3. **Why tools instead of asking the LLM for numbers?** Grounding and auditability: calories come from `CALORIE_DATABASE` (labelled `estimated=False`) or are explicitly flagged as LLM estimates; totals are computed in code, not by the model.
4. **How does the agent "plan"?** Implicitly via the model: multi-food input ("rice and chicken") yields several tool calls, possibly across iterations; the system prompt describes the desired sequence (lookup → add → report).
5. **How do you stop infinite loops?** `MAX_AGENT_STEPS` (env, default 8) bounds the `for` loop; exceeding it returns a `SYSTEM_LIMIT` final step.

### B. Tool calling
6. **How does tool calling work end to end?** Request includes `tools=TOOLS`; the response `message.tool_calls` holds name + JSON-string arguments; we `json.loads`, call `execute_tool`, and send back a `role: "tool"` message with matching `tool_call_id`.
7. **What if the model emits malformed arguments?** `json.JSONDecodeError` is caught → `arguments = {}` and a `tool_arguments_error` trace step; the tool then returns `success: False` ("Food name is required"), which the model sees and can retry.
8. **What if a tool throws?** `execute_tool` wraps everything in try/except and returns `{"success": False, "message": "Tool execution error: …"}`.
9. **Unknown foods?** `lookup_calories` returns `found: False` with guidance; the model supplies an estimate and calls `add_meal(estimated=True, source=…)`. `add_meal` without calories on an unknown food fails rather than guessing. This was the documented "honest failure" fix.
10. **Why `lookup_calories` has fuzzy matching?** Substring match against the DB returns `estimated=True, source="approximate database match"` so "grilled chicken wrap" still gets a grounded base value, flagged as approximate.

### C. Memory
11. **Where is memory and how is it keyed?** `MemoryStore` in `app/memory.py`, a dict of `session_id → SessionMemory`; the browser sends its own `session_id`, giving isolation per tab/user.
12. **How is memory read back?** At the start of each turn (`read_memory_context`) and again after every tool round, written into a single `CURRENT MEMORY` system message (`messages[1]`) so the model never sees stale duplicates.
13. **Why one refreshed snapshot rather than appending?** Avoids conflicting old numbers confusing the model; the prompt tells it to quote numbers only from the latest block or tool results.
14. **Thread safety?** A `threading.Lock` guards mutations; FastAPI sync endpoints run in a threadpool.
15. **Persistence?** None — in-process, lost on restart; resets automatically each day (`_get_session`). Production fix: SQLite/Redis.

### D. Plan-act loop
16. **Walk through "I ate an apple and set my budget to 2000".** Iter 1: model calls `lookup_calories(apple)` + `set_calorie_budget(2000)`; iter 2: `add_meal(apple, 95)`; iter 3: final text. Cards: `meal_logged` + `budget_set`.
17. **Can it call several tools in one iteration?** Yes — `for call in tool_calls` executes each and appends each result.
18. **How do you distinguish "ate" from "can I eat"?** System prompt rules + examples ("Do NOT add planned food"); verified through tool-call traces, not keyword code.

### E. Scope guard
19. **How is off-topic handled?** `looks_irrelevant()`: if no food keyword (`FOOD_PATTERNS`) and a blocked topic (`IRRELEVANT_PATTERNS`: python, weather…) matches → canned refusal before any LLM call (saves cost/latency, deterministic). The prompt also covers other off-topic asks.
20. **Weakness?** Regex lists are brittle; a stronger design is an LLM/classifier gate or embedding similarity.

### F. Deterministic cards
21. **What are cards and why deterministic?** `_build_cards()` builds `meal_logged`, `meals_logged`, `budget_set`, `meal_removed`, `breakdown` (fits/over verdict), `status` from tool results and memory deltas — so numbers on screen can't be hallucinated.
22. **How is the "fits / over" verdict computed?** Sum of looked-up calories vs `state_before["remaining"]`; `unknown` when no budget.
23. **Why `_clean_number`?** Normalizes floats (95.0 → 95) for display.

### G. API & streaming
24. **Streaming design?** `/api/chat/stream` returns `text/event-stream`; `run_stream` is a generator yielding `("trace", step)` then `("done", result)`, mapped to SSE `event: trace` / `event: done` (`event: error` on exceptions).
25. **Why a generator?** The UI can show the agent's steps live (agent trace) while the LLM is still working, and `run()` reuses it for the non-streaming endpoint.
26. **How is budget set via API?** `/api/budget` builds a natural-language message and runs the agent, so it goes through the same tool path.
27. **Static serving?** `/static` mounts `web/`, `/` returns `index.html`; one service serves API + UI (relative `/api/...` paths).

### H. Testing
28. **How do you test an LLM app without a key?** `tests/conftest.py` sets a fake `GROQ_API_KEY`; tests `patch.object(agent_instance, "client", fake)` with scripted responses (`_fake_client`) to drive multi-step loops, memory read-back, and the step-limit.
29. **What's covered?** Tools (unknown food doesn't hard-fail), memory isolation/persistence, agent loop, API routes.
30. **Limits of mock tests?** They verify our orchestration, not model behaviour; live runs (notebook / manual demo) cover that.

### I. Groq vs Gemini
31. **Why support both?** Resilience and choice: both offer OpenAI-compatible endpoints, so one `openai` client with a different `base_url`/key/model serves either (`LLM_PROVIDER`, `LLM_MODEL`; `auto` picks by available key — see `app/config.py`).
32. **Trade-offs?** Groq: very low latency, open-weight models (e.g. gpt-oss). Gemini: strong reasoning/multimodal, generous free tier, but its OpenAI-compat layer is newer — tool-call edge cases (e.g. reasoning/thought signatures on newer models) need testing.
33. **Why the OpenAI SDK instead of `groq`/`google-genai`?** One code path, mocks unchanged (`client.chat.completions.create`), trivial provider swap.

### J. Limitations & improvements
34. **Limitations?** In-memory state; small DB; estimates approximate; regex scope guard; no auth; Vercel serverless can't hold memory across invocations.
35. **What would you improve?** Persist to SQLite/Redis; integrate a nutrition API (USDA/Open Food Facts); portion/quantity handling; LLM-based scope classifier; auth + rate limiting; streaming tokens; evaluation set of prompts with expected tool calls.
36. **Safety stance?** Informational only; prompt states estimates aren't medical advice; the agent never claims an action unless the tool succeeded.

---

## 4. 3-minute live demo script

| Time | Say / type | Show |
|---|---|---|
| 0:00 | Open `http://localhost:8000`; toggle the **agent trace** panel. | UI, empty state |
| 0:20 | `My calorie budget is 2000 calories.` | `set_calorie_budget` tool call → budget card |
| 0:45 | `I had oatmeal and a banana for breakfast.` | 2× `lookup_calories`, 2× `add_meal`; `meals_logged` card; totals update |
| 1:15 | `Can I have an apple and yogurt as a snack?` | lookups only, **not added**; `breakdown` card with fits/over |
| 1:40 | `I ate dragon fruit cheesecake. Add it.` | not in DB → LLM estimate flagged "est." (the documented failure fix) |
| 2:05 | `How many calories do I have left?` | answer from memory (read_memory_context in trace) |
| 2:25 | `Remove the banana.` | `remove_meal` → `meal_removed` card |
| 2:40 | `Write me a Python script.` | scope guard: refused without an LLM call |
| 2:50 | Point to code: `run_stream` loop in `app/agent.py`, `TOOLS`, `memory_store`. | Wrap-up |

Backup if the network fails: `python -m pytest` (mocked agent loop passes offline).
