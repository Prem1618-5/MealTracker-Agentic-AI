import json
import re
import time
from typing import Any, Generator

from openai import OpenAI

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_AGENT_STEPS
from .memory import memory_store
from .tools import (
    add_meal,
    lookup_calories,
    remove_meal,
    set_calorie_budget,
)


# ============================================================
# LLM CLIENT (Groq or Gemini via OpenAI-compatible endpoint)
# ============================================================

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


# ============================================================
# TOOLS AVAILABLE TO THE AGENT
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_calories",
            "description": (
                "Look up approximate calories for a food. "
                "Use this when calorie information is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "food": {
                        "type": "string",
                        "description": "Name of the food.",
                    }
                },
                "required": ["food"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_meal",
            "description": (
                "Add food that the user actually ate to today's "
                "meal memory. For uncommon foods, use a reasonable "
                "approximate calorie value and mark it as estimated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "food": {
                        "type": "string",
                        "description": "Food eaten by the user.",
                    },
                    "calories": {
                        "type": "number",
                        "description": (
                            "Approximate calories for the food."
                        ),
                    },
                    "estimated": {
                        "type": "boolean",
                        "description": (
                            "True when the calorie value is estimated."
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Source or explanation of the calorie value."
                        ),
                    },
                },
                "required": ["food"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_meal",
            "description": (
                "Remove a previously recorded meal from today's memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "food": {
                        "type": "string",
                        "description": "Food to remove.",
                    }
                },
                "required": ["food"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_calorie_budget",
            "description": (
                "Set or update the user's daily calorie budget."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {
                        "type": "number",
                        "description": "Daily calorie budget.",
                    }
                },
                "required": ["budget"],
            },
        },
    },
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Meal Calorie Helper, an agentic AI assistant for tracking food,
meals, calories, and daily calorie budgets.

Your purpose is to help the user with:

- meals
- food
- calories
- calorie tracking
- daily calorie budgets
- breakfast
- lunch
- dinner
- snacks
- meal recommendations
- deciding whether planned food fits within the remaining budget

You are informational only.
Calorie values are approximate educational estimates and are not medical
advice.

============================================================
AGENTIC BEHAVIOUR
============================================================

You are a real tool-using AI agent.

When appropriate:

1. Understand the user's request.
2. Read the current memory.
3. Decide what action is needed.
4. Call the appropriate tool.
5. Inspect the tool result.
6. Call another tool if necessary.
7. Produce the final answer.

Do not pretend that a tool was called.

Only tell the user that something was added, removed, or saved when
the corresponding tool actually succeeded.

============================================================
MEMORY
============================================================

The current session memory is supplied to you in a system message
titled "CURRENT MEMORY". This block is refreshed with the latest
state before every model call, so it is always current.

Use it for:

- today's meals
- total calories consumed
- daily calorie budget
- remaining calories

Never invent previous meals.

When you quote any number (budget, consumed, remaining), quote it
ONLY from the latest CURRENT MEMORY block or from a tool result you
just received. Ignore numbers that appear anywhere else in the
conversation, including earlier turns.

============================================================
LOOKUP CALORIES
============================================================

Use lookup_calories whenever you need calorie information.

Example:

User:
"How many calories are in an apple?"

Action:
lookup_calories("apple")

============================================================
ADDING FOOD
============================================================

Only add food when the user actually ate it or explicitly asks
you to add it.

Examples:

"I ate an apple."
→ Add the apple.

"I had rice and chicken for lunch."
→ Add both foods.

"Can I eat an apple?"
→ Do NOT add the apple.

"Would yogurt be a good snack?"
→ Do NOT add the yogurt.

============================================================
UNCOMMON FOOD
============================================================

Do NOT reject a food because it is missing from the database.

Example:

"I ate dragon fruit cheesecake. Add it."

First:

lookup_calories("dragon fruit cheesecake")

If it is not found:

1. Make a reasonable approximate calorie estimate.
2. Clearly tell the user the value is estimated.
3. Call add_meal using the estimate.
4. Do not reject the food.

============================================================
SNACK DECISIONS
============================================================

When the user asks whether a planned snack fits:

1. Look up every food.
2. Add the calories together.
3. Check the current memory.
4. Determine the remaining calorie budget.
5. Compare the snack calories with the remaining calories.
6. Explain the result.

Example:

User:
"Can I have an apple and yogurt?"

Use:

lookup_calories("apple")
lookup_calories("yogurt")

Then calculate the combined calories.

IMPORTANT:

Do NOT add planned food to memory unless the user says they
actually ate it.

============================================================
DAILY STATUS
============================================================

When the user asks:

"How many calories have I eaten?"

or:

"How many calories do I have left?"

Use the actual memory state.

Report:

- daily budget
- consumed calories
- remaining calories

Never invent numbers.

============================================================
SETTING THE CALORIE BUDGET
============================================================

When the user provides a calorie budget, ALWAYS call the
set_calorie_budget tool.

Example:

User:
"My calorie budget is 2000 calories."

Action:

set_calorie_budget(2000)

Do not merely tell the user the budget was saved.

The tool must actually update memory.

After the tool succeeds, report:

- budget
- consumed calories
- remaining calories

============================================================
REMOVING FOOD
============================================================

When the user asks to remove a recorded meal:

Example:

"Remove the banana."

Action:

remove_meal("banana")

Then report the updated calorie state.

============================================================
RECOMMENDATIONS
============================================================

When the user asks for recommendations, use clear formatting.

Example:

**Dinner ideas under 600 calories**

1. Grilled chicken and rice
   - About 500 kcal
   - High-protein option

2. Vegetable wrap
   - About 420 kcal
   - Lighter option

3. Dal with roti
   - About 450 kcal
   - Vegetarian option

Avoid giant paragraphs.

============================================================
OUT-OF-SCOPE QUESTIONS
============================================================

You are focused on:

- food
- meals
- calories
- calorie tracking
- meal planning
- snacks
- nutrition-related meal questions

For clearly unrelated requests, politely explain that you are
focused on meal and calorie tracking.

However:

DO NOT reject a food or meal request simply because the food is
uncommon or missing from the database.

============================================================
FINAL ANSWERS
============================================================

Be friendly and concise. Two to five short sentences is usually
right.

Formatting rules:

- Use short markdown: **bold** for key numbers, bullet lists of at
  most 4 items, and short paragraphs.
- NEVER draw tables — no markdown pipe tables and no ASCII tables
  (never produce lines containing "|"). The app renders structured
  nutrition cards itself from your tool results, so the text never
  needs one.
- NEVER use markdown headings (#, ##, ###). Use a short **bold
  lead-in** instead.
- The app already shows the user updated totals, budget, and
  remaining calories after every change. Do not restate full meal
  lists or repeat every number; add only what those cards do not
  show.

Always base tracking answers on actual memory and tool results.
"""


# ============================================================
# SCOPE DETECTION
# ============================================================

FOOD_PATTERNS = [
    r"\bfood\b",
    r"\bmeal\b",
    r"\bmeals\b",
    r"\beat\b",
    r"\bate\b",
    r"\beating\b",
    r"\bcalorie\b",
    r"\bcalories\b",
    r"\bbreakfast\b",
    r"\blunch\b",
    r"\bdinner\b",
    r"\bsnack\b",
    r"\bdiet\b",
    r"\bnutrition\b",
    r"\bhungry\b",
    r"\bapple\b",
    r"\bbanana\b",
    r"\byogurt\b",
    r"\brice\b",
    r"\bchicken\b",
    r"\bpizza\b",
]


IRRELEVANT_PATTERNS = [
    r"\bpython\b",
    r"\bdjango\b",
    r"\bjavascript\b",
    r"\bhtml\b",
    r"\bcss\b",
    r"\bjava program\b",
    r"\bc\+\+\b",
    r"\bphysics\b",
    r"\bchemistry\b",
    r"\bpolitics\b",
    r"\bweather\b",
    r"\bmovie\b",
    r"\bcricket\b",
    r"\bfootball\b",
]


def looks_irrelevant(text: str) -> bool:
    text = text.lower()

    has_food_context = any(
        re.search(pattern, text)
        for pattern in FOOD_PATTERNS
    )

    if has_food_context:
        return False

    return any(
        re.search(pattern, text)
        for pattern in IRRELEVANT_PATTERNS
    )


# ============================================================
# CARD BUILDING (deterministic — computed from tool results and
# memory state, never parsed from model prose)
# ============================================================

def _clean_number(value: Any) -> Any:
    """Round to a clean int when near-integral, else 2 decimals."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    nearest = round(number)
    if abs(number - nearest) < 0.05:
        return int(nearest)
    return round(number, 2)


def _dedupe_lookups(lookups: list[dict]) -> list[dict]:
    """Keep the last lookup per food (the model may retry)."""
    by_food: dict[str, dict] = {}
    for item in lookups:
        by_food[item["food"].lower().strip()] = item
    return list(by_food.values())


def _build_cards(
    *,
    state_before: dict,
    state_after: dict,
    added: list[dict],
    removed: list[dict],
    budget_value: float | None,
    lookups: list[dict],
) -> list[dict]:
    cards: list[dict] = []

    consumed = _clean_number(state_after.get("consumed"))
    remaining = _clean_number(state_after.get("remaining"))

    # 1. Successful adds → meal_logged / meals_logged
    if len(added) == 1:
        meal = added[0]
        cards.append({
            "type": "meal_logged",
            "food": meal["food"],
            "calories": _clean_number(meal["calories"]),
            "estimated": meal["estimated"],
            "consumed": consumed,
            "remaining": remaining,
        })
    elif len(added) > 1:
        cards.append({
            "type": "meals_logged",
            "items": [
                {
                    "food": meal["food"],
                    "calories": _clean_number(meal["calories"]),
                    "estimated": meal["estimated"],
                }
                for meal in added
            ],
            "consumed": consumed,
            "remaining": remaining,
        })

    # 2. Successful budget set → budget_set
    if budget_value is not None:
        cards.append({
            "type": "budget_set",
            "budget": _clean_number(budget_value),
            "consumed": consumed,
            "remaining": remaining,
        })

    # 3. Successful removes → meal_removed
    for meal in removed:
        cards.append({
            "type": "meal_removed",
            "food": meal["food"],
            "calories": _clean_number(meal["calories"]),
            "consumed": consumed,
            "remaining": remaining,
        })

    # 4. Lookups with no add → breakdown card with fit verdict
    successful_lookups = [
        item for item in _dedupe_lookups(lookups)
        if item["found"] and item["calories"] is not None
    ]
    if successful_lookups and not added:
        items = [
            {
                "food": item["food"],
                "calories": _clean_number(item["calories"]),
                "estimated": item["estimated"],
            }
            for item in successful_lookups
        ]
        total = sum(float(item["calories"]) for item in items)
        remaining_before = state_before.get("remaining")

        if remaining_before is None:
            verdict = "unknown"
            remaining_after = None
        else:
            verdict = (
                "fits"
                if total <= float(remaining_before)
                else "over"
            )
            remaining_after = float(remaining_before) - total

        cards.append({
            "type": "breakdown",
            "title": " + ".join(item["food"] for item in successful_lookups),
            "items": items,
            "total": _clean_number(total),
            "remaining_before": _clean_number(remaining_before),
            "remaining_after": _clean_number(remaining_after),
            "verdict": verdict,
        })

    # 5. Status card only when state changed and nothing else fits
    if not cards and state_before != state_after:
        cards.append({
            "type": "status",
            "budget": _clean_number(state_after.get("budget")),
            "consumed": consumed,
            "remaining": remaining,
        })

    return cards


# ============================================================
# AGENT CLASS
# ============================================================

class MealCalorieAgent:

    def __init__(self):
        self.client = client
        self.model = LLM_MODEL

    # ========================================================
    # MEMORY
    # ========================================================

    def get_state(self, session_id: str) -> dict:
        return memory_store.summary(session_id)

    def get_memory_context(self, session_id: str) -> str:
        return json.dumps(self.get_state(session_id), indent=2)

    # ========================================================
    # TOOL EXECUTION
    # ========================================================

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str,
    ) -> dict:
        try:
            if tool_name == "lookup_calories":
                food = str(arguments.get("food", "")).strip()
                if not food:
                    return {
                        "success": False,
                        "message": "Food name is required.",
                    }
                return lookup_calories(food)

            if tool_name == "add_meal":
                food = str(arguments.get("food", "")).strip()
                if not food:
                    return {
                        "success": False,
                        "message": "Food name is required.",
                    }
                return add_meal(
                    session_id=session_id,
                    food=food,
                    calories=arguments.get("calories"),
                    estimated=arguments.get("estimated", False),
                    source=arguments.get("source", "agent estimate"),
                )

            if tool_name == "remove_meal":
                food = str(arguments.get("food", "")).strip()
                if not food:
                    return {
                        "success": False,
                        "message": "Food name is required.",
                    }
                return remove_meal(session_id=session_id, food=food)

            if tool_name == "set_calorie_budget":
                budget = arguments.get("budget")
                if budget is None:
                    return {
                        "success": False,
                        "message": "Calorie budget is required.",
                    }
                return set_calorie_budget(
                    session_id=session_id,
                    budget=float(budget),
                )

            return {
                "success": False,
                "message": f"Unknown tool requested: {tool_name}",
            }

        except Exception as exc:
            return {
                "success": False,
                "message": f"Tool execution error: {exc}",
            }

    # ========================================================
    # MAIN AGENT LOOP (streaming)
    # ========================================================

    def run_stream(
        self,
        user_message: str,
        session_id: str = "default",
    ) -> Generator[tuple[str, dict], None, None]:
        """Yield ("trace", step) as each step happens, then ("done", result).

        The done payload carries answer, trace, cards, state, session_id.
        """
        started = time.perf_counter()
        step_counter = 0

        def make_step(**fields: Any) -> dict:
            nonlocal step_counter
            step_counter += 1
            step = {
                "id": f"s{step_counter}",
                "ts": round((time.perf_counter() - started) * 1000, 1),
            }
            step.update(fields)
            return step

        user_message = (user_message or "").strip()

        # ----------------------------------------------------
        # EMPTY MESSAGE
        # ----------------------------------------------------
        if not user_message:
            yield ("done", {
                "answer": (
                    "Tell me what you ate, your calorie budget, "
                    "or what food you're considering."
                ),
                "trace": [],
                "cards": [],
                "state": self.get_state(session_id),
                "session_id": session_id,
            })
            return

        # ----------------------------------------------------
        # SCOPE CHECK
        # ----------------------------------------------------
        if looks_irrelevant(user_message):
            scope_step = make_step(step="scope_check", result="irrelevant")
            yield ("trace", scope_step)
            yield ("done", {
                "answer": (
                    "I'm focused on meals, food, calories, calorie "
                    "budgets, and meal recommendations. Ask me something "
                    "related to your meals."
                ),
                "trace": [scope_step],
                "cards": [],
                "state": self.get_state(session_id),
                "session_id": session_id,
            })
            return

        # ----------------------------------------------------
        # READ MEMORY
        # ----------------------------------------------------
        trace: list[dict] = []
        state_before = self.get_state(session_id)

        memory_context = self.get_memory_context(session_id)
        memory_step = make_step(
            step="read_memory_context",
            result=json.loads(memory_context),
        )
        trace.append(memory_step)
        yield ("trace", memory_step)

        # The memory snapshot lives in exactly ONE system message and
        # is refreshed in place after every tool round, so the model
        # never sees stale duplicated snapshots.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "CURRENT MEMORY:\n\n" + memory_context},
            {"role": "user", "content": user_message},
        ]

        # Card tracking
        added_meals: list[dict] = []
        removed_meals: list[dict] = []
        budget_value: float | None = None
        lookups: list[dict] = []

        # ----------------------------------------------------
        # PLAN / ACT LOOP
        # ----------------------------------------------------
        for iteration in range(MAX_AGENT_STEPS):
            iteration_step = make_step(
                step="agent_iteration",
                iteration=iteration + 1,
            )
            trace.append(iteration_step)
            yield ("trace", iteration_step)

            call_started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except Exception as exc:
                error_step = make_step(step="llm_error", result=str(exc))
                trace.append(error_step)
                yield ("trace", error_step)
                yield ("done", {
                    "answer": (
                        "I couldn't connect to the AI agent. "
                        "Please check your LLM API key/provider configuration."
                    ),
                    "trace": trace,
                    "cards": [],
                    "state": self.get_state(session_id),
                    "session_id": session_id,
                })
                return

            iteration_step["duration_ms"] = round(
                (time.perf_counter() - call_started) * 1000, 1
            )

            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls

            # ------------------------------------------------
            # FINAL ANSWER
            # ------------------------------------------------
            if not tool_calls:
                answer = (
                    assistant_message.content
                    or "I couldn't generate a response."
                )
                final_step = make_step(
                    step="final_answer",
                    result=answer,
                    duration_ms=iteration_step["duration_ms"],
                )
                trace.append(final_step)
                yield ("trace", final_step)

                state_after = self.get_state(session_id)
                cards = _build_cards(
                    state_before=state_before,
                    state_after=state_after,
                    added=added_meals,
                    removed=removed_meals,
                    budget_value=budget_value,
                    lookups=lookups,
                )
                yield ("done", {
                    "answer": answer,
                    "trace": trace,
                    "cards": cards,
                    "state": state_after,
                    "session_id": session_id,
                })
                return

            # ------------------------------------------------
            # ADD ASSISTANT TOOL CALL MESSAGE
            # ------------------------------------------------
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            })

            # ------------------------------------------------
            # EXECUTE TOOLS
            # ------------------------------------------------
            for call in tool_calls:
                tool_name = call.function.name
                raw_arguments = call.function.arguments or "{}"

                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        arguments = {}
                except json.JSONDecodeError:
                    arguments = {}
                    arg_error_step = make_step(
                        step="tool_arguments_error",
                        tool=tool_name,
                        raw_arguments=raw_arguments,
                    )
                    trace.append(arg_error_step)
                    yield ("trace", arg_error_step)

                call_step = make_step(
                    step="tool_call",
                    tool=tool_name,
                    arguments=arguments,
                )
                trace.append(call_step)
                yield ("trace", call_step)

                exec_started = time.perf_counter()
                result = self.execute_tool(
                    tool_name=tool_name,
                    arguments=arguments,
                    session_id=session_id,
                )
                result_step = make_step(
                    step="tool_result",
                    tool=tool_name,
                    result=result,
                    duration_ms=round(
                        (time.perf_counter() - exec_started) * 1000, 1
                    ),
                )
                trace.append(result_step)
                yield ("trace", result_step)

                # Track for cards
                if tool_name == "lookup_calories":
                    # lookup results carry "found" instead of "success"
                    lookups.append({
                        "food": result.get("food") or arguments.get("food", ""),
                        "calories": result.get("calories"),
                        "estimated": bool(result.get("estimated")),
                        "found": bool(result.get("found")),
                    })

                if result.get("success"):
                    if tool_name == "add_meal":
                        added_meals.append({
                            "food": result.get("food") or arguments.get("food", ""),
                            "calories": result.get("calories"),
                            "estimated": bool(result.get("estimated")),
                        })
                    elif tool_name == "remove_meal":
                        removed_meals.append({
                            "food": result.get("food", ""),
                            "calories": result.get("removed_calories"),
                        })
                    elif tool_name == "set_calorie_budget":
                        budget_value = result.get("budget")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                })

            # ------------------------------------------------
            # REFRESH THE SINGLE MEMORY SNAPSHOT IN PLACE
            # ------------------------------------------------
            latest_memory = self.get_memory_context(session_id)
            messages[1]["content"] = "CURRENT MEMORY:\n\n" + latest_memory

        # ----------------------------------------------------
        # MAX ITERATIONS
        # ----------------------------------------------------
        answer = (
            "I wasn't able to complete that request within the "
            "agent's allowed steps. Please try again."
        )
        final_step = make_step(
            step="final_answer",
            result=answer,
            decision_source="SYSTEM_LIMIT",
        )
        trace.append(final_step)
        yield ("trace", final_step)

        state_after = self.get_state(session_id)
        cards = _build_cards(
            state_before=state_before,
            state_after=state_after,
            added=added_meals,
            removed=removed_meals,
            budget_value=budget_value,
            lookups=lookups,
        )
        yield ("done", {
            "answer": answer,
            "trace": trace,
            "cards": cards,
            "state": state_after,
            "session_id": session_id,
        })

    # ========================================================
    # NON-STREAMING WRAPPER
    # ========================================================

    def run(
        self,
        user_message: str,
        session_id: str = "default",
    ) -> dict:
        result: dict | None = None
        for kind, payload in self.run_stream(
            user_message=user_message,
            session_id=session_id,
        ):
            if kind == "done":
                result = payload
        if result is None:
            raise RuntimeError("Agent stream ended without a done payload.")
        return result


# ============================================================
# GLOBAL AGENT INSTANCE
# ============================================================

agent = MealCalorieAgent()
