import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# LLM PROVIDER (Groq or Gemini, both via OpenAI-compatible APIs)
# ============================================================

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-3.8-flash",
    },
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


def _pick_provider() -> str:
    requested = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    if requested in PROVIDERS:
        return requested
    # auto: prefer Groq (fast), fall back to Gemini
    if GROQ_API_KEY:
        return "groq"
    if GEMINI_API_KEY:
        return "gemini"
    return "groq"


LLM_PROVIDER = _pick_provider()
_provider = PROVIDERS[LLM_PROVIDER]

LLM_BASE_URL = _provider["base_url"]
LLM_API_KEY = os.getenv(_provider["key_env"], "").strip()

# LLM_MODEL overrides; legacy GROQ_MODEL / GROQ_TEXT_MODEL still honoured.
LLM_MODEL = (
    os.getenv("LLM_MODEL")
    or (os.getenv("GROQ_MODEL") or os.getenv("GROQ_TEXT_MODEL")
        if LLM_PROVIDER == "groq" else None)
    or _provider["default_model"]
).strip()

# Backwards-compatible aliases
GROQ_MODEL = LLM_MODEL

try:
    MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8").strip())
except ValueError:
    MAX_AGENT_STEPS = 8

APP_NAME = "Meal Calorie Helper"


# Tests set a dummy GROQ_API_KEY; a real run without any key must fail loudly.
if not LLM_API_KEY:
    raise RuntimeError(
        f"{_provider['key_env']} is missing. Add it to {ENV_FILE} "
        "(see .env.example) or set LLM_PROVIDER=groq|gemini."
    )
