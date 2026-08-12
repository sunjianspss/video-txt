from __future__ import annotations

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TARGET_LANGUAGE = "Simplified Chinese"
DEFAULT_LANGUAGE_CODE = "zho"
DEFAULT_TERMS = ["Claude", "Claude Code", "Anthropic", "MCP"]
DEFAULT_BATCH_CHARS = 3200
DEFAULT_CONCURRENCY = 4
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

# Everything --provider fills in at once, so one flag replaces three.
PROVIDERS = {
    "openai": {
        "base_url": DEFAULT_BASE_URL,
        "api_key_env": DEFAULT_API_KEY_ENV,
        "model": None,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-v4-flash",
    },
}
