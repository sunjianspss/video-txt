from __future__ import annotations

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TARGET_LANGUAGE = "Simplified Chinese"
DEFAULT_LANGUAGE_CODE = "zho"
DEFAULT_BATCH_CHARS = 3200
LOCAL_BATCH_CHARS = 800
DEFAULT_CONCURRENCY = 4
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_TIMEOUT = 180.0

# "auto" means: say nothing about reasoning and let the model do whatever it defaults to.
REASONING_EFFORTS = ("auto", "none", "low", "medium", "high")

# AI terms that stay in the original language in every translation.
# Add terms here to keep them untranslated for good; --preserve-term adds per-run ones.
DEFAULT_TERMS = [
    "Claude",
    "Claude Code",
    "Claude Cowork",
    "Anthropic",
    "MCP",
    "Agent",
    "Agentic",
    "OpenAI",
    "Codex",
    "AI",
    "token",
    "harness",
    "Transformers",
    "CLAUDE.md",
    "AGENT.md",
]

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
    # A model running on this machine: no bill, no key, and no fixed model name
    # either — whichever one LM Studio has loaded is the one we ask for.
    # Thinking is off by default: local reasoning models spend thousands of tokens
    # deliberating over a subtitle line, which is minutes per batch and no better a
    # translation. Pass --reasoning-effort to get it back.
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key_env": "LMSTUDIO_API_KEY",
        "model": None,
        "requires_key": False,
        "discover_model": True,
        "reasoning_effort": "none",
        # Small batches too: asked for 70 lines in one go, a local model returns 60 and
        # calls it done. It never gets the ids wrong, it just stops early.
        "batch_chars": LOCAL_BATCH_CHARS,
    },
}
