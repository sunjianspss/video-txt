from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_SECRETS_FILE = Path("~/.secrets")

ASSIGNMENT_PATTERN = re.compile(
    r"^(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$"
)


def parse_secrets_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ASSIGNMENT_PATTERN.match(stripped)
        if not match:
            continue
        value = match.group("value").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        values[match.group("name")] = value
    return values


def load_secrets_file(path: Path | None = None, *, override: bool = False) -> list[str]:
    secrets_path = (path or DEFAULT_SECRETS_FILE).expanduser()
    if not secrets_path.is_file():
        return []

    loaded: list[str] = []
    for name, value in parse_secrets_text(secrets_path.read_text(encoding="utf-8")).items():
        if override or not os.environ.get(name):
            os.environ[name] = value
            loaded.append(name)
    return loaded


def resolve_api_key(env_name: str, *, secrets_file: Path | None = None) -> str:
    api_key = os.environ.get(env_name)
    if api_key:
        return api_key

    load_secrets_file(secrets_file)
    api_key = os.environ.get(env_name)
    if api_key:
        return api_key

    secrets_path = (secrets_file or DEFAULT_SECRETS_FILE).expanduser()
    raise SystemExit(
        f"Missing API key: {env_name} is not set.\n"
        f"Either export it in your shell, or add this line to {secrets_path}:\n"
        f"  export {env_name}='your-key'"
    )
