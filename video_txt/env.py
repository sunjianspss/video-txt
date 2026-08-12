from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_SECRETS_FILE = Path("~/.secrets")

ASSIGNMENT_PATTERN = re.compile(
    r"^(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$"
)


class CredentialError(RuntimeError):
    pass


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


def ensure_private_permissions(path: Path) -> None:
    if os.name != "posix":
        return
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise CredentialError(
            f"Refusing to read credentials from {path}: mode {mode:03o} allows other "
            f"users to read it. Tighten it with: chmod 600 {path}"
        )


def read_secrets_file(path: Path | None = None) -> dict[str, str]:
    """What the credentials file assigns, leaving the environment alone.

    Reading a key is not the same as handing it out. Exporting the whole file
    would pass every key in it to every subprocess the run goes on to spawn —
    ffmpeg, whisper, a cloning worker — none of which asked for any of them.
    """
    secrets_path = (path or DEFAULT_SECRETS_FILE).expanduser()
    if not secrets_path.is_file():
        return {}
    ensure_private_permissions(secrets_path)
    return parse_secrets_text(secrets_path.read_text(encoding="utf-8"))


def resolve_optional_key(env_name: str, *, secrets_file: Path | None = None) -> str:
    """The key if it can be found, empty when the caller can manage without one."""
    return os.environ.get(env_name) or read_secrets_file(secrets_file).get(env_name, "")


def resolve_api_key(env_name: str, *, secrets_file: Path | None = None) -> str:
    api_key = resolve_optional_key(env_name, secrets_file=secrets_file)
    if api_key:
        return api_key

    secrets_path = (secrets_file or DEFAULT_SECRETS_FILE).expanduser()
    raise CredentialError(
        f"Missing API key: {env_name} is not set.\n"
        f"Either export it in your shell, or add this line to {secrets_path}:\n"
        f"  export {env_name}='your-key'"
    )
