"""Settings for subtitle generation, independent of execution and storage."""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import (
    DEFAULT_BATCH_CHARS,
    DEFAULT_CONCURRENCY,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TIMEOUT,
)
from .terminology import Terminology


@dataclass
class TranslationConfig:
    base_url: str
    api_key: str
    model: str
    target_language: str = DEFAULT_TARGET_LANGUAGE
    source_language: str | None = None
    preserve_terms: list[str] = field(default_factory=list)
    terminology: Terminology | None = None
    note: str | None = None
    # The lines are written to be spoken by a voice, not read off the screen.
    spoken: bool = False
    batch_chars: int = DEFAULT_BATCH_CHARS
    retries: int = 2
    concurrency: int = DEFAULT_CONCURRENCY
    context_cues: int = 3
    # Provider-specific thinking toggle. "auto" leaves the request untouched.
    thinking: str = "auto"
    # "auto" (or None) leaves reasoning_effort out of the request entirely.
    reasoning_effort: str = "auto"
    timeout: float = DEFAULT_TIMEOUT
    backoff_base: float = 1.5
    backoff_cap: float = 30.0
