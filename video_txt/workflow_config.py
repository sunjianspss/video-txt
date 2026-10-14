"""Typed input shared by CLI and persisted project workflows."""

from __future__ import annotations

import argparse
import types
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import get_args, get_origin, get_type_hints


@dataclass
class WorkflowConfig:
    workflow: str = field(default="run", metadata={"choices": ("run", "dub")})
    video: Path = Path()
    subtitle: Path | None = None
    subtitle_output: Path | None = None
    video_output: Path | None = None
    output_dir: Path | None = None
    retranscribe: bool = False
    retranslate: bool = False
    whisper_model: str | None = None
    whisper_backend: str = field(
        default="auto", metadata={"choices": ("auto", "openai-whisper", "mlx-whisper")}
    )
    language: str | None = None
    device: str | None = None
    initial_prompt: str | None = None
    audio_stream: int | None = None
    refine_subtitles: bool = False
    whisper_args: list[str] = field(default_factory=list)
    skip_transcript_check: bool = False
    provider: str | None = field(
        default=None, metadata={"choices": ("deepseek", "lmstudio", "openai")}
    )
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    secrets_file: Path = Path("~/.secrets")
    target_language: str = "Simplified Chinese"
    source_language: str | None = None
    batch_chars: int | None = None
    retries: int = 2
    concurrency: int = 4
    context_cues: int = 3
    reasoning_effort: str | None = field(
        default=None, metadata={"choices": ("auto", "none", "low", "medium", "high", "max")}
    )
    timeout: float = 180.0
    resume: bool = True
    debug_dir: Path | None = None
    reuse: Path | None = None
    reuse_translation: Path | None = None
    note: str | None = None
    preserve_term: list[str] = field(default_factory=list)
    term_file: Path | None = None
    mux_mode: str = field(default="soft", metadata={"choices": ("soft", "hard")})
    language_code: str | None = None
    default_subtitle: bool = False
    subtitle_codec: str | None = None
    hard_subtitle_layout: str = field(
        default="normal", metadata={"choices": ("normal", "bottom-box", "top")}
    )
    hard_subtitle_font: str = "PingFang SC"
    hard_subtitle_font_size: int | None = None
    hard_subtitle_margin_v: int | None = None
    hard_subtitle_box_height: float = 0.22
    hard_subtitle_box_opacity: float = 0.68
    video_codec: str = "libx264"
    crf: int = 20
    preset: str = "medium"
    video_size: str | None = None
    ffmpeg_path: str | None = None
    overwrite_video: bool = False
    revisions: Path | None = None
    bilingual: bool = False
    bilingual_order: str = field(
        default="translation-first", metadata={"choices": ("translation-first", "source-first")}
    )
    dry_run: bool = False
    tts_engine: str | None = field(
        default=None,
        metadata={"choices": ("edge-tts", "say", "cosyvoice", "f5-tts", "index-tts")},
    )
    voice: str | None = None
    voice_unit: str = field(default="sentence", metadata={"choices": ("sentence", "line")})
    rate: str = "+0%"
    max_atempo: float = 1.35
    fit_duration: bool = False
    fit_tempo: float = 1.15
    fit_rounds: int = 2
    keep_bgm: bool = False
    separate_bgm: bool = False
    bgm_volume: float | None = None
    keep_dub_audio: bool = False
    soft_subtitle: bool = False
    tts_concurrency: int = 4
    cache_dir: Path | None = None
    prune_cache: bool = False
    audio_output: Path | None = None
    diarize: bool = False
    speaker_count: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    speaker_voices: list[str] = field(default_factory=list)
    diarize_model: str = "pyannote/speaker-diarization-3.1"
    hf_token_env: str = "HF_TOKEN"
    rediarize: bool = False
    clone_model: str | None = None
    clone_references: list[str] = field(default_factory=list)
    clone_repo: Path | None = None
    clone_python: str | None = None
    clone_device: str | None = None
    project_reuse_translation: bool = False
    thinking: str | None = field(
        default=None, metadata={"choices": ("auto", "enabled", "disabled")}
    )

    @classmethod
    def from_namespace(cls, args: argparse.Namespace, *, workflow: str) -> WorkflowConfig:
        names = {item.name for item in fields(cls)}
        return cls(**{**{k: v for k, v in vars(args).items() if k in names}, "workflow": workflow})

    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for item in fields(self):
            value = getattr(self, item.name)
            if not _matches_type(value, hints[item.name]):
                raise WorkflowConfigError(f"Invalid type for workflow setting {item.name}.")
            choices = item.metadata.get("choices")
            if value is not None and choices and value not in choices:
                raise WorkflowConfigError(
                    f"Invalid {item.name}: {value!r}. Expected one of {choices}."
                )


class WorkflowConfigError(ValueError):
    pass


def _matches_type(value: object, expected: object) -> bool:
    origin = get_origin(expected)
    if origin is types.UnionType:
        return any(_matches_type(value, alternative) for alternative in get_args(expected))
    if origin is list:
        return isinstance(value, list) and all(
            _matches_type(item, get_args(expected)[0]) for item in value
        )
    if expected is float:
        return type(value) in (int, float)
    if expected in (int, bool):
        return type(value) is expected
    return isinstance(value, expected)
