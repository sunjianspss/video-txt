from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .mux import MuxOptions, run_mux
from .subtitles import translated_subtitle_path
from .transcribe import TranscribeOptions, run_transcribe
from .translate import TranslationConfig, translate_subtitle_file


@dataclass
class TranscribeStage:
    model: str | None = None
    backend: str = "auto"
    language: str | None = None
    device: str | None = None
    initial_prompt: str | None = None
    extra_args: list[str] = field(default_factory=list)
    retranscribe: bool = False


@dataclass
class TranslateStage:
    config: TranslationConfig
    output_path: Path | None = None
    debug_dir: Path | None = None
    resume: bool = True
    retranslate: bool = False


def stage_prefix(label: str | None) -> str:
    return f"{label} " if label else ""


def ensure_source_subtitle(
    video: Path,
    *,
    subtitle: Path | None,
    output_dir: Path,
    stage: TranscribeStage,
    dry_run: bool = False,
    label: str | None = "[1/3]",
) -> Path:
    prefix = stage_prefix(label)
    if subtitle is not None:
        if not subtitle.is_file():
            raise SystemExit(f"Subtitle file not found: {subtitle}")
        print(f"{prefix}Source subtitle: reuse {subtitle}")
        return subtitle

    expected = output_dir / f"{video.stem}.srt"
    if expected.is_file() and not stage.retranscribe:
        print(f"{prefix}Transcribe: skip, reusing {expected}")
        return expected

    print(f"{prefix}Transcribe: running Whisper" + (" (dry run)" if dry_run else ""))
    return run_transcribe(
        TranscribeOptions(
            input_path=video,
            output_dir=output_dir,
            mode="transcribe",
            model=stage.model,
            output_format="srt",
            language=stage.language,
            backend=stage.backend,
            device=stage.device,
            initial_prompt=stage.initial_prompt,
            extra_args=stage.extra_args,
        ),
        dry_run=dry_run,
    )


def ensure_translated_subtitle(
    source_subtitle: Path,
    *,
    stage: TranslateStage,
    dry_run: bool = False,
    label: str | None = "[2/3]",
) -> Path:
    prefix = stage_prefix(label)
    output_path = stage.output_path or translated_subtitle_path(
        source_subtitle, stage.config.target_language
    )
    if output_path.is_file() and not stage.retranslate:
        print(f"{prefix}Translate: skip, reusing {output_path}")
        return output_path

    print(f"{prefix}Translate: calling the API" + (" (dry run)" if dry_run else ""))
    if dry_run and not source_subtitle.is_file():
        print(f"Would translate {source_subtitle} -> {output_path}")
        print()
        return output_path

    return translate_subtitle_file(
        input_path=source_subtitle,
        output_path=output_path,
        config=stage.config,
        debug_dir=stage.debug_dir,
        resume=stage.resume,
        dry_run=dry_run,
    )


def run_pipeline(
    *,
    video: Path,
    subtitle: Path | None,
    output_dir: Path,
    transcribe_stage: TranscribeStage,
    translate_stage: TranslateStage,
    mux_options_for: Callable[[Path], MuxOptions],
    dry_run: bool = False,
) -> Path:
    source_subtitle = ensure_source_subtitle(
        video,
        subtitle=subtitle,
        output_dir=output_dir,
        stage=transcribe_stage,
        dry_run=dry_run,
    )
    translated_subtitle = ensure_translated_subtitle(
        source_subtitle, stage=translate_stage, dry_run=dry_run
    )
    options = mux_options_for(translated_subtitle)
    print(f"[3/3] Mux: {options.mux_mode} subtitles -> {options.video_output}")
    return run_mux(options, dry_run=dry_run)
