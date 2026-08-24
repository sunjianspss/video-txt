from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from .media import (
    AudioStreamSelection,
    MediaError,
    find_ffmpeg,
    format_command,
    probe_audio_streams,
    select_audio_stream,
)
from .refine import (
    RefineError,
    document_from_whisper_result,
    refine_transcript,
    write_word_document,
)
from .subtitles import write_srt

BACKENDS = ("auto", "openai-whisper", "mlx-whisper")
OUTPUT_FORMATS = ("txt", "vtt", "srt", "tsv", "json", "all")
REFINEMENT_RESERVED_FLAGS = {
    "--output-dir",
    "--output-format",
    "--output-name",
    "--word-timestamps",
}

MLX_MODEL_ALIASES = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large": "mlx-community/whisper-large-v3",
    "large-v2": "mlx-community/whisper-large-v2",
    "large-v3": "mlx-community/whisper-large-v3",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


class TranscribeError(RuntimeError):
    pass


@dataclass
class TranscribeOptions:
    input_path: Path
    output_dir: Path
    mode: str = "transcribe"
    model: str | None = None
    output_format: str = "txt"
    language: str | None = None
    backend: str = "auto"
    device: str | None = None
    initial_prompt: str | None = None
    extra_args: list[str] = field(default_factory=list)
    refine_subtitles: bool = False
    audio_stream: int | None = None


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def resolve_backend(requested: str) -> str:
    if requested not in BACKENDS:
        raise TranscribeError(f"Unknown transcription backend: {requested}")

    if requested == "mlx-whisper":
        if not module_available("mlx_whisper"):
            raise TranscribeError(
                "Missing dependency: mlx-whisper.\nInstall it with: uv add mlx-whisper"
            )
        return "mlx-whisper"

    if requested == "openai-whisper":
        if not module_available("whisper"):
            raise TranscribeError(
                "Missing dependency: openai-whisper.\nInstall it with: uv add openai-whisper"
            )
        return "openai-whisper"

    if platform.machine() == "arm64" and module_available("mlx_whisper"):
        return "mlx-whisper"
    if module_available("whisper"):
        return "openai-whisper"
    raise TranscribeError(
        "No transcription backend available.\n"
        "Install one with: uv add openai-whisper\n"
        "On Apple Silicon, 'uv add mlx-whisper' is several times faster."
    )


def resolve_model(*, mode: str, model: str | None, backend: str) -> str:
    resolved = model or ("medium" if mode == "translate" else "turbo")
    if mode == "translate" and resolved == "turbo":
        raise TranscribeError(
            "The turbo model does not support translation.\n"
            "Use --model medium, --model large, or omit --model in translate mode."
        )
    if backend == "mlx-whisper" and "/" not in resolved:
        return MLX_MODEL_ALIASES.get(resolved, f"mlx-community/whisper-{resolved}")
    return resolved


def mlx_launcher() -> list[str]:
    console_script = shutil.which("mlx_whisper")
    if console_script:
        return [console_script]
    return [sys.executable, "-m", "mlx_whisper.cli"]


def build_command(options: TranscribeOptions, *, backend: str, model: str) -> list[str]:
    if backend == "mlx-whisper":
        command = [
            *mlx_launcher(),
            str(options.input_path),
            "--model",
            model,
            "--task",
            options.mode,
            "--output-format",
            options.output_format,
            "--output-dir",
            str(options.output_dir),
            "--output-name",
            options.input_path.stem,
        ]
        if options.language:
            command.extend(["--language", options.language])
        if options.initial_prompt:
            command.extend(["--initial-prompt", options.initial_prompt])
        if options.refine_subtitles:
            command.extend(["--word-timestamps", "True"])
        if options.device:
            print(
                f"Note: --device {options.device} is ignored by the mlx-whisper backend "
                "(it always runs on the Apple Metal GPU).",
                file=sys.stderr,
            )
        return [*command, *options.extra_args]

    command = [
        sys.executable,
        "-m",
        "whisper",
        str(options.input_path),
        "--model",
        model,
        "--task",
        options.mode,
        "--output_format",
        options.output_format,
        "--output_dir",
        str(options.output_dir),
    ]
    if options.language:
        command.extend(["--language", options.language])
    if options.device:
        command.extend(["--device", options.device])
    if options.initial_prompt:
        command.extend(["--initial_prompt", options.initial_prompt])
    if options.refine_subtitles:
        command.extend(["--word_timestamps", "True"])
    return [*command, *options.extra_args]


def expected_output_path(options: TranscribeOptions) -> Path:
    suffix = "txt" if options.output_format == "all" else options.output_format
    return options.output_dir / f"{options.input_path.stem}.{suffix}"


def audio_extract_command(
    input_path: Path,
    output_path: Path,
    *,
    stream_index: int,
    ffmpeg_path: str,
) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def describe_audio_selection(selection: AudioStreamSelection) -> str:
    stream = selection.stream
    metadata = ", ".join(
        value for value in (stream.title, stream.language, stream.layout) if value
    )
    details = f" ({metadata})" if metadata else ""
    return f"Audio stream: {stream.index}{details} — {selection.reason}"


def run_transcribe(options: TranscribeOptions, *, dry_run: bool = False) -> Path:
    if not options.input_path.is_file():
        raise TranscribeError(f"Input file not found: {options.input_path}")
    if options.refine_subtitles and options.output_format != "srt":
        raise TranscribeError("--refine-subtitles requires --format srt.")
    conflict = refinement_argument_conflict(options.extra_args)
    if options.refine_subtitles and conflict:
        raise TranscribeError(
            f"--whisper-arg {conflict} conflicts with --refine-subtitles; "
            "the refinement mode controls its JSON output and word timestamps."
        )

    ffmpeg_path = str(find_ffmpeg())
    backend = resolve_backend(options.backend)
    model = resolve_model(mode=options.mode, model=options.model, backend=backend)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = expected_output_path(options)

    try:
        streams = probe_audio_streams(options.input_path, ffmpeg_path=ffmpeg_path)
    except MediaError as exc:
        if not dry_run or options.audio_stream is not None:
            raise TranscribeError(str(exc)) from exc
        streams = []
        print(f"Audio stream: not inspected during dry run ({exc})")
    try:
        selection = (
            select_audio_stream(
                streams,
                preferred_language=options.language,
                requested_index=options.audio_stream,
            )
            if streams
            else None
        )
    except MediaError as exc:
        raise TranscribeError(str(exc)) from exc
    extracted_audio: Path | None = None
    if options.audio_stream is not None or len(streams) > 1:
        assert selection is not None
        extracted_audio = (
            options.output_dir
            / f".{options.input_path.stem}.audio-stream-{selection.stream.index}"
            / f"{options.input_path.stem}.wav"
        )

    if dry_run:
        command_options = (
            replace(options, output_format="json") if options.refine_subtitles else options
        )
        if extracted_audio is not None:
            assert selection is not None
            print(describe_audio_selection(selection))
            print(
                format_command(
                    audio_extract_command(
                        options.input_path,
                        extracted_audio,
                        stream_index=selection.stream.index,
                        ffmpeg_path=ffmpeg_path,
                    )
                )
            )
            command_options = replace(command_options, input_path=extracted_audio)
        command = build_command(command_options, backend=backend, model=model)
        print(format_command(command))
        return output_path

    print(f"Input: {options.input_path}")
    print(f"Output dir: {options.output_dir}")
    print(f"Backend: {backend}")
    print(f"Mode: {options.mode}")
    print(f"Model: {model}")
    if options.language:
        print(f"Language: {options.language}")
    if extracted_audio is not None:
        assert selection is not None
        print(describe_audio_selection(selection))
    print()

    if extracted_audio is None:
        return _run_whisper(
            options,
            backend=backend,
            model=model,
            output_path=output_path,
        )

    with tempfile.TemporaryDirectory(
        dir=options.output_dir,
        # selection cannot be None when extraction was requested.
        prefix=f".{options.input_path.stem}.audio-stream-{selection.stream.index}-",
    ) as temporary_dir:
        audio_path = Path(temporary_dir) / f"{options.input_path.stem}.wav"
        command = audio_extract_command(
            options.input_path,
            audio_path,
            stream_index=selection.stream.index,
            ffmpeg_path=ffmpeg_path,
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not audio_path.is_file():
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise TranscribeError(
                f"Could not extract audio stream {selection.stream.index}: {detail}"
            )
        return _run_whisper(
            replace(options, input_path=audio_path),
            backend=backend,
            model=model,
            output_path=output_path,
        )


def _run_whisper(
    options: TranscribeOptions,
    *,
    backend: str,
    model: str,
    output_path: Path,
) -> Path:
    if options.refine_subtitles:
        return _run_refined_transcription(
            options,
            backend=backend,
            model=model,
            output_path=output_path,
        )

    command = build_command(options, backend=backend, model=model)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise TranscribeError(f"Transcription failed with exit code {completed.returncode}.")
    return output_path


def refinement_argument_conflict(extra_args: list[str]) -> str | None:
    for argument in extra_args:
        flag = argument.partition("=")[0].replace("_", "-")
        if flag in REFINEMENT_RESERVED_FLAGS:
            return argument
    return None


def _run_refined_transcription(
    options: TranscribeOptions,
    *,
    backend: str,
    model: str,
    output_path: Path,
) -> Path:
    with tempfile.TemporaryDirectory(
        dir=options.output_dir,
        prefix=".video-txt-transcribe-",
    ) as temporary_dir:
        raw_options = replace(
            options,
            output_dir=Path(temporary_dir),
            output_format="json",
        )
        command = build_command(raw_options, backend=backend, model=model)
        completed = subprocess.run(command)
        if completed.returncode != 0:
            raise TranscribeError(
                f"Transcription failed with exit code {completed.returncode}."
            )

        raw_path = expected_output_path(raw_options)
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise RefineError("Whisper JSON root must be an object.")
            document = document_from_whisper_result(payload)
            word_path = options.output_dir / f"{options.input_path.stem}.words.json"
            result = refine_transcript(document)
            write_srt(output_path, result.cues)
            # This is the completion marker used by the pipeline's reuse check,
            # so write it only after the refined subtitle is safely on disk.
            write_word_document(
                word_path,
                document,
                backend=backend,
                model=model,
                subtitle_path=output_path,
            )
        except (OSError, json.JSONDecodeError, RefineError) as exc:
            raise TranscribeError(f"Cannot refine Whisper word timestamps: {exc}") from exc

    longest = max((cue.duration for cue in result.cues), default=0.0)
    print(
        f"Refined subtitles: {result.source_words} words -> {result.cue_count} cues "
        f"(longest {longest:.1f}s)."
    )
    print(f"Word timestamps: {word_path}")
    return output_path
