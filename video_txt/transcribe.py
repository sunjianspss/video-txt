from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .media import find_ffmpeg, format_command

BACKENDS = ("auto", "openai-whisper", "mlx-whisper")
OUTPUT_FORMATS = ("txt", "vtt", "srt", "tsv", "json", "all")

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
    return [*command, *options.extra_args]


def expected_output_path(options: TranscribeOptions) -> Path:
    suffix = "txt" if options.output_format == "all" else options.output_format
    return options.output_dir / f"{options.input_path.stem}.{suffix}"


def run_transcribe(options: TranscribeOptions, *, dry_run: bool = False) -> Path:
    if not options.input_path.is_file():
        raise TranscribeError(f"Input file not found: {options.input_path}")

    find_ffmpeg()
    backend = resolve_backend(options.backend)
    model = resolve_model(mode=options.mode, model=options.model, backend=backend)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(options, backend=backend, model=model)
    output_path = expected_output_path(options)

    if dry_run:
        print(format_command(command))
        return output_path

    print(f"Input: {options.input_path}")
    print(f"Output dir: {options.output_dir}")
    print(f"Backend: {backend}")
    print(f"Mode: {options.mode}")
    print(f"Model: {model}")
    if options.language:
        print(f"Language: {options.language}")
    print()

    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise TranscribeError(f"Transcription failed with exit code {completed.returncode}.")
    return output_path
