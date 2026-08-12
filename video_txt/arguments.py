from __future__ import annotations

import argparse
from pathlib import Path

from .constants import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BATCH_CHARS,
    DEFAULT_CONCURRENCY,
    DEFAULT_TARGET_LANGUAGE,
    PROVIDERS,
)
from .diarize import (
    DEFAULT_MODEL as DEFAULT_DIARIZE_MODEL,
)
from .diarize import (
    DEFAULT_TOKEN_ENV,
)
from .dub import ENGINES
from .env import DEFAULT_SECRETS_FILE
from .mux import HARD_LAYOUTS
from .timeline import VOICE_UNITS
from .transcribe import BACKENDS, OUTPUT_FORMATS
from .voices import DEFAULT_VOICE

# A parser or one of its argument groups: whatever an option can be hung on.
type ArgumentTarget = argparse.ArgumentParser | argparse._ArgumentGroup


def add_dry_run(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument("--dry-run", action="store_true", help=help_text)


def add_pipeline_arguments(parser: argparse.ArgumentParser, *, video_output_help: str) -> None:
    """What every video -> finished video command takes: a video, and where things go."""
    parser.add_argument("video", type=Path, help="Path to the source video file.")
    parser.add_argument(
        "--subtitle",
        type=Path,
        help="Existing source-language .srt. Skips the transcription step.",
    )
    parser.add_argument(
        "--subtitle-output",
        type=Path,
        help="Translated subtitle path. Defaults to '<video>.zh.srt'.",
    )
    parser.add_argument("-o", "--video-output", type=Path, help=video_output_help)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for intermediate subtitle files. Defaults to the video's folder.",
    )
    parser.add_argument(
        "--retranscribe", action="store_true", help="Transcribe again even if an .srt exists."
    )
    parser.add_argument(
        "--retranslate", action="store_true", help="Translate again even if a translation exists."
    )


def add_language_code_argument(group: ArgumentTarget) -> None:
    group.add_argument(
        "--language-code",
        help=(
            "Language code written into the soft subtitle metadata. "
            "Follows --target-language by default."
        ),
    )


def add_ffmpeg_arguments(group: ArgumentTarget) -> None:
    group.add_argument(
        "--ffmpeg",
        dest="ffmpeg_path",
        help="Path to the ffmpeg binary. Also read from FFMPEG_PATH.",
    )
    group.add_argument(
        "--overwrite-video",
        action="store_true",
        help="Overwrite the output video if it already exists.",
    )


def add_translation_arguments(parser: argparse.ArgumentParser, *, debug_flag: str) -> None:
    group = parser.add_argument_group("translation")
    group.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        help="Preset for base URL, API key variable and model. Defaults to openai.",
    )
    group.add_argument(
        "--model",
        help="Translation model name. Falls back to the provider default or OPENAI_MODEL.",
    )
    group.add_argument("--base-url", help="OpenAI-compatible API base URL.")
    group.add_argument(
        "--api-key-env",
        help=f"Environment variable holding the API key. Defaults to {DEFAULT_API_KEY_ENV}.",
    )
    group.add_argument(
        "--secrets-file",
        type=Path,
        default=DEFAULT_SECRETS_FILE,
        help=(
            "Shell-style file read when the API key variable is unset. "
            "Overrides the configured default."
        ),
    )
    group.add_argument(
        "--target-language",
        default=DEFAULT_TARGET_LANGUAGE,
        help="Target language for translation. Defaults to Simplified Chinese.",
    )
    group.add_argument(
        "--source-language",
        help="Optional hint for the source language, for example Japanese.",
    )
    group.add_argument(
        "--batch-chars",
        type=int,
        default=DEFAULT_BATCH_CHARS,
        help=f"Approximate text characters per API batch. Defaults to {DEFAULT_BATCH_CHARS}.",
    )
    group.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per batch when a request fails. Defaults to 2.",
    )
    group.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Batches translated in parallel. Defaults to {DEFAULT_CONCURRENCY}.",
    )
    group.add_argument(
        "--context-cues",
        type=int,
        default=3,
        help="Preceding subtitle lines sent as context for tone and terminology. Defaults to 3.",
    )
    group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not reuse or write the .partial.jsonl resume file.",
    )
    group.add_argument(
        debug_flag,
        dest="debug_dir",
        type=Path,
        help="Directory for invalid API response debug files.",
    )
    group.add_argument("--note", help="Extra translation note, for example 'keep a casual tone'.")
    group.add_argument(
        "--preserve-term",
        action="append",
        default=[],
        help="Term to keep in the original language. Can be passed multiple times.",
    )


def add_transcribe_arguments(parser: argparse.ArgumentParser, *, standalone: bool) -> None:
    group = parser.add_argument_group("transcription")
    model_flags = ["-m", "--model"] if standalone else ["--whisper-model"]
    group.add_argument(
        *model_flags,
        dest="whisper_model",
        help="Whisper model. Defaults to turbo for transcription and medium for translation.",
    )
    group.add_argument(
        "--backend",
        dest="whisper_backend",
        choices=BACKENDS,
        default="auto",
        help="Transcription backend. auto prefers mlx-whisper on Apple Silicon.",
    )
    language_flags = ["-l", "--language"] if standalone else ["--language"]
    group.add_argument(
        *language_flags,
        dest="language",
        help="Spoken language in the media, for example en or Chinese. Defaults to auto-detect.",
    )
    group.add_argument(
        "--device", help="Torch device for the openai-whisper backend, for example cpu."
    )
    group.add_argument(
        "--initial-prompt",
        help="Prompt that biases spelling of names and jargon, for example 'Claude Code, MCP'.",
    )
    group.add_argument(
        "--whisper-arg",
        action="append",
        default=[],
        dest="whisper_args",
        help="Extra raw argument passed to the backend. Can be passed multiple times.",
    )
    group.add_argument(
        "--skip-transcript-check",
        action="store_true",
        help="Do not look for repeated lines and wrong-language output in the transcript.",
    )


def add_mux_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("muxing")
    group.add_argument(
        "--mux-mode",
        choices=("soft", "hard"),
        default="soft",
        help="soft embeds a toggleable subtitle track; hard burns text into the image.",
    )
    add_language_code_argument(group)
    group.add_argument(
        "--default-subtitle",
        action="store_true",
        help="Mark the soft subtitle track as the default stream.",
    )
    group.add_argument(
        "--subtitle-codec",
        help="Override the soft subtitle encoder. Defaults to the one the output container needs.",
    )
    group.add_argument(
        "--hard-subtitle-layout",
        choices=HARD_LAYOUTS,
        default="normal",
        help=(
            "Layout for --mux-mode hard: normal sits at the bottom, bottom-box adds a "
            "translucent band over burned-in original subtitles, top moves text to the top."
        ),
    )
    group.add_argument(
        "--hard-subtitle-font",
        default="PingFang SC",
        help="Font family for hard subtitles. Defaults to PingFang SC.",
    )
    group.add_argument(
        "--hard-subtitle-font-size",
        type=int,
        help="Hard subtitle size in pixels. Defaults to about 4.5%% of the video height.",
    )
    group.add_argument(
        "--hard-subtitle-margin-v",
        type=int,
        help="Hard subtitle vertical margin in pixels. Defaults to about 5%% of the video height.",
    )
    group.add_argument(
        "--hard-subtitle-box-height",
        type=float,
        default=0.22,
        help="Bottom box height as a fraction of video height for bottom-box. Defaults to 0.22.",
    )
    group.add_argument(
        "--hard-subtitle-box-opacity",
        type=float,
        default=0.68,
        help="Bottom box opacity for bottom-box, from 0 to 1. Defaults to 0.68.",
    )
    group.add_argument(
        "--video-codec",
        default="libx264",
        help="Video encoder used when burning hard subtitles. Defaults to libx264.",
    )
    group.add_argument(
        "--crf",
        type=int,
        default=20,
        help="Quality for libx264/libx265, lower is better. Defaults to 20.",
    )
    group.add_argument(
        "--preset",
        default="medium",
        help="libx264/libx265 speed preset. Defaults to medium.",
    )
    group.add_argument(
        "--video-size",
        help="Video resolution as WIDTHxHEIGHT, used when ffprobe cannot detect it.",
    )
    add_ffmpeg_arguments(group)


def add_speaker_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("speakers")
    group.add_argument(
        "--diarize",
        action="store_true",
        help="Work out who speaks when, and give each speaker a voice of their own.",
    )
    group.add_argument(
        "--speakers",
        type=int,
        dest="speaker_count",
        help="How many people speak, when you already know. Guessed otherwise.",
    )
    group.add_argument("--min-speakers", type=int, help="Lower bound on the number of speakers.")
    group.add_argument("--max-speakers", type=int, help="Upper bound on the number of speakers.")
    group.add_argument(
        "--speaker-voice",
        action="append",
        default=[],
        dest="speaker_voices",
        metavar="SPEAKER=VOICE",
        help="Voice for one speaker, for example SPEAKER_01=zh-CN-YunxiNeural. Repeatable.",
    )
    group.add_argument(
        "--diarize-model",
        default=DEFAULT_DIARIZE_MODEL,
        help=f"Diarization pipeline. Defaults to {DEFAULT_DIARIZE_MODEL}.",
    )
    group.add_argument(
        "--hf-token-env",
        default=DEFAULT_TOKEN_ENV,
        help=(
            "Environment variable holding the Hugging Face token the gated "
            f"diarization models need. Defaults to {DEFAULT_TOKEN_ENV}."
        ),
    )
    group.add_argument(
        "--rediarize",
        action="store_true",
        help="Diarize again even if the speaker turns were already written.",
    )


def add_voice_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("voice-over")
    group.add_argument(
        "--tts-engine",
        choices=ENGINES,
        default="edge-tts",
        help=(
            "Speech engine. edge-tts sounds natural and is free; say works offline "
            "on macOS; cosyvoice and f5-tts clone the original speaker locally."
        ),
    )
    group.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Voice name for edge-tts or say. Defaults to {DEFAULT_VOICE}.",
    )
    group.add_argument(
        "--voice-unit",
        choices=VOICE_UNITS,
        default="sentence",
        help=(
            "How much text goes into one voice clip. sentence speaks whole sentences, "
            "which keeps the delivery natural; line speaks each subtitle line on its own."
        ),
    )
    group.add_argument(
        "--rate", default="+0%", help="edge-tts speaking rate, for example +10%%. Defaults to +0%%."
    )
    group.add_argument(
        "--max-atempo",
        type=float,
        default=1.35,
        help="Largest speed-up used to fit a line into its slot. Defaults to 1.35.",
    )
    group.add_argument(
        "--fit-duration",
        action="store_true",
        help="Rewrite lines whose voice clip overruns the on-screen slot, then synthesize again.",
    )
    group.add_argument(
        "--fit-tempo",
        type=float,
        default=1.15,
        help="Speed-up tolerated before a line is rewritten shorter. Defaults to 1.15.",
    )
    group.add_argument(
        "--fit-rounds",
        type=int,
        default=2,
        help="Rewrite passes over the lines that still overrun. Defaults to 2.",
    )
    group.add_argument(
        "--keep-bgm",
        action="store_true",
        help="Mix the original audio in quietly instead of replacing it.",
    )
    group.add_argument(
        "--bgm-volume",
        type=float,
        default=0.15,
        help="Original audio volume when --keep-bgm is set. Defaults to 0.15.",
    )
    group.add_argument(
        "--keep-dub-audio",
        action="store_true",
        help=(
            "Keep the voice track as a separate .dub.wav. It runs to a few hundred "
            "megabytes, so it is removed once it is inside the video, unless you "
            "named it yourself with --audio-output."
        ),
    )
    group.add_argument(
        "--soft-subtitle",
        action="store_true",
        help="Also embed the translated subtitles as a toggleable track.",
    )
    add_language_code_argument(group)
    group.add_argument(
        "--tts-concurrency",
        type=int,
        default=4,
        help="Voice clips synthesized in parallel. Defaults to 4.",
    )
    group.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory for cached voice clips. Defaults to '<video>.dub-cache'.",
    )
    group.add_argument(
        "--prune-cache",
        action="store_true",
        help=(
            "Delete cached voice clips this run did not use. Changing --voice or "
            "--rate resynthesizes everything and leaves the old clips behind."
        ),
    )
    group.add_argument(
        "--audio-output", type=Path, help="Path for the rendered Chinese audio track."
    )
    add_ffmpeg_arguments(group)


def add_clone_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("voice cloning")
    group.add_argument(
        "--clone-model",
        help=(
            "Model for --tts-engine cosyvoice or f5-tts. CosyVoice needs the "
            "directory it was downloaded to; F5-TTS defaults to F5TTS_v1_Base."
        ),
    )
    group.add_argument(
        "--clone-reference",
        action="append",
        default=[],
        dest="clone_references",
        metavar="[SPEAKER=]AUDIO",
        help=(
            "Clip to clone a voice from, instead of one cut out of the video. "
            "A .txt of the same name is read as its transcript. Repeatable."
        ),
    )
    group.add_argument(
        "--clone-repo",
        type=Path,
        help="Checkout to import the model from, for example a cloned CosyVoice.",
    )
    group.add_argument(
        "--clone-python",
        help="Interpreter that has the cloning model installed. Defaults to this one.",
    )
    group.add_argument(
        "--clone-device", help="Torch device for the cloning model, for example mps or cuda."
    )


def build_transcribe_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Path to a local audio or video file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory for generated files. Defaults to the input file's folder.",
    )
    parser.add_argument(
        "--mode",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="Whisper mode. translate means X->English, matching Whisper's built-in behavior.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=OUTPUT_FORMATS,
        default="txt",
        help="Output file format. Defaults to txt.",
    )
    add_transcribe_arguments(parser, standalone=True)
    add_dry_run(parser, "Print the transcription command without running it.")


def build_translate_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Path to the source .srt file.")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output .srt path. Defaults to '<input>.zh.srt'."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file if it exists."
    )
    add_translation_arguments(parser, debug_flag="--debug-dir")
    add_dry_run(parser, "Print the resolved configuration without calling the API.")


def build_mux_command(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", type=Path, help="Path to the source video file.")
    parser.add_argument("subtitle", type=Path, help="Path to the source .srt subtitle file.")
    parser.add_argument(
        "-o",
        "--video-output",
        type=Path,
        help="Output video path. Defaults to '<video>.zh-subbed.mp4'.",
    )
    parser.add_argument(
        "--subtitle-output",
        type=Path,
        help="Translated subtitle path. Defaults to '<subtitle>.zh.srt'.",
    )
    parser.add_argument(
        "--retranslate",
        action="store_true",
        help="Translate again even if the translated subtitle already exists.",
    )
    add_translation_arguments(parser, debug_flag="--translation-debug-dir")
    add_mux_arguments(parser)
    add_dry_run(parser, "Print the translation plan and ffmpeg command without running them.")


def build_run_command(parser: argparse.ArgumentParser) -> None:
    add_pipeline_arguments(
        parser, video_output_help="Output video path. Defaults to '<video>.zh-subbed.mp4'."
    )
    add_transcribe_arguments(parser, standalone=False)
    add_translation_arguments(parser, debug_flag="--translation-debug-dir")
    add_mux_arguments(parser)
    add_dry_run(
        parser, "Print every stage's plan without transcribing, calling the API or running ffmpeg."
    )


def build_dub_command(parser: argparse.ArgumentParser) -> None:
    add_pipeline_arguments(
        parser, video_output_help="Output video path. Defaults to '<video>.zh-dubbed.mp4'."
    )
    add_voice_arguments(parser)
    add_speaker_arguments(parser)
    add_clone_arguments(parser)
    add_transcribe_arguments(parser, standalone=False)
    add_translation_arguments(parser, debug_flag="--translation-debug-dir")
    add_dry_run(parser, "Print every stage's plan without synthesizing speech or running ffmpeg.")
