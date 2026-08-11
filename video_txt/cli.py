from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .constants import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_BATCH_CHARS,
    DEFAULT_CONCURRENCY,
    DEFAULT_LANGUAGE_CODE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TERMS,
)
from .dub import DubError, DubOptions, run_dub
from .dub import default_video_output as default_dub_output
from .env import DEFAULT_SECRETS_FILE, resolve_api_key
from .media import MediaError, parse_video_size
from .mux import HARD_LAYOUTS, MuxError, MuxOptions, default_video_output, run_mux
from .pipeline import (
    TranscribeStage,
    TranslateStage,
    ensure_source_subtitle,
    ensure_translated_subtitle,
    run_pipeline,
)
from .subtitles import SubtitleFormatError, translated_subtitle_path
from .transcribe import BACKENDS, OUTPUT_FORMATS, TranscribeError, TranscribeOptions, run_transcribe
from .translate import (
    TranslationConfig,
    TranslationError,
    translate_subtitle_file,
)

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


def add_dry_run(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument("--dry-run", action="store_true", help=help_text)


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
            f"Defaults to {DEFAULT_SECRETS_FILE}."
        ),
    )
    group.add_argument(
        "--target-language",
        default=DEFAULT_TARGET_LANGUAGE,
        help="Target language for translation. Defaults to Simplified Chinese.",
    )
    group.add_argument(
        "--source-language",
        help="Optional hint for the source language, for example English.",
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


def add_mux_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("muxing")
    group.add_argument(
        "--mux-mode",
        choices=("soft", "hard"),
        default="soft",
        help="soft embeds a toggleable subtitle track; hard burns text into the image.",
    )
    group.add_argument(
        "--language-code",
        default=DEFAULT_LANGUAGE_CODE,
        help=(
            "Language code written into soft subtitle metadata. "
            f"Defaults to {DEFAULT_LANGUAGE_CODE}."
        ),
    )
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


def resolve_provider_settings(args: argparse.Namespace) -> tuple[str, str, str]:
    provider = PROVIDERS[args.provider] if args.provider else {}
    base_url = (
        args.base_url
        or provider.get("base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    api_key_env = args.api_key_env or provider.get("api_key_env") or DEFAULT_API_KEY_ENV
    model = args.model or provider.get("model") or os.environ.get("OPENAI_MODEL")
    if not model:
        raise SystemExit(
            "Missing model name. Pass --model, use --provider deepseek, or set OPENAI_MODEL."
        )
    return base_url, api_key_env, model


def build_translation_config(args: argparse.Namespace, *, require_key: bool) -> TranslationConfig:
    base_url, api_key_env, model = resolve_provider_settings(args)
    api_key = resolve_api_key(api_key_env, secrets_file=args.secrets_file) if require_key else ""
    return TranslationConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        target_language=args.target_language,
        source_language=args.source_language,
        preserve_terms=list(dict.fromkeys([*DEFAULT_TERMS, *args.preserve_term])),
        note=args.note,
        batch_chars=args.batch_chars,
        retries=max(0, args.retries),
        concurrency=max(1, args.concurrency),
        context_cues=max(0, args.context_cues),
    )


def build_mux_options(
    args: argparse.Namespace,
    *,
    video: Path,
    subtitle: Path,
    video_output: Path,
) -> MuxOptions:
    return MuxOptions(
        video_input=video,
        subtitle_input=subtitle,
        video_output=video_output,
        mux_mode=args.mux_mode,
        language_code=args.language_code,
        default_subtitle=args.default_subtitle,
        subtitle_codec=args.subtitle_codec,
        overwrite=args.overwrite_video,
        hard_layout=args.hard_subtitle_layout,
        font=args.hard_subtitle_font,
        font_size=args.hard_subtitle_font_size,
        margin_v=args.hard_subtitle_margin_v,
        box_height=args.hard_subtitle_box_height,
        box_opacity=args.hard_subtitle_box_opacity,
        video_codec=args.video_codec,
        crf=args.crf,
        preset=args.preset,
        video_size=parse_video_size(args.video_size) if args.video_size else None,
        ffmpeg_path=args.ffmpeg_path,
    )


def build_translate_stage(
    args: argparse.Namespace, *, require_key: bool, output_path: Path | None
) -> TranslateStage:
    return TranslateStage(
        config=build_translation_config(args, require_key=require_key),
        output_path=output_path,
        debug_dir=args.debug_dir.expanduser().resolve() if args.debug_dir else None,
        resume=args.resume,
        retranslate=args.retranslate,
    )


def build_transcribe_stage(args: argparse.Namespace) -> TranscribeStage:
    return TranscribeStage(
        model=args.whisper_model,
        backend=args.whisper_backend,
        language=args.language,
        device=args.device,
        initial_prompt=args.initial_prompt,
        extra_args=args.whisper_args,
        retranscribe=args.retranscribe,
    )


def validate_hard_subtitle_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.hard_subtitle_font_size is not None and args.hard_subtitle_font_size <= 0:
        parser.error("--hard-subtitle-font-size must be greater than 0")
    if args.hard_subtitle_margin_v is not None and args.hard_subtitle_margin_v < 0:
        parser.error("--hard-subtitle-margin-v must be 0 or greater")
    if not 0 < args.hard_subtitle_box_height <= 0.5:
        parser.error("--hard-subtitle-box-height must be greater than 0 and no more than 0.5")
    if not 0 <= args.hard_subtitle_box_opacity <= 1:
        parser.error("--hard-subtitle-box-opacity must be between 0 and 1")


def existing_file(parser: argparse.ArgumentParser, path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        parser.error(f"{label} not found: {resolved}")
    return resolved


def command_transcribe(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    input_path = existing_file(parser, args.input, "Input file")
    output_dir = (args.output_dir or input_path.parent).expanduser().resolve()
    options = TranscribeOptions(
        input_path=input_path,
        output_dir=output_dir,
        mode=args.mode,
        model=args.whisper_model,
        output_format=args.format,
        language=args.language,
        backend=args.whisper_backend,
        device=args.device,
        initial_prompt=args.initial_prompt,
        extra_args=args.whisper_args,
    )
    output_path = run_transcribe(options, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Done: {output_path}")
    return 0


def command_translate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    input_path = existing_file(parser, args.input, "Input file")
    if input_path.suffix.lower() != ".srt":
        parser.error(f"Expected an .srt file, got: {input_path.name}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else translated_subtitle_path(input_path, args.target_language)
    )
    if output_path.exists() and not args.overwrite and not args.dry_run:
        parser.error(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")

    translate_subtitle_file(
        input_path=input_path,
        output_path=output_path,
        config=build_translation_config(args, require_key=not args.dry_run),
        debug_dir=args.debug_dir.expanduser().resolve() if args.debug_dir else None,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    return 0


def command_mux(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    validate_hard_subtitle_options(parser, args)
    video = existing_file(parser, args.video, "Video file")
    subtitle = existing_file(parser, args.subtitle, "Subtitle file")
    if subtitle.suffix.lower() != ".srt":
        parser.error(f"Expected an .srt subtitle file, got: {subtitle.name}")

    subtitle_output = (
        args.subtitle_output.expanduser().resolve()
        if args.subtitle_output
        else translated_subtitle_path(subtitle, args.target_language)
    )
    video_output = (
        args.video_output.expanduser().resolve()
        if args.video_output
        else default_video_output(
            video, mux_mode=args.mux_mode, target_language=args.target_language
        )
    )

    stage = build_translate_stage(args, require_key=not args.dry_run, output_path=subtitle_output)
    translated = ensure_translated_subtitle(subtitle, stage=stage, dry_run=args.dry_run, label=None)
    options = build_mux_options(args, video=video, subtitle=translated, video_output=video_output)
    output = run_mux(options, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Done: {output}")
    return 0


def command_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    validate_hard_subtitle_options(parser, args)
    video = existing_file(parser, args.video, "Video file")
    subtitle = args.subtitle.expanduser().resolve() if args.subtitle else None
    output_dir = (args.output_dir or video.parent).expanduser().resolve()
    video_output = (
        args.video_output.expanduser().resolve()
        if args.video_output
        else default_video_output(
            video, mux_mode=args.mux_mode, target_language=args.target_language
        )
    )

    output = run_pipeline(
        video=video,
        subtitle=subtitle,
        output_dir=output_dir,
        transcribe_stage=build_transcribe_stage(args),
        translate_stage=build_translate_stage(
            args,
            require_key=not args.dry_run,
            output_path=args.subtitle_output.expanduser().resolve()
            if args.subtitle_output
            else None,
        ),
        mux_options_for=lambda translated: build_mux_options(
            args, video=video, subtitle=translated, video_output=video_output
        ),
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"Done: {output}")
    return 0


def command_dub(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    video = existing_file(parser, args.video, "Video file")
    subtitle = args.subtitle.expanduser().resolve() if args.subtitle else None
    output_dir = (args.output_dir or video.parent).expanduser().resolve()
    video_output = (
        args.video_output.expanduser().resolve() if args.video_output else default_dub_output(video)
    )

    source_subtitle = ensure_source_subtitle(
        video,
        subtitle=subtitle,
        output_dir=output_dir,
        stage=build_transcribe_stage(args),
        dry_run=args.dry_run,
    )
    translated = ensure_translated_subtitle(
        source_subtitle,
        stage=build_translate_stage(
            args,
            require_key=not args.dry_run,
            output_path=args.subtitle_output.expanduser().resolve()
            if args.subtitle_output
            else None,
        ),
        dry_run=args.dry_run,
    )

    print(f"[3/3] Dub: {args.tts_engine} voice {args.voice}")
    options = DubOptions(
        video_input=video,
        subtitle_input=translated,
        video_output=video_output,
        audio_output=args.audio_output.expanduser().resolve() if args.audio_output else None,
        cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
        engine=args.tts_engine,
        voice=args.voice,
        rate=args.rate,
        max_atempo=args.max_atempo,
        keep_bgm=args.keep_bgm,
        bgm_volume=args.bgm_volume,
        soft_subtitle=args.soft_subtitle,
        concurrency=max(1, args.tts_concurrency),
        overwrite=args.overwrite_video,
        ffmpeg_path=args.ffmpeg_path,
    )
    output = run_dub(options, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Done: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-txt",
        description="Transcribe, translate, subtitle and dub local audio or video files.",
    )
    parser.add_argument("--version", action="version", version=f"video-txt {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser(
        "transcribe", help="Convert audio or video into text with Whisper."
    )
    transcribe.add_argument("input", type=Path, help="Path to a local audio or video file.")
    transcribe.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory for generated files. Defaults to the input file's folder.",
    )
    transcribe.add_argument(
        "--mode",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="Whisper mode. translate means X->English, matching Whisper's built-in behavior.",
    )
    transcribe.add_argument(
        "-f",
        "--format",
        choices=OUTPUT_FORMATS,
        default="txt",
        help="Output file format. Defaults to txt.",
    )
    add_transcribe_arguments(transcribe, standalone=True)
    add_dry_run(transcribe, "Print the transcription command without running it.")
    transcribe.set_defaults(handler=command_transcribe)

    translate = subparsers.add_parser(
        "translate", help="Translate an .srt subtitle file with an OpenAI-compatible API."
    )
    translate.add_argument("input", type=Path, help="Path to the source .srt file.")
    translate.add_argument(
        "-o", "--output", type=Path, help="Output .srt path. Defaults to '<input>.zh.srt'."
    )
    translate.add_argument(
        "--overwrite", action="store_true", help="Overwrite the output file if it exists."
    )
    add_translation_arguments(translate, debug_flag="--debug-dir")
    add_dry_run(translate, "Print the resolved configuration without calling the API.")
    translate.set_defaults(handler=command_translate)

    mux = subparsers.add_parser("mux", help="Translate an .srt file and mux it back into a video.")
    mux.add_argument("video", type=Path, help="Path to the source video file.")
    mux.add_argument("subtitle", type=Path, help="Path to the source .srt subtitle file.")
    mux.add_argument(
        "-o",
        "--video-output",
        type=Path,
        help="Output video path. Defaults to '<video>.zh-subbed.mp4'.",
    )
    mux.add_argument(
        "--subtitle-output",
        type=Path,
        help="Translated subtitle path. Defaults to '<subtitle>.zh.srt'.",
    )
    mux.add_argument(
        "--retranslate",
        action="store_true",
        help="Translate again even if the translated subtitle already exists.",
    )
    add_translation_arguments(mux, debug_flag="--translation-debug-dir")
    add_mux_arguments(mux)
    add_dry_run(mux, "Print the translation plan and ffmpeg command without running them.")
    mux.set_defaults(handler=command_mux)

    run = subparsers.add_parser(
        "run", help="One command: video -> transcript -> translation -> subtitled video."
    )
    run.add_argument("video", type=Path, help="Path to the source video file.")
    run.add_argument(
        "--subtitle",
        type=Path,
        help="Existing source-language .srt. Skips the transcription step.",
    )
    run.add_argument(
        "--subtitle-output",
        type=Path,
        help="Translated subtitle path. Defaults to '<video>.zh.srt'.",
    )
    run.add_argument(
        "-o",
        "--video-output",
        type=Path,
        help="Output video path. Defaults to '<video>.zh-subbed.mp4'.",
    )
    run.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for intermediate subtitle files. Defaults to the video's folder.",
    )
    run.add_argument(
        "--retranscribe", action="store_true", help="Transcribe again even if an .srt exists."
    )
    run.add_argument(
        "--retranslate", action="store_true", help="Translate again even if a translation exists."
    )
    add_transcribe_arguments(run, standalone=False)
    add_translation_arguments(run, debug_flag="--translation-debug-dir")
    add_mux_arguments(run)
    add_dry_run(
        run, "Print every stage's plan without transcribing, calling the API or running ffmpeg."
    )
    run.set_defaults(handler=command_run)

    dub = subparsers.add_parser("dub", help="One command: video -> Chinese voice-over video.")
    dub.add_argument("video", type=Path, help="Path to the source video file.")
    dub.add_argument(
        "--subtitle",
        type=Path,
        help="Existing source-language .srt. Skips the transcription step.",
    )
    dub.add_argument(
        "--subtitle-output",
        type=Path,
        help="Translated subtitle path. Defaults to '<video>.zh.srt'.",
    )
    dub.add_argument(
        "-o",
        "--video-output",
        type=Path,
        help="Output video path. Defaults to '<video>.zh-dubbed.mp4'.",
    )
    dub.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for intermediate subtitle files. Defaults to the video's folder.",
    )
    dub.add_argument(
        "--retranscribe", action="store_true", help="Transcribe again even if an .srt exists."
    )
    dub.add_argument(
        "--retranslate", action="store_true", help="Translate again even if a translation exists."
    )
    voice_group = dub.add_argument_group("voice-over")
    voice_group.add_argument(
        "--tts-engine",
        choices=("edge-tts", "say"),
        default="edge-tts",
        help="Speech engine. edge-tts sounds natural and is free; say works offline on macOS.",
    )
    voice_group.add_argument(
        "--voice",
        default="zh-CN-XiaoxiaoNeural",
        help="Voice name, for example zh-CN-XiaoxiaoNeural or zh-CN-YunxiNeural.",
    )
    voice_group.add_argument(
        "--rate", default="+0%", help="edge-tts speaking rate, for example +10%%. Defaults to +0%%."
    )
    voice_group.add_argument(
        "--max-atempo",
        type=float,
        default=1.35,
        help="Largest speed-up used to fit a line into its slot. Defaults to 1.35.",
    )
    voice_group.add_argument(
        "--keep-bgm",
        action="store_true",
        help="Mix the original audio in quietly instead of replacing it.",
    )
    voice_group.add_argument(
        "--bgm-volume",
        type=float,
        default=0.15,
        help="Original audio volume when --keep-bgm is set. Defaults to 0.15.",
    )
    voice_group.add_argument(
        "--soft-subtitle",
        action="store_true",
        help="Also embed the Chinese subtitles as a toggleable track.",
    )
    voice_group.add_argument(
        "--tts-concurrency",
        type=int,
        default=4,
        help="Voice clips synthesized in parallel. Defaults to 4.",
    )
    voice_group.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory for cached voice clips. Defaults to '<video>.dub-cache'.",
    )
    voice_group.add_argument(
        "--audio-output", type=Path, help="Path for the rendered Chinese audio track."
    )
    voice_group.add_argument(
        "--ffmpeg",
        dest="ffmpeg_path",
        help="Path to the ffmpeg binary. Also read from FFMPEG_PATH.",
    )
    voice_group.add_argument(
        "--overwrite-video",
        action="store_true",
        help="Overwrite the output video if it already exists.",
    )
    add_transcribe_arguments(dub, standalone=False)
    add_translation_arguments(dub, debug_flag="--translation-debug-dir")
    add_dry_run(dub, "Print every stage's plan without synthesizing speech or running ffmpeg.")
    dub.set_defaults(handler=command_dub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args, parser)
    except (
        DubError,
        MediaError,
        MuxError,
        SubtitleFormatError,
        TranscribeError,
        TranslationError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
