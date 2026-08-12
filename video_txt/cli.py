from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .arguments import (
    build_dub_command,
    build_mux_command,
    build_run_command,
    build_transcribe_command,
    build_translate_command,
)
from .clone import CLONE_ENGINES, CloneError, CloneOptions, is_rate, speed_from_rate
from .constants import DEFAULT_API_KEY_ENV, DEFAULT_BASE_URL, DEFAULT_TERMS, PROVIDERS
from .diarize import (
    DEFAULT_SPEAKER,
    DiarizeError,
    DiarizeOptions,
    SpeakerTurn,
    ensure_speakers,
)
from .dub import DubError, DubOptions, default_dubbed_output, run_dub
from .env import CredentialError, resolve_api_key, resolve_optional_key
from .fit import FitOptions
from .media import MediaError, parse_video_size
from .mux import MuxError, MuxOptions, default_video_output, run_mux
from .pipeline import (
    TranscribeStage,
    TranslateStage,
    ensure_source_subtitle,
    ensure_translated_subtitle,
    run_pipeline,
)
from .quality import check_transcript
from .subtitles import SubtitleFormatError, language_code, translated_subtitle_path
from .transcribe import TranscribeError, TranscribeOptions, run_transcribe
from .translate import (
    TranslationConfig,
    TranslationError,
    translate_subtitle_file,
)
from .voices import DEFAULT_SAY_VOICE, DEFAULT_VOICE, resolve_voice_name

SPEAKER_PATTERN = re.compile(r"^\w+$")


def resolved(path: Path | None) -> Path | None:
    """A path as it was typed, made absolute. None stays None, meaning 'pick a default'."""
    return path.expanduser().resolve() if path else None


def existing_file(parser: argparse.ArgumentParser, path: Path, label: str) -> Path:
    full_path = path.expanduser().resolve()
    if not full_path.is_file():
        parser.error(f"{label} not found: {full_path}")
    return full_path


def normalize_speaker(name: str) -> str:
    """pyannote calls them SPEAKER_00; take 0 and 00 to mean the same thing."""
    label = name.strip()
    return f"SPEAKER_{int(label):02d}" if label.isdigit() else label


def split_assignment(value: str) -> tuple[str, str]:
    """SPEAKER=value, or a bare value meaning the only speaker there is."""
    key, separator, rest = value.partition("=")
    if separator and SPEAKER_PATTERN.match(key.strip()):
        return normalize_speaker(key), rest.strip()
    return DEFAULT_SPEAKER, value.strip()


def parse_speaker_voices(parser: argparse.ArgumentParser, values: list[str]) -> dict[str, str]:
    voices: dict[str, str] = {}
    for value in values:
        speaker, voice = split_assignment(value)
        if not speaker or not voice:
            parser.error(
                "--speaker-voice expects SPEAKER=VOICE, for example "
                f"SPEAKER_01=zh-CN-YunxiNeural. Got: {value}"
            )
        voices[speaker] = voice
    return voices


def parse_clone_references(parser: argparse.ArgumentParser, values: list[str]) -> dict[str, Path]:
    references: dict[str, Path] = {}
    for value in values:
        speaker, audio = split_assignment(value)
        if not audio:
            parser.error(
                "--clone-reference expects an audio path, optionally prefixed with "
                f"SPEAKER=. Got: {value}"
            )
        references[speaker] = Path(audio)
    return references


def build_diarize_options(args: argparse.Namespace) -> DiarizeOptions:
    return DiarizeOptions(
        model=args.diarize_model,
        token=resolve_optional_key(args.hf_token_env, secrets_file=args.secrets_file),
        speakers=args.speaker_count,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        rediarize=args.rediarize,
    )


def build_clone_options(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> CloneOptions | None:
    if args.tts_engine not in CLONE_ENGINES:
        return None
    return CloneOptions(
        engine=args.tts_engine,
        model=args.clone_model,
        python=args.clone_python,
        repo=args.clone_repo,
        device=args.clone_device,
        speed=speed_from_rate(args.rate),
        references=parse_clone_references(parser, args.clone_references),
    )


def resolve_provider_settings(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[str, str, str]:
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
        parser.error(
            "Missing model name. Pass --model, use --provider deepseek, or set OPENAI_MODEL."
        )
    return base_url, api_key_env, model


def validate_translation_numbers(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Settings the translator would otherwise quietly pull back into range."""
    if args.batch_chars < 1:
        parser.error("--batch-chars must be 1 or greater")
    if args.retries < 0:
        parser.error("--retries must be 0 or greater")
    if args.concurrency < 1:
        parser.error("--concurrency must be 1 or greater")
    if args.context_cues < 0:
        parser.error("--context-cues must be 0 or greater")


def build_translation_config(
    args: argparse.Namespace, parser: argparse.ArgumentParser, *, require_key: bool
) -> TranslationConfig:
    validate_translation_numbers(parser, args)
    base_url, api_key_env, model = resolve_provider_settings(args, parser)
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
        retries=args.retries,
        concurrency=args.concurrency,
        context_cues=args.context_cues,
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
        language_code=args.language_code or language_code(args.target_language),
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


def dub_voicing(args: argparse.Namespace) -> str:
    """How the dub will be spoken, in the few words the stage banner has room for."""
    if args.tts_engine in CLONE_ENGINES:
        return "cloning the original voice"
    if args.diarize:
        return "one voice per speaker"
    return f"voice {resolve_voice_name(args.tts_engine, args.voice)}"


def build_dub_options(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    video: Path,
    subtitle: Path,
    source_subtitle: Path,
    video_output: Path,
    translation: Callable[[], TranslationConfig],
    turns: list[SpeakerTurn],
) -> DubOptions:
    return DubOptions(
        video_input=video,
        subtitle_input=subtitle,
        video_output=video_output,
        source_subtitle=source_subtitle,
        fit=build_fit_options(args, translation=translation),
        audio_output=resolved(args.audio_output),
        cache_dir=resolved(args.cache_dir),
        engine=args.tts_engine,
        voice=args.voice,
        voice_unit=args.voice_unit,
        rate=args.rate,
        max_atempo=args.max_atempo,
        keep_bgm=args.keep_bgm,
        keep_audio=args.keep_dub_audio,
        prune_cache=args.prune_cache,
        bgm_volume=args.bgm_volume,
        soft_subtitle=args.soft_subtitle,
        language_code=args.language_code or language_code(args.target_language),
        concurrency=args.tts_concurrency,
        overwrite=args.overwrite_video,
        ffmpeg_path=args.ffmpeg_path,
        turns=turns,
        speaker_voices=parse_speaker_voices(parser, args.speaker_voices),
        clone=build_clone_options(args, parser),
    )


def build_translate_stage(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    require_key: bool,
    output_path: Path | None,
) -> TranslateStage:
    return TranslateStage(
        config=TranslationConfig(
            base_url="",
            api_key="",
            model="",
            target_language=args.target_language,
        ),
        config_loader=lambda: build_translation_config(args, parser, require_key=require_key),
        output_path=output_path,
        debug_dir=resolved(args.debug_dir),
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
        skip_transcript_check=args.skip_transcript_check,
    )


def build_fit_options(
    args: argparse.Namespace, *, translation: Callable[[], TranslationConfig]
) -> FitOptions | None:
    if not args.fit_duration:
        return None
    return FitOptions(translation=translation(), tempo=args.fit_tempo, rounds=args.fit_rounds)


def validate_fit_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.fit_duration:
        return
    if args.fit_tempo < 1.0:
        parser.error("--fit-tempo must be 1.0 or greater")
    if args.fit_rounds < 1:
        parser.error("--fit-rounds must be 1 or greater")
    if args.fit_tempo > args.max_atempo:
        parser.error(
            f"--fit-tempo {args.fit_tempo} leaves lines needing more speed-up than "
            f"--max-atempo {args.max_atempo} can apply, so they would run past their slot. "
            "Lower --fit-tempo or raise --max-atempo."
        )


def validate_dub_numbers(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Settings the renderer would otherwise quietly pull back into range."""
    if not is_rate(args.rate):
        parser.error(f"--rate must be a signed percentage such as +10%. Got: {args.rate}")
    if args.max_atempo < 1.0:
        parser.error("--max-atempo must be 1.0 or greater: it can only speed a clip up.")
    if not 0 <= args.bgm_volume <= 1:
        parser.error("--bgm-volume must be between 0 and 1")
    if args.tts_concurrency < 1:
        parser.error("--tts-concurrency must be 1 or greater")


def validate_engine_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Flags addressed to an engine other than the one that will run.

    None of them would reach anything. The run finishes, reports success, and
    sounds nothing like what was asked for.
    """
    if args.tts_engine not in CLONE_ENGINES:
        used = [
            flag
            for flag, value in (
                ("--clone-model", args.clone_model),
                ("--clone-reference", args.clone_references),
                ("--clone-repo", args.clone_repo),
                ("--clone-python", args.clone_python),
                ("--clone-device", args.clone_device),
            )
            if value
        ]
        if used:
            engines = " or ".join(f"--tts-engine {engine}" for engine in CLONE_ENGINES)
            parser.error(f"{used[0]} only applies to a cloning engine. Add {engines}.")
        if args.tts_engine == "say":
            asked = [
                ("--voice", args.voice),
                *(("--speaker-voice", split_assignment(v)[1]) for v in args.speaker_voices),
            ]
            # Against the resolved name, not the one that was typed: the default
            # --voice is an edge-tts name that say already answers to with one of
            # its own, and nobody who left the flag alone should hear about it.
            neural = [
                (flag, name)
                for flag, name in asked
                if resolve_voice_name(args.tts_engine, name).endswith("Neural")
            ]
            if neural:
                flag, name = neural[0]
                parser.error(
                    f"{flag} {name} is an edge-tts voice, and say has names of its own. "
                    f"Leave it out for {DEFAULT_SAY_VOICE}, or list them with: say -v '?'"
                )
        return

    if args.voice != DEFAULT_VOICE:
        parser.error(
            f"--tts-engine {args.tts_engine} clones the voice out of the reference clip, so "
            "--voice has nothing to name. Choose the clip with --clone-reference instead."
        )
    if args.speaker_voices:
        parser.error(
            f"--tts-engine {args.tts_engine} gives every speaker the voice cloned from their "
            "own reference clip, so --speaker-voice has nothing to name."
        )


def validate_voice_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    validate_dub_numbers(parser, args)
    validate_engine_options(parser, args)
    if args.speaker_voices and not args.diarize:
        parser.error("--speaker-voice needs --diarize: without it every line has one speaker.")
    named_references = [
        value for value in args.clone_references if split_assignment(value)[0] != DEFAULT_SPEAKER
    ]
    if named_references and not args.diarize:
        parser.error(
            "--clone-reference SPEAKER=... needs --diarize: without it every line has one "
            f"speaker, so the name matches nobody. Drop it: --clone-reference "
            f"{split_assignment(named_references[0])[1]}"
        )

    counts = {
        "--speakers": args.speaker_count,
        "--min-speakers": args.min_speakers,
        "--max-speakers": args.max_speakers,
    }
    for flag, value in counts.items():
        if value is not None and value < 1:
            parser.error(f"{flag} must be 1 or greater")
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        parser.error("--min-speakers cannot be greater than --max-speakers")


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


def command_transcribe(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    input_path = existing_file(parser, args.input, "Input file")
    output_dir = resolved(args.output_dir) or input_path.parent
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
    if args.dry_run:
        return 0
    print(f"Done: {output_path}")
    if output_path.suffix.lower() != ".srt" or args.skip_transcript_check:
        return 0

    report = check_transcript(output_path, language=args.language)
    if not report:
        return 0
    # The transcript is written either way; the exit code is how a script chained
    # with && finds out not to spend an API bill on it.
    print()
    print(report, file=sys.stderr)
    return 1


def command_translate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    input_path = existing_file(parser, args.input, "Input file")
    if input_path.suffix.lower() != ".srt":
        parser.error(f"Expected an .srt file, got: {input_path.name}")

    output_path = resolved(args.output) or translated_subtitle_path(
        input_path, args.target_language
    )
    if output_path.exists() and not args.overwrite and not args.dry_run:
        parser.error(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")

    translate_subtitle_file(
        input_path=input_path,
        output_path=output_path,
        config=build_translation_config(args, parser, require_key=not args.dry_run),
        debug_dir=resolved(args.debug_dir),
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

    subtitle_output = resolved(args.subtitle_output) or translated_subtitle_path(
        subtitle, args.target_language
    )
    video_output = resolved(args.video_output) or default_video_output(
        video, mux_mode=args.mux_mode, target_language=args.target_language
    )

    stage = build_translate_stage(
        args, parser, require_key=not args.dry_run, output_path=subtitle_output
    )
    translated = ensure_translated_subtitle(subtitle, stage=stage, dry_run=args.dry_run, label=None)
    options = build_mux_options(args, video=video, subtitle=translated, video_output=video_output)
    output = run_mux(options, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Done: {output}")
    return 0


def command_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    validate_hard_subtitle_options(parser, args)
    video = existing_file(parser, args.video, "Video file")
    subtitle = existing_file(parser, args.subtitle, "Subtitle file") if args.subtitle else None
    output_dir = resolved(args.output_dir) or video.parent
    video_output = resolved(args.video_output) or default_video_output(
        video, mux_mode=args.mux_mode, target_language=args.target_language
    )

    output = run_pipeline(
        video=video,
        subtitle=subtitle,
        output_dir=output_dir,
        transcribe_stage=build_transcribe_stage(args),
        translate_stage=build_translate_stage(
            args,
            parser,
            require_key=not args.dry_run,
            output_path=resolved(args.subtitle_output),
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
    validate_fit_options(parser, args)
    validate_voice_options(parser, args)
    video = existing_file(parser, args.video, "Video file")
    subtitle = existing_file(parser, args.subtitle, "Subtitle file") if args.subtitle else None
    output_dir = resolved(args.output_dir) or video.parent
    video_output = resolved(args.video_output) or default_dubbed_output(
        video, target_language=args.target_language
    )

    # Diarization comes first because it is the stage most likely to be turned
    # away for a missing token, and nothing before it is worth paying for twice.
    stages = 4 if args.diarize else 3
    turns = []
    if args.diarize:
        turns = ensure_speakers(
            video,
            options=build_diarize_options(args),
            dry_run=args.dry_run,
            label=f"[1/{stages}]",
        )

    source_subtitle = ensure_source_subtitle(
        video,
        subtitle=subtitle,
        output_dir=output_dir,
        stage=build_transcribe_stage(args),
        dry_run=args.dry_run,
        label=f"[{stages - 2}/{stages}]",
    )
    translate_stage = build_translate_stage(
        args,
        parser,
        require_key=not args.dry_run,
        output_path=resolved(args.subtitle_output),
    )
    translated = ensure_translated_subtitle(
        source_subtitle,
        stage=translate_stage,
        dry_run=args.dry_run,
        label=f"[{stages - 1}/{stages}]",
    )

    print(f"[{stages}/{stages}] Dub: {args.tts_engine} {dub_voicing(args)}")
    options = build_dub_options(
        args,
        parser,
        video=video,
        subtitle=translated,
        source_subtitle=source_subtitle,
        video_output=video_output,
        translation=translate_stage.require_config,
        turns=turns,
    )
    output = run_dub(options, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Done: {output}")
    return 0


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    handler: Callable[[argparse.Namespace, argparse.ArgumentParser], int]


COMMANDS = (
    Command(
        "transcribe",
        "Convert audio or video into text with Whisper.",
        build_transcribe_command,
        command_transcribe,
    ),
    Command(
        "translate",
        "Translate an .srt subtitle file with an OpenAI-compatible API.",
        build_translate_command,
        command_translate,
    ),
    Command(
        "mux",
        "Translate an .srt file and mux it back into a video.",
        build_mux_command,
        command_mux,
    ),
    Command(
        "run",
        "One command: video -> transcript -> translation -> subtitled video.",
        build_run_command,
        command_run,
    ),
    Command(
        "dub",
        "One command: video -> Chinese voice-over video.",
        build_dub_command,
        command_dub,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-txt",
        description="Transcribe, translate, subtitle and dub local audio or video files.",
    )
    parser.add_argument("--version", action="version", version=f"video-txt {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparser = subparsers.add_parser(command.name, help=command.help)
        command.add_arguments(subparser)
        subparser.set_defaults(handler=command.handler)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args, parser)
    except (
        CloneError,
        CredentialError,
        DiarizeError,
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
