from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .arguments import (
    build_audit_command,
    build_bilingual_command,
    build_clean_command,
    build_draft_terms_command,
    build_dub_command,
    build_music_command,
    build_mux_command,
    build_project_command,
    build_retranscribe_range_command,
    build_revise_command,
    build_run_command,
    build_transcribe_command,
    build_translate_command,
    build_translation_audit_command,
)
from .bilingual import (
    BilingualError,
    bilingual_cues,
    bilingual_subtitle_path,
    merge_subtitles,
)
from .clone import CLONE_ENGINES, CloneError, CloneOptions, is_rate, speed_from_rate
from .constants import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_BATCH_CHARS,
    DEFAULT_TERMS,
    PROVIDERS,
)
from .diarize import (
    DEFAULT_SPEAKER,
    DiarizeError,
    DiarizeOptions,
    SpeakerTurn,
    ensure_speakers,
)
from .draft import draft_terminology, draft_to_dict
from .dub import DubError, DubOptions, default_dubbed_output, run_dub
from .env import CredentialError, resolve_api_key, resolve_optional_key
from .fit import FitOptions
from .media import (
    MediaError,
    find_ffmpeg,
    parse_video_size,
    probe_audio_streams,
    select_audio_stream,
)
from .music import (
    MusicError,
    MusicPiece,
    extract_command,
    find_music,
    loudness_envelope,
    piece_filename,
    sung_spans,
)
from .mux import MuxError, MuxOptions, default_video_output, run_mux
from .pipeline import (
    TranscribeStage,
    TranslateStage,
    ensure_bilingual_subtitle,
    ensure_revised_subtitle,
    ensure_source_subtitle,
    ensure_translated_subtitle,
    run_pipeline,
)
from .project import (
    ProjectError,
    build_project_state,
    create_project_config,
    inspect_project,
    project_legacy_argv,
    untracked_project_outputs,
)
from .project import write_json as write_project_json
from .quality import audit_transcript, check_transcript, probe_media_duration, repair_transcript
from .retranscribe import (
    RetranscribeError,
    RetranscribeOptions,
    parse_timecode,
    run_retranscribe,
)
from .reuse import PreviousTranslation, ReuseError
from .revise import (
    RevisionError,
    apply_revisions,
    load_revisions,
    plan_revisions,
    revised_subtitle_path,
    revision_report_path_for,
)
from .separate import (
    SeparateError,
    ensure_instrumental,
    ensure_stems,
    separated_bgm_path,
    separated_voice_path,
)
from .subtitles import (
    SubtitleFormatError,
    language_code,
    language_suffix,
    parse_srt,
    translated_subtitle_path,
    write_srt,
)
from .terminology import (
    Terminology,
    TerminologyError,
    audit_translation,
    load_terminology,
    translation_audit_path_for,
    write_translation_audit_report,
)
from .transcribe import TranscribeError, TranscribeOptions, run_transcribe
from .transcribe import resolve_backend as resolve_transcription_backend
from .transcribe import resolve_model as resolve_whisper_model
from .translate import (
    TranslationConfig,
    TranslationError,
    loaded_local_model,
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
        lang=language_suffix(args.target_language),
        references=parse_clone_references(parser, args.clone_references),
    )


def resolve_provider_settings(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    discover_model: bool = True,
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
    if not model and provider.get("discover_model") and discover_model:
        model = discover_local_model(base_url, parser)
    elif not model and provider.get("discover_model"):
        model = "<loaded-local-model>"
    if not model:
        parser.error(
            "Missing model name. Pass --model, use --provider deepseek, or set OPENAI_MODEL."
        )
    return base_url, api_key_env, model


def discover_local_model(base_url: str, parser: argparse.ArgumentParser) -> str:
    """Whatever the local server is serving, so --model can be left off."""
    try:
        model = loaded_local_model(base_url)
    except TranslationError as exc:
        parser.error(f"{exc}\nLoad a model in LM Studio and start its server, or pass --model.")
    print(f"Local model: {model}")
    return model


def provider_requires_key(args: argparse.Namespace) -> bool:
    """Whether a missing key should stop the run. A model on localhost asks for none."""
    provider = PROVIDERS[args.provider] if args.provider else {}
    return bool(provider.get("requires_key", True))


def resolve_reasoning_effort(args: argparse.Namespace) -> str:
    provider = PROVIDERS[args.provider] if args.provider else {}
    if args.provider == "deepseek" and args.reasoning_effort == "none":
        # DeepSeek controls the off state with `thinking.type`, not an effort named none.
        return "auto"
    return args.reasoning_effort or provider.get("reasoning_effort") or "auto"


def resolve_thinking(args: argparse.Namespace) -> str:
    provider = PROVIDERS[args.provider] if args.provider else {}
    if args.provider == "deepseek" and args.reasoning_effort:
        if args.reasoning_effort == "none":
            return "disabled"
        if args.reasoning_effort != "auto":
            return "enabled"
    return provider.get("thinking") or "auto"


def resolve_batch_chars(args: argparse.Namespace) -> int:
    provider = PROVIDERS[args.provider] if args.provider else {}
    return args.batch_chars or provider.get("batch_chars") or DEFAULT_BATCH_CHARS


def validate_translation_numbers(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Settings the translator would otherwise quietly pull back into range."""
    if args.batch_chars is not None and args.batch_chars < 1:
        parser.error("--batch-chars must be 1 or greater")
    if args.retries < 0:
        parser.error("--retries must be 0 or greater")
    if args.concurrency < 1:
        parser.error("--concurrency must be 1 or greater")
    if args.context_cues < 0:
        parser.error("--context-cues must be 0 or greater")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")


def translation_language_key(value: str) -> str:
    suffix = language_suffix(value)
    return suffix if suffix != "translated" else re.sub(r"[\s_-]+", "", value).casefold()


def validate_terminology_languages(
    terminology: Terminology, *, source_language: str | None, target_language: str
) -> None:
    if terminology.target_language and translation_language_key(
        terminology.target_language
    ) != translation_language_key(target_language):
        raise TerminologyError(
            f"Terminology target language {terminology.target_language!r} does not match "
            f"the requested target language {target_language!r}."
        )
    if (
        terminology.source_language
        and source_language
        and translation_language_key(terminology.source_language)
        != translation_language_key(source_language)
    ):
        raise TerminologyError(
            f"Terminology source language {terminology.source_language!r} does not match "
            f"the requested source language {source_language!r}."
        )


def terminology_from_args(args: argparse.Namespace) -> Terminology | None:
    if not args.term_file:
        return None
    term_path = resolved(args.term_file)
    assert term_path is not None
    return load_terminology(term_path)


def build_translation_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    require_key: bool,
    spoken: bool = False,
    discover_model: bool = True,
    terminology: Terminology | None = None,
) -> TranslationConfig:
    validate_translation_numbers(parser, args)
    if terminology is None:
        terminology = terminology_from_args(args)
    if terminology is not None:
        validate_terminology_languages(
            terminology,
            source_language=args.source_language,
            target_language=args.target_language,
        )
    base_url, api_key_env, model = resolve_provider_settings(
        args, parser, discover_model=discover_model
    )
    read_key = resolve_api_key if provider_requires_key(args) else resolve_optional_key
    api_key = read_key(api_key_env, secrets_file=args.secrets_file) if require_key else ""
    return TranslationConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        target_language=args.target_language,
        source_language=args.source_language or (
            terminology.source_language if terminology is not None else None
        ),
        preserve_terms=list(dict.fromkeys([*DEFAULT_TERMS, *args.preserve_term])),
        terminology=terminology,
        note=args.note,
        spoken=spoken,
        batch_chars=resolve_batch_chars(args),
        retries=args.retries,
        concurrency=args.concurrency,
        context_cues=args.context_cues,
        thinking=resolve_thinking(args),
        reasoning_effort=resolve_reasoning_effort(args),
        timeout=args.timeout,
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
        bilingual=args.bilingual,
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
        separate_bgm=args.separate_bgm,
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
    output_dir: Path | None = None,
    spoken: bool = False,
) -> TranslateStage:
    terminology = terminology_from_args(args)
    if terminology is not None:
        validate_terminology_languages(
            terminology,
            source_language=args.source_language,
            target_language=args.target_language,
        )
    return TranslateStage(
        config=TranslationConfig(
            base_url="",
            api_key="",
            model="",
            target_language=args.target_language,
            source_language=args.source_language or (
                terminology.source_language if terminology is not None else None
            ),
            terminology=terminology,
        ),
        config_loader=lambda: build_translation_config(
            args,
            parser,
            require_key=require_key,
            spoken=spoken,
            discover_model=require_key,
            terminology=terminology,
        ),
        output_path=output_path,
        output_dir=output_dir,
        debug_dir=resolved(args.debug_dir),
        resume=args.resume,
        retranslate=args.retranslate,
        reuse_if_exists=getattr(args, "project_reuse_translation", False),
        previous=resolve_previous_translation(args, parser),
    )


def build_transcribe_stage(args: argparse.Namespace) -> TranscribeStage:
    return TranscribeStage(
        model=args.whisper_model,
        backend=args.whisper_backend,
        language=args.language,
        device=args.device,
        initial_prompt=args.initial_prompt,
        extra_args=args.whisper_args,
        refine_subtitles=args.refine_subtitles,
        audio_stream=args.audio_stream,
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
    if args.bgm_volume is not None and not 0 <= args.bgm_volume <= 1:
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
    if not args.diarize:
        for flag, value in counts.items():
            if value is not None:
                parser.error(f"{flag} needs --diarize")
        if args.rediarize:
            parser.error("--rediarize needs --diarize")
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
        refine_subtitles=args.refine_subtitles,
        audio_stream=args.audio_stream,
    )
    output_path = run_transcribe(options, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    print(f"Done: {output_path}")
    if output_path.suffix.lower() != ".srt" or args.skip_transcript_check:
        return 0

    report = check_transcript(output_path, language=args.language, media_path=input_path)
    if not report:
        return 0
    # The transcript is written either way; the exit code is how a script chained
    # with && finds out not to spend an API bill on it.
    print()
    print(report, file=sys.stderr)
    return 1


def command_retranscribe_range(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    video = existing_file(parser, args.video, "Video file")
    subtitle = existing_file(parser, args.subtitle, "Subtitle file")
    if subtitle.suffix.lower() != ".srt":
        parser.error(f"Expected an .srt file, got: {subtitle.name}")
    timed_range = args.from_time is not None or args.to_time is not None
    cue_range = args.from_cue is not None or args.to_cue is not None
    if timed_range and cue_range:
        parser.error("Use either --from/--to or --from-cue/--to-cue, not both.")
    if not timed_range and not cue_range:
        parser.error("Pass --from and --to, or --from-cue and --to-cue.")
    if cue_range:
        if args.from_cue is None or args.to_cue is None:
            parser.error("--from-cue and --to-cue must be passed together.")
        source_cues = parse_srt(subtitle)
        if args.from_cue < 1 or args.to_cue < args.from_cue:
            parser.error("Cue range must start at 1 or greater and end at or after its start.")
        if args.to_cue > len(source_cues):
            parser.error(
                f"Cue range ends at {args.to_cue}, but {subtitle.name} has "
                f"only {len(source_cues)} cues."
            )
        start = source_cues[args.from_cue - 1].start_seconds
        end = source_cues[args.to_cue - 1].end_seconds
    else:
        if args.from_time is None or args.to_time is None:
            parser.error("--from and --to must be passed together.")
        start = parse_timecode(args.from_time)
        end = parse_timecode(args.to_time)

    output = resolved(args.output) or subtitle.with_name(f"{subtitle.stem}.repaired.srt")
    assert output is not None
    report = resolved(args.report) or output.with_name(
        f"{output.stem}.retranscription-report.json"
    )
    options = RetranscribeOptions(
        video_path=video,
        subtitle_path=subtitle,
        output_path=output,
        report_path=report,
        start=start,
        end=end,
        padding=args.padding,
        model=args.whisper_model,
        language=args.language,
        backend=args.whisper_backend,
        device=args.device,
        initial_prompt=args.initial_prompt,
        extra_args=args.whisper_args,
        refine_subtitles=args.refine_subtitles,
        audio_stream=args.audio_stream,
        overwrite=args.overwrite,
        from_cue=args.from_cue,
        to_cue=args.to_cue,
    )
    output = run_retranscribe(options, dry_run=args.dry_run)
    if args.dry_run:
        return 0
    print(f"Repaired: {output}")
    print(f"Report: {report}")
    return 0


def write_json(path: Path, payload: dict[str, object]) -> Path:
    """Atomically write a generated JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def command_audit(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    subtitle = existing_file(parser, args.input, "Subtitle file")
    if subtitle.suffix.lower() != ".srt":
        parser.error(f"Expected an .srt file, got: {subtitle.name}")
    media = existing_file(parser, args.media, "Media file") if args.media else None
    report_path = resolved(args.report) or subtitle.with_suffix(".audit.json")
    if report_path == subtitle:
        parser.error("The JSON report cannot replace the source subtitle.")
    if report_path.exists() and not args.overwrite:
        parser.error(f"Report already exists: {report_path}. Pass --overwrite to replace it.")

    report = audit_transcript(
        parse_srt(subtitle),
        language=args.language,
        media_duration=probe_media_duration(media) if media else None,
    )
    payload = {
        "schema": "video-txt.subtitle-audit",
        "version": 1,
        "source": str(subtitle),
        "media": str(media) if media else None,
        **report.to_dict(),
    }
    write_json(report_path, payload)
    noun = "finding" if len(report.findings) == 1 else "findings"
    print(f"Audit: {len(report.findings)} {noun} across {report.cue_count} cues")
    print(f"Report: {report_path}")
    return 0 if report.is_clean else 1


def command_translation_audit(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    source = existing_file(parser, args.source, "Source subtitle")
    translation = existing_file(parser, args.translation, "Translated subtitle")
    for label, path in (("Source", source), ("Translation", translation)):
        if path.suffix.lower() != ".srt":
            parser.error(f"{label} must be an .srt file, got: {path.name}")

    term_file = (
        existing_file(parser, args.term_file, "Terminology file") if args.term_file else None
    )
    terminology = load_terminology(term_file) if term_file is not None else None
    report_path = resolved(args.report) or translation_audit_path_for(translation)
    protected = {source, translation}
    if term_file is not None:
        protected.add(term_file)
    if report_path in protected:
        parser.error("The JSON report cannot replace an input subtitle or terminology file.")
    if report_path.exists() and not args.overwrite:
        parser.error(f"Report already exists: {report_path}. Pass --overwrite to replace it.")

    source_cues = parse_srt(source)
    revised: set[int] = set()
    if args.revisions:
        revisions_path = existing_file(parser, args.revisions, "Revision file")
        if report_path == revisions_path:
            parser.error("The JSON report cannot replace the revision file.")
        revised = set(plan_revisions(source_cues, load_revisions(revisions_path).revisions))

    report = audit_translation(
        source_cues, parse_srt(translation), terminology, revised=revised
    )
    write_translation_audit_report(
        source_path=source,
        translation_path=translation,
        terminology=terminology,
        audit=report,
        report_path=report_path,
        overwrite=args.overwrite,
    )
    noun = "finding" if len(report.findings) == 1 else "findings"
    accepted = report.summary["accepted"]
    print(
        f"Translation audit: {len(report.findings)} {noun}"
        + (f", {accepted} accepted on hand-revised lines" if accepted else "")
    )
    print(f"Report: {report_path}")
    return 0 if report.is_clean else 1


def default_music_dir(media: Path) -> Path:
    return media.with_name(f"{media.stem}.music")


def command_music(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    media = existing_file(parser, args.media, "Media file")
    output_dir = resolved(args.output_dir) or default_music_dir(media)
    cache_dir = resolved(args.cache_dir) or media.with_name(f"{media.stem}.dub-cache")
    if args.min_duration <= 0:
        parser.error("--min-duration must be greater than 0")
    named = args.from_time is not None or args.to_time is not None
    if named and (args.from_time is None or args.to_time is None):
        parser.error("--from and --to must be passed together.")
    ffmpeg_path = find_ffmpeg(explicit=None)

    print(f"Media: {media}")
    print(f"Separated audio cache: {cache_dir}")
    print(f"Music: {output_dir}")
    if args.dry_run and not cache_dir.is_dir():
        print("Would separate the soundtrack, then search it for music (dry run).")
        return 0

    # The dub's own cache, reused: a film already separated for --separate-bgm
    # costs nothing here.
    instrumental = ensure_instrumental(media, cache_dir=cache_dir, ffmpeg_path=ffmpeg_path)
    voice = separated_voice_path(cache_dir)
    if not voice.is_file():
        raise MusicError(
            f"The separated voice track is missing: {voice}. It is what says which "
            f"stretches nobody talks over. Delete {separated_bgm_path(cache_dir).parent} "
            "and run again to rebuild both."
        )

    if named:
        pieces = [
            MusicPiece(
                start=parse_timecode(args.from_time),
                end=parse_timecode(args.to_time),
                voice_share=0.0,
                peak_lufs=0.0,
                labelled="named",
            )
        ]
        sources = {0: media if args.source == "mix" else instrumental}
    else:
        print("Measuring the soundtrack...")
        pieces = find_music(
            loudness_envelope(instrumental, ffmpeg_path=ffmpeg_path),
            loudness_envelope(voice, ffmpeg_path=ffmpeg_path),
            music_floor=args.music_floor,
            voice_floor=args.voice_floor,
            min_duration=args.min_duration,
            clean_only=args.clean,
        )
        sources = dict.fromkeys(range(len(pieces)), instrumental)
        if args.subtitle:
            songs = sung_spans(parse_srt(existing_file(parser, args.subtitle, "Subtitle file")))
            print(f"Songs marked in the subtitle: {len(songs)}")
            # A song is cut from the soundtrack itself: without its singing it is
            # not the song, and the singing is the half separation takes out.
            for song in songs:
                sources[len(pieces)] = media
                pieces.append(song)

    stems: dict[str, Path] = {}
    if args.stems and not args.dry_run:
        stems = ensure_stems(
            media, model=args.model, cache_dir=cache_dir, ffmpeg_path=ffmpeg_path
        )

    print(f"Pieces found: {len(pieces)}")
    for number, piece in enumerate(pieces, start=1):
        print(
            f"  {number:02d}  {piece.kind:15s} {piece.clock_range} "
            f"({piece.duration:6.1f}s, voice {piece.voice_share:.0%})"
        )
    if args.dry_run:
        print(f"Would write {len(pieces)} file(s) to: {output_dir}")
        return 0
    if not pieces:
        print("Nothing long enough to write out. Lower --min-duration to widen the search.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, object]] = []
    for number, piece in enumerate(pieces, start=1):
        target = output_dir / piece_filename(media, number, piece)
        if target.exists() and not args.overwrite:
            parser.error(f"Music file already exists: {target}. Pass --overwrite to replace it.")
        completed = subprocess.run(
            extract_command(
                sources.get(number - 1, instrumental),
                piece,
                output=target,
                ffmpeg_path=ffmpeg_path,
                normalize=not args.raw_levels,
            ),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not target.is_file():
            raise MusicError(f"Could not cut {target.name}: {(completed.stderr or '').strip()}")
        written.append({**piece.to_dict(), "file": target.name})

    report = output_dir / f"{media.stem}.music.json"
    write_json(
        report,
        {
            "schema": "video-txt.music",
            "version": 1,
            "media": str(media),
            "instrumental": str(instrumental),
            "stems": {name: str(path) for name, path in stems.items()},
            "normalized": not args.raw_levels,
            "piece_count": len(written),
            "pieces": written,
        },
    )
    print(f"Wrote {len(written)} music file(s) to: {output_dir}")
    print(f"Whole-film instrumental: {instrumental}")
    if stems:
        print(f"Stems: {', '.join(sorted(stems))} in {next(iter(stems.values())).parent}")
    print(f"Report: {report}")
    return 0


def command_bilingual(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    source = existing_file(parser, args.source, "Source subtitle")
    translation = existing_file(parser, args.translation, "Translated subtitle")
    for label, path in (("Source", source), ("Translation", translation)):
        if path.suffix.lower() != ".srt":
            parser.error(f"{label} must be an .srt file, got: {path.name}")

    output = resolved(args.output) or bilingual_subtitle_path(translation)
    if output in {source, translation}:
        parser.error("--output must be a new path; bilingual never overwrites an input file.")
    if output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {output}. Pass --overwrite to replace it.")

    merged = merge_subtitles(parse_srt(source), parse_srt(translation), order=args.order)
    write_srt(output, bilingual_cues(merged))
    print(f"Bilingual: {len(merged)} cue(s) in both languages ({args.order})")
    print(f"Wrote: {output}")
    return 0


def command_draft_terms(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    subtitles = [
        existing_file(parser, path, "Subtitle file") for path in args.subtitles
    ]
    for path in subtitles:
        if path.suffix.lower() != ".srt":
            parser.error(f"Expected an .srt file, got: {path.name}")
    if args.min_count < 1:
        parser.error("--min-count must be 1 or greater")
    if args.limit < 1:
        parser.error("--limit must be 1 or greater")

    existing_terms = None
    against = None
    if args.against:
        against = existing_file(parser, args.against, "Terminology file")
        existing_terms = load_terminology(against)

    output = resolved(args.output)
    assert output is not None
    if output in {*subtitles, against}:
        parser.error("--output must be a new path; the draft never replaces an input file.")
    if output.exists() and not args.overwrite:
        parser.error(f"Draft already exists: {output}. Pass --overwrite to replace it.")

    cues = [cue for path in subtitles for cue in parse_srt(path)]
    draft = draft_terminology(
        cues,
        existing=existing_terms,
        min_count=args.min_count,
        max_terms=args.limit,
    )
    write_json(
        output,
        draft_to_dict(
            draft,
            source_language=args.source_language,
            target_language=args.target_language,
        ),
    )

    print(
        f"Scanned {draft.scanned_cues} cues in {len(subtitles)} subtitle file(s); "
        f"proposing {len(draft.candidates)} name(s)."
    )
    if draft.already_covered:
        print(f"Already in {against}: {len(draft.already_covered)} name(s), left out.")
    for candidate in draft.candidates:
        if candidate.warning:
            print(f"  {candidate.source}: {candidate.warning}", file=sys.stderr)
    print(f"Draft: {output}")
    print(
        "Every 'target' is blank, so the file will not load as a glossary until they are "
        "filled in. Delete the names that are not worth an entry, then pass it as --term-file."
    )
    return 0


def command_clean(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    subtitle = existing_file(parser, args.input, "Subtitle file")
    if subtitle.suffix.lower() != ".srt":
        parser.error(f"Expected an .srt file, got: {subtitle.name}")
    media = existing_file(parser, args.media, "Media file") if args.media else None
    output = resolved(args.output)
    assert output is not None
    if output.suffix.lower() != ".srt":
        parser.error(f"Expected an .srt output file, got: {output.name}")
    if output == subtitle:
        parser.error("--output must be a new path; clean never overwrites the source subtitle.")
    report_path = resolved(args.report) or output.with_suffix(".audit.json")
    if report_path in {subtitle, output}:
        parser.error("The JSON report must not replace the source or cleaned subtitle.")
    for label, path in (("Output file", output), ("Report", report_path)):
        if path.exists() and not args.overwrite and not args.dry_run:
            parser.error(f"{label} already exists: {path}. Pass --overwrite to replace it.")

    cues = parse_srt(subtitle)
    duration = probe_media_duration(media) if media else None
    report = audit_transcript(cues, language=args.language, media_duration=duration)
    result = repair_transcript(cues, report)
    post_repair = audit_transcript(
        result.cues,
        language=args.language,
        media_duration=duration,
    )
    if args.dry_run:
        print(
            f"Would clean {subtitle}: remove {result.removed_count}, "
            f"merge {result.merged_count}, leave {len(post_repair.findings)} findings"
        )
        print(f"Would write: {output}")
        print(f"Would report: {report_path}")
        return 1 if post_repair.has_errors else 0

    write_srt(output, result.cues)
    payload = {
        "schema": "video-txt.subtitle-audit",
        "version": 1,
        "source": str(subtitle),
        "media": str(media) if media else None,
        "output": str(output),
        **report.to_dict(),
        "repair": result.to_dict(),
        "post_repair": post_repair.to_dict(),
    }
    write_json(report_path, payload)
    print(
        f"Cleaned: {output} "
        f"(removed {result.removed_count}, merged {result.merged_count})"
    )
    print(f"Report: {report_path}")
    if not post_repair.is_clean:
        print(f"Manual review still needed: {len(post_repair.findings)} findings", file=sys.stderr)
    return 1 if post_repair.has_errors else 0


def command_revise(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    source = existing_file(parser, args.source, "Source subtitle")
    translation = existing_file(parser, args.translation, "Translated subtitle")
    revisions_path = existing_file(parser, args.revisions, "Revision file")
    for label, path in (("Source", source), ("Translation", translation)):
        if path.suffix.lower() != ".srt":
            parser.error(f"{label} must be an .srt file, got: {path.name}")

    output = resolved(args.output) or revised_subtitle_path(translation)
    report_path = resolved(args.report) or revision_report_path_for(output)
    inputs = {source, translation, revisions_path}
    if output in inputs:
        parser.error("--output must be a new path; revise never overwrites an input file.")
    if report_path in inputs | {output}:
        parser.error("The JSON report must not replace an input subtitle or the output.")
    for label, path in (("Output file", output), ("Report", report_path)):
        if path.exists() and not args.overwrite and not args.dry_run:
            parser.error(f"{label} already exists: {path}. Pass --overwrite to replace it.")

    revision_set = load_revisions(revisions_path)
    result = apply_revisions(parse_srt(source), parse_srt(translation), revision_set)

    for item in result.drifted:
        print(
            f"Anchor moved: cue {item.cue_index} is now {item.drift:.1f}s from where "
            f"{item.revision.source!r} used to be.",
            file=sys.stderr,
        )
    if args.dry_run:
        for item in result.applied:
            state = "would change" if item.changed else "already current"
            print(f"  cue {item.cue_index} ({state}): {item.before!r} -> {item.after!r}")
        print(f"Would write: {output}")
        print(f"Would report: {report_path}")
        return 0

    write_srt(output, result.cues)
    write_json(
        report_path,
        {
            "schema": "video-txt.revision-report",
            "version": 1,
            "source": str(source),
            "translation": str(translation),
            "revisions": str(revisions_path),
            "output": str(output),
            **result.to_dict(),
        },
    )
    already = len(result.already_current)
    print(
        f"Revised: {output} ({result.changed_count} of {len(result.applied)} line(s) changed"
        + (f", {already} already current" if already else "")
        + ")"
    )
    print(f"Report: {report_path}")
    return 0


def resolve_previous_translation(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> PreviousTranslation | None:
    """The source/translation pair --reuse points at, with the naming rule filled in."""
    if args.reuse is None:
        if args.reuse_translation is not None:
            parser.error("--reuse-translation needs --reuse, the source it was translated from.")
        return None
    source = existing_file(parser, args.reuse, "Previous source subtitle")
    translation = resolved(args.reuse_translation) or translated_subtitle_path(
        source, args.target_language
    )
    if not translation.is_file():
        parser.error(
            f"No translation of {source.name} at {translation}. "
            "Point at it with --reuse-translation."
        )
    return PreviousTranslation(source=source, translation=translation)


def command_translate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    input_path = existing_file(parser, args.input, "Input file")
    if input_path.suffix.lower() != ".srt":
        parser.error(f"Expected an .srt file, got: {input_path.name}")

    output_path = resolved(args.output) or translated_subtitle_path(
        input_path, args.target_language
    )
    if output_path.exists() and not args.overwrite and not args.dry_run:
        parser.error(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")
    report_path = translation_audit_path_for(output_path)
    if report_path.exists() and not args.overwrite and not args.dry_run:
        parser.error(f"Report already exists: {report_path}. Pass --overwrite to replace it.")

    previous = resolve_previous_translation(args, parser)

    translate_subtitle_file(
        input_path=input_path,
        output_path=output_path,
        previous=previous,
        config=build_translation_config(
            args,
            parser,
            require_key=not args.dry_run,
            discover_model=not args.dry_run,
        ),
        debug_dir=resolved(args.debug_dir),
        resume=args.resume,
        dry_run=args.dry_run,
        overwrite_audit=args.overwrite,
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
    translated = ensure_revised_subtitle(
        subtitle,
        translated,
        revisions=resolved(args.revisions),
        terminology=stage.config.terminology,
        dry_run=args.dry_run,
    )
    translated = ensure_bilingual_subtitle(
        subtitle,
        translated,
        bilingual=args.bilingual,
        order=args.bilingual_order,
        dry_run=args.dry_run,
    )
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
            output_dir=output_dir,
        ),
        mux_options_for=lambda translated: build_mux_options(
            args, video=video, subtitle=translated, video_output=video_output
        ),
        revisions=resolved(args.revisions),
        bilingual=args.bilingual,
        bilingual_order=args.bilingual_order,
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
        output_dir=output_dir,
        # The lines are going to be spoken, so they should read like speech.
        # An already-translated subtitle is reused as is: --retranslate redoes it.
        spoken=True,
    )
    translated = ensure_translated_subtitle(
        source_subtitle,
        stage=translate_stage,
        dry_run=args.dry_run,
        label=f"[{stages - 1}/{stages}]",
    )
    translated = ensure_revised_subtitle(
        source_subtitle,
        translated,
        revisions=resolved(args.revisions),
        terminology=translate_stage.config.terminology,
        dry_run=args.dry_run,
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


def command_project(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.project_action == "status":
        inspection = inspect_project(args.project_file)
        print(f"Project: {inspection.project_path}")
        print(f"Video: {inspection.paths['video']}")
        audio_selection = inspection.state.get("audio_selection")
        if isinstance(audio_selection, dict):
            selected = audio_selection.get("selected_stream")
            reason = audio_selection.get("reason")
            print(
                f"Selected audio stream: {selected or 'unknown'} — "
                f"{reason or 'no reason recorded'}"
            )
        transcription_resolution = inspection.state.get("transcription_resolution")
        if isinstance(transcription_resolution, dict):
            print(
                "Whisper: "
                f"{transcription_resolution.get('backend') or 'unknown'} / "
                f"{transcription_resolution.get('model') or 'unknown'}"
            )
        print(f"Translated subtitle: {inspection.paths['translated_subtitle']}")
        print(f"Video output: {inspection.paths['video_output']}")
        for name, status in inspection.statuses.items():
            label = "stale" if status.stale else status.reason
            print(f"{name}: {label} — {status.reason}" if status.stale else f"{name}: {label}")
        return 1 if inspection.stale else 0
    if args.project_action == "run":
        inspection = inspect_project(args.project_file)
        transcription = inspection.config["transcription"]
        translation = inspection.config["translation"]
        assert isinstance(transcription, dict)
        assert isinstance(translation, dict)
        print(f"Project: {inspection.project_path}")
        print(f"Requested audio stream: {transcription.get('audio_stream') or 'auto'}")
        print(f"Spoken language: {transcription.get('language') or 'auto'}")
        if "terminology" in inspection.paths:
            print(f"Terminology file: {inspection.paths['terminology']}")
        print(f"Translation model: {translation.get('model') or 'provider default'}")
        print(f"Translated subtitle: {inspection.paths['translated_subtitle']}")
        print(f"Video output: {inspection.paths['video_output']}")
        if not inspection.stale:
            print("All enabled stages are current; nothing to run.")
            return 0
        if not args.dry_run:
            untracked = untracked_project_outputs(inspection)
            if untracked:
                listed = "\n".join(f"  {path}" for path in untracked)
                raise ProjectError(
                    "Refusing to replace output files that are not recorded in project state:\n"
                    f"{listed}\nMove them aside, choose new output paths, "
                    "or use --dry-run to inspect."
                )
        legacy_argv = project_legacy_argv(inspection, dry_run=args.dry_run)
        legacy_args = build_parser().parse_args(legacy_argv)
        legacy_args.project_reuse_translation = not inspection.statuses["translate"].stale
        code = legacy_args.handler(legacy_args, parser)
        if code != 0 or args.dry_run:
            return code
        requested_stream = transcription.get("audio_stream")
        if isinstance(requested_stream, int):
            audio_selection: dict[str, object] = {
                "requested_stream": requested_stream,
                "selected_stream": requested_stream,
                "reason": f"selected explicitly with --audio-stream {requested_stream}",
            }
        else:
            try:
                selected = select_audio_stream(
                    probe_audio_streams(inspection.paths["video"]),
                    preferred_language=(
                        str(transcription["language"])
                        if transcription.get("language") is not None
                        else None
                    ),
                )
                audio_selection = {
                    "requested_stream": None,
                    "selected_stream": selected.stream.index,
                    "reason": selected.reason,
                }
            except MediaError as exc:
                audio_selection = {
                    "requested_stream": None,
                    "selected_stream": None,
                    "reason": f"could not inspect after run: {exc}",
                }
        refreshed = inspect_project(inspection.project_path)
        transcription_resolution = None
        if transcription.get("enabled", True):
            actual_backend = resolve_transcription_backend(
                str(transcription.get("backend") or "auto")
            )
            transcription_resolution = {
                "backend": actual_backend,
                "model": resolve_whisper_model(
                    mode="transcribe",
                    model=(
                        str(transcription["model"])
                        if transcription.get("model") is not None
                        else None
                    ),
                    backend=actual_backend,
                ),
                "language": transcription.get("language"),
            }
        write_project_json(
            refreshed.state_path,
            build_project_state(
                refreshed,
                audio_selection=audio_selection,
                transcription_resolution=transcription_resolution,
            ),
        )
        print(f"State: {refreshed.state_path}")
        return 0
    if args.project_action != "init":
        parser.error(f"Unknown project action: {args.project_action}")
    # The same checks the workflow's own command runs, run now: `project run`
    # replays this configuration as that command, so a combination it would
    # reject has to be caught while it is being typed, not a stage later.
    if args.workflow == "dub":
        validate_fit_options(parser, args)
        validate_voice_options(parser, args)
    else:
        validate_hard_subtitle_options(parser, args)
    project_path = resolved(args.output)
    assert project_path is not None
    if project_path.exists() and not args.overwrite:
        parser.error(
            f"Project file already exists: {project_path}. Pass --overwrite to replace it."
        )
    video = existing_file(parser, args.video, "Video file")
    subtitle = existing_file(parser, args.subtitle, "Subtitle file") if args.subtitle else None
    term_file = (
        existing_file(parser, args.term_file, "Terminology file") if args.term_file else None
    )
    if term_file is not None:
        validate_terminology_languages(
            load_terminology(term_file),
            source_language=args.source_language,
            target_language=args.target_language,
        )
    revisions_file = (
        existing_file(parser, args.revisions, "Revision file") if args.revisions else None
    )
    if revisions_file is not None:
        # Read it now: a malformed revision file is worth hearing about while the
        # project is being set up, not on the run that was meant to use it.
        load_revisions(revisions_file)
    payload = create_project_config(
        project_path=project_path,
        video=video,
        args=args,
        term_file=term_file,
        revisions_file=revisions_file,
        subtitle=subtitle,
    )
    write_project_json(project_path, payload)
    print(f"Project: {project_path}")
    return 0


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    handler: Callable[[argparse.Namespace, argparse.ArgumentParser], int]


COMMANDS = (
    Command(
        "project",
        "Create, inspect and run reproducible video projects.",
        build_project_command,
        command_project,
    ),
    Command(
        "audit",
        "Inspect an .srt file for timing, hallucination and layout defects.",
        build_audit_command,
        command_audit,
    ),
    Command(
        "audit-translation",
        "Compare source and translated subtitles for structure and terminology defects.",
        build_translation_audit_command,
        command_translation_audit,
    ),
    Command(
        "music",
        "Separate a soundtrack and cut the music out of it as playable files.",
        build_music_command,
        command_music,
    ),
    Command(
        "bilingual",
        "Merge a source and a translated .srt into one subtitle carrying both.",
        build_bilingual_command,
        command_bilingual,
    ),
    Command(
        "draft-terms",
        "Propose a project glossary from the names a source subtitle keeps using.",
        build_draft_terms_command,
        command_draft_terms,
    ),
    Command(
        "clean",
        "Write a new .srt with only high-confidence safe repairs applied.",
        build_clean_command,
        command_clean,
    ),
    Command(
        "revise",
        "Apply hand-corrected lines to a translated .srt, anchored on the source text.",
        build_revise_command,
        command_revise,
    ),
    Command(
        "transcribe",
        "Convert audio or video into text with Whisper.",
        build_transcribe_command,
        command_transcribe,
    ),
    Command(
        "retranscribe-range",
        "Re-run Whisper for one subtitle range and splice it into a new SRT.",
        build_retranscribe_range_command,
        command_retranscribe_range,
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
        MusicError,
        MuxError,
        ProjectError,
        RetranscribeError,
        BilingualError,
        ReuseError,
        RevisionError,
        SeparateError,
        SubtitleFormatError,
        TerminologyError,
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
