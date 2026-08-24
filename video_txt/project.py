from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_BATCH_CHARS,
    DEFAULT_CONCURRENCY,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TERMS,
    DEFAULT_TIMEOUT,
    PROVIDERS,
)
from .diarize import speakers_path
from .dub import default_audio_output, default_cache_dir, default_dubbed_output
from .mux import default_video_output
from .subtitles import translated_subtitle_path
from .terminology import load_terminology, translation_audit_path_for

PROJECT_SCHEMA = "video-txt.project"
PROJECT_VERSION = 1
STATE_SCHEMA = "video-txt.project-state"
STATE_VERSION = 1


class ProjectError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageStatus:
    name: str
    stale: bool
    reason: str


@dataclass(frozen=True)
class ProjectInspection:
    project_path: Path
    config: dict[str, object]
    state_path: Path
    state: dict[str, object]
    paths: dict[str, Path]
    desired: dict[str, dict[str, object]]
    artifacts: dict[str, bool]
    enabled: set[str]
    statuses: dict[str, StageStatus]

    @property
    def stale(self) -> bool:
        return any(self.statuses[name].stale for name in self.enabled)


STAGE_ORDER = ("transcribe", "translate", "mux", "tts")
STAGE_DEPENDENCIES = {
    "transcribe": (),
    "translate": ("transcribe",),
    "mux": ("translate",),
    "tts": ("translate",),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProjectError(f"Could not fingerprint {path}: {exc}") from exc
    return digest.hexdigest()


def fingerprint_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stage_record(inputs: Mapping[str, object]) -> dict[str, object]:
    return {"fingerprint": fingerprint_payload(inputs), "inputs": dict(inputs)}


def _changed_input_reason(
    desired: Mapping[str, object], recorded: Mapping[str, object]
) -> str:
    earlier = recorded.get("inputs")
    if not isinstance(earlier, Mapping):
        return "configuration or content changed"
    changed = [
        key
        for key in sorted(set(desired) | set(earlier))
        if desired.get(key) != earlier.get(key)
    ]
    labels = [key.removesuffix("_sha256").replace("_", " ") for key in changed]
    return f"{', '.join(labels)} changed" if labels else "configuration or content changed"


def evaluate_stages(
    desired: Mapping[str, Mapping[str, object]],
    *,
    recorded: Mapping[str, object],
    artifacts: Mapping[str, bool],
    enabled: set[str],
) -> dict[str, StageStatus]:
    """Compare desired stage inputs with one successful state, then propagate the DAG."""
    statuses: dict[str, StageStatus] = {}
    for name in STAGE_ORDER:
        if name not in enabled:
            statuses[name] = StageStatus(name, False, "disabled for this workflow")
            continue
        stale_upstream = next(
            (
                dependency
                for dependency in STAGE_DEPENDENCIES[name]
                if dependency in enabled and statuses[dependency].stale
            ),
            None,
        )
        if stale_upstream is not None:
            statuses[name] = StageStatus(name, True, f"upstream {stale_upstream} is stale")
            continue
        if not artifacts.get(name, False):
            statuses[name] = StageStatus(name, True, "artifact missing")
            continue
        previous = recorded.get(name)
        if not isinstance(previous, Mapping):
            statuses[name] = StageStatus(name, True, "no recorded successful run")
            continue
        inputs = desired[name]
        if previous.get("fingerprint") != fingerprint_payload(inputs):
            statuses[name] = StageStatus(name, True, _changed_input_reason(inputs, previous))
            continue
        statuses[name] = StageStatus(name, False, "current")
    return statuses


def relative_path(path: Path, *, base: Path) -> str:
    return os.path.relpath(path.expanduser().resolve(), base.expanduser().resolve())


def resolve_output_path(value: Path | None, *, base: Path, default: Path) -> Path:
    if value is None:
        return default
    expanded = value.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def _path_record(path: Path, *, base: Path) -> dict[str, str]:
    return {"path": relative_path(path, base=base), "sha256": sha256_file(path)}


def _portable_executable(value: str | None, *, base: Path) -> str | None:
    if value is None or ("/" not in value and not value.startswith((".", "~"))):
        return value
    return relative_path(Path(value), base=base)


def _clone_reference_records(values: list[str], *, base: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for value in values:
        speaker, separator, raw_path = value.partition("=")
        if not separator:
            speaker, raw_path = "", speaker
        reference = Path(raw_path).expanduser().resolve()
        records.append(
            {"speaker": speaker.strip(), "path": relative_path(reference, base=base)}
        )
    return records


def create_project_config(
    *,
    project_path: Path,
    video: Path,
    args: Any,
    term_file: Path | None,
    subtitle: Path | None,
) -> dict[str, object]:
    base = project_path.parent.resolve()
    video = video.resolve()
    output_dir = resolve_output_path(args.output_dir, base=base, default=video.parent)
    source_subtitle = subtitle or output_dir / f"{video.stem}.srt"
    translated_subtitle = resolve_output_path(
        args.subtitle_output,
        base=base,
        default=output_dir / translated_subtitle_path(source_subtitle, args.target_language).name,
    )
    workflow = args.workflow
    default_output = (
        default_dubbed_output(video, target_language=args.target_language)
        if workflow == "dub"
        else default_video_output(
            video,
            mux_mode=args.mux_mode,
            target_language=args.target_language,
        )
    )
    video_output = resolve_output_path(args.video_output, base=base, default=default_output)
    cache_dir = resolve_output_path(
        args.cache_dir, base=base, default=default_cache_dir(video)
    )
    dub_audio = resolve_output_path(
        args.audio_output, base=base, default=default_audio_output(video_output)
    )

    provider_name = args.provider or "openai"
    provider = PROVIDERS[provider_name]
    translation_model = args.model or provider.get("model")
    loaded_terminology = load_terminology(term_file) if term_file is not None else None
    terminology = _path_record(term_file, base=base) if term_file is not None else None
    translation_debug = (
        resolve_output_path(args.debug_dir, base=base, default=base)
        if args.debug_dir is not None
        else None
    )

    return {
        "schema": PROJECT_SCHEMA,
        "version": PROJECT_VERSION,
        "workflow": workflow,
        "video": _path_record(video, base=base),
        "paths": {
            "source_subtitle": relative_path(source_subtitle, base=base),
            "translated_subtitle": relative_path(translated_subtitle, base=base),
            "translation_audit": relative_path(
                translation_audit_path_for(translated_subtitle), base=base
            ),
            "video_output": relative_path(video_output, base=base),
            "output_dir": relative_path(output_dir, base=base),
            "subtitle_audit": relative_path(
                source_subtitle.with_suffix(".audit.json"), base=base
            ),
            "word_timestamps": relative_path(
                output_dir / f"{video.stem}.words.json", base=base
            ),
            "speaker_turns": relative_path(speakers_path(video), base=base),
            "dub_audio": relative_path(dub_audio, base=base),
            "cache_dir": relative_path(cache_dir, base=base),
            "translation_debug": (
                relative_path(translation_debug, base=base)
                if translation_debug is not None
                else None
            ),
        },
        "transcription": {
            "enabled": subtitle is None,
            "audio_stream": args.audio_stream,
            "backend": args.whisper_backend,
            "model": args.whisper_model,
            "language": args.language,
            "device": args.device,
            "initial_prompt": args.initial_prompt,
            "whisper_args": args.whisper_args,
            "refine_subtitles": args.refine_subtitles,
            "skip_transcript_check": args.skip_transcript_check,
        },
        "translation": {
            "provider": provider_name,
            "model": translation_model,
            "base_url": args.base_url or provider.get("base_url"),
            "credential_env": (
                args.api_key_env or provider.get("api_key_env") or "OPENAI_API_KEY"
            ),
            "credentials_file": str(args.secrets_file),
            "target_language": args.target_language or DEFAULT_TARGET_LANGUAGE,
            "source_language": args.source_language or (
                loaded_terminology.source_language if loaded_terminology is not None else None
            ),
            "batch_chars": args.batch_chars or provider.get("batch_chars") or DEFAULT_BATCH_CHARS,
            "retries": args.retries,
            "concurrency": args.concurrency or DEFAULT_CONCURRENCY,
            "context_cues": args.context_cues,
            "reasoning_effort": (
                args.reasoning_effort or provider.get("reasoning_effort") or "auto"
            ),
            "thinking": provider.get("thinking") or "auto",
            "timeout": args.timeout or DEFAULT_TIMEOUT,
            "resume": args.resume,
            "note": args.note,
            "preserve_terms": list(dict.fromkeys([*DEFAULT_TERMS, *args.preserve_term])),
            "terminology": terminology,
        },
        "mux": {
            "mode": args.mux_mode,
            "language_code": args.language_code,
            "default_subtitle": args.default_subtitle,
            "subtitle_codec": args.subtitle_codec,
            "hard_subtitle_layout": args.hard_subtitle_layout,
            "hard_subtitle_font": args.hard_subtitle_font,
            "hard_subtitle_font_size": args.hard_subtitle_font_size,
            "hard_subtitle_margin_v": args.hard_subtitle_margin_v,
            "hard_subtitle_box_height": args.hard_subtitle_box_height,
            "hard_subtitle_box_opacity": args.hard_subtitle_box_opacity,
            "video_codec": args.video_codec,
            "crf": args.crf,
            "preset": args.preset,
            "video_size": args.video_size,
            "ffmpeg": _portable_executable(args.ffmpeg_path, base=base),
        },
        "tts": {
            "engine": args.tts_engine,
            "voice": args.voice,
            "voice_unit": args.voice_unit,
            "rate": args.rate,
            "max_atempo": args.max_atempo,
            "fit_duration": args.fit_duration,
            "fit_tempo": args.fit_tempo,
            "fit_rounds": args.fit_rounds,
            "keep_bgm": args.keep_bgm,
            "separate_bgm": args.separate_bgm,
            "bgm_volume": args.bgm_volume,
            "keep_dub_audio": args.keep_dub_audio,
            "soft_subtitle": args.soft_subtitle,
            "concurrency": args.tts_concurrency,
            "prune_cache": args.prune_cache,
        },
        "speakers": {
            "enabled": args.diarize,
            "count": args.speaker_count,
            "min": args.min_speakers,
            "max": args.max_speakers,
            "voices": args.speaker_voices,
            "model": args.diarize_model,
            "credential_env": args.hf_token_env,
        },
        "voice_clone": {
            "model": args.clone_model,
            "references": _clone_reference_records(args.clone_references, base=base),
            "repo": (
                relative_path(args.clone_repo, base=base) if args.clone_repo is not None else None
            ),
            "python": _portable_executable(args.clone_python, base=base),
            "device": args.clone_device,
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
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


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ProjectError(f"Could not read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProjectError(f"{label.capitalize()} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProjectError(f"{label.capitalize()} must contain one JSON object: {path}")
    return payload


def _reject_secret_fields(value: object, *, location: str = "project") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in {"api_key", "token", "access_token", "password", "secret"}:
                raise ProjectError(
                    f"Secret field {location}.{key} is not allowed in a project file."
                )
            _reject_secret_fields(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_fields(child, location=f"{location}[{index}]")


def load_project(project_path: Path) -> dict[str, object]:
    path = project_path.expanduser().resolve()
    payload = _read_json_object(path, label="project file")
    if payload.get("schema") != PROJECT_SCHEMA:
        raise ProjectError(f"Project schema must be {PROJECT_SCHEMA!r}: {path}")
    if payload.get("version") != PROJECT_VERSION:
        raise ProjectError(f"Project version must be {PROJECT_VERSION}: {path}")
    if payload.get("workflow") not in {"run", "dub"}:
        raise ProjectError("Project workflow must be 'run' or 'dub'.")
    _reject_secret_fields(payload)
    return payload


def project_state_path(project_path: Path) -> Path:
    return project_path.expanduser().resolve().parent / ".video-txt" / "state.json"


def load_project_state(project_path: Path) -> dict[str, object]:
    path = project_state_path(project_path)
    if not path.is_file():
        return {}
    payload = _read_json_object(path, label="project state")
    if payload.get("schema") != STATE_SCHEMA or payload.get("version") != STATE_VERSION:
        raise ProjectError(f"Unsupported project state schema or version: {path}")
    _reject_secret_fields(payload, location="state")
    return payload


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProjectError(f"Project field {key!r} must be a JSON object.")
    return value


def _project_relative_path(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProjectError(f"Project path {field!r} must be a non-empty string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_project_paths(
    project_path: Path, config: Mapping[str, object]
) -> dict[str, Path]:
    base = project_path.expanduser().resolve().parent
    video = _required_mapping(config, "video")
    declared = _required_mapping(config, "paths")
    paths = {
        key: _project_relative_path(base, value, field=f"paths.{key}")
        for key, value in declared.items()
        if isinstance(key, str) and value is not None
    }
    paths["video"] = _project_relative_path(base, video.get("path"), field="video.path")
    translation = _required_mapping(config, "translation")
    terminology = translation.get("terminology")
    if terminology is not None:
        if not isinstance(terminology, Mapping):
            raise ProjectError("Project field 'translation.terminology' must be an object or null.")
        paths["terminology"] = _project_relative_path(
            base, terminology.get("path"), field="translation.terminology.path"
        )
    return paths


def _optional_sha256(path: Path | None) -> str | None:
    return sha256_file(path) if path is not None and path.is_file() else None


def inspect_project(project_path: Path) -> ProjectInspection:
    path = project_path.expanduser().resolve()
    config = load_project(path)
    state = load_project_state(path)
    paths = resolve_project_paths(path, config)
    transcription = dict(_required_mapping(config, "transcription"))
    translation = dict(_required_mapping(config, "translation"))
    mux = dict(_required_mapping(config, "mux"))
    tts = dict(_required_mapping(config, "tts"))
    speakers = dict(_required_mapping(config, "speakers"))
    voice_clone = dict(_required_mapping(config, "voice_clone"))
    translation.pop("terminology", None)

    video_sha = _optional_sha256(paths["video"])
    source_sha = _optional_sha256(paths.get("source_subtitle"))
    translation_sha = _optional_sha256(paths.get("translated_subtitle"))
    term_sha = _optional_sha256(paths.get("terminology"))
    desired = {
        "transcribe": {
            "video_sha256": video_sha,
            "output_path": str(paths["source_subtitle"]),
            "settings": transcription,
        },
        "translate": {
            "source_sha256": source_sha,
            "terminology_sha256": term_sha,
            "output_path": str(paths["translated_subtitle"]),
            "audit_path": str(paths["translation_audit"]),
            "settings": translation,
        },
        "mux": {
            "video_sha256": video_sha,
            "translation_sha256": translation_sha,
            "output_path": str(paths["video_output"]),
            "settings": mux,
        },
        "tts": {
            "video_sha256": video_sha,
            "source_sha256": source_sha,
            "translation_sha256": translation_sha,
            "output_path": str(paths["video_output"]),
            "audio_path": str(paths["dub_audio"]),
            "cache_path": str(paths["cache_dir"]),
            "settings": {"tts": tts, "speakers": speakers, "voice_clone": voice_clone},
        },
    }
    workflow = config["workflow"]
    enabled = {"translate", "mux"} if workflow == "run" else {"translate", "tts"}
    if transcription.get("enabled", True):
        enabled.add("transcribe")
    artifacts = {
        "transcribe": paths.get("source_subtitle", Path()).is_file(),
        "translate": (
            paths.get("translated_subtitle", Path()).is_file()
            and paths.get("translation_audit", Path()).is_file()
        ),
        "mux": paths.get("video_output", Path()).is_file(),
        "tts": paths.get("video_output", Path()).is_file(),
    }
    recorded = state.get("stages")
    statuses = evaluate_stages(
        desired,
        recorded=recorded if isinstance(recorded, Mapping) else {},
        artifacts=artifacts,
        enabled=enabled,
    )
    return ProjectInspection(
        project_path=path,
        config=config,
        state_path=project_state_path(path),
        state=state,
        paths=paths,
        desired=desired,
        artifacts=artifacts,
        enabled=enabled,
        statuses=statuses,
    )


def build_project_state(
    inspection: ProjectInspection,
    *,
    audio_selection: Mapping[str, object] | None = None,
    transcription_resolution: Mapping[str, object] | None = None,
) -> dict[str, object]:
    missing = [name for name in inspection.enabled if not inspection.artifacts.get(name, False)]
    if missing:
        raise ProjectError(
            "Cannot record a successful project state; missing artifacts for: "
            + ", ".join(sorted(missing))
        )
    artifact_paths = {
        "transcribe": [inspection.paths.get("source_subtitle")],
        "translate": [
            inspection.paths.get("translated_subtitle"),
            inspection.paths.get("translation_audit"),
        ],
        "mux": [inspection.paths.get("video_output")],
        "tts": [inspection.paths.get("video_output")],
    }
    stages: dict[str, object] = {}
    for name in inspection.enabled:
        record = stage_record(inspection.desired[name])
        record["status"] = "complete"
        record["artifacts"] = [
            relative_path(path, base=inspection.project_path.parent)
            for path in artifact_paths[name]
            if path is not None
        ]
        stages[name] = record
    return {
        "schema": STATE_SCHEMA,
        "version": STATE_VERSION,
        "updated_at": datetime.now(UTC).isoformat(),
        "project_fingerprint": fingerprint_payload(inspection.config),
        "audio_selection": dict(audio_selection) if audio_selection is not None else None,
        "transcription_resolution": (
            dict(transcription_resolution) if transcription_resolution is not None else None
        ),
        "content": {
            "video_sha256": inspection.desired["transcribe"]["video_sha256"],
            "source_subtitle_sha256": inspection.desired["translate"]["source_sha256"],
            "translation_sha256": inspection.desired["mux"]["translation_sha256"],
            "terminology_sha256": inspection.desired["translate"]["terminology_sha256"],
            "translation_audit_sha256": _optional_sha256(
                inspection.paths.get("translation_audit")
            ),
            "video_output_sha256": _optional_sha256(inspection.paths.get("video_output")),
            "dub_audio_sha256": _optional_sha256(inspection.paths.get("dub_audio")),
        },
        "stages": stages,
    }


def untracked_project_outputs(inspection: ProjectInspection) -> list[Path]:
    recorded = inspection.state.get("stages")
    recorded_stages = recorded if isinstance(recorded, Mapping) else {}
    candidates = {
        "transcribe": (inspection.paths["source_subtitle"],),
        "translate": (
            inspection.paths["translated_subtitle"],
            inspection.paths["translation_audit"],
        ),
        "mux": (inspection.paths["video_output"],),
        "tts": (inspection.paths["video_output"], inspection.paths["dub_audio"]),
    }
    return [
        path
        for name in inspection.enabled
        if inspection.statuses[name].stale and name not in recorded_stages
        for path in candidates[name]
        if path.exists()
    ]


def _append_option(arguments: list[str], flag: str, value: object) -> None:
    if value is not None:
        arguments.extend([flag, str(value)])


def _resolved_executable(project_path: Path, value: object) -> object:
    if not isinstance(value, str) or ("/" not in value and not value.startswith((".", "~"))):
        return value
    return _project_relative_path(project_path.parent, value, field="executable path")


def project_legacy_argv(inspection: ProjectInspection, *, dry_run: bool) -> list[str]:
    config = inspection.config
    transcription = _required_mapping(config, "transcription")
    translation = _required_mapping(config, "translation")
    mux = _required_mapping(config, "mux")
    workflow = str(config["workflow"])
    arguments = [workflow, str(inspection.paths["video"])]
    if not transcription.get("enabled", True):
        arguments.extend(["--subtitle", str(inspection.paths["source_subtitle"])])
    arguments.extend(
        [
            "--subtitle-output",
            str(inspection.paths["translated_subtitle"]),
            "--output-dir",
            str(inspection.paths["output_dir"]),
            "--video-output",
            str(inspection.paths["video_output"]),
        ]
    )

    _append_option(arguments, "--backend", transcription.get("backend"))
    _append_option(arguments, "--whisper-model", transcription.get("model"))
    _append_option(arguments, "--language", transcription.get("language"))
    _append_option(arguments, "--device", transcription.get("device"))
    _append_option(arguments, "--initial-prompt", transcription.get("initial_prompt"))
    _append_option(arguments, "--audio-stream", transcription.get("audio_stream"))
    for value in transcription.get("whisper_args", []):
        _append_option(arguments, "--whisper-arg", value)
    if transcription.get("refine_subtitles"):
        arguments.append("--refine-subtitles")
    if transcription.get("skip_transcript_check"):
        arguments.append("--skip-transcript-check")

    _append_option(arguments, "--provider", translation.get("provider"))
    _append_option(arguments, "--model", translation.get("model"))
    _append_option(arguments, "--base-url", translation.get("base_url"))
    _append_option(arguments, "--api-key-env", translation.get("credential_env"))
    credentials_file = translation.get("credentials_file")
    if credentials_file is not None:
        arguments.extend(
            [
                "--secrets-file",
                str(
                    _project_relative_path(
                        inspection.project_path.parent,
                        credentials_file,
                        field="translation.credentials_file",
                    )
                ),
            ]
        )
    _append_option(arguments, "--target-language", translation.get("target_language"))
    _append_option(arguments, "--source-language", translation.get("source_language"))
    _append_option(arguments, "--batch-chars", translation.get("batch_chars"))
    _append_option(arguments, "--retries", translation.get("retries"))
    _append_option(arguments, "--concurrency", translation.get("concurrency"))
    _append_option(arguments, "--context-cues", translation.get("context_cues"))
    _append_option(arguments, "--reasoning-effort", translation.get("reasoning_effort"))
    _append_option(arguments, "--timeout", translation.get("timeout"))
    _append_option(arguments, "--note", translation.get("note"))
    for value in translation.get("preserve_terms", []):
        _append_option(arguments, "--preserve-term", value)
    if not translation.get("resume", True):
        arguments.append("--no-resume")
    if "terminology" in inspection.paths:
        arguments.extend(["--term-file", str(inspection.paths["terminology"])])
    translation_debug = inspection.paths.get("translation_debug")
    if translation_debug is not None:
        arguments.extend(["--translation-debug-dir", str(translation_debug)])

    if workflow == "run":
        _append_option(arguments, "--mux-mode", mux.get("mode"))
        _append_option(arguments, "--language-code", mux.get("language_code"))
        _append_option(arguments, "--subtitle-codec", mux.get("subtitle_codec"))
        _append_option(
            arguments, "--hard-subtitle-layout", mux.get("hard_subtitle_layout")
        )
        _append_option(arguments, "--hard-subtitle-font", mux.get("hard_subtitle_font"))
        _append_option(
            arguments, "--hard-subtitle-font-size", mux.get("hard_subtitle_font_size")
        )
        _append_option(
            arguments, "--hard-subtitle-margin-v", mux.get("hard_subtitle_margin_v")
        )
        _append_option(
            arguments, "--hard-subtitle-box-height", mux.get("hard_subtitle_box_height")
        )
        _append_option(
            arguments, "--hard-subtitle-box-opacity", mux.get("hard_subtitle_box_opacity")
        )
        _append_option(arguments, "--video-codec", mux.get("video_codec"))
        _append_option(arguments, "--crf", mux.get("crf"))
        _append_option(arguments, "--preset", mux.get("preset"))
        _append_option(arguments, "--video-size", mux.get("video_size"))
        _append_option(
            arguments,
            "--ffmpeg",
            _resolved_executable(inspection.project_path, mux.get("ffmpeg")),
        )
        if mux.get("default_subtitle"):
            arguments.append("--default-subtitle")
    else:
        tts = _required_mapping(config, "tts")
        speakers = _required_mapping(config, "speakers")
        voice_clone = _required_mapping(config, "voice_clone")
        _append_option(arguments, "--tts-engine", tts.get("engine"))
        _append_option(arguments, "--voice", tts.get("voice"))
        _append_option(arguments, "--voice-unit", tts.get("voice_unit"))
        _append_option(arguments, "--rate", tts.get("rate"))
        _append_option(arguments, "--max-atempo", tts.get("max_atempo"))
        _append_option(arguments, "--fit-tempo", tts.get("fit_tempo"))
        _append_option(arguments, "--fit-rounds", tts.get("fit_rounds"))
        _append_option(arguments, "--bgm-volume", tts.get("bgm_volume"))
        _append_option(arguments, "--tts-concurrency", tts.get("concurrency"))
        arguments.extend(["--cache-dir", str(inspection.paths["cache_dir"])])
        arguments.extend(["--audio-output", str(inspection.paths["dub_audio"])])
        _append_option(arguments, "--language-code", mux.get("language_code"))
        _append_option(
            arguments,
            "--ffmpeg",
            _resolved_executable(inspection.project_path, mux.get("ffmpeg")),
        )
        for key, flag in (
            ("fit_duration", "--fit-duration"),
            ("keep_bgm", "--keep-bgm"),
            ("separate_bgm", "--separate-bgm"),
            ("keep_dub_audio", "--keep-dub-audio"),
            ("soft_subtitle", "--soft-subtitle"),
            ("prune_cache", "--prune-cache"),
        ):
            if tts.get(key):
                arguments.append(flag)
        if speakers.get("enabled"):
            arguments.append("--diarize")
        _append_option(arguments, "--speakers", speakers.get("count"))
        _append_option(arguments, "--min-speakers", speakers.get("min"))
        _append_option(arguments, "--max-speakers", speakers.get("max"))
        _append_option(arguments, "--diarize-model", speakers.get("model"))
        _append_option(arguments, "--hf-token-env", speakers.get("credential_env"))
        for value in speakers.get("voices", []):
            _append_option(arguments, "--speaker-voice", value)

        _append_option(arguments, "--clone-model", voice_clone.get("model"))
        repo = voice_clone.get("repo")
        if repo is not None:
            arguments.extend(
                [
                    "--clone-repo",
                    str(
                        _project_relative_path(
                            inspection.project_path.parent,
                            repo,
                            field="voice_clone.repo",
                        )
                    ),
                ]
            )
        _append_option(
            arguments,
            "--clone-python",
            _resolved_executable(inspection.project_path, voice_clone.get("python")),
        )
        _append_option(arguments, "--clone-device", voice_clone.get("device"))
        references = voice_clone.get("references", [])
        if not isinstance(references, list):
            raise ProjectError("Project field 'voice_clone.references' must be a list.")
        for index, reference in enumerate(references):
            if not isinstance(reference, Mapping):
                raise ProjectError(f"Voice clone reference #{index + 1} must be an object.")
            reference_path = _project_relative_path(
                inspection.project_path.parent,
                reference.get("path"),
                field=f"voice_clone.references[{index}].path",
            )
            speaker = reference.get("speaker")
            value = f"{speaker}={reference_path}" if speaker else str(reference_path)
            arguments.extend(["--clone-reference", value])

    if inspection.statuses["transcribe"].stale and "transcribe" in inspection.enabled:
        arguments.append("--retranscribe")
    if inspection.statuses["translate"].stale:
        arguments.append("--retranslate")
    final_stage = "mux" if workflow == "run" else "tts"
    recorded_stages = inspection.state.get("stages")
    if (
        inspection.statuses[final_stage].stale
        and inspection.paths["video_output"].exists()
        and isinstance(recorded_stages, Mapping)
        and final_stage in recorded_stages
    ):
        arguments.append("--overwrite-video")
    if dry_run:
        arguments.append("--dry-run")
    return arguments
