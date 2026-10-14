from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import write_json
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
from .revise import revised_subtitle_path
from .subtitles import translated_subtitle_path
from .terminology import load_terminology, translation_audit_path_for
from .workflow_config import WorkflowConfig

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


STAGE_ORDER = ("transcribe", "translate", "revise", "mux", "tts")
# `revise` sits between the translation and everything that consumes it. Editing
# the revision file has to reach the finished video, but it must never reach the
# translation: retranslating a whole film to restore one hand-corrected line is
# exactly the cost this stage exists to avoid. So the revision fingerprint lives
# here, and mux/tts go stale through the graph rather than through translate.
STAGE_DEPENDENCIES = {
    "transcribe": (),
    "translate": ("transcribe",),
    "revise": ("translate",),
    "mux": ("translate", "revise"),
    "tts": ("translate", "revise"),
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
        if previous.get("status", "complete") != "complete":
            statuses[name] = StageStatus(name, True, "previous run incomplete")
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
    args: WorkflowConfig,
    term_file: Path | None,
    revisions_file: Path | None,
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
    revisions = (
        _path_record(revisions_file, base=base) if revisions_file is not None else None
    )
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
            "revised_subtitle": relative_path(
                revised_subtitle_path(translated_subtitle), base=base
            ),
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
            **_project_settings(args, "transcription"),
            "enabled": subtitle is None,
        },
        "translation": {
            **_project_settings(args, "translation"),
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
            "concurrency": args.concurrency or DEFAULT_CONCURRENCY,
            "reasoning_effort": (
                args.reasoning_effort or provider.get("reasoning_effort") or "auto"
            ),
            "thinking": provider.get("thinking") or "auto",
            "timeout": args.timeout or DEFAULT_TIMEOUT,
            "preserve_terms": list(dict.fromkeys([*DEFAULT_TERMS, *args.preserve_term])),
            "terminology": terminology,
            "revisions": revisions,
        },
        "mux": {
            **_project_settings(args, "mux"),
            "ffmpeg": _portable_executable(args.ffmpeg_path, base=base),
        },
        "tts": _project_settings(args, "tts"),
        "speakers": _project_settings(args, "speakers"),
        "voice_clone": {
            **_project_settings(args, "voice_clone"),
            "references": _clone_reference_records(args.clone_references, base=base),
            "repo": (
                relative_path(args.clone_repo, base=base) if args.clone_repo is not None else None
            ),
            "python": _portable_executable(args.clone_python, base=base),
        },
    }


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
    for field, key in (("terminology", "terminology"), ("revisions", "revisions")):
        declared_file = translation.get(field)
        if declared_file is None:
            continue
        if not isinstance(declared_file, Mapping):
            raise ProjectError(
                f"Project field 'translation.{field}' must be an object or null."
            )
        paths[key] = _project_relative_path(
            base, declared_file.get("path"), field=f"translation.{field}.path"
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
    # Not part of the translation settings: what the model is asked for does not
    # change when a hand-corrected line does.
    translation.pop("revisions", None)

    workflow = config["workflow"]
    enabled = {"translate", "mux"} if workflow == "run" else {"translate", "tts"}
    if transcription.get("enabled", True):
        enabled.add("transcribe")
    if "revisions" in paths:
        enabled.add("revise")

    video_sha = _optional_sha256(paths["video"])
    source_sha = _optional_sha256(paths.get("source_subtitle"))
    translation_sha = _optional_sha256(paths.get("translated_subtitle"))
    term_sha = _optional_sha256(paths.get("terminology"))
    revisions_sha = _optional_sha256(paths.get("revisions"))
    revised_subtitle = paths.get("revised_subtitle", Path())
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
        "revise": {
            "translation_sha256": translation_sha,
            "revisions_sha256": revisions_sha,
            "output_path": str(revised_subtitle),
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
    artifacts = {
        "transcribe": paths.get("source_subtitle", Path()).is_file(),
        "revise": revised_subtitle.is_file(),
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
    artifact_paths = project_artifacts(inspection)
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


def project_artifacts(inspection: ProjectInspection) -> dict[str, tuple[Path, ...]]:
    paths = inspection.paths
    return {
        "transcribe": (paths["source_subtitle"],),
        "revise": (paths["revised_subtitle"],) if "revised_subtitle" in paths else (),
        "translate": (paths["translated_subtitle"], paths["translation_audit"]),
        "mux": (paths["video_output"],),
        "tts": (paths["video_output"], paths["dub_audio"]),
    }


def untracked_project_outputs(inspection: ProjectInspection) -> list[Path]:
    recorded = inspection.state.get("stages", {})
    if not isinstance(recorded, Mapping):
        raise ProjectError("Project state stages must be an object.")
    owned = {
        (inspection.project_path.parent / value).resolve()
        for record in recorded.values()
        if isinstance(record, Mapping)
        for value in record.get("artifacts", [])
        if isinstance(value, str)
    }
    content = inspection.state.get("content", {})
    audio = inspection.paths["dub_audio"]
    tts = recorded.get("tts", {})
    if (
        isinstance(content, Mapping)
        and isinstance(tts, Mapping)
        and tts.get("status") == "complete"
        and audio.is_file()
        and content.get("dub_audio_sha256") == sha256_file(audio)
    ):
        owned.add(audio.resolve())
    return [
        path
        for name, paths in project_artifacts(inspection).items()
        if name in inspection.enabled and inspection.statuses[name].stale
        for path in paths
        if path.exists() and path.resolve() not in owned
    ]


def begin_project_run(inspection: ProjectInspection) -> None:
    """Claim checked output paths before work, retaining completed checkpoints."""
    untracked = untracked_project_outputs(inspection)
    if untracked:
        raise ProjectError(
            "Refusing to replace output files that are not recorded in project state:\n"
            + "\n".join(f"  {path}" for path in untracked)
            + "\nMove them aside, choose new output paths, or use --dry-run to inspect."
        )
    state = dict(inspection.state)
    stages = dict(state.get("stages", {}))
    for name, paths in project_artifacts(inspection).items():
        if name in inspection.enabled and inspection.statuses[name].stale:
            stages[name] = {
                "status": "pending",
                "artifacts": [relative_path(p, base=inspection.project_path.parent) for p in paths],
            }
    state.update(schema=STATE_SCHEMA, version=STATE_VERSION, stages=stages)
    write_json(inspection.state_path, state)


def checkpoint_project_stage(project_path: Path, name: str) -> None:
    inspection = inspect_project(project_path)
    if name not in inspection.enabled:
        return
    if not inspection.artifacts[name]:
        raise ProjectError(f"Cannot complete {name}: expected artifacts are missing.")
    state = dict(inspection.state)
    stages = dict(state.get("stages", {}))
    stages[name] = {
        **stage_record(inspection.desired[name]),
        "status": "complete",
        "artifacts": [
            relative_path(p, base=inspection.project_path.parent)
            for p in project_artifacts(inspection)[name]
        ],
    }
    state.update(stages=stages, updated_at=datetime.now(UTC).isoformat())
    write_json(inspection.state_path, state)


# One correspondence for writing and reading settings; paths are resolved separately.
PROJECT_OPTION_FIELDS = {
    "whisper_backend": ("transcription", "backend"),
    "whisper_model": ("transcription", "model"),
    "language": ("transcription", "language"),
    "device": ("transcription", "device"),
    "initial_prompt": ("transcription", "initial_prompt"),
    "audio_stream": ("transcription", "audio_stream"),
    "provider": ("translation", "provider"),
    "model": ("translation", "model"),
    "base_url": ("translation", "base_url"),
    "api_key_env": ("translation", "credential_env"),
    "target_language": ("translation", "target_language"),
    "source_language": ("translation", "source_language"),
    "batch_chars": ("translation", "batch_chars"),
    "retries": ("translation", "retries"),
    "concurrency": ("translation", "concurrency"),
    "context_cues": ("translation", "context_cues"),
    "reasoning_effort": ("translation", "reasoning_effort"),
    "timeout": ("translation", "timeout"),
    "note": ("translation", "note"),
    "mux_mode": ("mux", "mode"),
    "language_code": ("mux", "language_code"),
    "subtitle_codec": ("mux", "subtitle_codec"),
    "hard_subtitle_layout": ("mux", "hard_subtitle_layout"),
    "hard_subtitle_font": ("mux", "hard_subtitle_font"),
    "hard_subtitle_font_size": ("mux", "hard_subtitle_font_size"),
    "hard_subtitle_margin_v": ("mux", "hard_subtitle_margin_v"),
    "hard_subtitle_box_height": ("mux", "hard_subtitle_box_height"),
    "hard_subtitle_box_opacity": ("mux", "hard_subtitle_box_opacity"),
    "video_codec": ("mux", "video_codec"),
    "crf": ("mux", "crf"),
    "preset": ("mux", "preset"),
    "video_size": ("mux", "video_size"),
    "tts_engine": ("tts", "engine"),
    "voice": ("tts", "voice"),
    "voice_unit": ("tts", "voice_unit"),
    "rate": ("tts", "rate"),
    "max_atempo": ("tts", "max_atempo"),
    "fit_tempo": ("tts", "fit_tempo"),
    "fit_rounds": ("tts", "fit_rounds"),
    "bgm_volume": ("tts", "bgm_volume"),
    "tts_concurrency": ("tts", "concurrency"),
    "speaker_count": ("speakers", "count"),
    "min_speakers": ("speakers", "min"),
    "max_speakers": ("speakers", "max"),
    "diarize_model": ("speakers", "model"),
    "hf_token_env": ("speakers", "credential_env"),
    "clone_model": ("voice_clone", "model"),
    "clone_device": ("voice_clone", "device"),
    "refine_subtitles": ("transcription", "refine_subtitles"),
    "skip_transcript_check": ("transcription", "skip_transcript_check"),
    "whisper_args": ("transcription", "whisper_args"),
    "preserve_term": ("translation", "preserve_terms"),
    "resume": ("translation", "resume"),
    "thinking": ("translation", "thinking"),
    "default_subtitle": ("mux", "default_subtitle"),
    "bilingual": ("mux", "bilingual"),
    "bilingual_order": ("mux", "bilingual_order"),
    "diarize": ("speakers", "enabled"),
    "speaker_voices": ("speakers", "voices"),
    "fit_duration": ("tts", "fit_duration"),
    "keep_bgm": ("tts", "keep_bgm"),
    "separate_bgm": ("tts", "separate_bgm"),
    "keep_dub_audio": ("tts", "keep_dub_audio"),
    "soft_subtitle": ("tts", "soft_subtitle"),
    "prune_cache": ("tts", "prune_cache"),
}


def project_workflow_config(inspection: ProjectInspection, *, dry_run: bool) -> WorkflowConfig:
    values: dict[str, object] = {"workflow": inspection.config["workflow"], "dry_run": dry_run}
    for option, (section, key) in PROJECT_OPTION_FIELDS.items():
        settings = _required_mapping(inspection.config, section)
        if key in settings and settings[key] is not None:
            values[option] = settings[key]
    paths = inspection.paths
    values.update(
        video=paths["video"],
        output_dir=paths["output_dir"],
        subtitle_output=paths["translated_subtitle"],
        video_output=paths["video_output"],
        term_file=paths.get("terminology"),
        revisions=paths.get("revisions"),
        debug_dir=paths.get("translation_debug"),
        cache_dir=paths["cache_dir"],
        audio_output=paths["dub_audio"],
        project_reuse_translation=not inspection.statuses["translate"].stale,
        retranscribe=inspection.statuses["transcribe"].stale,
        retranslate=inspection.statuses["translate"].stale,
    )
    if not _required_mapping(inspection.config, "transcription").get("enabled", True):
        values["subtitle"] = paths["source_subtitle"]
    for section, key, option in (
        ("translation", "credentials_file", "secrets_file"),
        ("voice_clone", "repo", "clone_repo"),
    ):
        value = _required_mapping(inspection.config, section).get(key)
        if value is not None:
            values[option] = _project_relative_path(
                inspection.project_path.parent, value, field=f"{section}.{key}"
            )
    for section, key, option in (
        ("mux", "ffmpeg", "ffmpeg_path"),
        ("voice_clone", "python", "clone_python"),
    ):
        value = _required_mapping(inspection.config, section).get(key)
        if isinstance(value, str) and ("/" in value or value.startswith((".", "~"))):
            value = str(_project_relative_path(inspection.project_path.parent, value, field=key))
        values[option] = value
    references = _required_mapping(inspection.config, "voice_clone").get("references", [])
    if not isinstance(references, list):
        raise ProjectError("Project field 'voice_clone.references' must be a list.")
    values["clone_references"] = []
    for reference in references:
        if not isinstance(reference, Mapping):
            raise ProjectError("Each clone reference must be an object.")
        path = _project_relative_path(
            inspection.project_path.parent, reference.get("path"), field="clone reference"
        )
        speaker = reference.get("speaker")
        values["clone_references"].append(f"{speaker}={path}" if speaker else str(path))
    final_stage = "mux" if values["workflow"] == "run" else "tts"
    values["overwrite_video"] = (
        inspection.statuses[final_stage].stale
        and paths["video_output"].exists()
        and paths["video_output"] not in untracked_project_outputs(inspection)
    )
    return WorkflowConfig(**values)


def _project_settings(args: WorkflowConfig, section: str) -> dict[str, object]:
    return {
        key: getattr(args, option)
        for option, (owner, key) in PROJECT_OPTION_FIELDS.items()
        if owner == section
    }
