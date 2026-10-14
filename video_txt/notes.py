"""Immutable, timestamped reading copies of existing subtitles for an Obsidian vault.

SRT remains the source of truth. A complete revision directory is published in one
rename; a small head receipt is advanced last. Human summaries and drafts are only
read for freshness, never rewritten. This exporter performs no model calls.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .artifacts import write_json
from .media import probe_duration
from .quality import audit_transcript
from .subtitles import (
    SubtitleCue,
    SubtitleFormatError,
    join_text_parts,
    parse_srt_text,
)
from .terminology import audit_translation

SCHEMA = "video-txt.notes"
VERSION = 1
ROLLUP_HEADER = re.compile(
    r"(?m)^\s*(\d+)\s*\n(\d+:\d{2}:\d{2}[,.]\d{3} --> "
    r"\d+:\d{2}:\d{2}[,.]\d{3})[^\S\n]*\n"
)


class NotesError(ValueError):
    pass


@dataclass(frozen=True)
class NotesOptions:
    video: Path
    source: Path
    translation: Path
    vault: Path
    folder: str
    source_id: str
    title: str
    author: str = ""
    source_url: str = ""
    published: str = ""
    source_language: str = "en"
    target_language: str = "zh-CN"
    alignment: str = "strict"
    source_rollup: bool = False
    dry_run: bool = False


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def read_rollup(raw: str) -> tuple[list[SubtitleCue], dict]:
    """Explicit YouTube roll-up import: remove only adjacent shared whole lines.

    Blank lines after a timing header are accepted here, not in the general SRT
    parser. Original cue numbers and every removal remain in the import report.
    Words inside a line (including spoken repetitions) are never deduplicated.
    """
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    headers = list(ROLLUP_HEADER.finditer(normalized))
    if not headers or normalized[: headers[0].start()].strip():
        raise NotesError("Unrecognized numbered roll-up SRT; no content was imported.")
    cues: list[SubtitleCue] = []
    previous_lines: list[str] = []
    previous_end = -1.0
    actions = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(normalized)
        lines = [
            line.strip() for line in normalized[header.end() : end].splitlines() if line.strip()
        ]
        if any("-->" in line for line in lines):
            raise NotesError(f"Invalid roll-up cue {header[1]}; inspect the original SRT.")
        cue = SubtitleCue(header[1], header[2], lines)
        if cue.end_seconds <= cue.start_seconds or cue.start_seconds < previous_end - 0.01:
            raise NotesError(f"Invalid/overlapping roll-up timing at cue {cue.index}.")
        if not lines:
            actions.append({"cue": cue.index, "empty_display_cleared": True})
            previous_lines, previous_end = [], cue.end_seconds
            continue
        overlap = 0
        if abs(cue.start_seconds - previous_end) <= 0.25:
            for size in range(1, min(len(previous_lines), len(lines)) + 1):
                if previous_lines[-size:] == lines[:size]:
                    overlap = size
        if overlap:
            actions.append({"cue": cue.index, "removed_display_lines": lines[:overlap]})
        remaining = lines[overlap:]
        if remaining:
            cues.append(SubtitleCue(cue.index, cue.timing, remaining))
        previous_lines, previous_end = lines, cue.end_seconds
    if not cues:
        raise NotesError("Roll-up input contains no readable text.")
    return cues, {"input_cues": len(headers), "reading_cues": len(cues), "actions": actions}


def stamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{ms:03}"


def paragraph_groups(cues: list[SubtitleCue]) -> list[list[int]]:
    groups: list[list[int]] = []
    pending: list[int] = []
    for i, cue in enumerate(cues):
        if pending and cue.start_seconds - cues[pending[-1]].end_seconds >= 2:
            groups.append(pending)
            pending = []
        pending.append(i)
        duration = cue.end_seconds - cues[pending[0]].start_seconds
        sentence = re.search(r"[.!?。！？][\"'’”）)]*$", cue.text.strip())
        if duration >= 45 or (duration >= 15 and sentence):
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    return groups


def md_text(value: str) -> str:
    # Subtitle text is data, including apparent Markdown links or HTML.
    return re.sub(r"([\\`*_\[\]<>#^!])", r"\\\1", value)


def render_note(
    options: NotesOptions,
    revision: str,
    role: str,
    cues: list[SubtitleCue],
    groups: list[list[int]],
    ids: list[str],
    peer: str,
    input_sha: str,
) -> str:
    label = (
        "英文原文"
        if role == "source" and options.source_language == "en"
        else (
            "原文"
            if role == "source"
            else "中文译文"
            if options.target_language.startswith("zh")
            else "译文"
        )
    )
    metadata = {
        "title": f"{options.title} — {label}",
        "type": "video-transcript",
        "status": "needs-review",
        "created": date.today().isoformat(),
        "video_source_id": options.source_id,
        "video_revision": revision,
        "language": options.source_language if role == "source" else options.target_language,
        "author": options.author,
        "source": options.source_url,
        "published": options.published or None,
        "media_path": str(options.video.resolve()),
        "subtitle_sha256": input_sha,
        "alignment": options.alignment,
        "tags": ["视频来源", "字幕", "待校对"],
    }
    lines = [
        "---",
        *[f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in metadata.items()],
        "---",
        "",
        f"# {md_text(options.title)} — {label}",
        "",
        f"[▶ 打开本地视频]({options.video.resolve().as_uri()}) · [[{peer}|另一种语言]]",
        "",
        f"来源版本：`{revision}`。自动导出的来源稿，尚未经逐句听校。",
        "",
    ]
    if options.alignment == "independent":
        lines += ["> 两份字幕独立分段，仅按视频时间检索；不表示逐句翻译对齐。", ""]
    if options.source_rollup and role == "source":
        lines += [
            "> 英文滚动字幕已合并相邻重复显示行，未改写用词；原始 SRT 保持不变，"
            "处理记录见本目录 manifest.json。",
            "",
        ]
    for group, block_id in zip(groups, ids, strict=True):
        first, last = cues[group[0]], cues[group[-1]]
        text = join_text_parts([join_text_parts(cues[i].text_lines) for i in group])
        lines += [
            f"**{stamp(first.start_seconds)} — {stamp(last.end_seconds)}**",
            "",
            f"{md_text(text)} ^{block_id}",
            "",
        ]
    return "\n".join(lines)


def dependent_notes(vault: Path, source_id: str, revision: str) -> list[str]:
    """Read just the flat provenance fields; never parse or rewrite human prose."""
    stale = []
    for root, directories, filenames in os.walk(vault):
        directories[:] = [d for d in directories if not d.startswith(".")]
        for filename in filenames:
            if not filename.endswith(".md"):
                continue
            path = Path(root) / filename
            with path.open(encoding="utf-8") as handle:
                header = handle.read(8192)
            if not header.startswith("---\n"):
                continue
            header = header.split("\n---", 1)[0]
            fields = {}
            for key in ("video_source_id", "video_revision", "type"):
                match = re.search(rf"^{key}:\s*([^\n]*)$", header, re.M)
                fields[key] = match[1].strip().strip("\"'") if match else ""
            if (
                fields["video_source_id"] == source_id
                and fields["type"] != "video-transcript"
                and fields["video_revision"] != revision
            ):
                stale.append(path.relative_to(vault).as_posix())
    return sorted(stale)


def validate_options(options: NotesOptions) -> tuple[Path, Path]:
    vault = options.vault.resolve()
    if not vault.is_dir():
        raise NotesError(f"Vault directory does not exist: {vault}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", options.source_id):
        raise NotesError(
            "--source-id must be a stable lowercase identifier (letters, digits, '-')."
        )
    destination = (vault / options.folder / options.source_id).resolve()
    if not destination.is_relative_to(vault) or destination == vault:
        raise NotesError("--folder must stay inside the vault.")
    if any(c in destination.relative_to(vault).as_posix() for c in "#|[]^\n\r"):
        raise NotesError("Output folder contains characters unsafe for Obsidian links.")
    if not options.video.is_file():
        raise NotesError(f"Media not found: {options.video}")
    if options.published:
        date.fromisoformat(options.published)
    if ".fitted." in options.translation.name or ".bilingual." in options.translation.name:
        raise NotesError(
            "Use the complete monolingual translation, not fitted/bilingual subtitles."
        )
    if options.source_rollup and options.alignment != "independent":
        raise NotesError(
            "Roll-up captions need --alignment independent; cue pairing is unavailable."
        )
    return vault, destination


def prepare(options: NotesOptions, vault: Path, destination: Path) -> tuple[str, dict, dict]:
    source_bytes = options.source.read_bytes()
    translation_bytes = options.translation.read_bytes()
    raw_source = source_bytes.decode("utf-8-sig")
    normalization = None
    if options.source_rollup:
        source, normalization = read_rollup(raw_source)
    else:
        source = parse_srt_text(raw_source, source=str(options.source))
    translation = parse_srt_text(
        translation_bytes.decode("utf-8-sig"), source=str(options.translation)
    )
    duration = probe_duration(options.video)
    audits = {}
    for role, cues, language in (
        ("source", source, options.source_language),
        ("translation", translation, options.target_language),
    ):
        if any(c.end_seconds <= c.start_seconds for c in cues):
            raise NotesError(f"{role}: zero/reversed duration; repair the subtitle first.")
        if any(a.start_seconds > b.start_seconds for a, b in zip(cues, cues[1:], strict=False)):
            raise NotesError(f"{role}: out-of-order timeline; repair the subtitle first.")
        report = audit_transcript(cues, language=language, media_duration=duration)
        audits[role] = report.to_dict()
        if report.has_errors:
            raise NotesError(f"{role} audit failed: {_error_summary(report.findings)}")
    if options.alignment == "strict":
        audit = audit_translation(source, translation)
        if audit.has_errors:
            raise NotesError(
                "Source/translation do not align. Run audit-translation. "
                "Use --alignment independent only for separately produced captions "
                "of the same verified video; it does not claim sentence alignment."
            )
        audits["translation_pair"] = audit.to_dict()
    metadata = {
        key: getattr(options, key)
        for key in (
            "source_id",
            "title",
            "author",
            "source_url",
            "published",
            "source_language",
            "target_language",
            "alignment",
            "source_rollup",
        )
    }
    identity = {
        "schema": SCHEMA,
        "version": VERSION,
        "metadata": metadata,
        "source_sha256": digest(source_bytes),
        "translation_sha256": digest(translation_bytes),
    }
    revision = digest(json_bytes(identity))[:20]
    relative = (destination / revision).relative_to(vault)
    filenames = {
        "source": "原文.md",
        "translation": "中文译文.md" if options.target_language.startswith("zh") else "译文.md",
    }
    source_groups = paragraph_groups(source)
    groups = {
        "source": source_groups,
        "translation": source_groups
        if options.alignment == "strict"
        else paragraph_groups(translation),
    }
    cue_sets = {"source": source, "translation": translation}
    block_ids = {}
    for role in cue_sets:
        if role == "translation" and options.alignment == "strict":
            block_ids[role] = block_ids["source"]
            continue
        cues = cue_sets[role]
        block_ids[role] = [
            "p-"
            + digest(
                json_bytes(
                    [[[cues[i].start_seconds, cues[i].end_seconds, cues[i].text] for i in group]]
                )
            )[:16]
            for group in groups[role]
        ]
    documents = {}
    for role in cue_sets:
        peer = "translation" if role == "source" else "source"
        documents[filenames[role]] = render_note(
            options,
            revision,
            role,
            cue_sets[role],
            groups[role],
            block_ids[role],
            (relative / filenames[peer]).with_suffix("").as_posix(),
            identity[f"{role}_sha256"],
        ).encode()
    manifest = {
        **identity,
        "revision": revision,
        "created": date.today().isoformat(),
        "video": str(options.video.resolve()),
        "duration_seconds": duration,
        "inputs": {
            "source": str(options.source.resolve()),
            "translation": str(options.translation.resolve()),
        },
        "audits": audits,
        "rollup_import": normalization,
        "documents": {name: digest(data) for name, data in documents.items()},
        "paragraphs": {
            role: [
                {
                    "block_id": block_id,
                    "start": cue_sets[role][group[0]].start_seconds,
                    "end": cue_sets[role][group[-1]].end_seconds,
                    "source_cues": [cue_sets[role][i].index for i in group],
                }
                for group, block_id in zip(groups[role], block_ids[role], strict=True)
            ]
            for role in cue_sets
        },
    }
    return revision, documents, manifest


def verify_revision(directory: Path, revision: str) -> dict:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("invalid receipt")
        identity = {
            key: manifest[key]
            for key in ("schema", "version", "metadata", "source_sha256", "translation_sha256")
        }
        if (
            manifest["schema"] != SCHEMA
            or manifest["version"] != VERSION
            or manifest["revision"] != revision
            or not manifest["documents"]
            or digest(json_bytes(identity))[:20] != revision
            or len(manifest["documents"]) != 2
        ):
            raise ValueError("incompatible receipt")
        for filename, expected in manifest["documents"].items():
            if Path(filename).name != filename or not filename.endswith(".md"):
                raise ValueError("unsafe document path")
            if digest((directory / filename).read_bytes()) != expected:
                raise ValueError(f"human modification or damaged export: {filename}")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise NotesError(f"Preserving existing revision {directory}: {exc}") from exc


def verify_head(destination: Path, source_id: str) -> None:
    head = destination / "latest.json"
    if not head.exists():
        return
    previous = json.loads(head.read_text())
    if (
        not isinstance(previous, dict)
        or previous.get("schema") != SCHEMA
        or previous.get("version") != VERSION
        or previous.get("source_id") != source_id
        or not re.fullmatch(r"[a-f0-9]{20}", str(previous.get("revision", "")))
    ):
        raise NotesError("Unknown export receipt; preserving existing files.")
    verify_revision(destination / previous["revision"], previous["revision"])


def publish(destination: Path, revision: str, documents: dict, manifest: dict) -> None:
    target = destination / revision
    if target.exists():
        verify_revision(target, revision)
    else:
        with tempfile.TemporaryDirectory(prefix=".notes-", dir=destination) as staging:
            stage = Path(staging)
            for filename, data in documents.items():
                (stage / filename).write_bytes(data)
            (stage / "manifest.json").write_bytes(json_bytes(manifest))
            verify_revision(stage, revision)
            stage.rename(target)
    # The directory is already complete. A failure here is recoverable by rerun.
    head = {
        "schema": SCHEMA,
        "version": VERSION,
        "source_id": manifest["metadata"]["source_id"],
        "revision": revision,
    }
    head_path = destination / "latest.json"
    if not head_path.exists() or json.loads(head_path.read_text()) != head:
        write_json(head_path, head)


def _error_summary(findings, limit: int = 5) -> str:
    """Error codes with the first few offending cue numbers, so the repair target is findable."""
    where: dict[str, list[int]] = {}
    for finding in findings:
        if finding.severity == "error":
            where.setdefault(finding.code, []).extend(finding.cue_indexes)
    parts = []
    for code in sorted(where):
        cues = sorted(set(where[code]))
        if cues:
            more = f" 等 {len(cues)} 条" if len(cues) > limit else ""
            parts.append(f"{code}（第 {', '.join(map(str, cues[:limit]))} 条{more}）")
        else:
            parts.append(code)
    return ", ".join(parts)


def export_notes(options: NotesOptions) -> dict:
    try:
        vault, destination = validate_options(options)
        revision, documents, manifest = prepare(options, vault, destination)
        if options.dry_run:
            verify_head(destination, options.source_id)
            if (destination / revision).exists():
                verify_revision(destination / revision, revision)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / ".export.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise NotesError(
                        "Another export is running for this source. Retry later."
                    ) from exc
                verify_head(destination, options.source_id)
                for role in ("source", "translation"):
                    if digest(getattr(options, role).read_bytes()) != manifest[f"{role}_sha256"]:
                        raise NotesError(
                            "Subtitles changed during export; retry with current inputs."
                        )
                publish(destination, revision, documents, manifest)
        return {
            "source_id": options.source_id,
            "revision": revision,
            "dry_run": options.dry_run,
            "documents": [str(destination / revision / name) for name in documents],
            "paragraph_counts": {
                role: len(items) for role, items in manifest["paragraphs"].items()
            },
            "warning_counts": {
                role: audit.get("warning_count", audit.get("summary", {}).get("warning", 0))
                for role, audit in manifest["audits"].items()
            },
            "needs_review": dependent_notes(vault, options.source_id, revision),
        }
    except (OSError, ValueError) as exc:
        if isinstance(exc, (NotesError, SubtitleFormatError)):
            raise
        raise NotesError(str(exc)) from exc


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--source", type=Path, required=True, help="Complete original-language SRT."
    )
    parser.add_argument(
        "--translation",
        type=Path,
        required=True,
        help="Explicitly selected complete, preferably revised, translation SRT.",
    )
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--folder", default="raw/videos", help="Folder relative to the vault.")
    parser.add_argument(
        "--source-id", required=True, help="Stable ID reused for all future exports."
    )
    parser.add_argument("--title", required=True)
    parser.add_argument("--author", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument(
        "--published", default="", help="Known source date, YYYY-MM-DD; omit if unknown."
    )
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--target-language", default="zh-CN")
    parser.add_argument(
        "--alignment",
        choices=("strict", "independent"),
        default="strict",
        help="Independent timelines must be verified to belong to the same video.",
    )
    parser.add_argument(
        "--source-rollup",
        action="store_true",
        help="Explicitly import YouTube rolling display captions; log line deduplication.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report; write nothing."
    )


def command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    result = export_notes(
        NotesOptions(**{key: getattr(args, key) for key in NotesOptions.__dataclass_fields__})
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
