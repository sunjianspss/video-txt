"""Reviewed screen captions and speech share one timeline and the existing muxer.

OCR samples are evidence, not subtitles. The editable, versioned plan owns the
source corrections, translations and timing; SRTs and finished media are projections.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .arguments import add_mux_arguments, add_translation_arguments
from .artifacts import write_json
from .media import find_ffmpeg, probe_duration
from .mux import MuxOptions, run_mux
from .subtitles import (
    SubtitleCue,
    display_width,
    parse_srt,
    seconds_to_srt_time,
    serialize_srt,
    write_srt,
)
from .translate import TranslationConfig, translate_subtitle_file

SCHEMA = "video-txt.screen-plan"
VERSION = 1


class ScreenError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScreenError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ScreenError(f"Expected a JSON object: {path}")
    return data


def resolve_path(base: Path, value: str) -> Path:
    return (base / value).expanduser().resolve()


def number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ScreenError(f"{label} must be a finite number.")
    return float(value)


def load_plan(path: Path, *, ready: bool = False) -> tuple[dict, Path]:
    plan = read_json(path)
    if plan.get("schema") != SCHEMA or plan.get("version") != VERSION:
        raise ScreenError("Unsupported screen-plan schema/version.")
    media = plan.get("video", {})
    if not isinstance(media, dict) or not isinstance(media.get("path"), str):
        raise ScreenError("Plan needs video.path and video.sha256.")
    video = resolve_path(path.parent, media["path"])
    if not video.is_file() or sha256(video) != media.get("sha256"):
        raise ScreenError("Video changed or is missing; scan this source again in a new directory.")
    duration = number(media.get("duration"), "video.duration")
    if duration <= 0:
        raise ScreenError("video.duration must be positive.")
    cues = plan.get("captions")
    if not isinstance(cues, list):
        raise ScreenError("Plan captions must be a list.")
    ids = set()
    for item in cues:
        if not isinstance(item, dict):
            raise ScreenError("Every caption must be an object.")
        ident = item.get("id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ScreenError("Caption IDs must be nonempty and unique.")
        ids.add(ident)
        if item.get("status") not in ("draft", "ready", "omit"):
            raise ScreenError(f"{ident}: status must be draft, ready or omit.")
        if item["status"] == "omit":
            if not str(item.get("note", "")).strip():
                raise ScreenError(f"{ident}: an omitted caption needs a note.")
            continue
        start = number(item.get("start"), f"{ident}.start")
        end = number(item.get("end"), f"{ident}.end")
        if not 0 <= start < end <= duration:
            raise ScreenError(f"{ident}: invalid timing {start}–{end} for {duration}s video.")
        if not isinstance(item.get("source"), str) or not item["source"].strip():
            raise ScreenError(f"{ident}: source text is empty.")
        if not isinstance(item.get("translation", ""), str):
            raise ScreenError(f"{ident}: translation must be text.")
        if ready:
            if item["status"] != "ready" or not item.get("translation", "").strip():
                raise ScreenError(
                    f"{ident}: review source, translation and timing; set status=ready."
                )
            evidence = item.get("evidence", [])
            if not isinstance(evidence, list) or not evidence:
                raise ScreenError(f"{ident}: screen caption needs screenshot evidence.")
            for sample in evidence:
                if not isinstance(sample, dict) or not isinstance(sample.get("image"), str):
                    raise ScreenError(f"{ident}: invalid screenshot evidence.")
                image = resolve_path(path.parent, sample["image"])
                if not image.is_file() or sha256(image) != sample.get("sha256"):
                    raise ScreenError(f"{ident}: screenshot changed or missing: {image}")
                if not 0 <= number(sample.get("time"), "evidence.time") < duration:
                    raise ScreenError(f"{ident}: screenshot time outside video.")
    return plan, video


def subtitle(item: dict, text: str, index: int) -> SubtitleCue:
    timing = f"{seconds_to_srt_time(item['start'])} --> {seconds_to_srt_time(item['end'])}"
    return SubtitleCue(str(index), timing, text.splitlines())


def wrap_caption(text: str, *, columns: int = 64) -> str:
    """Wrap without dropping text. Dense pages must be edited in the plan."""
    lines = []
    for paragraph in text.strip().splitlines():
        line = ""
        for char in paragraph:
            if display_width(line + char) > columns:
                lines.append(line.rstrip())
                line = ""
            line += char
        if line.strip():
            lines.append(line.strip())
    if len(lines) > 2:
        raise ScreenError("Caption exceeds two lines; shorten or split it in the plan.")
    return "\n".join(lines)


def compile_plan(
    path: Path,
    *,
    speech_source: Path | None = None,
    speech_translation: Path | None = None,
) -> tuple[Path, list[SubtitleCue], list[SubtitleCue], list[str]]:
    plan, video = load_plan(path, ready=True)
    rows = []
    warnings = []
    for item in plan["captions"]:
        if item["status"] == "omit":
            continue
        try:
            text = wrap_caption(item["translation"])
        except ScreenError as exc:
            raise ScreenError(f"{item['id']}: {exc}") from exc
        rows.append((item, item["source"], text))
    if bool(speech_source) != bool(speech_translation):
        raise ScreenError("Pass both --speech-source and --speech-translation.")
    if speech_source and speech_translation:
        sources = parse_srt(speech_source)
        translations = parse_srt(speech_translation)
        if len(sources) != len(translations):
            raise ScreenError("Speech source/translation cue counts differ.")
        for i, (source, translated) in enumerate(zip(sources, translations, strict=True), 1):
            if source.time_range != translated.time_range or source.is_empty or translated.is_empty:
                raise ScreenError(f"Speech cue {i}: timing mismatch or empty text.")
            start, end = source.time_range
            if not 0 <= start < end <= plan["video"]["duration"]:
                raise ScreenError(f"Speech cue {i}: outside video duration.")
            item = {"id": f"speech-{i}", "start": start, "end": end}
            rows.append((item, source.text, wrap_caption(translated.text)))
    rows.sort(key=lambda row: row[0]["start"])
    if not rows:
        raise ScreenError("No captions to render.")
    for i, (item, _source, translation) in enumerate(rows):
        if i and round(item["start"] * 1000) < round(rows[i - 1][0]["end"] * 1000):
            raise ScreenError(
                f"Timeline collision: {rows[i - 1][0]['id']} and {item['id']}. "
                "Edit the screen plan or the speech range; neither track was silently dropped."
            )
        cps = len(translation.replace("\n", "")) / (item["end"] - item["start"])
        if cps > 16:
            warnings.append(f"{item['id']}: {cps:.1f} characters/s; consider shortening.")
    source_cues = [subtitle(item, text, i) for i, (item, text, _) in enumerate(rows, 1)]
    target_cues = [subtitle(item, text, i) for i, (item, _, text) in enumerate(rows, 1)]
    return video, source_cues, target_cues, warnings


def translate_plan(path: Path, output: Path, config: TranslationConfig) -> Path:
    plan, _video = load_plan(path)
    if output.exists():
        raise ScreenError(f"Output already exists: {output}. Choose a new path.")
    items = [item for item in plan["captions"] if item["status"] != "omit"]
    if not items:
        raise ScreenError("No screen text to translate.")
    if any(item.get("translation", "").strip() for item in items):
        raise ScreenError(
            "Plan already contains translations; edit that plan or use a fresh draft."
        )
    with TemporaryDirectory(prefix="video-txt-screen-translate-") as directory:
        source = Path(directory) / "screen.srt"
        target = Path(directory) / "screen.zh.srt"
        write_srt(source, [subtitle(item, item["source"], i) for i, item in enumerate(items, 1)])
        translate_subtitle_file(input_path=source, output_path=target, config=config)
        translated = parse_srt(target)
        if len(translated) != len(items):
            raise ScreenError("Translation returned a different number of cues.")
        for item, cue in zip(items, translated, strict=True):
            item["translation"] = cue.text
            item["translation_full"] = cue.text
            item["status"] = "draft"
    # Relative video/evidence paths remain relative to the input plan's directory.
    relocated = copy.deepcopy(plan)
    relocated["target_language"] = config.target_language
    if output.parent.resolve() != path.parent.resolve():
        relocated["video"]["path"] = os.path.relpath(
            resolve_path(path.parent, plan["video"]["path"]), output.parent
        )
        for item in relocated["captions"]:
            for evidence in item.get("evidence", []):
                evidence["image"] = os.path.relpath(
                    resolve_path(path.parent, evidence["image"]), output.parent
                )
    return write_json(output, relocated)


def render_plan(
    path: Path,
    options: MuxOptions,
    *,
    speech_source: Path | None = None,
    speech_translation: Path | None = None,
    dry_run: bool = False,
) -> Path:
    video, source, target, warnings = compile_plan(
        path, speech_source=speech_source, speech_translation=speech_translation
    )
    protected = {p.resolve() for p in (path, video, speech_source, speech_translation) if p}
    raw = read_json(path)
    protected.update(
        resolve_path(path.parent, e["image"])
        for c in raw["captions"]
        if c["status"] != "omit"
        for e in c.get("evidence", [])
    )
    output = options.video_output.resolve()
    if output in protected:
        raise ScreenError("Output cannot replace the video, plan, speech or screenshot inputs.")
    report = output.with_suffix(".screen-render.json")
    if report in protected:
        raise ScreenError("Render report would replace an input file.")
    if report.exists() and not options.overwrite:
        raise ScreenError(f"Render report already exists: {report}; choose a new output name.")
    if output.exists() and not options.overwrite:
        raise ScreenError(f"Output already exists: {output}; use a new name or --overwrite-video.")
    ffmpeg = find_ffmpeg(explicit=options.ffmpeg_path, need_subtitles_filter=True)
    actual_duration = probe_duration(video, ffmpeg_path=ffmpeg)
    if abs(actual_duration - raw["video"]["duration"]) > 0.05:
        raise ScreenError("Plan duration does not match the source video.")
    for warning in warnings:
        print(f"Warning: {warning}")
    print(f"Screen + speech: {len(target)} cues; {actual_duration:.3f}s; original audio retained.")
    options.video_input = video
    options.mux_mode = "hard"
    if dry_run:
        options.subtitle_input = output.with_suffix(".srt")
        return run_mux(options, dry_run=True)
    # The muxer's atomic publish protects an existing movie if encoding fails.
    with TemporaryDirectory(prefix="video-txt-screen-render-") as directory:
        options.subtitle_input = Path(directory) / "combined.zh.srt"
        write_srt(options.subtitle_input, target)
        result = run_mux(options)
    # Sidecars are provenance for this exact render, never the editable authority.
    # Embed their contents in one atomic report to avoid overwriting unrelated SRTs.
    write_json(
        report,
        {
            "schema": "video-txt.screen-render",
            "version": 1,
            "plan": str(path.resolve()),
            "plan_sha256": sha256(path),
            "video_sha256": raw["video"]["sha256"],
            "output": str(output),
            "speech_source_sha256": sha256(speech_source) if speech_source else None,
            "speech_translation_sha256": sha256(speech_translation) if speech_translation else None,
            "source_srt": serialize_srt(source),
            "translation_srt": serialize_srt(target),
            "cue_count": len(target),
            "warnings": warnings,
        },
    )
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    subs = parser.add_subparsers(dest="screen_action", required=True)
    scan = subs.add_parser(
        "scan", help="Collect timestamped screenshots and OCR; create a draft plan."
    )
    scan.add_argument("video", type=Path)
    scan.add_argument("-o", "--output-dir", type=Path)
    scan.add_argument("--interval", type=float, default=1.0)
    scan.add_argument("--from", dest="start", type=float, default=0.0)
    scan.add_argument("--to", dest="end", type=float)
    scan.add_argument("--ocr-language", default="eng")
    scan.add_argument("--max-frames", type=int, default=600)
    scan.add_argument("--ffmpeg", dest="ffmpeg_path")
    scan.add_argument("--dry-run", action="store_true")
    translate = subs.add_parser("translate", help="Translate corrected OCR into a new draft plan.")
    translate.add_argument("plan", type=Path)
    translate.add_argument("-o", "--output", type=Path)
    add_translation_arguments(translate, debug_flag="--debug-dir")
    translate.set_defaults(provider="lmstudio")
    translate.add_argument("--dry-run", action="store_true")
    for action in ("validate", "render"):
        command = subs.add_parser(
            action, help=f"{action.title()} reviewed screen and speech captions."
        )
        command.add_argument("plan", type=Path)
        command.add_argument("--speech-source", type=Path)
        command.add_argument("--speech-translation", type=Path)
        if action == "render":
            command.add_argument("-o", "--video-output", type=Path)
            add_mux_arguments(command)
            command.set_defaults(
                mux_mode="hard",
                hard_subtitle_layout="bottom-box",
                hard_subtitle_box_height=0.15,
                hard_subtitle_box_opacity=0.65,
                target_language="Simplified Chinese",
                bilingual=False,
            )
            command.add_argument("--dry-run", action="store_true")
