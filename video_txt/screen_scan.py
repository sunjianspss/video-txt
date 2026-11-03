"""Bounded, restartable local OCR acquisition. Never promotes OCR to reviewed text."""

from __future__ import annotations

import csv
import html
import io
import math
import os
import re
import shutil
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

from .artifacts import write_json
from .media import find_ffmpeg, probe_duration
from .screen import SCHEMA, VERSION, ScreenError, number, read_json, sha256


def call(args: list[str]) -> str:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScreenError(f"Tool failed: {args[0]}: {exc}") from exc
    if completed.returncode:
        raise ScreenError(f"{args[0]} failed: {completed.stderr[-1500:]}")
    return completed.stdout


def parse_ocr(tsv: str) -> tuple[str, float, list[dict]]:
    rows = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    lines: dict[tuple, list[str]] = {}
    boxes = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text or row.get("level") != "5":
            continue
        try:
            confidence = float(row["conf"])
            box = {key: int(row[key]) for key in ("left", "top", "width", "height")}
        except (ValueError, KeyError) as exc:
            raise ScreenError("Invalid Tesseract TSV response.") from exc
        if confidence < 0:
            continue
        key = tuple(row.get(k) for k in ("page_num", "block_num", "par_num", "line_num"))
        lines.setdefault(key, []).append(text)
        boxes.append({"text": text, "confidence": confidence, **box})
    text = "\n".join(" ".join(words) for words in lines.values())
    confidence = sum(b["confidence"] for b in boxes) / len(boxes) if boxes else 0
    return text, round(confidence, 2), boxes


def skeleton(text: str) -> str:
    # Digits group animation frames; all original variants are retained below.
    return re.sub(r"\d", "#", " ".join(text.casefold().split()))


def draft_captions(samples: list[dict], end: float) -> list[dict]:
    groups: list[list[dict]] = []
    for sample in samples:
        if not sample["text"].strip():
            groups.append([sample])
            continue
        if (
            groups
            and groups[-1][-1]["text"].strip()
            and SequenceMatcher(
                None, skeleton(groups[-1][-1]["text"]), skeleton(sample["text"])
            ).ratio()
            >= 0.88
        ):
            groups[-1].append(sample)
        else:
            groups.append([sample])
    captions = []
    for i, group in enumerate(groups):
        if not group[0]["text"].strip():
            continue
        # Use the latest sample as a suggestion, never max length or an average
        # of animated values. A reviewer must still inspect the original frames.
        chosen = group[-1]
        variants = list(dict.fromkeys(s["text"] for s in group))
        captions.append(
            {
                "id": f"screen-{len(captions) + 1:04}",
                "start": group[0]["time"],
                "end": groups[i + 1][0]["time"] if i + 1 < len(groups) else end,
                "source": chosen["text"],
                "translation": "",
                "translation_full": "",
                "status": "draft",
                "note": "Check screenshot, final animated values, blur, timing and reading speed.",
                "variants": variants,
                "evidence": [{k: s[k] for k in ("time", "image", "sha256")} for s in group],
            }
        )
    return captions


def scan_video(
    video: Path,
    output: Path,
    *,
    interval: float = 1,
    start: float = 0,
    end: float | None = None,
    ocr_language: str = "eng",
    max_frames: int = 600,
    ffmpeg_path: str | None = None,
    dry_run: bool = False,
) -> Path:
    video, output = video.expanduser().resolve(), output.expanduser().resolve()
    if not video.is_file():
        raise ScreenError(f"Video not found: {video}")
    interval, start = number(interval, "interval"), number(start, "start")
    if interval <= 0 or max_frames < 1:
        raise ScreenError("interval and max-frames must be positive.")
    ffmpeg = find_ffmpeg(explicit=ffmpeg_path)
    duration = probe_duration(video, ffmpeg_path=ffmpeg)
    end = duration if end is None else number(end, "end")
    if not 0 <= start < end <= duration:
        raise ScreenError("Scan range must be inside the video duration.")
    count = math.ceil((end - start) / interval)
    if count > max_frames:
        raise ScreenError(
            f"Scan needs {count} frames (limit {max_frames}); select a range or interval."
        )
    if dry_run:
        print(f"Would sample {count} frames at exact seek times {start}–{end}s -> {output}")
        return output / "plan.json"
    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise ScreenError(
            "Missing local OCR dependency: tesseract. Install it with brew install tesseract."
        )
    settings = {
        "schema": "video-txt.screen-scan",
        "version": 1,
        "video_sha256": sha256(video),
        "interval": interval,
        "start": start,
        "end": end,
        "ocr_language": ocr_language,
        "ocr_version": call([tesseract, "--version"]).splitlines()[0],
    }
    state = output / "scan.json"
    plan_path = output / "plan.json"
    if state.exists():
        if read_json(state).get("settings") != settings:
            raise ScreenError(
                "Existing scan has different inputs/settings; use a new output directory."
            )
    elif output.exists() and any(output.iterdir()):
        raise ScreenError(
            "Scan directory is not empty and has no matching scan.json; use a new one."
        )
    output.mkdir(parents=True, exist_ok=True)
    frames = output / "frames"
    frames.mkdir(exist_ok=True)
    if not state.exists():
        write_json(state, {"settings": settings, "complete": False})
    samples = []
    for i in range(count):
        time = round(start + i * interval, 6)
        image = frames / f"{i:06}.jpg"
        record = frames / f"{i:06}.json"
        cached = read_json(record) if record.exists() else None
        if cached and image.is_file() and cached.get("sha256") == sha256(image):
            if cached.get("time") != time:
                raise ScreenError("Cached sample time changed; use a new scan directory.")
            samples.append(cached)
            continue
        call(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-ss",
                str(time),
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(image),
            ]
        )
        if not image.is_file() or image.stat().st_size == 0:
            raise ScreenError(f"No frame returned at {time}s; reduce the scan end time.")
        text, confidence, boxes = parse_ocr(
            call(
                [
                    tesseract,
                    str(image),
                    "stdout",
                    "-l",
                    ocr_language,
                    "tsv",
                ]
            )
        )
        sample = {
            "time": time,
            "image": os.path.relpath(image, output),
            "sha256": sha256(image),
            "text": text,
            "confidence": confidence,
            "boxes": boxes,
        }
        write_json(record, sample)
        samples.append(sample)
        print(f"OCR {i + 1}/{count}: {time:.2f}s, {len(text)} characters", flush=True)
    write_json(state, {"settings": settings, "complete": True, "samples": samples})
    if not plan_path.exists():
        write_json(
            plan_path,
            {
                "schema": SCHEMA,
                "version": VERSION,
                "video": {
                    "path": os.path.relpath(video, output),
                    "sha256": settings["video_sha256"],
                    "duration": duration,
                },
                "source_language": ocr_language,
                "target_language": "zh-CN",
                "captions": draft_captions(samples, end),
            },
        )
    # An ordinary local HTML artifact; no remote assets or script execution.
    cards = []
    for sample in samples:
        cards.append(
            f"<article><h2>{sample['time']:.2f}s · confidence {sample['confidence']}</h2>"
            f'<img src="{html.escape(sample["image"])}" loading="lazy">'
            f"<pre>{html.escape(sample['text'])}</pre></article>"
        )
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Screen OCR evidence</title>'
        "<style>body{max-width:1200px;margin:24px auto;font-family:sans-serif}"
        "article{border-bottom:1px solid #aaa;margin-bottom:28px}img{width:100%}"
        "pre{white-space:pre-wrap}</style><h1>画面文字识别证据</h1>" + "".join(cards),
        encoding="utf-8",
    )
    print(f"Draft: {plan_path}\nEvidence: {output / 'review.html'}")
    return plan_path
