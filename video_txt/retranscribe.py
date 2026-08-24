from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .media import (
    AudioStreamSelection,
    find_ffmpeg,
    format_command,
    probe_audio_streams,
    select_audio_stream,
)
from .subtitles import SubtitleCue, parse_srt, seconds_to_srt_time, write_srt
from .transcribe import (
    TranscribeOptions,
    build_command,
    describe_audio_selection,
    resolve_backend,
    resolve_model,
    run_transcribe,
)

TIMECODE_PATTERN = re.compile(
    r"(?:(?P<hours>\d+):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{2})"
    r"(?:[.,](?P<fraction>\d{1,3}))?"
)


class RetranscribeError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetranscribeOptions:
    video_path: Path
    subtitle_path: Path
    output_path: Path
    report_path: Path
    start: float
    end: float
    padding: float = 1.5
    model: str | None = None
    language: str | None = None
    backend: str = "auto"
    device: str | None = None
    initial_prompt: str | None = None
    extra_args: list[str] = field(default_factory=list)
    refine_subtitles: bool = False
    audio_stream: int | None = None
    overwrite: bool = False
    from_cue: int | None = None
    to_cue: int | None = None


def parse_timecode(value: str) -> float:
    match = TIMECODE_PATTERN.fullmatch(value.strip())
    if not match:
        raise RetranscribeError(
            f"Invalid time {value!r}. Expected HH:MM:SS, optionally with milliseconds."
        )
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise RetranscribeError(
            f"Invalid time {value!r}. Minutes and seconds must be less than 60."
        )
    fraction = int((match.group("fraction") or "0").ljust(3, "0")) / 1000
    return hours * 3600 + minutes * 60 + seconds + fraction


def range_extract_command(
    video_path: Path,
    audio_path: Path,
    *,
    start: float,
    end: float,
    stream_index: int,
    ffmpeg_path: str,
) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{end - start:.3f}",
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]


def _overlaps(cue: SubtitleCue, start: float, end: float) -> bool:
    return cue.end_seconds > start and cue.start_seconds < end


def _absolute_replacements(
    cues: list[SubtitleCue],
    *,
    extraction_start: float,
    start: float,
    end: float,
    boundary_cues: list[SubtitleCue],
) -> list[SubtitleCue]:
    replacements: list[SubtitleCue] = []
    previous_end = start
    boundary_texts = {" ".join(cue.text.split()).casefold() for cue in boundary_cues}
    for cue in sorted(cues, key=lambda item: (item.start_seconds, item.end_seconds)):
        absolute_start = extraction_start + cue.start_seconds
        absolute_end = extraction_start + cue.end_seconds
        midpoint = (absolute_start + absolute_end) / 2
        if cue.is_empty or not start <= midpoint <= end:
            continue
        normalized_text = " ".join(cue.text.split()).casefold()
        crosses_core_boundary = absolute_start < start or absolute_end > end
        if crosses_core_boundary and normalized_text in boundary_texts:
            continue
        clipped_start = max(start, absolute_start, previous_end)
        clipped_end = min(end, absolute_end)
        if clipped_end <= clipped_start:
            continue
        replacements.append(
            SubtitleCue(
                index=str(len(replacements) + 1),
                timing=(
                    f"{seconds_to_srt_time(clipped_start)} --> "
                    f"{seconds_to_srt_time(clipped_end)}"
                ),
                text_lines=list(cue.text_lines),
            )
        )
        previous_end = clipped_end
    return replacements


def _renumber(cues: list[SubtitleCue]) -> list[SubtitleCue]:
    return [
        SubtitleCue(index=str(index), timing=cue.timing, text_lines=list(cue.text_lines))
        for index, cue in enumerate(cues, start=1)
    ]


def _cue_record(cue: SubtitleCue) -> dict[str, object]:
    return {
        "index": cue.index,
        "start": cue.start_seconds,
        "end": cue.end_seconds,
        "timing": cue.timing,
        "text": cue.text,
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
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


def _validate_outputs(options: RetranscribeOptions) -> None:
    source = options.subtitle_path.resolve()
    if options.output_path.resolve() == options.report_path.resolve():
        raise RetranscribeError(
            "The repaired subtitle and retranscription report need different paths."
        )
    if options.output_path.resolve() == source:
        raise RetranscribeError("The repaired subtitle cannot replace the source subtitle.")
    if options.report_path.resolve() == source:
        raise RetranscribeError("The retranscription report cannot replace the source subtitle.")
    for label, path in (("Output", options.output_path), ("Report", options.report_path)):
        if path.exists() and not options.overwrite:
            raise RetranscribeError(
                f"{label} already exists: {path}. Pass --overwrite to replace it."
            )


def _select_stream(options: RetranscribeOptions, *, ffmpeg_path: str) -> AudioStreamSelection:
    streams = probe_audio_streams(options.video_path, ffmpeg_path=ffmpeg_path)
    return select_audio_stream(
        streams,
        preferred_language=options.language,
        requested_index=options.audio_stream,
    )


def run_retranscribe(options: RetranscribeOptions, *, dry_run: bool = False) -> Path:
    if options.start < 0 or options.end <= options.start:
        raise RetranscribeError("The retranscription range must have an end after its start.")
    if options.padding < 0:
        raise RetranscribeError("--padding must be 0 or greater.")
    _validate_outputs(options)

    source_cues = parse_srt(options.subtitle_path)
    removed = [cue for cue in source_cues if _overlaps(cue, options.start, options.end)]
    kept = [cue for cue in source_cues if not _overlaps(cue, options.start, options.end)]
    before = [cue for cue in kept if cue.end_seconds <= options.start]
    after = [cue for cue in kept if cue.start_seconds >= options.end]
    boundary_cues: list[SubtitleCue] = []
    if before:
        boundary_cues.append(max(before, key=lambda cue: cue.end_seconds))
    if after:
        boundary_cues.append(min(after, key=lambda cue: cue.start_seconds))
    extraction_start = max(0.0, options.start - options.padding)
    extraction_end = options.end + options.padding
    ffmpeg_path = str(find_ffmpeg())
    selection = _select_stream(options, ffmpeg_path=ffmpeg_path)
    backend = resolve_backend(options.backend)
    model = resolve_model(mode="transcribe", model=options.model, backend=backend)

    if dry_run:
        audio_path = options.output_path.parent / f".{options.video_path.stem}.range.wav"
        extraction = range_extract_command(
            options.video_path,
            audio_path,
            start=extraction_start,
            end=extraction_end,
            stream_index=selection.stream.index,
            ffmpeg_path=ffmpeg_path,
        )
        command = build_command(
            TranscribeOptions(
                input_path=audio_path,
                output_dir=options.output_path.parent,
                output_format="json" if options.refine_subtitles else "srt",
                model=options.model,
                language=options.language,
                backend=options.backend,
                device=options.device,
                initial_prompt=options.initial_prompt,
                extra_args=options.extra_args,
                refine_subtitles=options.refine_subtitles,
            ),
            backend=backend,
            model=model,
        )
        print(describe_audio_selection(selection))
        print(
            f"Core range: {seconds_to_srt_time(options.start)} --> "
            f"{seconds_to_srt_time(options.end)}"
        )
        print(
            f"Extraction range: {seconds_to_srt_time(extraction_start)} --> "
            f"{seconds_to_srt_time(extraction_end)} (padding {options.padding:g}s)"
        )
        print(f"Would replace {len(removed)} source cues and keep {len(kept)} unchanged.")
        print(format_command(extraction))
        print(format_command(command))
        print(f"Would write: {options.output_path}")
        print(f"Would report: {options.report_path}")
        return options.output_path

    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=options.output_path.parent,
        prefix=f".{options.output_path.stem}.retranscribe-",
    ) as temporary_dir:
        work_dir = Path(temporary_dir)
        audio_path = work_dir / "range.wav"
        extraction = range_extract_command(
            options.video_path,
            audio_path,
            start=extraction_start,
            end=extraction_end,
            stream_index=selection.stream.index,
            ffmpeg_path=ffmpeg_path,
        )
        completed = subprocess.run(extraction, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not audio_path.is_file():
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise RetranscribeError(f"Could not extract the retranscription range: {detail}")

        local_subtitle = run_transcribe(
            TranscribeOptions(
                input_path=audio_path,
                output_dir=work_dir,
                output_format="srt",
                model=model,
                language=options.language,
                backend=backend,
                device=options.device,
                initial_prompt=options.initial_prompt,
                extra_args=options.extra_args,
                refine_subtitles=options.refine_subtitles,
            )
        )
        replacement = _absolute_replacements(
            parse_srt(local_subtitle),
            extraction_start=extraction_start,
            start=options.start,
            end=options.end,
            boundary_cues=boundary_cues,
        )

    merged = _renumber(
        sorted([*kept, *replacement], key=lambda cue: (cue.start_seconds, cue.end_seconds))
    )
    write_srt(options.output_path, merged)
    _write_json(
        options.report_path,
        {
            "schema": "video-txt.retranscription-report",
            "version": 1,
            "video": str(options.video_path.resolve()),
            "source_subtitle": str(options.subtitle_path.resolve()),
            "output_subtitle": str(options.output_path.resolve()),
            "range": {
                "start": options.start,
                "end": options.end,
                "padding": options.padding,
            },
            "requested_range": (
                {"from_cue": options.from_cue, "to_cue": options.to_cue}
                if options.from_cue is not None and options.to_cue is not None
                else {"start": options.start, "end": options.end}
            ),
            "extraction": {"start": extraction_start, "end": extraction_end},
            "audio_stream": {
                "requested": options.audio_stream,
                "selected": selection.stream.index,
                "reason": selection.reason,
            },
            "transcription": {
                "requested_backend": options.backend,
                "backend": backend,
                "model": model,
                "language": options.language,
                "refine_subtitles": options.refine_subtitles,
            },
            "removed_cues": [_cue_record(cue) for cue in removed],
            "added_cues": [_cue_record(cue) for cue in replacement],
        },
    )
    return options.output_path
