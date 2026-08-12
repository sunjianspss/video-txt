from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .media import MediaError, probe_duration
from .subtitles import SubtitleCue, SubtitleFormatError, language_suffix, parse_srt

# Whisper decodes in ~30 second windows and emits one line per window, so a model that
# has stopped hearing speech repeats a single phrase for as long as it stays lost.
# Real dialogue does repeat a word a few times over, but not across whole minutes.
MIN_RUN_LENGTH = 4
MIN_RUN_SPAN = 60.0
MAX_REPORTED_RUNS = 3

# A transcript that stops far short of the end of the media means Whisper was fed a
# truncated file -- most often a download that had not finished when it was read.
# Half is generous: real dialogue thins out towards the credits, but does not vanish
# for the entire second half. Short clips are exempt; one long musical stretch could
# legitimately be most of their runtime.
MIN_COVERAGE_SHARE = 0.5
MIN_JUDGED_MEDIA_DURATION = 300.0

# Scripts that are obvious on sight. Latin-script languages cannot be told apart this
# cheaply, so asking for French and getting English goes unnoticed here.
SCRIPTS: dict[str, tuple[str, re.Pattern[str]]] = {
    "ja": ("Japanese", re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")),
    "zh": ("Chinese", re.compile(r"[\u4e00-\u9fff]")),
    "zh-hant": ("Chinese", re.compile(r"[\u4e00-\u9fff]")),
    "ko": ("Korean", re.compile(r"[\u1100-\u11ff\uac00-\ud7af]")),
    "ru": ("Russian", re.compile(r"[\u0400-\u04ff]")),
    "ar": ("Arabic", re.compile(r"[\u0600-\u06ff]")),
    "hi": ("Hindi", re.compile(r"[\u0900-\u097f]")),
    "th": ("Thai", re.compile(r"[\u0e00-\u0e7f]")),
}
MIN_SCRIPT_SHARE = 0.6


@dataclass(frozen=True)
class StuckRun:
    text: str
    count: int
    start: float
    end: float

    @property
    def span(self) -> float:
        return max(0.0, self.end - self.start)


def clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600}:{total // 60 % 60:02d}:{total % 60:02d}"


def repeated_runs(cues: list[SubtitleCue]) -> Iterator[list[SubtitleCue]]:
    run: list[SubtitleCue] = []
    for cue in cues:
        if cue.is_empty:
            continue
        if run and run[-1].text.strip() != cue.text.strip():
            yield run
            run = []
        run.append(cue)
    if run:
        yield run


def stuck_runs(cues: list[SubtitleCue]) -> list[StuckRun]:
    found: list[StuckRun] = []
    for run in repeated_runs(cues):
        if len(run) < MIN_RUN_LENGTH:
            continue
        candidate = StuckRun(
            text=run[0].text.strip(),
            count=len(run),
            start=run[0].start_seconds,
            end=run[-1].end_seconds,
        )
        if candidate.span >= MIN_RUN_SPAN:
            found.append(candidate)
    return sorted(found, key=lambda run: run.count, reverse=True)


def script_share(cues: list[SubtitleCue], pattern: re.Pattern[str]) -> float:
    texts = [cue.text for cue in cues if not cue.is_empty]
    if not texts:
        return 0.0
    return sum(1 for text in texts if pattern.search(text)) / len(texts)


def transcript_problems(cues: list[SubtitleCue], *, language: str | None = None) -> list[str]:
    problems = [
        f"{run.text!r} repeats {run.count} times in a row ({clock(run.start)} -> {clock(run.end)})"
        for run in stuck_runs(cues)[:MAX_REPORTED_RUNS]
    ]
    named_script = SCRIPTS.get(language_suffix(language)) if language else None
    if named_script:
        name, pattern = named_script
        share = script_share(cues, pattern)
        if share < MIN_SCRIPT_SHARE:
            problems.append(
                f"--language asked for {name}, but only {share:.0%} of the lines "
                f"contain {name} characters"
            )
    return problems


def coverage_problem(cues: list[SubtitleCue], media_duration: float) -> str | None:
    if media_duration < MIN_JUDGED_MEDIA_DURATION:
        return None
    ends = [cue.end_seconds for cue in cues if not cue.is_empty]
    if not ends:
        return None
    covered = max(ends)
    if covered >= media_duration * MIN_COVERAGE_SHARE:
        return None
    return (
        f"the transcript stops at {clock(covered)} but the media runs until "
        f"{clock(media_duration)} -- was the file fully downloaded before transcribing?"
    )


def probe_media_duration(media_path: Path) -> float | None:
    """Best effort; a quality check must not fail just because ffprobe could not run."""
    try:
        return probe_duration(media_path)
    except (MediaError, OSError):
        return None


def describe_problems(
    path: Path, problems: list[str], *, language: str | None, whisper_note: bool = True
) -> str:
    asked = f" --language {language}" if language else " --language <code>"
    if whisper_note:
        explanation = [
            "Whisper does that when it stops hearing speech, most often because it guessed",
            "the language wrong -- it only listens to the first 30 seconds, so a musical",
            "intro throws it off. Worth trying:",
            f"  --retranscribe{asked} "
            "--whisper-arg=--condition_on_previous_text --whisper-arg=False",
        ]
    else:
        explanation = [
            "That usually means the media file was cut short: a download that was still",
            "running, or an interrupted copy. Once the full file is in place, try:",
            f"  --retranscribe{asked}",
        ]
    return "\n".join(
        [
            f"This transcript looks broken: {path}",
            *(f"  * {problem}" for problem in problems),
            *explanation,
            "Pass --skip-transcript-check to use this transcript as it is.",
        ]
    )


def check_transcript(
    path: Path, *, language: str | None = None, media_path: Path | None = None
) -> str | None:
    """Report on an SRT that Whisper may have botched. None means it looks fine."""
    try:
        cues = parse_srt(path)
    except (OSError, SubtitleFormatError):
        return None
    problems = transcript_problems(cues, language=language)
    whisper_note = bool(problems)
    duration = probe_media_duration(media_path) if media_path is not None else None
    if duration is not None:
        gap = coverage_problem(cues, duration)
        if gap is not None:
            problems.append(gap)
    if not problems:
        return None
    return describe_problems(path, problems, language=language, whisper_note=whisper_note)
