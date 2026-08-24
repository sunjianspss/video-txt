from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .media import MediaError, probe_duration
from .subtitles import (
    SubtitleCue,
    SubtitleFormatError,
    display_width,
    join_text_parts,
    language_suffix,
    parse_srt,
)

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

Severity = Literal["warning", "error"]


@dataclass(frozen=True)
class AuditPolicy:
    """Conservative thresholds for subtitle defects we can identify reliably."""

    full_window_seconds: float = 29.5
    full_window_max_words: int = 6
    full_window_max_chars: int = 40
    max_cue_duration: float = 10.0
    max_lines: int = 2
    max_line_width: int = 42


@dataclass(frozen=True)
class AuditFinding:
    code: str
    message: str
    cue_positions: tuple[int, ...] = ()
    cue_indexes: tuple[str, ...] = ()
    severity: Severity = "warning"
    repair: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "cue_positions": list(self.cue_positions),
            "cue_indexes": list(self.cue_indexes),
            "repair": self.repair,
        }


@dataclass(frozen=True)
class AuditReport:
    cue_count: int
    findings: tuple[AuditFinding, ...]

    @property
    def is_clean(self) -> bool:
        return not self.findings

    @property
    def has_errors(self) -> bool:
        return any(finding.severity == "error" for finding in self.findings)

    @property
    def counts_by_code(self) -> dict[str, int]:
        return dict(sorted(Counter(finding.code for finding in self.findings).items()))

    @property
    def repairable_count(self) -> int:
        return sum(finding.repair is not None for finding in self.findings)

    def to_dict(self) -> dict[str, object]:
        return {
            "cue_count": self.cue_count,
            "is_clean": self.is_clean,
            "has_errors": self.has_errors,
            "finding_count": len(self.findings),
            "error_count": sum(finding.severity == "error" for finding in self.findings),
            "warning_count": sum(finding.severity == "warning" for finding in self.findings),
            "repairable_count": self.repairable_count,
            "summary": self.counts_by_code,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class RepairAction:
    code: str
    cue_positions: tuple[int, ...]
    cue_indexes: tuple[str, ...]
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "cue_positions": list(self.cue_positions),
            "cue_indexes": list(self.cue_indexes),
            "message": self.message,
        }


@dataclass(frozen=True)
class RepairResult:
    cues: list[SubtitleCue]
    actions: tuple[RepairAction, ...]
    unresolved: tuple[AuditFinding, ...]

    @property
    def removed_count(self) -> int:
        return sum(action.code.startswith("remove_") for action in self.actions)

    @property
    def merged_count(self) -> int:
        return sum(action.code == "merge_zero_duration" for action in self.actions)

    def to_dict(self) -> dict[str, object]:
        return {
            "output_cue_count": len(self.cues),
            "removed_count": self.removed_count,
            "merged_count": self.merged_count,
            "actions": [action.to_dict() for action in self.actions],
            "unresolved": [finding.to_dict() for finding in self.unresolved],
        }


def audit_transcript(
    cues: list[SubtitleCue],
    *,
    language: str | None = None,
    media_duration: float | None = None,
    policy: AuditPolicy | None = None,
) -> AuditReport:
    """Inspect subtitle cues without changing them."""
    selected = policy or AuditPolicy()
    findings: list[AuditFinding] = []
    mismatched_indexes = [
        (position, cue.index)
        for position, cue in enumerate(cues, start=1)
        if cue.index != str(position)
    ]
    if mismatched_indexes:
        findings.append(
            AuditFinding(
                code="non_sequential_indexes",
                message="Subtitle indexes are missing, duplicated, or out of sequence.",
                cue_positions=tuple(position for position, _index in mismatched_indexes),
                cue_indexes=tuple(index for _position, index in mismatched_indexes),
                repair="renumber",
            )
        )
    previous_valid: tuple[int, SubtitleCue, float] | None = None
    for position, cue in enumerate(cues, start=1):
        start, end = cue.time_range
        if cue.is_empty:
            findings.append(
                AuditFinding(
                    code="empty_cue",
                    message=f"Cue {cue.index} contains no subtitle text.",
                    cue_positions=(position,),
                    cue_indexes=(cue.index,),
                    severity="error",
                    repair="remove",
                )
            )
            continue
        if end < start:
            findings.append(
                AuditFinding(
                    code="reversed_timing",
                    message=f"Cue {cue.index} ends before it starts.",
                    cue_positions=(position,),
                    cue_indexes=(cue.index,),
                    severity="error",
                )
            )
            continue
        if previous_valid is not None and start < previous_valid[2]:
            previous_position, previous_cue, previous_end = previous_valid
            findings.append(
                AuditFinding(
                    code="overlap",
                    message=(
                        f"Cues {previous_cue.index} and {cue.index} overlap by "
                        f"{previous_end - start:.2f} seconds."
                    ),
                    cue_positions=(previous_position, position),
                    cue_indexes=(previous_cue.index, cue.index),
                )
            )
        words = cue.text.split()
        compact_text = re.sub(r"\s+", "", cue.text)
        if (
            cue.duration >= selected.full_window_seconds
            and len(words) <= selected.full_window_max_words
            and len(compact_text) <= selected.full_window_max_chars
        ):
            findings.append(
                AuditFinding(
                    code="full_window_hallucination",
                    message=(
                        f"Cue {cue.index} shows a short phrase for {cue.duration:.2f} seconds, "
                        "matching a full Whisper decoding window."
                    ),
                    cue_positions=(position,),
                    cue_indexes=(cue.index,),
                    severity="error",
                    repair="remove",
                )
            )
        elif cue.duration > selected.max_cue_duration:
            findings.append(
                AuditFinding(
                    code="overlong_duration",
                    message=(
                        f"Cue {cue.index} remains visible for {cue.duration:.2f} seconds; "
                        f"the configured maximum is {selected.max_cue_duration:.2f}."
                    ),
                    cue_positions=(position,),
                    cue_indexes=(cue.index,),
                )
            )
        if len(cue.text_lines) > selected.max_lines:
            findings.append(
                AuditFinding(
                    code="too_many_lines",
                    message=(
                        f"Cue {cue.index} uses {len(cue.text_lines)} lines; "
                        f"the configured maximum is {selected.max_lines}."
                    ),
                    cue_positions=(position,),
                    cue_indexes=(cue.index,),
                )
            )
        for line_number, line in enumerate(cue.text_lines, start=1):
            width = display_width(line)
            if width <= selected.max_line_width:
                continue
            findings.append(
                AuditFinding(
                    code="line_too_wide",
                    message=(
                        f"Cue {cue.index}, line {line_number} uses {width} display columns; "
                        f"the configured maximum is {selected.max_line_width}."
                    ),
                    cue_positions=(position,),
                    cue_indexes=(cue.index,),
                )
            )
        if end == start:
            preceding_is_removable = any(
                finding.repair == "remove" and position - 1 in finding.cue_positions
                for finding in findings
            )
            can_merge_previous = (
                previous_valid is not None
                and previous_valid[0] == position - 1
                and not preceding_is_removable
            )
            findings.append(
                AuditFinding(
                    code="zero_duration",
                    message=f"Cue {cue.index} has zero display duration.",
                    cue_positions=(position,),
                    cue_indexes=(cue.index,),
                    severity="error",
                    repair="merge_previous" if can_merge_previous else None,
                )
            )
        previous_valid = (position, cue, end)
    for run in stuck_runs(cues)[:MAX_REPORTED_RUNS]:
        positions_and_indexes = [
            (position, cue.index)
            for position, cue in enumerate(cues, start=1)
            if cue.text.strip() == run.text
            and cue.start_seconds >= run.start
            and cue.end_seconds <= run.end
        ]
        findings.append(
            AuditFinding(
                code="repeated_text",
                message=(
                    f"{run.text!r} repeats {run.count} times in a row "
                    f"({clock(run.start)} -> {clock(run.end)})."
                ),
                cue_positions=tuple(position for position, _index in positions_and_indexes),
                cue_indexes=tuple(index for _position, index in positions_and_indexes),
                severity="error",
            )
        )
    named_script = SCRIPTS.get(language_suffix(language)) if language else None
    if named_script:
        name, pattern = named_script
        share = script_share(cues, pattern)
        if share < MIN_SCRIPT_SHARE:
            findings.append(
                AuditFinding(
                    code="wrong_script",
                    message=(
                        f"--language asked for {name}, but only {share:.0%} of the lines "
                        f"contain {name} characters."
                    ),
                    severity="error",
                )
            )
    gap = coverage_problem(cues, media_duration) if media_duration is not None else None
    if gap is not None:
        findings.append(AuditFinding(code="low_coverage", message=gap, severity="error"))
    return AuditReport(cue_count=len(cues), findings=tuple(findings))


def repair_transcript(
    cues: list[SubtitleCue],
    report: AuditReport,
    *,
    policy: AuditPolicy | None = None,
) -> RepairResult:
    """Apply only the repairs explicitly marked safe by an audit report."""
    del policy
    removable = {
        position: finding.code
        for finding in report.findings
        if finding.repair == "remove"
        for position in finding.cue_positions
    }
    mergeable = {
        position: finding
        for finding in report.findings
        if finding.code == "zero_duration" and finding.repair == "merge_previous"
        for position in finding.cue_positions
    }
    actions: list[RepairAction] = []
    repaired: list[SubtitleCue] = []
    unresolved = [finding for finding in report.findings if finding.repair is None]
    last_kept_source_position: int | None = None
    for position, cue in enumerate(cues, start=1):
        if position in removable:
            finding_code = removable[position]
            action_code = (
                "remove_full_window_hallucination"
                if finding_code == "full_window_hallucination"
                else f"remove_{finding_code}"
            )
            actions.append(
                RepairAction(
                    code=action_code,
                    cue_positions=(position,),
                    cue_indexes=(cue.index,),
                    message=f"Removed {finding_code.replace('_', ' ')} at cue {cue.index}.",
                )
            )
            continue
        if position in mergeable:
            if repaired and last_kept_source_position == position - 1:
                previous = repaired[-1]
                previous_text = previous.text.rstrip()
                fragment = cue.text.lstrip()
                if previous_text.endswith(("-", "…")):
                    merged_text = previous_text + fragment
                else:
                    merged_text = join_text_parts([previous_text, fragment])
                repaired[-1] = previous.with_text(merged_text)
                actions.append(
                    RepairAction(
                        code="merge_zero_duration",
                        cue_positions=(position - 1, position),
                        cue_indexes=(previous.index, cue.index),
                        message=f"Merged zero-duration cue {cue.index} into the preceding cue.",
                    )
                )
                last_kept_source_position = position
                continue
            unresolved.append(mergeable[position])
        repaired.append(
            SubtitleCue(
                index=str(len(repaired) + 1),
                timing=cue.timing,
                text_lines=list(cue.text_lines),
            )
        )
        last_kept_source_position = position
    for index, cue in enumerate(repaired, start=1):
        cue.index = str(index)
    renumbered = next(
        (finding for finding in report.findings if finding.repair == "renumber"), None
    )
    if renumbered is not None:
        actions.append(
            RepairAction(
                code="renumber_cues",
                cue_positions=renumbered.cue_positions,
                cue_indexes=renumbered.cue_indexes,
                message="Renumbered subtitle cues sequentially.",
            )
        )
    return RepairResult(repaired, tuple(actions), tuple(unresolved))


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
    duration = probe_media_duration(media_path) if media_path is not None else None
    report = audit_transcript(cues, language=language, media_duration=duration)
    blockers = [finding for finding in report.findings if finding.severity == "error"]
    if not blockers:
        return None
    whisper_note = any(finding.code != "low_coverage" for finding in blockers)
    return describe_problems(
        path,
        [finding.message for finding in blockers],
        language=language,
        whisper_note=whisper_note,
    )
