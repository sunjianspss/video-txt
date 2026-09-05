"""Keeping the lines a person fixed by hand.

A translation is a generated file: every rerun replaces it. The corrections
somebody made while reading along -- a name the model got wrong, a joke it took
literally, a line that says the opposite of what happens on screen -- are not
generated, and they are the most expensive text in the project. Written into the
translation, they last exactly until the next run.

So they live in a file of their own, anchored to the source line rather than to
a cue number: `clean` renumbers, `retranscribe-range` splices, and a number that
means cue 110 today means somebody else's line tomorrow. The anchor is the
source text plus roughly when it is said, which is how `reuse` already decides
that two cues are the same cue.

An anchor that matches nothing is an error, never a silent no-op. A revision
file that has quietly stopped applying is worse than no revision file at all:
the run still reports success, and every corrected line is back to what the
model said.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .reuse import normalized
from .subtitles import (
    SubtitleCue,
    SubtitleFormatError,
    seconds_to_srt_time,
    srt_time_to_seconds,
)

REVISION_SCHEMA = "video-txt.revisions"
REVISION_VERSION = 1
# An anchor landing this far from where the revision was written is still
# applied -- re-transcribing moves a line by a second or two -- but the report
# says so, because a line that moved by minutes is usually a different line.
NOTABLE_DRIFT = 5.0
# How many timestamps an "which one did you mean" error lists before giving up.
MAX_LISTED_ANCHORS = 3


class RevisionError(ValueError):
    pass


@dataclass(frozen=True)
class Revision:
    """One hand-corrected line, and the source line it belongs to.

    `at` is the start time of that source cue. It is only needed when the same
    line is said more than once, which is what makes the text alone ambiguous.
    """

    source: str
    text: str
    at: float | None = None
    note: str = ""


@dataclass(frozen=True)
class RevisionSet:
    source_language: str | None
    target_language: str | None
    revisions: tuple[Revision, ...]


@dataclass(frozen=True)
class AppliedRevision:
    revision: Revision
    position: int
    cue_index: str
    before: str
    after: str
    drift: float

    @property
    def changed(self) -> bool:
        return self.before != self.after

    def to_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "cue_index": self.cue_index,
            "source": self.revision.source,
            "at": seconds_to_srt_time(self.revision.at) if self.revision.at is not None else None,
            "anchor_drift_seconds": round(self.drift, 3),
            "changed": self.changed,
            "before": self.before,
            "after": self.after,
            "note": self.revision.note or None,
        }


@dataclass(frozen=True)
class RevisionResult:
    cues: list[SubtitleCue]
    applied: tuple[AppliedRevision, ...]

    @property
    def changed_count(self) -> int:
        return sum(item.changed for item in self.applied)

    @property
    def already_current(self) -> tuple[AppliedRevision, ...]:
        """Revisions the translation already agreed with.

        Not a problem: applying the file twice has to be safe, and a model that
        has since learned to get the line right is good news. Worth counting,
        because a revision that is permanently redundant can be deleted.
        """
        return tuple(item for item in self.applied if not item.changed)

    @property
    def drifted(self) -> tuple[AppliedRevision, ...]:
        return tuple(item for item in self.applied if item.drift > NOTABLE_DRIFT)

    def to_dict(self) -> dict[str, object]:
        return {
            "revision_count": len(self.applied),
            "changed_count": self.changed_count,
            "already_current_count": len(self.already_current),
            "drifted_count": len(self.drifted),
            "revisions": [item.to_dict() for item in self.applied],
        }


def anchor_positions(source_cues: list[SubtitleCue], revision: Revision) -> list[int]:
    """Every cue position whose source line reads the same as the anchor."""
    key = normalized(revision.source)
    return [
        position
        for position, cue in enumerate(source_cues, start=1)
        if not cue.is_empty and normalized(cue.text) == key
    ]


def locate_revision(
    source_cues: list[SubtitleCue], revision: Revision, *, number: int
) -> tuple[int, float]:
    """The cue a revision corrects, and how far its anchor time was from it."""
    positions = anchor_positions(source_cues, revision)
    if not positions:
        raise RevisionError(
            f"Revision #{number} anchors to a line that is not in the source subtitle: "
            f"{revision.source!r}. It may have been re-transcribed or removed. "
            "Update the anchor to the line as it reads now, or drop the revision."
        )
    if revision.at is None:
        if len(positions) > 1:
            listed = ", ".join(
                seconds_to_srt_time(source_cues[position - 1].start_seconds)
                for position in positions[:MAX_LISTED_ANCHORS]
            )
            more = ", ..." if len(positions) > MAX_LISTED_ANCHORS else ""
            raise RevisionError(
                f"Revision #{number} anchors to {revision.source!r}, which is said "
                f"{len(positions)} times ({listed}{more}). Add \"at\" with the timestamp "
                "of the one you mean."
            )
        return positions[0], 0.0

    at = revision.at
    position = min(
        positions, key=lambda candidate: abs(source_cues[candidate - 1].start_seconds - at)
    )
    return position, abs(source_cues[position - 1].start_seconds - at)


def plan_revisions(
    source_cues: list[SubtitleCue], revisions: tuple[Revision, ...]
) -> dict[int, tuple[int, Revision, float]]:
    """Which cue each revision lands on, refusing to let two of them share one."""
    planned: dict[int, tuple[int, Revision, float]] = {}
    for number, revision in enumerate(revisions, start=1):
        position, drift = locate_revision(source_cues, revision, number=number)
        if position in planned:
            earlier, _revision, _drift = planned[position]
            raise RevisionError(
                f"Revisions #{earlier} and #{number} both correct cue "
                f"{source_cues[position - 1].index}. Keep whichever one is right."
            )
        planned[position] = (number, revision, drift)
    return planned


def apply_revisions(
    source_cues: list[SubtitleCue],
    translated_cues: list[SubtitleCue],
    revision_set: RevisionSet,
) -> RevisionResult:
    """Put every hand-corrected line back into a freshly translated subtitle."""
    if len(source_cues) != len(translated_cues):
        raise RevisionError(
            f"The source has {len(source_cues)} cues and the translation "
            f"{len(translated_cues)}. Revisions anchor on the source line, so the two have "
            "to be the same subtitle. Run audit-translation to see where they part ways."
        )
    planned = plan_revisions(source_cues, revision_set.revisions)
    cues = list(translated_cues)
    applied: list[AppliedRevision] = []
    for position in sorted(planned):
        _number, revision, drift = planned[position]
        current = cues[position - 1]
        applied.append(
            AppliedRevision(
                revision=revision,
                position=position,
                cue_index=current.index,
                before=current.text,
                after=revision.text,
                drift=drift,
            )
        )
        cues[position - 1] = current.with_text(revision.text)
    return RevisionResult(cues=cues, applied=tuple(applied))


def _required_text(item: dict[str, object], key: str, *, number: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RevisionError(f"Revision #{number} {key!r} must be a non-empty string.")
    return value.strip()


def _optional_timestamp(value: object, *, number: int) -> float | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RevisionError(
            f"Revision #{number} 'at' must be an SRT timestamp such as \"00:10:33,280\"."
        )
    try:
        return srt_time_to_seconds(value)
    except SubtitleFormatError as exc:
        raise RevisionError(
            f"Revision #{number} 'at' is not an SRT timestamp: {value!r}. "
            'Copy it from the source subtitle, for example "00:10:33,280".'
        ) from exc


def _optional_language(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise RevisionError(f"{key} must be a non-empty string when provided.")
    return value.strip() if isinstance(value, str) else None


def load_revisions(path: Path) -> RevisionSet:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise RevisionError(f"Could not read revision file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RevisionError(f"Revision file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RevisionError("Revision file must contain one JSON object.")
    if payload.get("schema") != REVISION_SCHEMA:
        raise RevisionError(f"Revision schema must be {REVISION_SCHEMA!r}: {path}")
    if payload.get("version") != REVISION_VERSION:
        raise RevisionError(f"Revision version must be {REVISION_VERSION}: {path}")
    raw_revisions = payload.get("revisions")
    if not isinstance(raw_revisions, list):
        raise RevisionError("Revision file 'revisions' must be a JSON list.")

    revisions: list[Revision] = []
    seen: set[tuple[str, float | None]] = set()
    for number, item in enumerate(raw_revisions, start=1):
        if not isinstance(item, dict):
            raise RevisionError(f"Revision #{number} must be a JSON object.")
        source = _required_text(item, "source", number=number)
        text = _required_text(item, "text", number=number)
        at = _optional_timestamp(item.get("at"), number=number)
        note = item.get("note", "")
        if not isinstance(note, str):
            raise RevisionError(f"Revision #{number} 'note' must be a string.")
        anchor = (normalized(source), at)
        if anchor in seen:
            raise RevisionError(
                f"Revision #{number} repeats an earlier anchor: {source!r}. "
                "Two corrections of one line cannot both be right; keep one."
            )
        seen.add(anchor)
        revisions.append(Revision(source=source, text=text, at=at, note=note.strip()))

    return RevisionSet(
        source_language=_optional_language(payload, "source_language"),
        target_language=_optional_language(payload, "target_language"),
        revisions=tuple(revisions),
    )


def revised_subtitle_path(translation: Path) -> Path:
    return translation.with_name(f"{translation.stem}.revised.srt")


def revision_report_path_for(output: Path) -> Path:
    return output.with_name(f"{output.stem}.revision-report.json")
