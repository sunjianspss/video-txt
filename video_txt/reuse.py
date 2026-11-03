"""Keeping the translation of every line a repair did not touch.

`retranscribe-range` rewrites a few minutes of a transcript and leaves the rest
of the file exactly as it was. Translating the repaired file from scratch pays
for the whole feature again -- an hour of local inference for a dozen changed
lines -- and hands back new wording for lines nobody asked about, including any
a person had already corrected by hand.

Matching the repaired source against the one it came from says which lines are
the same line. Their translations are carried over, and only what actually
changed is sent to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .subtitles import SubtitleCue, parse_srt


class ReuseError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreviousTranslation:
    """A source subtitle and the translation that was made from it."""

    source: Path
    translation: Path


@dataclass(frozen=True)
class ReusePlan:
    by_position: dict[str, str]
    translatable: int
    previous: PreviousTranslation

    @property
    def reused(self) -> int:
        return len(self.by_position)

    @property
    def fresh(self) -> int:
        return self.translatable - self.reused


def normalized(text: str) -> str:
    """Two lines are the same line when they read the same, whitespace aside."""
    return " ".join(text.split()).casefold()


def match_translations(
    cues: list[SubtitleCue],
    *,
    previous_source: list[SubtitleCue],
    previous_translation: list[SubtitleCue],
) -> dict[str, str]:
    """Translations of the previous cues that say the same thing, by new position."""
    if len(previous_source) != len(previous_translation):
        raise ReuseError(
            f"The previous subtitles do not line up: {len(previous_source)} source cues "
            f"against {len(previous_translation)} translated ones. Translating the source "
            "again is the only safe move."
        )
    candidates: dict[str, list[tuple[float, str]]] = {}
    for source, translation in zip(previous_source, previous_translation, strict=True):
        if source.is_empty or translation.is_empty:
            continue
        candidates.setdefault(normalized(source.text), []).append(
            (source.start_seconds, translation.text)
        )

    matched: dict[str, str] = {}
    for position, cue in enumerate(cues, start=1):
        if cue.is_empty:
            continue
        options = candidates.get(normalized(cue.text))
        if not options:
            continue
        # A line like "Yes." is said all through a film. The one nearest in time
        # is the one this cue is; the others are other moments that read alike.
        _, text = min(options, key=lambda option: abs(option[0] - cue.start_seconds))
        matched[str(position)] = text
    return matched


def plan_reuse(cues: list[SubtitleCue], previous: PreviousTranslation) -> ReusePlan:
    return ReusePlan(
        by_position=match_translations(
            cues,
            previous_source=parse_srt(previous.source),
            previous_translation=parse_srt(previous.translation),
        ),
        translatable=sum(1 for cue in cues if not cue.is_empty),
        previous=previous,
    )
