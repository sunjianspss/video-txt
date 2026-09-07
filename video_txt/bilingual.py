"""Showing the original line and the translation together.

Two languages on screen is what a subtitle is for when the point is not only to
follow the film: somebody learning the language reads both, and somebody
proofreading a translation cannot judge a line without the line it came from.
Both of those are read side by side, and both are badly served by opening two
files and scrolling them in step.

The two subtitles are the same subtitle -- the translator keeps the cue count,
the indexes and the timings, and the audit fails the pair when it does not -- so
merging them is positional, and a mismatch is refused rather than guessed at.

Each language is flattened onto one line. A source cue broken over two lines and
a translation broken over two would stack four deep and take half the frame; the
break was a decision about one language's line width, and it does not survive
being put next to another language anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .subtitles import SubtitleCue, join_text_parts

# Which language reads first. Chinese subtitles conventionally lead with the
# translation and keep the original underneath, smaller.
ORDERS = ("translation-first", "source-first")
DEFAULT_ORDER = "translation-first"


class BilingualError(ValueError):
    pass


@dataclass(frozen=True)
class BilingualCue:
    """One cue in two languages, and which of them is the secondary line."""

    index: str
    timing: str
    primary: str
    secondary: str

    @property
    def lines(self) -> list[str]:
        return [line for line in (self.primary, self.secondary) if line]

    @property
    def secondary_lines(self) -> int:
        """How many trailing lines are the quieter language, for the ASS styling."""
        return 1 if self.secondary and self.primary else 0

    def to_cue(self) -> SubtitleCue:
        return SubtitleCue(index=self.index, timing=self.timing, text_lines=self.lines)


def _flattened(cue: SubtitleCue) -> str:
    return join_text_parts([line.strip() for line in cue.text_lines])


def merge_subtitles(
    source_cues: list[SubtitleCue],
    translated_cues: list[SubtitleCue],
    *,
    order: str = DEFAULT_ORDER,
) -> list[BilingualCue]:
    """Put the two subtitles on one timeline, one language per line."""
    if order not in ORDERS:
        raise BilingualError(f"Unknown bilingual order: {order!r}. Use one of {', '.join(ORDERS)}.")
    if len(source_cues) != len(translated_cues):
        raise BilingualError(
            f"The source has {len(source_cues)} cues and the translation "
            f"{len(translated_cues)}. A bilingual subtitle pairs them line for line, so the two "
            "have to be the same subtitle. Run audit-translation to see where they part ways."
        )

    merged: list[BilingualCue] = []
    for source, translated in zip(source_cues, translated_cues, strict=True):
        if source.time_range != translated.time_range:
            raise BilingualError(
                f"Cue {source.index} is timed differently in the two files. "
                "Run audit-translation before merging them."
            )
        original, translation = _flattened(source), _flattened(translated)
        first, second = (
            (translation, original) if order == "translation-first" else (original, translation)
        )
        merged.append(
            BilingualCue(
                index=translated.index,
                timing=translated.timing,
                primary=first,
                secondary=second,
            )
        )
    return merged


def bilingual_cues(merged: list[BilingualCue]) -> list[SubtitleCue]:
    return [item.to_cue() for item in merged]


def secondary_line_counts(merged: list[BilingualCue]) -> list[int]:
    return [item.secondary_lines for item in merged]


def bilingual_subtitle_path(translation: Path) -> Path:
    return translation.with_name(f"{translation.stem}.bilingual.srt")
