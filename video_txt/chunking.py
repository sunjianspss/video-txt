"""Split long audio at silences so every Whisper run starts with the prompt.

Whisper only feeds ``--initial-prompt`` to the first 30-second window; later
windows are primed with the text decoded before them. On a long file the prompt
is therefore forgotten within minutes, and a wrongly spelled name (Claude ->
Quad) then repeats to the end. Transcribing short pieces, each started with the
prompt, keeps it in effect throughout. Everything here is pure so that the
cutting and stitching can be tested without Whisper.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .subtitles import SubtitleCue, seconds_to_srt_time

CHUNK_SECONDS = 120.0
# How far from the ideal cut point to look for a silence.
CHUNK_SEARCH_SECONDS = 20.0
# A run shorter than this still has the prompt in context for all of it.
MIN_CHUNKED_DURATION = 240.0
SILENCE_FILTER = "silencedetect=noise=-35dB:d=0.4"

_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class Chunk:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_silences(ffmpeg_log: str) -> list[tuple[float, float]]:
    """Silent stretches reported by ffmpeg's silencedetect, as (start, end)."""
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for line in ffmpeg_log.splitlines():
        if (match := _SILENCE_START.search(line)) is not None:
            start = max(0.0, float(match[1]))
        elif (match := _SILENCE_END.search(line)) is not None and start is not None:
            silences.append((start, float(match[1])))
            start = None
    return silences


def plan_chunks(
    duration: float,
    silences: Sequence[tuple[float, float]],
    *,
    target: float = CHUNK_SECONDS,
    search: float = CHUNK_SEARCH_SECONDS,
) -> list[Chunk]:
    """Cut ``[0, duration]`` into pieces of about ``target`` seconds.

    Each cut goes to the middle of the silence nearest the ideal point, so no
    word is split; with no silence in reach it falls back to the ideal point.
    """
    if duration <= 0:
        return []
    middles = [(start + end) / 2 for start, end in silences if end > start]
    chunks: list[Chunk] = []
    position = 0.0
    while duration - position > target + search:
        ideal = position + target
        near = [m for m in middles if abs(m - ideal) <= search and m > position]
        cut = min(near, key=lambda m: abs(m - ideal)) if near else ideal
        chunks.append(Chunk(position, cut))
        position = cut
    chunks.append(Chunk(position, duration))
    return chunks


# Whisper "loops" when it loses the thread: the same line over and over. Real
# speech does not produce four identical lines in a row.
REPEAT_RUN = 4


_SENTENCE_BREAK = re.compile(r"(?<=[.?!])\s+")


def sentences(texts: Sequence[str]) -> list[str]:
    """Texts split into sentences, so a loop inside one segment shows up too."""
    return [part for text in texts for part in _SENTENCE_BREAK.split(text.strip()) if part]


def repeated_runs(texts: Sequence[str]) -> list[tuple[int, int]]:
    """(first index, length) of every stretch of REPEAT_RUN+ identical lines."""
    keys = [" ".join(text.lower().split()) for text in texts]
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(keys):
        end = index
        while end + 1 < len(keys) and keys[end + 1] == keys[index]:
            end += 1
        if end - index + 1 >= REPEAT_RUN and keys[index]:
            runs.append((index, end - index + 1))
        index = end + 1
    return runs


def collapse_repeats(items: Sequence[Any], texts: Sequence[str]) -> list[Any]:
    """``items`` with every looped stretch cut down to its first line."""
    drop: set[int] = set()
    for first, length in repeated_runs(texts):
        drop.update(range(first + 1, first + length))
    return [item for number, item in enumerate(items) if number not in drop]


def _has_text(text: str) -> bool:
    return any(char.isalnum() for char in text)


def clamp_cues(cues: Sequence[SubtitleCue], duration: float) -> list[SubtitleCue]:
    """Keep the cues that lie inside a piece of ``duration`` seconds.

    At the end of a short audio Whisper likes to add a cue, often empty or a bare
    "!", whose times run past the audio or even backwards. Left in, they land
    inside the next piece once offset. A cue starting at or after the end is
    dropped, as is one with no letters or digits; the rest are cut to fit.
    """
    kept = []
    for cue in cues:
        start, end = cue.time_range
        if start >= duration or not _has_text(cue.text):
            continue
        end = min(max(end, start), duration)
        timing = f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}"
        kept.append(SubtitleCue(index=cue.index, timing=timing, text_lines=list(cue.text_lines)))
    return kept


def clamp_payload(payload: Mapping[str, Any], duration: float) -> dict[str, Any]:
    """The same for Whisper JSON: drop words and segments starting past the end."""
    segments = []
    for segment in payload.get("segments") or []:
        moved = dict(segment)
        if isinstance(moved.get("start"), (int, float)) and moved["start"] >= duration:
            continue
        if isinstance(moved.get("words"), list):
            moved["words"] = [
                {**word, "end": min(max(word["end"], word["start"]), duration)}
                for word in moved["words"]
                if isinstance(word.get("start"), (int, float))
                and isinstance(word.get("end"), (int, float))
                and word["start"] < duration
            ]
        if isinstance(moved.get("end"), (int, float)):
            moved["end"] = min(moved["end"], duration)
        segments.append(moved)
    return {**payload, "segments": segments}


def shift_cues(cues: Sequence[SubtitleCue], offset: float) -> list[SubtitleCue]:
    shifted = []
    for cue in cues:
        start, end = cue.time_range
        timing = f"{seconds_to_srt_time(start + offset)} --> {seconds_to_srt_time(end + offset)}"
        shifted.append(SubtitleCue(index=cue.index, timing=timing, text_lines=list(cue.text_lines)))
    return shifted


def renumber(cues: Sequence[SubtitleCue]) -> list[SubtitleCue]:
    return [
        SubtitleCue(index=str(number), timing=cue.timing, text_lines=cue.text_lines)
        for number, cue in enumerate(cues, start=1)
    ]


def merge_whisper_payloads(parts: Sequence[tuple[float, Mapping[str, Any]]]) -> dict[str, Any]:
    """Join per-chunk Whisper JSON into one result on the whole file's clock."""
    segments: list[dict[str, Any]] = []
    language: Any = None
    for offset, payload in parts:
        language = language or payload.get("language")
        for segment in payload.get("segments") or []:
            moved = dict(segment)
            for key in ("start", "end"):
                if isinstance(moved.get(key), (int, float)):
                    moved[key] = moved[key] + offset
            if isinstance(moved.get("words"), list):
                moved["words"] = [
                    {
                        **word,
                        **{
                            key: word[key] + offset
                            for key in ("start", "end")
                            if isinstance(word.get(key), (int, float))
                        },
                    }
                    for word in moved["words"]
                ]
            segments.append(moved)
    return {"language": language, "segments": segments}
