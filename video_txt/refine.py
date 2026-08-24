from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .subtitles import SubtitleCue, display_width, is_cjk, seconds_to_srt_time

SENTENCE_ENDINGS = (".", "?", "!", "。", "？", "！", "…")
CLAUSE_ENDINGS = (",", ";", ":", "，", "；", "：", "、")
NO_SPACE_BEFORE = frozenset(",.!?;:%)]}，。！？；：、…）】》」』")
NO_SPACE_AFTER = frozenset("([{（【《「『")
WORD_DOCUMENT_SCHEMA = "video-txt.word-timestamps"
WORD_DOCUMENT_VERSION = 1


class RefineError(ValueError):
    """The word-level transcript cannot be safely converted to subtitles."""


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    probability: float | None = None


@dataclass(frozen=True)
class TranscriptDocument:
    words: Sequence[Word]
    language: str | None = None

    def __post_init__(self) -> None:
        words = tuple(self.words)
        if not words:
            raise RefineError("Transcript has no usable word timestamps.")
        previous_start = -1.0
        for word in words:
            try:
                finite = math.isfinite(word.start) and math.isfinite(word.end)
            except TypeError as exc:
                raise RefineError("Transcript contains a non-numeric word timestamp.") from exc
            if (
                not word.text.strip()
                or not finite
                or word.start < 0
                or word.end < word.start
                or word.start < previous_start
            ):
                raise RefineError("Transcript contains an unusable word timestamp.")
            if word.probability is not None and (
                not math.isfinite(word.probability) or not 0 <= word.probability <= 1
            ):
                raise RefineError("Transcript contains an unusable word timestamp probability.")
            previous_start = word.start
        object.__setattr__(self, "words", words)


@dataclass(frozen=True)
class RefinePolicy:
    max_duration: float = 6.0
    max_line_width: int = 42
    max_lines: int = 2
    pause_split: float = 0.8


@dataclass(frozen=True)
class RefineResult:
    cues: list[SubtitleCue]
    source_words: int

    @property
    def cue_count(self) -> int:
        return len(self.cues)


def document_from_whisper_result(payload: Mapping[str, Any]) -> TranscriptDocument:
    """Normalize the JSON shape shared by the supported Whisper backends."""
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise RefineError("Whisper JSON has no segments list.")

    words: list[Word] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise RefineError("Whisper JSON contains a malformed segment.")
        raw_words = segment.get("words", [])
        if not isinstance(raw_words, list):
            raise RefineError("Whisper JSON contains a malformed words list.")
        for raw_word in raw_words:
            if not isinstance(raw_word, Mapping):
                raise RefineError("Whisper JSON contains a malformed word timestamp.")
            text = str(raw_word.get("word", "")).strip()
            if not text:
                continue
            try:
                start = float(raw_word["start"])
                end = float(raw_word["end"])
                raw_probability = raw_word.get("probability")
                probability = (
                    float(raw_probability) if raw_probability is not None else None
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RefineError("Whisper JSON contains an invalid word timestamp.") from exc
            if start < 0 or end < start:
                raise RefineError("Whisper JSON contains a reversed word timestamp.")
            if words and start < words[-1].start:
                raise RefineError("Whisper word timestamps are not chronological.")
            words.append(Word(text, start, end, probability))

    if not words:
        raise RefineError(
            "Whisper JSON contains no word timestamps; make sure word timestamps are enabled."
        )
    raw_language = payload.get("language")
    language = str(raw_language) if raw_language else None
    return TranscriptDocument(words=words, language=language)


def write_word_document(
    path: Path,
    document: TranscriptDocument,
    *,
    backend: str,
    model: str,
    subtitle_path: Path | None = None,
) -> Path:
    payload = {
        "schema": WORD_DOCUMENT_SCHEMA,
        "version": WORD_DOCUMENT_VERSION,
        "language": document.language,
        "backend": backend,
        "model": model,
        "subtitle_sha256": file_sha256(subtitle_path) if subtitle_path else None,
        "words": [
            {
                "text": word.text,
                "start": word.start,
                "end": word.end,
                "probability": word.probability,
            }
            for word in document.words
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        with suppress(OSError):
            temporary.unlink()
        raise
    return path


def file_sha256(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise RefineError(f"Cannot hash subtitle file: {path}") from exc


def word_document_matches_subtitle(word_path: Path, subtitle_path: Path) -> bool:
    try:
        load_word_document(word_path)
        payload = json.loads(word_path.read_text(encoding="utf-8"))
        expected = payload.get("subtitle_sha256")
        return isinstance(expected, str) and hmac.compare_digest(
            expected,
            file_sha256(subtitle_path),
        )
    except (AttributeError, OSError, json.JSONDecodeError, RefineError):
        return False


def load_word_document(path: Path) -> TranscriptDocument:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RefineError(f"Cannot read word timestamp document: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RefineError(f"Word timestamp document is malformed: {path}")
    if payload.get("schema") != WORD_DOCUMENT_SCHEMA:
        raise RefineError(f"Unknown word timestamp schema in: {path}")
    if payload.get("version") != WORD_DOCUMENT_VERSION:
        raise RefineError(f"Unsupported word timestamp version in: {path}")

    raw_words = payload.get("words")
    if not isinstance(raw_words, list):
        raise RefineError(f"Word timestamp document has no words list: {path}")
    words: list[Word] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, Mapping):
            raise RefineError(f"Word timestamp document has a malformed word: {path}")
        try:
            raw_probability = raw_word.get("probability")
            words.append(
                Word(
                    text=str(raw_word["text"]),
                    start=float(raw_word["start"]),
                    end=float(raw_word["end"]),
                    probability=(
                        float(raw_probability) if raw_probability is not None else None
                    ),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RefineError(f"Word timestamp document has invalid values: {path}") from exc
    if not words:
        raise RefineError(f"Word timestamp document contains no words: {path}")
    raw_language = payload.get("language")
    return TranscriptDocument(
        words=words,
        language=str(raw_language) if raw_language else None,
    )


def refine_transcript(
    document: TranscriptDocument,
    policy: RefinePolicy | None = None,
) -> RefineResult:
    """Turn word timestamps into readable, independently timed subtitle cues."""
    policy = policy or RefinePolicy()
    _validate_policy(policy)
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in document.words:
        if not word.text.strip():
            continue
        follows_pause = current and word.start - current[-1].end >= policy.pause_split
        if follows_pause:
            groups.append(current)
            current = []
        current.append(word)
        while len(current) > 1 and _exceeds_constraints(current, policy):
            split_at = _last_clause_boundary(current[:-1]) or len(current) - 1
            groups.append(current[:split_at])
            current = current[split_at:]
        if word.text.rstrip().endswith(SENTENCE_ENDINGS):
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    cues = [
        _cue_from_words(index, words, policy) for index, words in enumerate(groups, start=1)
    ]
    return RefineResult(cues=cues, source_words=len(document.words))


def _exceeds_constraints(words: list[Word], policy: RefinePolicy) -> bool:
    if words[-1].end - words[0].start > policy.max_duration:
        return True
    text = _join_words(words)
    return len(_wrap_text(text, policy.max_line_width)) > policy.max_lines


def _last_clause_boundary(words: list[Word]) -> int | None:
    for position in range(len(words), 0, -1):
        if words[position - 1].text.rstrip().endswith(CLAUSE_ENDINGS):
            return position
    return None


def _validate_policy(policy: RefinePolicy) -> None:
    if policy.max_duration <= 0:
        raise RefineError("Refinement policy max_duration must be greater than zero.")
    if policy.max_line_width < 2:
        raise RefineError("Refinement policy max_line_width must be at least two.")
    if policy.max_lines < 1:
        raise RefineError("Refinement policy max_lines must be at least one.")
    if policy.pause_split < 0:
        raise RefineError("Refinement policy pause_split cannot be negative.")


def _cue_from_words(index: int, words: list[Word], policy: RefinePolicy) -> SubtitleCue:
    start = seconds_to_srt_time(words[0].start)
    end = seconds_to_srt_time(words[-1].end)
    text = _join_words(words)
    return SubtitleCue(
        str(index),
        f"{start} --> {end}",
        _wrap_text(text, policy.max_line_width),
    )


def _join_words(words: list[Word]) -> str:
    joined = ""
    for word in words:
        part = word.text.strip()
        if not part:
            continue
        needs_space = (
            joined
            and not is_cjk(joined[-1])
            and not is_cjk(part[0])
            and part[0] not in NO_SPACE_BEFORE
            and joined[-1] not in NO_SPACE_AFTER
        )
        if needs_space:
            joined += " "
        joined += part
    return joined


def _wrap_text(text: str, max_width: int) -> list[str]:
    remaining = text.strip()
    lines: list[str] = []
    while remaining:
        if display_width(remaining) <= max_width:
            lines.append(remaining)
            break

        width = 0
        hard_break = 0
        for index, char in enumerate(remaining):
            char_width = display_width(char)
            if width + char_width > max_width:
                break
            width += char_width
            hard_break = index + 1

        prefix = remaining[:hard_break]
        whitespace_break = max(prefix.rfind(" "), prefix.rfind("\t"))
        ends_at_word_boundary = hard_break < len(remaining) and remaining[hard_break].isspace()
        break_at = (
            hard_break
            if ends_at_word_boundary or whitespace_break <= 0
            else whitespace_break
        )
        lines.append(remaining[:break_at].rstrip())
        remaining = remaining[break_at:].lstrip()
    return lines
