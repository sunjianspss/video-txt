"""Speaker diarization.

Whisper says what was said, not who said it. A panel or an interview dubbed
from that transcript alone comes back as one narrator reading every part, which
is exactly the information the original had and the dub threw away. pyannote
answers "who spoke when"; this module turns that answer into a speaker per
subtitle line, which is the unit everything downstream already works in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .media import file_identity
from .subtitles import SubtitleCue

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_TOKEN_ENV = "HF_TOKEN"
SPEAKERS_VERSION = 2
# The label every line carries when nobody ran diarization: one speaker, no name.
DEFAULT_SPEAKER = ""
# Diarization returns slivers where two turns touch. They are not speech anyone
# would attribute, and they distort the "who owns this line" tally.
MIN_TURN = 0.2
# A speaker who pauses this briefly is still holding the floor.
MERGE_GAP = 0.5


class DiarizeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap(self, start: float, end: float) -> float:
        return max(0.0, min(self.end, end) - max(self.start, start))


@dataclass
class DiarizeOptions:
    model: str = DEFAULT_MODEL
    token: str = ""
    speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    rediarize: bool = False


def speakers_path(video: Path) -> Path:
    return video.with_name(f"{video.stem}.speakers.json")


def merge_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    merged: list[SpeakerTurn] = []
    for turn in sorted(turns, key=lambda item: (item.start, item.end)):
        if turn.duration < MIN_TURN:
            continue
        previous = merged[-1] if merged else None
        # Only merge turns that are neighbours in time: a gap with somebody else
        # inside it is a real exchange, however short.
        if previous and previous.speaker == turn.speaker and turn.start - previous.end <= MERGE_GAP:
            merged[-1] = SpeakerTurn(previous.start, max(previous.end, turn.end), turn.speaker)
            continue
        merged.append(turn)
    return merged


def collect_turns(diarization: object) -> list[SpeakerTurn]:
    """Read the turns out of whatever shape the installed pyannote returns."""
    annotation = getattr(diarization, "speaker_diarization", diarization)
    itertracks = getattr(annotation, "itertracks", None)
    if itertracks is None:
        raise DiarizeError(
            f"Unexpected diarization result of type {type(diarization).__name__}. "
            "This build of pyannote.audio is not supported."
        )
    return merge_turns(
        [
            SpeakerTurn(float(segment.start), float(segment.end), str(label))
            for segment, _, label in itertracks(yield_label=True)
        ]
    )


def resolve_device() -> str | None:
    """CUDA when it is there. MPS is left alone: pyannote is unreliable on it."""
    try:
        import torch
    except ImportError:
        return None
    return "cuda" if torch.cuda.is_available() else None


def load_pipeline(options: DiarizeOptions) -> object:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizeError(
            "Missing dependency: pyannote.audio.\n"
            "Install it with: uv sync --extra diarize\n"
            "Then accept the model terms at https://hf.co/pyannote/speaker-diarization-3.1"
        ) from exc

    token = options.token or None
    try:
        pipeline = Pipeline.from_pretrained(options.model, token=token)
    except TypeError:
        # pyannote.audio 3.x spells the same argument use_auth_token.
        pipeline = Pipeline.from_pretrained(options.model, use_auth_token=token)

    if pipeline is None:
        raise DiarizeError(
            f"Could not load {options.model}. The model is gated, so a Hugging Face "
            "token is required and the terms have to be accepted first:\n"
            "  https://hf.co/pyannote/segmentation-3.0\n"
            "  https://hf.co/pyannote/speaker-diarization-3.1\n"
            f"Then export {DEFAULT_TOKEN_ENV} with a token that has read access."
        )

    device = resolve_device()
    if device:
        import torch

        pipeline.to(torch.device(device))
    return pipeline


def diarize_media(media: Path, options: DiarizeOptions) -> list[SpeakerTurn]:
    pipeline = load_pipeline(options)
    limits = {
        name: value
        for name, value in (
            ("num_speakers", options.speakers),
            ("min_speakers", options.min_speakers),
            ("max_speakers", options.max_speakers),
        )
        if value is not None
    }
    return collect_turns(pipeline(str(media), **limits))


def diarize_cache_key(video: Path, options: DiarizeOptions) -> dict[str, object]:
    return {
        "source": file_identity(video),
        "model": options.model,
        "speakers": options.speakers,
        "min_speakers": options.min_speakers,
        "max_speakers": options.max_speakers,
    }


def save_turns(
    path: Path,
    turns: list[SpeakerTurn],
    *,
    model: str,
    cache_key: dict[str, object] | None = None,
) -> Path:
    payload = {
        "version": SPEAKERS_VERSION,
        "model": model,
        "cache_key": cache_key,
        "turns": [
            {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": turn.speaker}
            for turn in turns
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_turns(
    path: Path,
    *,
    model: str,
    cache_key: dict[str, object] | None = None,
) -> list[SpeakerTurn] | None:
    """Turns from an earlier run, or None when there is nothing usable to reuse."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != SPEAKERS_VERSION:
        return None
    if payload.get("model") != model:
        return None
    if cache_key is not None and payload.get("cache_key") != cache_key:
        return None

    turns: list[SpeakerTurn] = []
    for record in payload.get("turns", []):
        try:
            turns.append(
                SpeakerTurn(
                    start=float(record["start"]),
                    end=float(record["end"]),
                    speaker=str(record["speaker"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            return None
    return turns


def speaker_totals(turns: list[SpeakerTurn]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for turn in turns:
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + turn.duration
    return totals


def speaker_order(turns: list[SpeakerTurn]) -> list[str]:
    """Speakers from the one who talks most to the one who talks least."""
    totals = speaker_totals(turns)
    return sorted(totals, key=lambda speaker: (-totals[speaker], speaker))


def assign_speakers(
    cues: list[tuple[int, SubtitleCue]], turns: list[SpeakerTurn]
) -> dict[int, str]:
    """Give every cue position the speaker who covers most of its span.

    A line that overlaps nobody keeps the previous line's speaker rather than
    starting a new one: silence in the diarization is usually a missed breath,
    not a change of voice.
    """
    if not turns or not cues:
        return {}

    ordered = sorted(turns, key=lambda turn: turn.start)
    current = speaker_order(turns)[0]
    assigned: dict[int, str] = {}
    first_unfinished = 0

    for position, cue in sorted(cues, key=lambda item: item[1].time_range[0]):
        start, end = cue.time_range
        while first_unfinished < len(ordered) and ordered[first_unfinished].end <= start:
            first_unfinished += 1

        totals: dict[str, float] = {}
        for turn in ordered[first_unfinished:]:
            if turn.start >= end:
                break
            overlap = turn.overlap(start, end)
            if overlap > 0:
                totals[turn.speaker] = totals.get(turn.speaker, 0.0) + overlap
        if totals:
            current = max(totals, key=lambda speaker: totals[speaker])
        assigned[position] = current

    return assigned


def describe_speakers(turns: list[SpeakerTurn]) -> str:
    totals = speaker_totals(turns)
    spoken = sum(totals.values()) or 1.0
    parts = [f"{speaker} {totals[speaker] / spoken:.0%}" for speaker in speaker_order(turns)]
    return ", ".join(parts)


def ensure_speakers(
    video: Path,
    *,
    options: DiarizeOptions,
    dry_run: bool = False,
    label: str | None = None,
) -> list[SpeakerTurn]:
    """Diarize the video, or reuse the turns an earlier run already wrote."""
    prefix = f"{label} " if label else ""
    path = speakers_path(video)
    cache_key = diarize_cache_key(video, options)
    if not options.rediarize:
        cached = load_turns(path, model=options.model, cache_key=cache_key)
        if cached is not None:
            print(f"{prefix}Diarize: skip, reusing {path}")
            print(f"  {len(speaker_totals(cached))} speakers: {describe_speakers(cached)}")
            return cached

    if dry_run:
        print(f"{prefix}Diarize: would run {options.model} (dry run)")
        print(f"  Speaker turns would be written to: {path}")
        return []

    print(f"{prefix}Diarize: running {options.model}")
    turns = diarize_media(video, options)
    if not turns:
        raise DiarizeError(
            f"{options.model} found no speech in {video.name}. "
            "Drop --diarize, or check that the video has an audio track."
        )
    save_turns(path, turns, model=options.model, cache_key=cache_key)
    print(f"  {len(speaker_totals(turns))} speakers: {describe_speakers(turns)}")
    print(f"  Wrote the speaker turns to: {path}")
    return turns
