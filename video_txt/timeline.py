"""Putting spoken clips back on the subtitle's clock.

The dub has to land where the original speaker did. This module works out how
much room each line really has, groups the lines back into the sentences they
were cut out of, and streams the clips into one continuous track at the times
the subtitle asks for.

Nothing here knows about speech engines or voices: it takes clips that already
exist and decides when each one plays and how fast.
"""

from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from .fit import speech_units
from .media import MediaError
from .subtitles import SubtitleCue, join_text_parts, seconds_to_srt_time

SAMPLE_WIDTH = 2
SILENCE_FLOOR = "-45dB"
SILENCE_CHUNK_FRAMES = 48_000

VOICE_UNITS = ("sentence", "line")
SENTENCE_ENDINGS = tuple("。！？!?…")
# A merged clip still has to be re-timed as one piece, so keep sentences short
# enough that a single overrun cannot drag a long stretch of the track with it.
MAX_UNIT_CHARS = 120
# A pause this long is a real break in delivery; splitting there costs no prosody.
MAX_UNIT_GAP = 1.5


@dataclass
class Segment:
    cue: SubtitleCue
    audio_path: Path
    start: float
    slot: float


@dataclass
class PlacedSegment:
    start: float
    duration: float
    tempo: float
    overflow: float


def compute_slots(cues: list[tuple[int, SubtitleCue]]) -> list[float]:
    """Seconds each line may occupy: its own span plus the silence before the next one."""
    slots: list[float] = []
    for index, (_, cue) in enumerate(cues):
        start, end = cue.time_range
        if index + 1 < len(cues):
            next_start = cues[index + 1][1].time_range[0]
            slot = max(end - start, next_start - start - 0.05)
        else:
            slot = max(end - start, 0.0) + 2.0
        slots.append(max(slot, 0.2))
    return slots


def merge_cues(group: list[tuple[int, SubtitleCue]]) -> tuple[int, SubtitleCue]:
    position, first = group[0]
    if len(group) == 1:
        return position, first

    parts = [cue.text.replace("\n", " ").strip() for _, cue in group]
    start = first.time_range[0]
    end = group[-1][1].time_range[1]
    return position, SubtitleCue(
        index=first.index,
        timing=f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}",
        text_lines=[join_text_parts(parts)],
    )


def split_into_sentences(
    cues: list[tuple[int, SubtitleCue]],
    *,
    speakers: dict[int, str] | None = None,
    max_chars: int = MAX_UNIT_CHARS,
    max_gap: float = MAX_UNIT_GAP,
) -> list[list[tuple[int, SubtitleCue]]]:
    """Gather consecutive cues into the sentences they were split out of."""
    sentences: list[list[tuple[int, SubtitleCue]]] = []
    group: list[tuple[int, SubtitleCue]] = []
    voices = speakers or {}

    for index, entry in enumerate(cues):
        group.append(entry)
        position, cue = entry
        gap_after = float("inf")
        handover = False
        if index + 1 < len(cues):
            next_position, next_cue = cues[index + 1]
            gap_after = next_cue.time_range[0] - cue.time_range[1]
            # Two people never share a clip: one voice has to speak the whole of it.
            handover = voices.get(position) != voices.get(next_position)
        pending = sum(len(item.text) for _, item in group)
        if (
            handover
            or cue.text.rstrip().endswith(SENTENCE_ENDINGS)
            or pending >= max_chars
            or gap_after > max_gap
        ):
            sentences.append(group)
            group = []

    if group:
        sentences.append(group)
    return sentences


def group_cues_into_sentences(
    cues: list[tuple[int, SubtitleCue]],
    *,
    speakers: dict[int, str] | None = None,
    **limits: float,
) -> list[tuple[int, SubtitleCue]]:
    """Merge consecutive cues that belong to one spoken sentence.

    Synthesizing one clip per subtitle line makes every line land on a
    sentence-final fall, and each clip pays a fixed lead-in cost. Over a long talk
    that is both a chopped-up delivery and minutes of speaking time the original
    speaker never used, which then has to be clawed back by speeding the voice up.
    Speaking whole sentences avoids paying either price.
    """
    return [merge_cues(group) for group in split_into_sentences(cues, speakers=speakers, **limits)]


def share_out(group: list[tuple[int, SubtitleCue]], total: float) -> dict[int, float]:
    """Split one clip's seconds between the lines it was merged from.

    The slot a clip has to fit into and the time it takes to say are divided the
    same way, so a line is always judged against the share of the sentence it is.
    """
    weights = [speech_units(cue.text) for _, cue in group]
    spoken = sum(weights) or 1.0
    return {
        position: total * weight / spoken
        for (position, _), weight in zip(group, weights, strict=True)
    }


def speaking_budgets(
    cues: list[tuple[int, SubtitleCue]],
    *,
    voice_unit: str,
    speakers: dict[int, str] | None = None,
) -> list[float]:
    """Seconds each line can claim once the clips are spoken.

    A line spoken inside a sentence is not held to its own on-screen span: it
    takes a share of the sentence's time proportional to how much of the sentence
    it is. Held to the span alone, a line that merely reads fast on screen looks
    like it needs rewriting when the sentence around it has room to spare.
    """
    if voice_unit != "sentence":
        return compute_slots(cues)

    sentences = split_into_sentences(cues, speakers=speakers)
    slots = compute_slots([merge_cues(group) for group in sentences])
    budgets: dict[int, float] = {}
    for group, slot in zip(sentences, slots, strict=True):
        budgets.update(share_out(group, slot))
    return [budgets[position] for position, _ in cues]


def build_segments(cues: list[tuple[int, SubtitleCue]], paths: list[Path]) -> list[Segment]:
    return [
        Segment(cue=cue, audio_path=path, start=cue.time_range[0], slot=slot)
        for (_, cue), path, slot in zip(cues, paths, compute_slots(cues), strict=True)
    ]


def run_audio_filter(command: list[str], *, source: bytes | None, label: str) -> bytes:
    completed = subprocess.run(command, input=source, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise MediaError(f"Could not decode {label}: {detail}")
    return completed.stdout


def decode_spoken_audio(audio_path: Path, *, ffmpeg_path: str, sample_rate: int) -> bytes:
    """Decode the speech renderer will place, without an engine's leading pad."""
    return run_audio_filter(
        [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(audio_path),
            # Engines pad the front of every clip. That padding makes the voice
            # come in late and spends slot the line needs for actual speech.
            "-af",
            f"silenceremove=start_periods=1:start_duration=0:start_threshold={SILENCE_FLOOR}",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ],
        source=None,
        label=audio_path.name,
    )


def spoken_duration(audio_path: Path, *, ffmpeg_path: str, sample_rate: int) -> float:
    """Measure the same trimmed PCM that rendering will put on the timeline."""
    audio = decode_spoken_audio(audio_path, ffmpeg_path=ffmpeg_path, sample_rate=sample_rate)
    return len(audio) / (sample_rate * SAMPLE_WIDTH)


def decode_segment(
    segment: Segment, *, ffmpeg_path: str, sample_rate: int, max_atempo: float
) -> tuple[bytes, float]:
    """Decode one clip to raw mono PCM, sped up only as far as its slot demands."""
    audio = decode_spoken_audio(
        segment.audio_path, ffmpeg_path=ffmpeg_path, sample_rate=sample_rate
    )

    natural = len(audio) / (sample_rate * SAMPLE_WIDTH)
    if segment.slot <= 0 or natural <= segment.slot:
        return audio, 1.0

    tempo = min(natural / segment.slot, max(1.0, max_atempo))
    if tempo <= 1.001:
        return audio, 1.0

    stretched = run_audio_filter(
        [
            ffmpeg_path,
            "-v",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "-",
            "-filter:a",
            f"atempo={tempo:.4f}",
            "-f",
            "s16le",
            "-",
        ],
        source=audio,
        label=segment.audio_path.name,
    )
    return stretched, tempo


def write_silence(handle: wave.Wave_write, frames: int) -> None:
    while frames > 0:
        chunk = min(frames, SILENCE_CHUNK_FRAMES)
        handle.writeframes(bytes(chunk * SAMPLE_WIDTH))
        frames -= chunk


def render_audio_track(
    segments: list[Segment],
    *,
    ffmpeg_path: str,
    sample_rate: int,
    max_atempo: float,
    total_duration: float,
    output_path: Path,
) -> list[PlacedSegment]:
    # Segments are sorted and clips never overlap (later ones get pushed back),
    # so the track can be streamed to disk instead of assembled in memory.
    placed: list[PlacedSegment] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)

        written_frames = 0
        for segment in segments:
            pcm, tempo = decode_segment(
                segment,
                ffmpeg_path=ffmpeg_path,
                sample_rate=sample_rate,
                max_atempo=max_atempo,
            )
            frames = len(pcm) // SAMPLE_WIDTH
            duration = frames / sample_rate
            start_seconds = max(segment.start, written_frames / sample_rate)
            start_frame = max(int(start_seconds * sample_rate), written_frames)
            write_silence(handle, start_frame - written_frames)
            handle.writeframes(pcm)
            written_frames = start_frame + frames
            placed.append(
                PlacedSegment(
                    start=start_frame / sample_rate,
                    duration=duration,
                    tempo=tempo,
                    overflow=max(0.0, duration - segment.slot),
                )
            )

        total_frames = int(total_duration * sample_rate) + sample_rate
        write_silence(handle, total_frames - written_frames)
    return placed


def report(placed: list[PlacedSegment], segments: list[Segment]) -> None:
    stretched = sum(1 for item in placed if item.tempo > 1.001)
    overflowing = [item for item in placed if item.overflow > 0.05]
    drift = max(
        (item.start - segment.start for item, segment in zip(placed, segments, strict=True)),
        default=0.0,
    )

    print(f"Voice clips: {len(placed)}")
    print(f"Speed-adjusted to fit: {stretched}")
    if overflowing:
        worst = max(item.overflow for item in overflowing)
        print(
            f"Still longer than their subtitle slot: {len(overflowing)} "
            f"(worst overshoot {worst:.1f}s)"
        )
    print(f"Largest timeline drift: {drift:.1f}s")
    if drift > 2.0:
        print(
            "Tip: add --rate +10%. The voice is then spoken faster, which sounds "
            "better than stretching the clip afterwards to catch up."
        )
