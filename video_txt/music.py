"""Lifting a film's music out of its soundtrack.

The score is mixed under the dialogue, so there is no track to copy it from:
what exists is one soundtrack with everybody talking over it. Separation takes
the dialogue out -- that part is solved, and `separate` already does it -- but it
leaves a two-hour instrumental with long silences in it, not a set of pieces
anybody would play.

Finding the pieces is the work here, and it is done from the audio alone. A
subtitle is a luxury this cannot count on: the film may be something seen online
with nothing beside it. ffmpeg's ebur128 filter reports momentary loudness ten
times a second -- a 109-minute film measures in under two seconds -- and where
the instrumental stays above the floor for long enough, music is playing.

What this deliberately does not do is decide whether somebody is singing. A
separator hears one human voice and puts sung and spoken alike in the same stem,
and measuring how much of a stretch has a voice over it does not separate them
either: on a real film the first thing that test called a song was a
seventy-second conversation. So no piece is labelled a song from audio. What is
measured is how much of a piece has any voice over it at all, which is reported
for every piece and is worth knowing for a different reason -- separation leaves
its worst artefacts where the dialogue was loudest, so the pieces nobody talks
over are the ones that come out clean enough to use somewhere else.

A song can still be cut out, by naming its range or by pointing at a subtitle
whose sung lines are marked. That is knowledge from outside the audio, which is
the only place it can come from.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .media import MediaError
from .subtitles import SubtitleCue

# ebur128 reports momentary loudness every 100 ms.
MEASURE_INTERVAL = 0.1
ENVELOPE_FILTER = "ebur128=metadata=1,ametadata=print:key=lavfi.r128.M:file=-"
LOUDNESS_PATTERN = re.compile(r"lavfi\.r128\.M=(-?\d+(?:\.\d+)?|-inf|nan)")
SILENT_LUFS = -120.0

# Where an instrument is not playing Demucs leaves the stem genuinely quiet, so
# the floor only has to clear the residue it does leave behind. Measured on a
# feature film: the instrumental sits at -60 LUFS through its silences and above
# -35 through its music.
MUSIC_FLOOR_LUFS = -45.0
# Above this, somebody is audible over the music.
VOICE_FLOOR_LUFS = -45.0
# A pause this short is a bar of rest, not the end of the piece.
BRIDGE_SECONDS = 2.5
# Under --clean a gap is somebody talking, not a rest, so almost nothing bridges
# it: a piece that promises nobody talks over it has to keep that promise.
CLEAN_BRIDGE_SECONDS = 0.5
# Shorter than this is a sting or a scene transition, not something to play.
MIN_PIECE_SECONDS = 20.0
# What counts as "nobody talks over this" for --clean.
CLEAN_VOICE_SHARE = 0.10

EDGE_FADE = 0.05
LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
# A sung line is marked this way in subtitles written by people. Whisper does not
# write it, which is why this is an optional hint and never a requirement.
LYRIC_MARK = "♪"


class MusicError(RuntimeError):
    pass


@dataclass(frozen=True)
class MusicPiece:
    """One stretch of the film where music plays, and how busy it is."""

    start: float
    end: float
    voice_share: float
    peak_lufs: float
    # Set only from outside the audio: a named range, or a subtitle's sung lines.
    labelled: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def clock_range(self) -> str:
        return f'{clock(self.start)}-{clock(self.end)}'

    @property
    def kind(self) -> str:
        if self.labelled:
            return self.labelled
        return "clean" if self.voice_share <= CLEAN_VOICE_SHARE else "under-dialogue"

    def to_dict(self) -> dict[str, object]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "clock": self.clock_range,
            "kind": self.kind,
            "voice_share": round(self.voice_share, 3),
            "peak_lufs": round(self.peak_lufs, 1),
        }


def clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}h{total // 60 % 60:02d}m{total % 60:02d}s"


def loudness_envelope(audio: Path, *, ffmpeg_path: str) -> list[float]:
    """Momentary loudness ten times a second, as ffmpeg measures it."""
    completed = subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio),
            "-af",
            ENVELOPE_FILTER,
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()
        raise MediaError(
            f"Could not measure the loudness of {audio.name}: "
            f"{tail[-1] if tail else 'unknown error'}"
        )

    levels = [
        SILENT_LUFS if match.group(1) in {"-inf", "nan"} else float(match.group(1))
        for line in completed.stdout.splitlines()
        if (match := LOUDNESS_PATTERN.search(line))
    ]
    if not levels:
        raise MediaError(f"ffmpeg measured no loudness at all in {audio.name}.")
    return levels


def runs_of(flags: list[bool], *, bridge: int) -> list[tuple[int, int]]:
    """Index ranges where the flag holds, joining gaps no longer than `bridge`."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for index, on in enumerate(flags):
        if on:
            if start is None:
                start = index
            gap = 0
            continue
        if start is None:
            continue
        gap += 1
        if gap > bridge:
            runs.append((start, index - gap + 1))
            start = None
            gap = 0
    if start is not None:
        runs.append((start, len(flags) - gap))
    return runs


def find_music(
    instrumental: list[float],
    voice: list[float],
    *,
    music_floor: float = MUSIC_FLOOR_LUFS,
    voice_floor: float = VOICE_FLOOR_LUFS,
    bridge: float = BRIDGE_SECONDS,
    min_duration: float = MIN_PIECE_SECONDS,
    clean_only: bool = False,
) -> list[MusicPiece]:
    """Every stretch long enough to be a piece of music.

    `clean_only` keeps just the stretches nobody talks over, which are the ones
    separation renders without artefacts and the only ones worth dropping into
    somebody else's video.
    """
    if len(instrumental) != len(voice):
        shortest = min(len(instrumental), len(voice))
        instrumental, voice = instrumental[:shortest], voice[:shortest]
    speaking = [level > voice_floor for level in voice]
    playing = [
        level > music_floor and not (clean_only and talking)
        for level, talking in zip(instrumental, speaking, strict=True)
    ]
    if clean_only:
        bridge = min(bridge, CLEAN_BRIDGE_SECONDS)

    pieces: list[MusicPiece] = []
    for first, last in runs_of(playing, bridge=max(0, round(bridge / MEASURE_INTERVAL))):
        start, end = first * MEASURE_INTERVAL, last * MEASURE_INTERVAL
        if end - start < min_duration:
            continue
        window = speaking[first:last]
        pieces.append(
            MusicPiece(
                start=start,
                end=end,
                voice_share=(sum(window) / len(window)) if window else 0.0,
                peak_lufs=max(instrumental[first:last], default=SILENT_LUFS),
            )
        )
    return pieces


def sung_spans(cues: list[SubtitleCue], *, bridge: float = BRIDGE_SECONDS) -> list[MusicPiece]:
    """Songs, from the lines a person marked as sung.

    Only subtitles written by people carry these marks; Whisper writes none, so
    an empty result means the subtitle is silent on the question, never that the
    film has no songs in it.
    """
    marked = [cue for cue in cues if LYRIC_MARK in cue.text]
    if not marked:
        return []

    spans: list[MusicPiece] = []
    start, end = marked[0].start_seconds, marked[0].end_seconds
    for cue in marked[1:]:
        if cue.start_seconds - end > bridge:
            spans.append(MusicPiece(start=start, end=end, voice_share=1.0,
                                    peak_lufs=SILENT_LUFS, labelled="song"))
            start = cue.start_seconds
        end = max(end, cue.end_seconds)
    spans.append(MusicPiece(start=start, end=end, voice_share=1.0,
                            peak_lufs=SILENT_LUFS, labelled="song"))
    return spans


def piece_filename(media: Path, number: int, piece: MusicPiece) -> str:
    return f"{media.stem}.music-{number:02d}.{piece.kind}.{clock(piece.start)}.flac"


def extract_command(
    source: Path,
    piece: MusicPiece,
    *,
    output: Path,
    ffmpeg_path: str,
    normalize: bool = True,
) -> list[str]:
    """Cut one piece out, faded at both ends so it does not open with a click."""
    filters = [
        f"afade=t=in:st=0:d={EDGE_FADE}",
        f"afade=t=out:st={max(0.0, piece.duration - EDGE_FADE):.3f}:d={EDGE_FADE}",
    ]
    if normalize:
        filters.append(LOUDNORM_FILTER)
    return [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{piece.start:.3f}",
        "-t",
        f"{piece.duration:.3f}",
        "-i",
        str(source),
        "-af",
        ",".join(filters),
        "-c:a",
        "flac",
        str(output),
    ]
