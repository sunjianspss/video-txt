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
from collections.abc import Sequence
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
# Sung lines are subtitled sparsely -- an instrumental bar, a breath, a verse
# nobody wrote down -- so the gaps between them are much wider than a gap in the
# audio. Measured on one episode's song: 3 to 4 seconds between every pair of
# lines, which the audio bridge would have cut into six fragments of one song.
SUNG_BRIDGE_SECONDS = 12.0
# Shorter than this is a sting or a scene transition, not something to play.
MIN_PIECE_SECONDS = 20.0
# What counts as "nobody talks over this" for --clean.
CLEAN_VOICE_SHARE = 0.10

# Separation puts everything that is not a voice into one instrumental, and a
# film is full of things that are not voices: footsteps, traffic, room tone, a
# door. Loudness alone cannot tell those from music -- listened to, twelve of
# fourteen pieces found that way on a real episode were background noise.
#
# What music has and a noise does not is more than one pitched instrument
# sounding at the same time. Drums are deliberately not among these: Demucs puts
# transients there, so footsteps and impacts arrive as drums and one background
# stretch measured 77% "drums" while carrying no music at all.
PITCHED_STEMS = ("bass", "guitar", "piano")
SIMULTANEOUS_PITCHED = 2
# Share of a piece that has to sound like music for the piece to be music.
# Measured over two episodes: the two pieces a listener called music scored 45%
# and 99%, the loudest thing that was not music scored 24%.
MIN_MUSICALITY = 0.25

EDGE_FADE = 0.05
LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
# loudnorm works internally at 192 kHz and hands that to the encoder, which turns
# a 44.1 kHz stem into a file four times the size carrying no more information.
# The dub already guards against this; so does every cut made here.
FALLBACK_SAMPLE_RATE = 48000

# FLAC keeps the separated audio exactly as it came out, which matters because
# this is working audio: it goes into somebody's video and gets encoded again,
# and a lossy file re-encoded is a second generation of loss. The lossy formats
# are here for the other reason somebody wants these files -- listening to them.
FORMATS = {
    "flac": ("flac", None, "flac"),
    "m4a": ("aac", "192k", "m4a"),
    "mp3": ("libmp3lame", "192k", "mp3"),
    "alac": ("alac", None, "m4a"),
}
DEFAULT_FORMAT = "flac"
# A sung line is marked this way in subtitles written by people. Whisper does not
# write it, which is why this is an optional hint and never a requirement.
LYRIC_MARK = "♪"
# The same mark on its own means only that music is playing -- a sting, a scene
# change, a radio in the background. Measured across one season: 72 of the 128
# short marked cues carry no words at all. A song is the ones with words in them.
WORD_PATTERN = re.compile(r"\w")
MIN_LYRIC_CHARACTERS = 2


class MusicError(RuntimeError):
    pass


@dataclass(frozen=True)
class MusicPiece:
    """One stretch of the film where music plays, and how busy it is."""

    start: float
    end: float
    voice_share: float
    peak_lufs: float
    # How much of it has two pitched instruments going at once. None when the
    # stems were not separated and the question was not asked.
    musicality: float | None = None
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
            "musicality": round(self.musicality, 3) if self.musicality is not None else None,
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


def musicality(pitched: Sequence[list[float]], first: int, last: int, *, floor: float) -> float:
    """Share of the window where two pitched instruments sound together."""
    if len(pitched) < SIMULTANEOUS_PITCHED or last <= first:
        return 0.0
    span = range(first, min(last, min(len(envelope) for envelope in pitched)))
    if not span:
        return 0.0
    together = sum(
        1
        for index in span
        if sum(envelope[index] > floor for envelope in pitched) >= SIMULTANEOUS_PITCHED
    )
    return together / len(span)


def find_music(
    instrumental: list[float],
    voice: list[float],
    *,
    music_floor: float = MUSIC_FLOOR_LUFS,
    voice_floor: float = VOICE_FLOOR_LUFS,
    bridge: float = BRIDGE_SECONDS,
    min_duration: float = MIN_PIECE_SECONDS,
    clean_only: bool = False,
    exclude: Sequence[tuple[float, float]] = (),
    pitched: Sequence[list[float]] = (),
    min_musicality: float = 0.0,
) -> list[MusicPiece]:
    """Every stretch long enough to be a piece of music.

    `clean_only` keeps just the stretches nobody talks over, which are the ones
    separation renders without artefacts and the only ones worth dropping into
    somebody else's video.

    `exclude` masks out what has already been claimed -- a song located from the
    subtitle sits inside whatever the audio finds around it, and writing both
    puts the same music on disk twice. Masking before the runs are grouped also
    keeps each remaining piece's own measurements honest.

    `pitched` holds the separated pitched stems. With them a piece is kept only
    when `min_musicality` of it has two of them sounding together, which is what
    separates a music cue from a street, a room or a door.
    """
    if len(instrumental) != len(voice):
        shortest = min(len(instrumental), len(voice))
        instrumental, voice = instrumental[:shortest], voice[:shortest]
    speaking = [level > voice_floor for level in voice]
    claimed = [False] * len(instrumental)
    for start, end in exclude:
        for index in range(
            max(0, int(start / MEASURE_INTERVAL)),
            min(len(claimed), round(end / MEASURE_INTERVAL)),
        ):
            claimed[index] = True
    playing = [
        level > music_floor and not taken and not (clean_only and talking)
        for level, talking, taken in zip(instrumental, speaking, claimed, strict=True)
    ]
    if clean_only:
        bridge = min(bridge, CLEAN_BRIDGE_SECONDS)

    pieces: list[MusicPiece] = []
    for first, last in runs_of(playing, bridge=max(0, round(bridge / MEASURE_INTERVAL))):
        start, end = first * MEASURE_INTERVAL, last * MEASURE_INTERVAL
        if end - start < min_duration:
            continue
        window = speaking[first:last]
        score = musicality(pitched, first, last, floor=music_floor) if pitched else None
        if score is not None and score < min_musicality:
            continue
        pieces.append(
            MusicPiece(
                start=start,
                end=end,
                voice_share=(sum(window) / len(window)) if window else 0.0,
                peak_lufs=max(instrumental[first:last], default=SILENT_LUFS),
                musicality=score,
            )
        )
    return pieces


def is_sung(cue: SubtitleCue) -> bool:
    """Whether this line is somebody singing words, not just a note in the margin."""
    if LYRIC_MARK not in cue.text:
        return False
    lyric = cue.text.replace(LYRIC_MARK, " ")
    return len(WORD_PATTERN.findall(lyric)) >= MIN_LYRIC_CHARACTERS


def sung_spans(
    cues: list[SubtitleCue], *, bridge: float = SUNG_BRIDGE_SECONDS
) -> list[MusicPiece]:
    """Songs, from the lines a person marked as sung.

    Only subtitles written by people carry these marks; Whisper writes none, so
    an empty result means the subtitle is silent on the question, never that the
    film has no songs in it.
    """
    marked = [cue for cue in cues if is_sung(cue)]
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


def piece_filename(
    media: Path, number: int, piece: MusicPiece, *, audio_format: str = DEFAULT_FORMAT
) -> str:
    suffix = FORMATS[audio_format][2]
    return f"{media.stem}.music-{number:02d}.{piece.kind}.{clock(piece.start)}.{suffix}"


def extract_command(
    source: Path,
    piece: MusicPiece,
    *,
    output: Path,
    ffmpeg_path: str,
    normalize: bool = True,
    sample_rate: int | None = None,
    audio_format: str = DEFAULT_FORMAT,
) -> list[str]:
    """Cut one piece out, faded at both ends so it does not open with a click."""
    codec, bitrate, _suffix = FORMATS[audio_format]
    filters = [
        f"afade=t=in:st=0:d={EDGE_FADE}",
        f"afade=t=out:st={max(0.0, piece.duration - EDGE_FADE):.3f}:d={EDGE_FADE}",
    ]
    if normalize:
        filters.append(LOUDNORM_FILTER)
    quality = ["-b:a", bitrate] if bitrate else []
    # Written back at the rate it was read at. Without this loudnorm's internal
    # 192 kHz reaches the encoder and quadruples the file for nothing.
    depth = ["-sample_fmt", "s16"] if codec in {"flac", "alac"} else []
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
        # A song is cut straight out of the video file. Without this the picture
        # rides along into any container that will hold it -- flac and mp3 drop
        # it, m4a keeps 720p of it in what is supposed to be an audio file.
        "-vn",
        "-af",
        ",".join(filters),
        "-ar",
        str(sample_rate or FALLBACK_SAMPLE_RATE),
        *depth,
        "-c:a",
        codec,
        *quality,
        str(output),
    ]
