from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .media import (
    find_ffmpeg,
    find_ffprobe,
    format_command,
    probe_duration,
    run_ffprobe,
)
from .mux import FASTSTART_CONTAINERS, resolve_subtitle_codec
from .subtitles import SubtitleCue, parse_srt

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_SAY_VOICE = "Tingting"
ENGINES = ("edge-tts", "say")
SAMPLE_WIDTH = 2


class DubError(RuntimeError):
    pass


@dataclass
class DubOptions:
    video_input: Path
    subtitle_input: Path
    video_output: Path
    audio_output: Path | None = None
    cache_dir: Path | None = None
    engine: str = "edge-tts"
    voice: str = DEFAULT_VOICE
    rate: str = "+0%"
    max_atempo: float = 1.35
    keep_bgm: bool = False
    bgm_volume: float = 0.15
    soft_subtitle: bool = False
    concurrency: int = 4
    sample_rate: int = 48000
    overwrite: bool = False
    ffmpeg_path: str | None = None


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


def default_audio_output(video_output: Path) -> Path:
    return video_output.with_name(f"{video_output.stem}.dub.wav")


def default_video_output(video: Path) -> Path:
    suffix = video.suffix.lower() if video.suffix.lower() in {".mp4", ".mov", ".mkv"} else ".mp4"
    return video.with_name(f"{video.stem}.zh-dubbed{suffix}")


def default_cache_dir(video: Path) -> Path:
    return video.with_name(f"{video.stem}.dub-cache")


def tts_launcher(engine: str) -> list[str]:
    if engine == "say":
        say = shutil.which("say")
        if not say:
            raise DubError("The 'say' engine needs macOS. Use --tts-engine edge-tts instead.")
        return [say]

    console_script = shutil.which("edge-tts")
    if console_script:
        return [console_script]
    try:
        __import__("edge_tts")
    except ImportError as exc:
        raise DubError(
            "Missing dependency: edge-tts.\n"
            "Install it with: uv add --optional dub edge-tts\n"
            "Or use the offline fallback: --tts-engine say"
        ) from exc
    return [sys.executable, "-m", "edge_tts"]


def build_tts_command(
    options: DubOptions, *, launcher: list[str], text: str, output_path: Path
) -> list[str]:
    if options.engine == "say":
        voice = options.voice if options.voice != DEFAULT_VOICE else DEFAULT_SAY_VOICE
        return [*launcher, "-v", voice, "-o", str(output_path), text]
    return [
        *launcher,
        "--voice",
        options.voice,
        f"--rate={options.rate}",
        "--text",
        text,
        "--write-media",
        str(output_path),
    ]


def segment_filename(engine: str, position: int, text: str, voice: str, rate: str) -> str:
    digest = hashlib.sha1(f"{engine}|{voice}|{rate}|{text}".encode()).hexdigest()[:12]
    extension = "aiff" if engine == "say" else "mp3"
    return f"cue-{position:05d}-{digest}.{extension}"


def synthesize_segments(
    cues: list[tuple[int, SubtitleCue]], options: DubOptions, cache_dir: Path
) -> list[Path]:
    launcher = tts_launcher(options.engine)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        cache_dir
        / segment_filename(options.engine, position, cue.text, options.voice, options.rate)
        for position, cue in cues
    ]

    pending = [
        (cue.text, path)
        for (_, cue), path in zip(cues, paths, strict=True)
        if not path.is_file() or path.stat().st_size == 0
    ]
    cached = len(paths) - len(pending)
    if cached:
        print(f"Reusing {cached} cached voice clip{'s' if cached > 1 else ''}.")
    if not pending:
        return paths

    print(f"Synthesizing {len(pending)} voice clips with {options.engine} ({options.voice})...")
    done = 0
    progress_lock = threading.Lock()

    def synthesize(job: tuple[str, Path]) -> None:
        nonlocal done
        text, path = job
        temp_path = path.with_suffix(path.suffix + ".part")
        command = build_tts_command(options, launcher=launcher, text=text, output_path=temp_path)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not temp_path.is_file():
            temp_path.unlink(missing_ok=True)
            detail = (completed.stderr or completed.stdout or "").strip()
            raise DubError(f"Text-to-speech failed for {text[:40]!r}: {detail}")
        temp_path.replace(path)
        with progress_lock:
            done += 1
            current = done
        if current % 10 == 0 or current == len(pending):
            print(f"  {current}/{len(pending)} clips done")

    workers = max(1, min(options.concurrency, len(pending)))
    if workers == 1:
        for job in pending:
            synthesize(job)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(synthesize, job) for job in pending]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                for queued in futures:
                    queued.cancel()
                raise
    return paths


def decode_segment(
    segment: Segment, *, ffmpeg_path: str, ffprobe_path: str, options: DubOptions
) -> tuple[bytes, float]:
    natural = probe_duration(segment.audio_path, ffprobe_path=ffprobe_path)
    tempo = 1.0
    if segment.slot > 0 and natural > segment.slot:
        tempo = min(natural / segment.slot, max(1.0, options.max_atempo))

    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-i",
        str(segment.audio_path),
        "-ar",
        str(options.sample_rate),
        "-ac",
        "1",
    ]
    if tempo > 1.001:
        command.extend(["-filter:a", f"atempo={tempo:.4f}"])
    command.extend(["-f", "s16le", "-"])

    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise DubError(f"Could not decode {segment.audio_path.name}: {detail}")
    return completed.stdout, tempo


def build_segments(cues: list[tuple[int, SubtitleCue]], paths: list[Path]) -> list[Segment]:
    segments: list[Segment] = []
    for index, ((_, cue), path) in enumerate(zip(cues, paths, strict=True)):
        start, end = cue.time_range
        if index + 1 < len(cues):
            next_start = cues[index + 1][1].time_range[0]
            slot = max(end - start, next_start - start - 0.05)
        else:
            slot = max(end - start, 0.0) + 2.0
        segments.append(Segment(cue=cue, audio_path=path, start=start, slot=max(slot, 0.2)))
    return segments


SILENCE_CHUNK_FRAMES = 48_000


def write_silence(handle: wave.Wave_write, frames: int) -> None:
    while frames > 0:
        chunk = min(frames, SILENCE_CHUNK_FRAMES)
        handle.writeframes(bytes(chunk * SAMPLE_WIDTH))
        frames -= chunk


def render_audio_track(
    segments: list[Segment],
    options: DubOptions,
    *,
    ffmpeg_path: str,
    total_duration: float,
    output_path: Path,
) -> list[PlacedSegment]:
    # Segments are sorted and clips never overlap (later ones get pushed back),
    # so the track can be streamed to disk instead of assembled in memory.
    ffprobe_path = find_ffprobe(ffmpeg_path)
    placed: list[PlacedSegment] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(options.sample_rate)

        written_frames = 0
        for segment in segments:
            pcm, tempo = decode_segment(
                segment, ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path, options=options
            )
            frames = len(pcm) // SAMPLE_WIDTH
            duration = frames / options.sample_rate
            start_seconds = max(segment.start, written_frames / options.sample_rate)
            start_frame = max(int(start_seconds * options.sample_rate), written_frames)
            write_silence(handle, start_frame - written_frames)
            handle.writeframes(pcm)
            written_frames = start_frame + frames
            placed.append(
                PlacedSegment(
                    start=start_frame / options.sample_rate,
                    duration=duration,
                    tempo=tempo,
                    overflow=max(0.0, duration - segment.slot),
                )
            )

        total_frames = int(total_duration * options.sample_rate) + options.sample_rate
        write_silence(handle, total_frames - written_frames)
    return placed


def has_audio_stream(video: Path, *, ffprobe_path: str) -> bool:
    output = run_ffprobe(
        ffprobe_path,
        ["-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
    )
    return bool(output.strip())


def build_dub_mux_command(
    options: DubOptions, *, ffmpeg_path: str, audio_path: Path, keep_bgm: bool
) -> list[str]:
    command = [
        ffmpeg_path,
        "-y" if options.overwrite else "-n",
        "-i",
        str(options.video_input),
        "-i",
        str(audio_path),
    ]
    if options.soft_subtitle:
        command.extend(["-i", str(options.subtitle_input)])

    if keep_bgm:
        volume = min(max(options.bgm_volume, 0.0), 1.0)
        command.extend(
            [
                "-filter_complex",
                f"[0:a]volume={volume:.3f}[bg];[1:a]volume=1.0[vo];"
                "[bg][vo]amix=inputs=2:duration=first:normalize=0[aout]",
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
            ]
        )
    else:
        command.extend(["-map", "0:v:0", "-map", "1:a:0"])

    if options.soft_subtitle:
        codec = resolve_subtitle_codec(options.video_output)
        command.extend(["-map", "2:0", "-c:s", codec, "-metadata:s:s:0", "language=zho"])

    command.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"])
    if options.video_output.suffix.lower() in FASTSTART_CONTAINERS:
        command.extend(["-movflags", "+faststart"])
    command.append(str(options.video_output))
    return command


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
            "Tip: raise --max-atempo, or shorten the Chinese lines "
            "(about 4 characters per second reads naturally)."
        )


def run_dub(options: DubOptions, *, dry_run: bool = False) -> Path:
    ffmpeg_path = find_ffmpeg(explicit=options.ffmpeg_path)
    audio_output = options.audio_output or default_audio_output(options.video_output)
    cache_dir = options.cache_dir or default_cache_dir(options.video_input)

    if dry_run:
        print(f"Voice clip cache: {cache_dir}")
        print(f"Dub audio track: {audio_output}")
        print("Mux command:")
        print(
            format_command(
                build_dub_mux_command(
                    options,
                    ffmpeg_path=ffmpeg_path,
                    audio_path=audio_output,
                    keep_bgm=options.keep_bgm,
                )
            )
        )
        return options.video_output

    if options.video_output.exists() and not options.overwrite:
        raise DubError(
            f"Output video already exists: {options.video_output}. "
            "Pass --overwrite-video to replace it."
        )

    cues = [
        (position, cue)
        for position, cue in enumerate(parse_srt(options.subtitle_input), start=1)
        if not cue.is_empty
    ]
    if not cues:
        raise DubError(f"No spoken lines found in {options.subtitle_input}")

    paths = synthesize_segments(cues, options, cache_dir)
    segments = build_segments(cues, paths)
    total_duration = probe_duration(options.video_input, ffmpeg_path=ffmpeg_path)

    print("Aligning voice clips to the subtitle timeline...")
    placed = render_audio_track(
        segments,
        options,
        ffmpeg_path=ffmpeg_path,
        total_duration=total_duration,
        output_path=audio_output,
    )
    report(placed, segments)

    keep_bgm = options.keep_bgm
    if keep_bgm and not has_audio_stream(
        options.video_input, ffprobe_path=find_ffprobe(ffmpeg_path)
    ):
        print("Source video has no audio track, so --keep-bgm has nothing to mix.", file=sys.stderr)
        keep_bgm = False

    print("Muxing the Chinese voice track into the video...")
    command = build_dub_mux_command(
        options, ffmpeg_path=ffmpeg_path, audio_path=audio_output, keep_bgm=keep_bgm
    )
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise DubError(f"ffmpeg failed with exit code {completed.returncode}.")
    return options.video_output
