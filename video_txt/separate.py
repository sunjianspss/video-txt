"""Splitting the original soundtrack into voice and everything else.

--keep-bgm keeps the music by keeping the whole original mix, speech included:
turned down far enough not to fight the dub, it takes the music and the room
tone down with it. Separating the track first removes only the original voices,
so everything else can stay at the volume the producer chose.

Demucs does the split. It costs minutes of CPU and a one-time model download,
so the finished background lives in the clip cache and reruns pick it up.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

SEPARATE_MODEL = "htdemucs"
BGM_DIR_NAME = "bgm"
STEM_FILENAME = "no_vocals.flac"


class SeparateError(RuntimeError):
    pass


def separated_bgm_path(cache_dir: Path) -> Path:
    return cache_dir / BGM_DIR_NAME / STEM_FILENAME


def require_demucs() -> None:
    if find_spec("demucs") is None:
        raise SeparateError(
            "Missing dependency: demucs, which separates voice from background.\n"
            "Install it with: uv sync --extra separate\n"
            "Or drop --separate-bgm and use --keep-bgm to mix the whole original in quietly."
        )


def extract_audio_command(video: Path, *, target: Path, ffmpeg_path: str) -> list[str]:
    """Demucs reads plain audio; feeding it the video trips over exotic containers."""
    return [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        str(target),
    ]


def separate_command(audio: Path, *, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "-n",
        SEPARATE_MODEL,
        "-o",
        str(output_dir),
        str(audio),
    ]


def compress_command(stem: Path, *, target: Path, ffmpeg_path: str) -> list[str]:
    """The stem as FLAC: half the size of the WAV Demucs writes, still lossless."""
    return [ffmpeg_path, "-y", "-v", "error", "-i", str(stem), "-c:a", "flac", str(target)]


def ensure_instrumental(video: Path, *, cache_dir: Path, ffmpeg_path: str) -> Path:
    """The original audio with the voices removed, made once and cached.

    Like the reference clips, the finished track is not swept by --prune-cache:
    it is expensive to remake and every rerun of the same video wants it.
    """
    target = separated_bgm_path(cache_dir)
    if target.is_file() and target.stat().st_size > 0:
        print(f"Reusing the separated background track: {target.name}")
        return target
    require_demucs()

    work_dir = target.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / "source.wav"
    completed = subprocess.run(
        extract_audio_command(video, target=source, ffmpeg_path=ffmpeg_path),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not source.is_file():
        detail = (completed.stderr or "").strip()
        raise SeparateError(f"Could not extract the audio track for separation: {detail}")

    print("Separating voice from background (one-time per video, takes a few minutes)...")
    returncode = subprocess.run(separate_command(source, output_dir=work_dir)).returncode
    stem = work_dir / SEPARATE_MODEL / source.stem / "no_vocals.wav"
    if returncode != 0 or not stem.is_file():
        raise SeparateError(
            f"Demucs did not produce the background track (exit code {returncode}). "
            "Its own report is printed above."
        )

    completed = subprocess.run(
        compress_command(stem, target=target, ffmpeg_path=ffmpeg_path),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not target.is_file():
        detail = (completed.stderr or "").strip()
        raise SeparateError(f"Could not compress the background track: {detail}")

    # Only the finished FLAC is worth keeping: the WAV stems are hundreds of
    # megabytes, and the vocal stem was never wanted in the first place.
    shutil.rmtree(work_dir / SEPARATE_MODEL, ignore_errors=True)
    source.unlink(missing_ok=True)
    print(f"Background track cached: {target}")
    return target
