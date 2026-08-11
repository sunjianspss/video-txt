from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

FFMPEG_FULL_PATH = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*[x×]\s*(\d+)\s*$", re.IGNORECASE)


class MediaError(RuntimeError):
    pass


class ProbeError(MediaError):
    pass


def parse_video_size(value: str) -> tuple[int, int]:
    match = SIZE_PATTERN.match(value)
    if not match:
        raise MediaError(f"Invalid video size {value!r}. Expected a value like 1920x1080.")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise MediaError(f"Invalid video size {value!r}. Width and height must be positive.")
    return width, height


def ffmpeg_supports_filter(ffmpeg_path: str, filter_name: str) -> bool:
    completed = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False
    return bool(re.search(rf"\b{re.escape(filter_name)}\b", completed.stdout))


def find_ffmpeg(*, explicit: str | None = None, need_subtitles_filter: bool = False) -> str:
    requested = explicit or os.environ.get("FFMPEG_PATH")
    if requested:
        resolved = shutil.which(requested) or (
            str(Path(requested).expanduser()) if Path(requested).expanduser().is_file() else None
        )
        if not resolved:
            raise MediaError(f"ffmpeg not found at: {requested}")
        if need_subtitles_filter and not ffmpeg_supports_filter(resolved, "subtitles"):
            raise MediaError(
                "This ffmpeg build has no 'subtitles' filter, "
                f"so it cannot burn hard subtitles: {resolved}"
            )
        return resolved

    candidates: list[str] = []
    if need_subtitles_filter and FFMPEG_FULL_PATH.is_file():
        candidates.append(str(FFMPEG_FULL_PATH))
    default_path = shutil.which("ffmpeg")
    if default_path:
        candidates.append(default_path)

    if not candidates:
        raise MediaError("Missing dependency: ffmpeg. Install it with: brew install ffmpeg")

    if not need_subtitles_filter:
        return candidates[0]

    for candidate in dict.fromkeys(candidates):
        if ffmpeg_supports_filter(candidate, "subtitles"):
            return candidate

    raise MediaError(
        "Hard subtitle mode requires an ffmpeg build with the 'subtitles' filter.\n"
        "Install a full build and point at it, for example:\n"
        "  brew install ffmpeg-full\n"
        "  export FFMPEG_PATH=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
    )


def find_ffprobe(ffmpeg_path: str | None = None, *, explicit: str | None = None) -> str:
    requested = explicit or os.environ.get("FFPROBE_PATH")
    if requested:
        resolved = shutil.which(requested) or (
            str(Path(requested).expanduser()) if Path(requested).expanduser().is_file() else None
        )
        if not resolved:
            raise MediaError(f"ffprobe not found at: {requested}")
        return resolved

    if ffmpeg_path:
        sibling = Path(ffmpeg_path).with_name("ffprobe")
        if sibling.is_file():
            return str(sibling)

    found = shutil.which("ffprobe")
    if found:
        return found
    raise ProbeError(
        "Missing dependency: ffprobe (ships with ffmpeg). Install it with: brew install ffmpeg"
    )


def run_ffprobe(ffprobe_path: str, args: list[str]) -> str:
    completed = subprocess.run(
        [ffprobe_path, "-v", "error", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise ProbeError(f"ffprobe failed: {detail}")
    return completed.stdout.strip()


def probe_video_size(video_path: Path, *, ffmpeg_path: str | None = None) -> tuple[int, int]:
    ffprobe_path = find_ffprobe(ffmpeg_path)
    output = run_ffprobe(
        ffprobe_path,
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(video_path),
        ],
    )
    match = re.search(r"(\d+)x(\d+)", output)
    if not match:
        raise ProbeError(
            f"Could not read the video resolution of {video_path.name} "
            f"(ffprobe returned {output!r}). Pass --video-size WIDTHxHEIGHT to continue."
        )
    return int(match.group(1)), int(match.group(2))


def probe_duration(
    media_path: Path, *, ffmpeg_path: str | None = None, ffprobe_path: str | None = None
) -> float:
    resolved = ffprobe_path or find_ffprobe(ffmpeg_path)
    output = run_ffprobe(
        resolved,
        [
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
    )
    try:
        return float(output.splitlines()[0])
    except (ValueError, IndexError) as exc:
        raise ProbeError(f"Could not read the duration of {media_path.name}: {output!r}") from exc


def quote_filter_value(value: str) -> str:
    # FFmpeg's filter parser needs an extra escaping layer for literal single quotes.
    escaped = value.replace("\\", "\\\\").replace("'", "'\\\\\\''")
    return f"'{escaped}'"


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def run_command(command: list[str], *, label: str | None = None) -> int:
    if label:
        print(label)
    completed = subprocess.run(command)
    return completed.returncode
