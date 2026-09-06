from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

FFMPEG_FULL_PATH = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*[x×]\s*(\d+)\s*$", re.IGNORECASE)


class MediaError(RuntimeError):
    pass


class ProbeError(MediaError):
    pass


@dataclass(frozen=True)
class AudioStream:
    index: int
    codec: str | None = None
    channels: int | None = None
    layout: str | None = None
    language: str | None = None
    title: str | None = None
    is_default: bool = False
    is_forced: bool = False
    is_hearing_impaired: bool = False
    is_visual_impaired: bool = False
    is_commentary: bool = False


@dataclass(frozen=True)
class AudioStreamSelection:
    stream: AudioStream
    reason: str


LANGUAGE_LABELS: dict[str, frozenset[str]] = {
    "en": frozenset({"en", "eng", "english"}),
    "es": frozenset(
        {"es", "spa", "spanish", "espanol", "español", "castellano", "lat", "latino"}
    ),
    "ja": frozenset({"ja", "jpn", "japanese", "日本語", "日语", "日文"}),
    "zh": frozenset({"zh", "zho", "chi", "chinese", "中文", "简体中文", "繁體中文"}),
    "ko": frozenset({"ko", "kor", "korean", "한국어", "韩语"}),
    "fr": frozenset({"fr", "fra", "fre", "french", "français"}),
    "de": frozenset({"de", "deu", "ger", "german", "deutsch"}),
    "pt": frozenset({"pt", "por", "portuguese", "português"}),
    "it": frozenset({"it", "ita", "italian", "italiano"}),
    "ru": frozenset({"ru", "rus", "russian", "русский"}),
    "ar": frozenset({"ar", "ara", "arabic", "العربية"}),
    "hi": frozenset({"hi", "hin", "hindi", "हिन्दी"}),
    "id": frozenset({"id", "ind", "indonesian", "bahasa"}),
    "vi": frozenset({"vi", "vie", "vietnamese", "tiếng"}),
    "th": frozenset({"th", "tha", "thai", "ไทย"}),
}
NON_DIALOGUE_LABELS = frozenset(
    {"commentary", "director", "description", "descriptive", "audiodescription", "ad"}
)
MIN_AUDIO_STREAM_SCORE_GAP = 25


def normalized_label(value: str | None) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold()).strip() if value else ""


def matched_language(value: str | None) -> str | None:
    words = set(normalized_label(value).split())
    for language, labels in LANGUAGE_LABELS.items():
        if words & labels:
            return language
    return None


def describe_audio_stream(stream: AudioStream) -> str:
    details = [value for value in (stream.title, stream.language, stream.layout) if value]
    suffix = f" ({', '.join(details)})" if details else ""
    return f"stream {stream.index}{suffix}"


def select_audio_stream(
    streams: list[AudioStream],
    *,
    preferred_language: str | None = None,
    requested_index: int | None = None,
) -> AudioStreamSelection:
    """Choose the most likely main-dialogue stream from already-probed metadata."""
    if not streams:
        raise MediaError("The media file has no audio streams.")
    if requested_index is not None:
        selected = next((stream for stream in streams if stream.index == requested_index), None)
        if selected is None:
            available = ", ".join(str(stream.index) for stream in streams) or "none"
            raise MediaError(
                f"Audio stream {requested_index} does not exist. Available indexes: {available}."
            )
        return AudioStreamSelection(
            selected,
            f"selected explicitly with --audio-stream {requested_index}",
        )
    if len(streams) == 1:
        return AudioStreamSelection(streams[0], "only available audio stream")
    preferred = matched_language(preferred_language) or normalized_label(preferred_language)
    if preferred and not any(
        matched_language(stream.title) == preferred
        or matched_language(stream.language) == preferred
        for stream in streams
    ):
        available = "; ".join(describe_audio_stream(stream) for stream in streams)
        raise MediaError(
            f"No audio stream matches {preferred}. Available: {available}. "
            "Pass --audio-stream INDEX to override the metadata."
        )
    scored: list[tuple[int, AudioStream, list[str]]] = []
    for stream in streams:
        score = 0
        reasons: list[str] = []
        title_language = matched_language(stream.title)
        tagged_language = matched_language(stream.language)
        if preferred and title_language == preferred:
            score += 200
            reasons.append(f"title {stream.title!r} matches {preferred}")
        elif preferred and title_language:
            score -= 120
        if preferred and tagged_language == preferred:
            score += 160
            reasons.append(f"language tag {stream.language!r} matches {preferred}")
        elif preferred and tagged_language:
            score -= 60
        if stream.is_default:
            score += 40
            reasons.append("marked default")
        if stream.is_forced:
            score -= 10
        if stream.is_hearing_impaired:
            score -= 20
        title_words = set(normalized_label(stream.title).split())
        if stream.is_commentary or stream.is_visual_impaired or title_words & NON_DIALOGUE_LABELS:
            score -= 240
        scored.append((score, stream, reasons))

    ranked = sorted(scored, key=lambda candidate: candidate[0], reverse=True)
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < MIN_AUDIO_STREAM_SCORE_GAP:
        choices = "; ".join(describe_audio_stream(candidate[1]) for candidate in ranked)
        raise MediaError(
            f"Cannot safely choose an audio stream. Available: {choices}. "
            "Pass --audio-stream INDEX to select one explicitly."
        )
    _score, selected, reasons = ranked[0]
    return AudioStreamSelection(
        selected,
        ", ".join(reasons) or "best main-dialogue metadata match",
    )


def file_identity(path: Path) -> dict[str, object] | None:
    """A cheap, content-sensitive identity for a potentially large media file."""
    resolved = path.expanduser().resolve()
    try:
        stat = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
            if stat.st_size > 1024 * 1024:
                handle.seek(max(0, stat.st_size - 1024 * 1024))
                digest.update(handle.read(1024 * 1024))
    except OSError:
        return None
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sample_sha256": digest.hexdigest(),
    }


def temporary_output_path(path: Path) -> Path:
    """A unique same-directory path with the final container suffix intact."""
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.part{path.suffix}")


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


def explicit_executable(requested: str, *, name: str) -> str:
    found = shutil.which(requested)
    if found:
        return found
    candidate = Path(requested).expanduser()
    if candidate.is_file() and not os.access(candidate, os.X_OK):
        raise MediaError(f"{name} is not executable: {candidate}")
    if candidate.is_file():
        return str(candidate)
    raise MediaError(f"{name} not found at: {requested}")


def find_ffmpeg(*, explicit: str | None = None, need_subtitles_filter: bool = False) -> str:
    requested = explicit or os.environ.get("FFMPEG_PATH")
    if requested:
        resolved = explicit_executable(requested, name="ffmpeg")
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
        return explicit_executable(requested, name="ffprobe")

    if ffmpeg_path:
        sibling = Path(ffmpeg_path).with_name("ffprobe")
        if sibling.is_file() and os.access(sibling, os.X_OK):
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


def probe_audio_streams(
    media_path: Path,
    *,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
) -> list[AudioStream]:
    """Read the audio metadata needed by the stream selector."""
    resolved = ffprobe_path or find_ffprobe(ffmpeg_path)
    output = run_ffprobe(
        resolved,
        [
            "-select_streams",
            "a",
            "-show_entries",
            (
                "stream=index,codec_name,channels,channel_layout:"
                "stream_tags=language,title:"
                "stream_disposition=default,forced,comment,hearing_impaired,visual_impaired"
            ),
            "-of",
            "json",
            str(media_path),
        ],
    )
    try:
        payload = json.loads(output)
        raw_streams = payload["streams"]
        if not isinstance(raw_streams, list):
            raise TypeError
        streams: list[AudioStream] = []
        for raw in raw_streams:
            if not isinstance(raw, dict):
                raise TypeError
            tags = raw.get("tags") or {}
            disposition = raw.get("disposition") or {}
            if not isinstance(tags, dict) or not isinstance(disposition, dict):
                raise TypeError
            streams.append(
                AudioStream(
                    index=int(raw["index"]),
                    codec=str(raw["codec_name"]) if raw.get("codec_name") else None,
                    channels=int(raw["channels"]) if raw.get("channels") is not None else None,
                    layout=(
                        str(raw["channel_layout"]) if raw.get("channel_layout") else None
                    ),
                    language=str(tags["language"]) if tags.get("language") else None,
                    title=str(tags["title"]) if tags.get("title") else None,
                    is_default=disposition.get("default") in {1, "1"},
                    is_forced=disposition.get("forced") in {1, "1"},
                    is_hearing_impaired=(
                        disposition.get("hearing_impaired") in {1, "1"}
                    ),
                    is_visual_impaired=(
                        disposition.get("visual_impaired") in {1, "1"}
                    ),
                    is_commentary=disposition.get("comment") in {1, "1"},
                )
            )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ProbeError(f"Could not read audio streams of {media_path.name}.") from exc
    return streams


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


def probe_sample_rate(
    media_path: Path, *, ffmpeg_path: str | None = None, ffprobe_path: str | None = None
) -> int | None:
    """The first audio stream's sample rate, or None when it cannot be read."""
    resolved = ffprobe_path or find_ffprobe(ffmpeg_path)
    try:
        output = run_ffprobe(
            resolved,
            [
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=nw=1:nk=1",
                str(media_path),
            ],
        )
        return int(output.splitlines()[0])
    except (ProbeError, ValueError, IndexError):
        return None


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
