from __future__ import annotations

import html
import os
import re
import tempfile
import unicodedata
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

# What a target language is called on disk, and how it is labelled inside a
# container: a filename suffix plus the ISO 639-2/B code players read.
LANGUAGES: dict[str, tuple[str, str]] = {
    "zh": ("zh", "zho"),
    "zh-cn": ("zh", "zho"),
    "zh_cn": ("zh", "zho"),
    "chinese": ("zh", "zho"),
    "simplified chinese": ("zh", "zho"),
    "中文": ("zh", "zho"),
    "简体中文": ("zh", "zho"),
    "zh-tw": ("zh-hant", "zho"),
    "zh-hk": ("zh-hant", "zho"),
    "traditional chinese": ("zh-hant", "zho"),
    "繁体中文": ("zh-hant", "zho"),
    "繁體中文": ("zh-hant", "zho"),
    "en": ("en", "eng"),
    "en-us": ("en", "eng"),
    "english": ("en", "eng"),
    "英语": ("en", "eng"),
    "英文": ("en", "eng"),
    "ja": ("ja", "jpn"),
    "ja-jp": ("ja", "jpn"),
    "japanese": ("ja", "jpn"),
    "日语": ("ja", "jpn"),
    "日文": ("ja", "jpn"),
    "日本語": ("ja", "jpn"),
    "ko": ("ko", "kor"),
    "korean": ("ko", "kor"),
    "韩语": ("ko", "kor"),
    "韩文": ("ko", "kor"),
    "fr": ("fr", "fra"),
    "french": ("fr", "fra"),
    "法语": ("fr", "fra"),
    "de": ("de", "deu"),
    "german": ("de", "deu"),
    "德语": ("de", "deu"),
    "es": ("es", "spa"),
    "spanish": ("es", "spa"),
    "西班牙语": ("es", "spa"),
    "pt": ("pt", "por"),
    "pt-br": ("pt", "por"),
    "portuguese": ("pt", "por"),
    "葡萄牙语": ("pt", "por"),
    "it": ("it", "ita"),
    "italian": ("it", "ita"),
    "意大利语": ("it", "ita"),
    "ru": ("ru", "rus"),
    "russian": ("ru", "rus"),
    "俄语": ("ru", "rus"),
    "ar": ("ar", "ara"),
    "arabic": ("ar", "ara"),
    "阿拉伯语": ("ar", "ara"),
    "hi": ("hi", "hin"),
    "hindi": ("hi", "hin"),
    "印地语": ("hi", "hin"),
    "id": ("id", "ind"),
    "indonesian": ("id", "ind"),
    "印尼语": ("id", "ind"),
    "vi": ("vi", "vie"),
    "vietnamese": ("vi", "vie"),
    "越南语": ("vi", "vie"),
    "th": ("th", "tha"),
    "thai": ("th", "tha"),
    "泰语": ("th", "tha"),
}
UNKNOWN_LANGUAGE_CODE = "und"

TIMING_PATTERN = re.compile(
    r"(?P<start>\d+:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(?P<end>\d+:\d{2}:\d{2}[,.]\d{1,3})"
)
TIMESTAMP_PATTERN = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")
SUBTITLE_TAG_PATTERN = re.compile(
    r"</?(?:b|i|u|s|font|span|ruby|rt|br|c|v|lang)(?:\s+[^<>]*)?\s*/?>",
    re.IGNORECASE,
)


class SubtitleFormatError(ValueError):
    pass


@dataclass
class SubtitleCue:
    index: str
    timing: str
    text_lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.text_lines)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def start_seconds(self) -> float:
        return self.time_range[0]

    @property
    def end_seconds(self) -> float:
        return self.time_range[1]

    @property
    def duration(self) -> float:
        start, end = self.time_range
        return max(0.0, end - start)

    @property
    def time_range(self) -> tuple[float, float]:
        match = TIMING_PATTERN.search(self.timing)
        if not match:
            raise SubtitleFormatError(f"Invalid SRT time range: {self.timing!r}")
        return (
            srt_time_to_seconds(match.group("start")),
            srt_time_to_seconds(match.group("end")),
        )

    def with_text(self, text: str) -> SubtitleCue:
        return SubtitleCue(
            index=self.index,
            timing=self.timing,
            text_lines=text.split("\n") if text else [],
        )


def is_cjk(char: str) -> bool:
    return "\u3000" <= char <= "\u9fff" or "\uff00" <= char <= "\uffef"


def display_width(text: str) -> int:
    """Approximate terminal/subtitle width, counting wide CJK glyphs as two columns."""
    return sum(2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1 for char in text)


def join_text_parts(parts: list[str]) -> str:
    """Join fragments the way the language writes them: no space between CJK."""
    joined = ""
    for part in parts:
        if not part:
            continue
        if joined and not (is_cjk(joined[-1]) or is_cjk(part[0])):
            joined += " "
        joined += part
    return joined


def resolve_language(target_language: str) -> tuple[str, str] | None:
    """The filename suffix and container code for a language, if we know it."""
    return LANGUAGES.get(target_language.strip().lower())


def language_suffix(target_language: str) -> str:
    known = resolve_language(target_language)
    return known[0] if known else "translated"


def language_code(target_language: str) -> str:
    """The code players show in their subtitle menu.

    An unrecognised language is labelled undetermined rather than guessed at, so a
    track never claims to be a language it is not.
    """
    known = resolve_language(target_language)
    return known[1] if known else UNKNOWN_LANGUAGE_CODE


def translated_subtitle_path(source: Path, target_language: str) -> Path:
    return source.with_name(f"{source.stem}.{language_suffix(target_language)}.srt")


def srt_time_to_seconds(value: str) -> float:
    match = TIMESTAMP_PATTERN.fullmatch(value.strip())
    if not match:
        raise SubtitleFormatError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, fraction = match.groups()
    millis = int(fraction.ljust(3, "0"))
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + millis / 1000


def seconds_to_srt_time(value: float) -> str:
    total_millis = max(0, round(value * 1000))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def srt_time_to_ass(value: str) -> str:
    match = TIMESTAMP_PATTERN.fullmatch(value.strip())
    if not match:
        raise SubtitleFormatError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, fraction = match.groups()
    centiseconds = int(fraction.ljust(3, "0")) // 10
    return f"{int(hours)}:{minutes}:{seconds}.{centiseconds:02d}"


def parse_srt_text(raw: str, *, source: str = "<string>") -> list[SubtitleCue]:
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not normalized.strip():
        raise SubtitleFormatError(f"Subtitle input is empty: {source}")

    cues: list[SubtitleCue] = []
    for block_number, block in enumerate(re.split(r"\n\s*\n", normalized), start=1):
        lines = [line.rstrip() for line in block.split("\n")]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            continue

        if TIMING_PATTERN.search(lines[0]):
            index = str(len(cues) + 1)
            timing_line, text_lines = lines[0], lines[1:]
        elif len(lines) >= 2 and TIMING_PATTERN.search(lines[1]):
            index = lines[0].strip()
            timing_line, text_lines = lines[1], lines[2:]
        else:
            raise SubtitleFormatError(
                f"Invalid SRT block #{block_number} in {source}: no '-->' time range found."
            )

        cues.append(
            SubtitleCue(
                index=index,
                timing=timing_line.strip(),
                text_lines=[line for line in text_lines if line.strip()],
            )
        )

    if not cues:
        raise SubtitleFormatError(f"No subtitle blocks found in {source}")
    return cues


def parse_srt(path: Path) -> list[SubtitleCue]:
    return parse_srt_text(path.read_text(encoding="utf-8-sig"), source=str(path))


def serialize_srt(cues: list[SubtitleCue]) -> str:
    blocks = ["\n".join([cue.index, cue.timing, *cue.text_lines]) for cue in cues]
    return "\n\n".join(blocks) + "\n"


def write_srt(path: Path, cues: list[SubtitleCue]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        fd = -1
        with handle:
            handle.write(serialize_srt(cues))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        if fd >= 0:
            with suppress(OSError):
                os.close(fd)
        temporary.unlink(missing_ok=True)
        raise
    return path


def chunk_by_chars[T](
    items: list[T], batch_chars: int, size_of: Callable[[T], int]
) -> list[list[T]]:
    limit = max(1, batch_chars)
    batches: list[list[T]] = []
    current: list[T] = []
    current_chars = 0

    for item in items:
        item_chars = max(1, size_of(item))
        if current and current_chars + item_chars > limit:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars

    if current:
        batches.append(current)
    return batches


def chunk_cues(cues: list[SubtitleCue], batch_chars: int) -> list[list[SubtitleCue]]:
    return chunk_by_chars(cues, batch_chars, lambda cue: len(cue.text))


def escape_ass_text(value: str) -> str:
    text = html.unescape(value.strip())
    text = SUBTITLE_TAG_PATTERN.sub("", text)
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def sanitize_style_value(value: str) -> str:
    return value.replace("\\", "").replace(",", " ").replace("{", "").replace("}", "").strip()


def resolve_style_metrics(
    *, video_height: int, font_size: int | None, margin_v: int | None
) -> tuple[int, int]:
    resolved_size = font_size if font_size else max(16, round(video_height * 0.045))
    resolved_margin = margin_v if margin_v is not None else max(8, round(video_height * 0.05))
    return resolved_size, resolved_margin


def build_ass_subtitle(
    *,
    cues: list[SubtitleCue],
    video_width: int,
    video_height: int,
    layout: str,
    font: str,
    font_size: int,
    margin_v: int,
) -> str:
    alignment = 8 if layout == "top" else 2
    font_size = max(8, font_size)
    margin_v = max(0, margin_v)
    outline = max(1, round(font_size / 12))
    side_margin = max(20, round(video_width * 0.04))
    font = sanitize_style_value(font) or "PingFang SC"

    events: list[str] = []
    for cue in cues:
        text = r"\N".join(escape_ass_text(line) for line in cue.text_lines)
        if not text:
            continue
        match = TIMING_PATTERN.search(cue.timing)
        if not match:
            raise SubtitleFormatError(f"Invalid SRT time range: {cue.timing!r}")
        start = srt_time_to_ass(match.group("start"))
        end = srt_time_to_ass(match.group("end"))
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {video_width}",
            f"PlayResY: {video_height}",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,"
            " BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,"
            " BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{font},{font_size},&H00FFFFFF,&H000000FF,&H99000000,&H99000000,"
            f"0,0,0,0,100,100,0,0,1,{outline},0,{alignment},{side_margin},{side_margin},{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            *events,
            "",
        ]
    )


def write_ass_subtitle(
    path: Path,
    *,
    cues: list[SubtitleCue],
    video_width: int,
    video_height: int,
    layout: str,
    font: str,
    font_size: int,
    margin_v: int,
) -> Path:
    content = build_ass_subtitle(
        cues=cues,
        video_width=video_width,
        video_height=video_height,
        layout=layout,
        font=font,
        font_size=font_size,
        margin_v=margin_v,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
