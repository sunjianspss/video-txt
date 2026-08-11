from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path

CHINESE_ALIASES = {
    "simplified chinese",
    "chinese",
    "zh",
    "zh-cn",
    "zh_cn",
    "中文",
    "简体中文",
}
ENGLISH_ALIASES = {"english", "en", "en-us", "英语", "英文"}

TIMING_PATTERN = re.compile(
    r"(?P<start>\d+:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(?P<end>\d+:\d{2}:\d{2}[,.]\d{1,3})"
)
TIMESTAMP_PATTERN = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


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


def language_suffix(target_language: str) -> str:
    normalized = target_language.strip().lower()
    if normalized in CHINESE_ALIASES:
        return "zh"
    if normalized in ENGLISH_ALIASES:
        return "en"
    return "translated"


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
    path.write_text(serialize_srt(cues), encoding="utf-8")
    return path


def chunk_cues(cues: list[SubtitleCue], batch_chars: int) -> list[list[SubtitleCue]]:
    limit = max(1, batch_chars)
    batches: list[list[SubtitleCue]] = []
    current: list[SubtitleCue] = []
    current_chars = 0

    for cue in cues:
        cue_chars = max(1, len(cue.text))
        if current and current_chars + cue_chars > limit:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(cue)
        current_chars += cue_chars

    if current:
        batches.append(current)
    return batches


def escape_ass_text(value: str) -> str:
    text = html.unescape(value.strip())
    text = re.sub(r"<[^>]+>", "", text)
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
