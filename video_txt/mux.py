from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_LANGUAGE_CODE
from .media import find_ffmpeg, format_command, probe_video_size, quote_filter_value
from .subtitles import (
    SubtitleCue,
    language_suffix,
    parse_srt,
    resolve_style_metrics,
    write_ass_subtitle,
)

SUBTITLE_CODEC_BY_CONTAINER = {
    ".mp4": "mov_text",
    ".m4v": "mov_text",
    ".mov": "mov_text",
    ".mkv": "srt",
    ".mka": "srt",
    ".webm": "webvtt",
}
HARD_SUBTITLE_CONTAINERS = {".mp4", ".m4v", ".mov", ".mkv"}
FASTSTART_CONTAINERS = {".mp4", ".m4v", ".mov"}
HARD_LAYOUTS = ("normal", "bottom-box", "top")
H26X_CODECS = {"libx264", "libx265"}


class MuxError(RuntimeError):
    pass


@dataclass
class MuxOptions:
    video_input: Path
    subtitle_input: Path
    video_output: Path
    mux_mode: str = "soft"
    language_code: str = DEFAULT_LANGUAGE_CODE
    default_subtitle: bool = False
    subtitle_codec: str | None = None
    overwrite: bool = False
    hard_layout: str = "normal"
    font: str = "PingFang SC"
    font_size: int | None = None
    margin_v: int | None = None
    box_height: float = 0.22
    box_opacity: float = 0.68
    video_codec: str = "libx264"
    crf: int = 20
    preset: str = "medium"
    video_size: tuple[int, int] | None = None
    ffmpeg_path: str | None = None


def default_video_output(video: Path, *, mux_mode: str, target_language: str) -> Path:
    lang = language_suffix(target_language)
    style = "subbed" if mux_mode == "soft" else "burned"
    suffix = video.suffix.lower()
    allowed = HARD_SUBTITLE_CONTAINERS if mux_mode == "hard" else set(SUBTITLE_CODEC_BY_CONTAINER)
    container = suffix if suffix in allowed else ".mp4"
    return video.with_name(f"{video.stem}.{lang}-{style}{container}")


def resolve_subtitle_codec(video_output: Path, *, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    codec = SUBTITLE_CODEC_BY_CONTAINER.get(video_output.suffix.lower())
    if not codec:
        supported = ", ".join(sorted(SUBTITLE_CODEC_BY_CONTAINER))
        raise MuxError(
            f"Soft subtitles are not supported for the '{video_output.suffix}' container.\n"
            f"Use one of {supported}, or pass --subtitle-codec to choose the encoder yourself."
        )
    return codec


def subtitle_track_arguments(
    video_output: Path, *, stream: str, language_code: str, codec: str | None = None
) -> list[str]:
    """Carry one input in as a subtitle track, labelled so players can name it."""
    return [
        "-map",
        stream,
        "-c:s",
        resolve_subtitle_codec(video_output, explicit=codec),
        "-metadata:s:s:0",
        f"language={language_code}",
    ]


def container_arguments(video_output: Path) -> list[str]:
    """Whatever the output container itself asks for, beyond the streams."""
    if video_output.suffix.lower() in FASTSTART_CONTAINERS:
        # Without this the player has to read the whole file before it can start.
        return ["-movflags", "+faststart"]
    return []


def build_subtitles_filter(
    path: Path,
    *,
    layout: str = "normal",
    box_height: float = 0.22,
    box_opacity: float = 0.68,
) -> str:
    subtitles = f"subtitles=filename={quote_filter_value(str(path))}"
    if layout != "bottom-box":
        return subtitles

    height = min(max(box_height, 0.05), 0.5)
    opacity = min(max(box_opacity, 0.0), 1.0)
    box = (
        f"drawbox=x=0:y=ih*(1-{height:.4f}):w=iw:h=ih*{height:.4f}:color=black@{opacity:.3f}:t=fill"
    )
    return f"{box},{subtitles}"


def styled_subtitle_path(subtitle: Path, layout: str) -> Path:
    return subtitle.with_name(f"{subtitle.stem}.{layout}.ass")


def prepare_styled_subtitle(
    options: MuxOptions,
    *,
    ffmpeg_path: str,
    cues: list[SubtitleCue] | None = None,
) -> Path:
    width, height = options.video_size or probe_video_size(
        options.video_input, ffmpeg_path=ffmpeg_path
    )
    font_size, margin_v = resolve_style_metrics(
        video_height=height, font_size=options.font_size, margin_v=options.margin_v
    )
    print(
        f"Subtitle style: {options.font} {font_size}px, margin {margin_v}px, video {width}x{height}"
    )
    return write_ass_subtitle(
        styled_subtitle_path(options.subtitle_input, options.hard_layout),
        cues=cues if cues is not None else parse_srt(options.subtitle_input),
        video_width=width,
        video_height=height,
        layout=options.hard_layout,
        font=options.font,
        font_size=font_size,
        margin_v=margin_v,
    )


def build_mux_command(
    options: MuxOptions, *, ffmpeg_path: str, subtitle_for_mux: Path
) -> list[str]:
    overwrite_flag = "-y" if options.overwrite else "-n"

    if options.mux_mode == "soft":
        command = [
            ffmpeg_path,
            overwrite_flag,
            "-i",
            str(options.video_input),
            "-i",
            str(subtitle_for_mux),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            *subtitle_track_arguments(
                options.video_output,
                stream="1:0",
                language_code=options.language_code,
                codec=options.subtitle_codec,
            ),
            "-c:v",
            "copy",
            "-c:a",
            "copy",
        ]
        if options.default_subtitle:
            command.extend(["-disposition:s:0", "default"])
        command.extend(container_arguments(options.video_output))
        command.append(str(options.video_output))
        return command

    command = [
        ffmpeg_path,
        overwrite_flag,
        "-i",
        str(options.video_input),
        "-vf",
        build_subtitles_filter(
            subtitle_for_mux,
            layout=options.hard_layout,
            box_height=options.box_height,
            box_opacity=options.box_opacity,
        ),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        # The words are in the picture now. Left to pick for itself, ffmpeg also
        # carries over a subtitle track the source already had, and one the
        # output container cannot hold takes the whole burn down with it.
        "-sn",
        "-c:v",
        options.video_codec,
    ]
    if options.video_codec in H26X_CODECS:
        command.extend(["-preset", options.preset, "-crf", str(options.crf), "-pix_fmt", "yuv420p"])
    command.extend(["-c:a", "copy"])
    command.extend(container_arguments(options.video_output))
    command.append(str(options.video_output))
    return command


def validate_output_container(options: MuxOptions) -> None:
    if options.mux_mode == "soft":
        resolve_subtitle_codec(options.video_output, explicit=options.subtitle_codec)
        return

    if (
        options.video_codec in H26X_CODECS
        and options.video_output.suffix.lower() not in HARD_SUBTITLE_CONTAINERS
    ):
        raise MuxError(
            f"The '{options.video_output.suffix}' container cannot hold "
            f"{options.video_codec} video, which hard subtitles re-encode to.\n"
            "Pass --video-output with an .mp4 / .mkv name, or pick a matching --video-codec."
        )


def run_mux(options: MuxOptions, *, dry_run: bool = False) -> Path:
    validate_output_container(options)
    ffmpeg_path = find_ffmpeg(
        explicit=options.ffmpeg_path,
        need_subtitles_filter=options.mux_mode == "hard",
    )

    subtitle_for_mux = options.subtitle_input
    if options.mux_mode == "hard":
        subtitle_for_mux = styled_subtitle_path(options.subtitle_input, options.hard_layout)

    if dry_run:
        print(
            format_command(
                build_mux_command(
                    options, ffmpeg_path=ffmpeg_path, subtitle_for_mux=subtitle_for_mux
                )
            )
        )
        return options.video_output

    if options.video_output.exists() and not options.overwrite:
        raise MuxError(
            f"Output video already exists: {options.video_output}. "
            "Pass --overwrite-video to replace it."
        )

    if options.mux_mode == "hard":
        print("Preparing styled hard subtitle layout...")
        subtitle_for_mux = prepare_styled_subtitle(options, ffmpeg_path=ffmpeg_path)

    command = build_mux_command(options, ffmpeg_path=ffmpeg_path, subtitle_for_mux=subtitle_for_mux)
    if options.mux_mode == "soft":
        print("Muxing soft subtitles into the video container...")
    else:
        print("Burning hard subtitles into the video frames...")

    options.video_output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise MuxError(f"ffmpeg failed with exit code {completed.returncode}.")
    return options.video_output
