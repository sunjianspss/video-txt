from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from .constants import DEFAULT_LANGUAGE_CODE
from .media import (
    find_ffmpeg,
    format_command,
    probe_video_size,
    quote_filter_value,
    temporary_output_path,
)
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
    # The subtitle carries both languages, so its second line is the original
    # and gets burned in smaller and quieter than the translation above it.
    bilingual: bool = False
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
    output_path: Path | None = None,
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
    styled = cues if cues is not None else parse_srt(options.subtitle_input)
    # A bilingual cue is exactly [translation, original], so the trailing line is
    # the quieter one wherever there are two.
    secondary = (
        [1 if len(cue.text_lines) == 2 else 0 for cue in styled] if options.bilingual else []
    )
    return write_ass_subtitle(
        output_path or styled_subtitle_path(options.subtitle_input, options.hard_layout),
        cues=styled,
        video_width=width,
        video_height=height,
        layout=options.hard_layout,
        font=options.font,
        font_size=font_size,
        margin_v=margin_v,
        secondary_lines=secondary,
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

    temporary_directory: TemporaryDirectory[str] | None = None
    temporary_video: Path | None = None
    try:
        if options.mux_mode == "hard":
            print("Preparing styled hard subtitle layout...")
            temporary_directory = TemporaryDirectory(prefix="video-txt-ass-")
            temporary_ass = (
                Path(temporary_directory.name)
                / styled_subtitle_path(options.subtitle_input, options.hard_layout).name
            )
            subtitle_for_mux = prepare_styled_subtitle(
                options,
                ffmpeg_path=ffmpeg_path,
                output_path=temporary_ass,
            )

        options.video_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_video = temporary_output_path(options.video_output)
        command_options = replace(options, video_output=temporary_video, overwrite=True)
        command = build_mux_command(
            command_options, ffmpeg_path=ffmpeg_path, subtitle_for_mux=subtitle_for_mux
        )
        if options.mux_mode == "soft":
            print("Muxing soft subtitles into the video container...")
        else:
            print("Burning hard subtitles into the video frames...")

        completed = subprocess.run(command)
        if completed.returncode != 0:
            raise MuxError(f"ffmpeg failed with exit code {completed.returncode}.")
        if not temporary_video.is_file() or temporary_video.stat().st_size == 0:
            raise MuxError("ffmpeg reported success but did not write the output video.")
        temporary_video.replace(options.video_output)
        return options.video_output
    finally:
        if temporary_video is not None:
            temporary_video.unlink(missing_ok=True)
        if temporary_directory is not None:
            temporary_directory.cleanup()
