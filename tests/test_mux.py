from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from video_txt import mux as mux_module
from video_txt.media import MediaError, parse_video_size, quote_filter_value
from video_txt.mux import (
    MuxError,
    MuxOptions,
    build_mux_command,
    build_subtitles_filter,
    default_video_output,
    resolve_subtitle_codec,
    run_mux,
    styled_subtitle_path,
    validate_output_container,
)


def make_options(**overrides) -> MuxOptions:
    defaults = dict(
        video_input=Path("/videos/clip.mp4"),
        subtitle_input=Path("/videos/clip.zh.srt"),
        video_output=Path("/videos/clip.zh-subbed.mp4"),
    )
    return MuxOptions(**{**defaults, **overrides})


def test_default_video_output_keeps_known_containers():
    assert (
        default_video_output(Path("/v/a.mkv"), mux_mode="soft", target_language="zh").name
        == "a.zh-subbed.mkv"
    )
    assert (
        default_video_output(Path("/v/a.mp4"), mux_mode="hard", target_language="zh").name
        == "a.zh-burned.mp4"
    )


def test_default_video_output_falls_back_to_mp4_for_unsupported_containers():
    assert (
        default_video_output(Path("/v/a.webm"), mux_mode="hard", target_language="zh").name
        == "a.zh-burned.mp4"
    )
    assert (
        default_video_output(Path("/v/a.avi"), mux_mode="soft", target_language="English").name
        == "a.en-subbed.mp4"
    )


def test_resolve_subtitle_codec_matches_the_container():
    assert resolve_subtitle_codec(Path("out.mp4")) == "mov_text"
    assert resolve_subtitle_codec(Path("out.MOV")) == "mov_text"
    assert resolve_subtitle_codec(Path("out.mkv")) == "srt"
    assert resolve_subtitle_codec(Path("out.webm")) == "webvtt"
    assert resolve_subtitle_codec(Path("out.avi"), explicit="srt") == "srt"


def test_resolve_subtitle_codec_reports_unsupported_containers():
    with pytest.raises(MuxError, match="not supported"):
        resolve_subtitle_codec(Path("out.avi"))


def test_soft_mux_command_uses_the_container_codec():
    command = build_mux_command(
        make_options(video_output=Path("/v/out.mkv")),
        ffmpeg_path="/bin/ffmpeg",
        subtitle_for_mux=Path("/v/clip.zh.srt"),
    )
    assert command[command.index("-c:s") + 1] == "srt"
    assert "-movflags" not in command
    assert command[:2] == ["/bin/ffmpeg", "-n"]
    assert "0:a?" in command


def test_soft_mux_command_adds_faststart_and_default_disposition_for_mp4():
    command = build_mux_command(
        make_options(default_subtitle=True, overwrite=True),
        ffmpeg_path="/bin/ffmpeg",
        subtitle_for_mux=Path("/v/clip.zh.srt"),
    )
    assert command[1] == "-y"
    assert command[command.index("-c:s") + 1] == "mov_text"
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert command[command.index("-disposition:s:0") + 1] == "default"


def test_hard_mux_command_carries_encoder_settings():
    command = build_mux_command(
        make_options(mux_mode="hard", crf=18, preset="slow"),
        ffmpeg_path="/bin/ffmpeg",
        subtitle_for_mux=Path("/v/clip.zh.normal.ass"),
    )
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-crf") + 1] == "18"
    assert command[command.index("-preset") + 1] == "slow"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-c:a") + 1] == "copy"


def test_hard_mux_command_picks_its_streams_rather_than_letting_ffmpeg_choose():
    """A source subtitle track ffmpeg selects for itself fails a burn it has no part in."""
    command = build_mux_command(
        make_options(mux_mode="hard"),
        ffmpeg_path="/bin/ffmpeg",
        subtitle_for_mux=Path("/v/clip.zh.normal.ass"),
    )
    assert [command[index + 1] for index, arg in enumerate(command) if arg == "-map"] == [
        "0:v:0",
        "0:a?",
    ]
    assert "-sn" in command


def test_hard_mux_command_skips_crf_for_other_encoders():
    command = build_mux_command(
        make_options(mux_mode="hard", video_codec="hevc_videotoolbox"),
        ffmpeg_path="/bin/ffmpeg",
        subtitle_for_mux=Path("/v/clip.zh.normal.ass"),
    )
    assert "-crf" not in command
    assert "-preset" not in command


def test_hard_output_rejects_a_container_that_cannot_hold_h264():
    with pytest.raises(MuxError, match="cannot hold libx264"):
        validate_output_container(make_options(video_output=Path("/v/out.webm"), mux_mode="hard"))


def test_hard_output_allows_other_containers_with_a_matching_codec():
    validate_output_container(
        make_options(video_output=Path("/v/out.webm"), mux_mode="hard", video_codec="libvpx-vp9")
    )


def test_soft_output_container_is_validated_up_front():
    with pytest.raises(MuxError, match="not supported"):
        validate_output_container(make_options(video_output=Path("/v/out.avi")))
    validate_output_container(make_options(video_output=Path("/v/out.avi"), subtitle_codec="srt"))


def test_subtitles_filter_quotes_paths_and_adds_the_bottom_box():
    plain = build_subtitles_filter(Path("/v/a b's.ass"), layout="normal")
    assert plain.startswith("subtitles=filename='")
    assert "force_style" not in plain

    boxed = build_subtitles_filter(Path("/v/a.ass"), layout="bottom-box", box_height=0.3)
    assert boxed.startswith("drawbox=")
    assert "ih*0.3000" in boxed
    assert boxed.count("subtitles=filename=") == 1


def test_subtitles_filter_clamps_the_box_geometry():
    boxed = build_subtitles_filter(
        Path("/v/a.ass"), layout="bottom-box", box_height=9.0, box_opacity=5.0
    )
    assert "ih*0.5000" in boxed
    assert "black@1.000" in boxed


def test_quote_filter_value_escapes_single_quotes():
    assert quote_filter_value("/v/plain.srt") == "'/v/plain.srt'"
    assert quote_filter_value("/v/it's.srt") == "'/v/it'\\\\\\''s.srt'"


def test_styled_subtitle_path_names_the_layout():
    assert styled_subtitle_path(Path("/v/a.zh.srt"), "top").name == "a.zh.top.ass"
    assert styled_subtitle_path(Path("/v/a.zh.srt"), "normal").name == "a.zh.normal.ass"


def test_hard_mux_never_overwrites_a_same_named_ass_file(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    subtitle = tmp_path / "clip.zh.srt"
    output = tmp_path / "out.mp4"
    video.write_bytes(b"video")
    subtitle.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    caller_ass = styled_subtitle_path(subtitle, "normal")
    caller_ass.write_text("caller-owned\n", encoding="utf-8")
    commands: list[list[str]] = []

    monkeypatch.setattr(mux_module, "find_ffmpeg", lambda **_kwargs: "ffmpeg")
    monkeypatch.setattr(mux_module, "probe_video_size", lambda *_args, **_kwargs: (1280, 720))
    monkeypatch.setattr(
        mux_module.subprocess,
        "run",
        lambda command: (
            commands.append(command),
            Path(command[-1]).write_bytes(b"video"),
            SimpleNamespace(returncode=0),
        )[-1],
    )

    run_mux(
        MuxOptions(
            video_input=video,
            subtitle_input=subtitle,
            video_output=output,
            mux_mode="hard",
        )
    )

    assert caller_ass.read_text(encoding="utf-8") == "caller-owned\n"
    assert str(caller_ass) not in commands[0][commands[0].index("-vf") + 1]


def test_failed_mux_preserves_an_existing_output_video(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    subtitle = tmp_path / "clip.zh.srt"
    output = tmp_path / "out.mp4"
    video.write_bytes(b"video")
    subtitle.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    output.write_bytes(b"known-good video")

    def interrupted(command):
        Path(command[-1]).write_bytes(b"partial replacement")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(mux_module, "find_ffmpeg", lambda **_kwargs: "ffmpeg")
    monkeypatch.setattr(mux_module.subprocess, "run", interrupted)

    with pytest.raises(MuxError, match="exit code 1"):
        run_mux(
            MuxOptions(
                video_input=video,
                subtitle_input=subtitle,
                video_output=output,
                overwrite=True,
            )
        )

    assert output.read_bytes() == b"known-good video"
    assert list(tmp_path.glob(".out.*.part.mp4")) == []


def test_parse_video_size():
    assert parse_video_size("1920x1080") == (1920, 1080)
    assert parse_video_size(" 1280 X 720 ") == (1280, 720)
    with pytest.raises(MediaError):
        parse_video_size("1920")
    with pytest.raises(MediaError):
        parse_video_size("0x100")
