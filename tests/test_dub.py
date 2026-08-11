from __future__ import annotations

from pathlib import Path

from video_txt.dub import (
    DubOptions,
    build_dub_mux_command,
    build_segments,
    build_tts_command,
    default_video_output,
    segment_filename,
)
from video_txt.subtitles import parse_srt_text

TIMED = (
    "1\n00:00:01,000 --> 00:00:03,000\nfirst\n\n"
    "2\n00:00:05,000 --> 00:00:06,000\nsecond\n\n"
    "3\n00:00:10,000 --> 00:00:12,000\nthird\n"
)


def make_options(**overrides) -> DubOptions:
    defaults = dict(
        video_input=Path("/videos/clip.mp4"),
        subtitle_input=Path("/videos/clip.zh.srt"),
        video_output=Path("/videos/clip.zh-dubbed.mp4"),
    )
    return DubOptions(**{**defaults, **overrides})


def test_default_video_output_names_the_dub():
    assert default_video_output(Path("/v/a.mp4")).name == "a.zh-dubbed.mp4"
    assert default_video_output(Path("/v/a.webm")).name == "a.zh-dubbed.mp4"


def test_build_segments_borrows_the_gap_before_the_next_line():
    cues = list(enumerate(parse_srt_text(TIMED), start=1))
    paths = [Path(f"/cache/{index}.mp3") for index, _ in cues]
    segments = build_segments(cues, paths)

    assert [segment.start for segment in segments] == [1.0, 5.0, 10.0]
    assert segments[0].slot == 3.95
    assert segments[1].slot == 4.95
    assert segments[2].slot == 4.0


def test_edge_tts_command_shape():
    command = build_tts_command(
        make_options(voice="zh-CN-YunxiNeural", rate="+10%"),
        launcher=["/bin/edge-tts"],
        text="你好",
        output_path=Path("/cache/a.mp3"),
    )
    assert command[command.index("--voice") + 1] == "zh-CN-YunxiNeural"
    assert "--rate=+10%" in command
    assert command[command.index("--text") + 1] == "你好"
    assert command[command.index("--write-media") + 1] == "/cache/a.mp3"


def test_say_command_falls_back_to_a_mac_voice():
    command = build_tts_command(
        make_options(engine="say"),
        launcher=["/usr/bin/say"],
        text="你好",
        output_path=Path("/cache/a.aiff"),
    )
    assert command == ["/usr/bin/say", "-v", "Tingting", "-o", "/cache/a.aiff", "你好"]


def test_segment_filename_is_stable_and_text_sensitive():
    first = segment_filename("edge-tts", 7, "你好", "v", "+0%")
    assert first == segment_filename("edge-tts", 7, "你好", "v", "+0%")
    assert first != segment_filename("edge-tts", 7, "你好啊", "v", "+0%")
    assert first != segment_filename("edge-tts", 7, "你好", "v", "+10%")
    assert first.startswith("cue-00007-")
    assert first.endswith(".mp3")
    assert segment_filename("say", 1, "x", "v", "+0%").endswith(".aiff")


def test_dub_mux_command_replaces_the_audio_track_by_default():
    command = build_dub_mux_command(
        make_options(),
        ffmpeg_path="/bin/ffmpeg",
        audio_path=Path("/videos/clip.dub.wav"),
        keep_bgm=False,
    )
    assert "-filter_complex" not in command
    assert command[command.index("-map") + 1] == "0:v:0"
    assert "1:a:0" in command
    assert command[command.index("-c:a") + 1] == "aac"
    assert "-shortest" in command
    assert command[command.index("-movflags") + 1] == "+faststart"


def test_dub_mux_command_mixes_the_original_audio_when_keeping_bgm():
    command = build_dub_mux_command(
        make_options(bgm_volume=0.2),
        ffmpeg_path="/bin/ffmpeg",
        audio_path=Path("/videos/clip.dub.wav"),
        keep_bgm=True,
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "volume=0.200" in graph
    assert "amix=inputs=2:duration=first:normalize=0[aout]" in graph
    assert "[aout]" in command


def test_dub_mux_command_can_add_the_subtitle_track():
    command = build_dub_mux_command(
        make_options(soft_subtitle=True),
        ffmpeg_path="/bin/ffmpeg",
        audio_path=Path("/videos/clip.dub.wav"),
        keep_bgm=False,
    )
    assert command.count("-i") == 3
    assert command[command.index("-c:s") + 1] == "mov_text"
    assert "2:0" in command
