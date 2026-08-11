from __future__ import annotations

import wave
from pathlib import Path

from video_txt import dub as dub_module
from video_txt.dub import (
    DubOptions,
    Segment,
    build_dub_mux_command,
    build_segments,
    build_tts_command,
    default_video_output,
    render_audio_track,
    segment_filename,
)
from video_txt.subtitles import SubtitleCue, parse_srt_text

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


def test_render_audio_track_streams_gaps_clips_overlaps_and_tail_padding(tmp_path, monkeypatch):
    clip_a = b"\x11\x11" * 50  # 0.5 s at 100 Hz
    clip_b = b"\x22\x22" * 30  # 0.3 s at 100 Hz
    clips = {Path("/cache/a.mp3"): (clip_a, 1.0), Path("/cache/b.mp3"): (clip_b, 1.2)}
    monkeypatch.setattr(dub_module, "find_ffprobe", lambda *_args, **_kwargs: "ffprobe")
    monkeypatch.setattr(
        dub_module, "decode_segment", lambda segment, **_kwargs: clips[segment.audio_path]
    )

    segments = [
        Segment(
            cue=SubtitleCue("1", "00:00:01,000 --> 00:00:02,000", ["一"]),
            audio_path=Path("/cache/a.mp3"),
            start=1.0,
            slot=1.0,
        ),
        # Starts at 1.2 s, but the first clip plays until 1.5 s, so it gets pushed.
        Segment(
            cue=SubtitleCue("2", "00:00:01,200 --> 00:00:02,000", ["二"]),
            audio_path=Path("/cache/b.mp3"),
            start=1.2,
            slot=0.8,
        ),
    ]
    output = tmp_path / "track.wav"
    placed = render_audio_track(
        segments,
        make_options(sample_rate=100),
        ffmpeg_path="ffmpeg",
        total_duration=3.0,
        output_path=output,
    )

    with wave.open(str(output)) as handle:
        assert handle.getframerate() == 100
        assert handle.getnframes() == 400  # video length + 1 s of padding
        data = handle.readframes(400)
    assert data[:200] == bytes(200)  # silence until 1.0 s
    assert data[200:300] == clip_a
    assert data[300:360] == clip_b  # pushed from 1.2 s to 1.5 s
    assert data[360:] == bytes(len(data) - 360)
    assert [item.start for item in placed] == [1.0, 1.5]
    assert placed[0].tempo == 1.0
    assert placed[1].tempo == 1.2


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
