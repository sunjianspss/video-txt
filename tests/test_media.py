from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from video_txt import media as media_module
from video_txt.media import (
    AudioStream,
    MediaError,
    find_ffmpeg,
    find_ffprobe,
    probe_audio_streams,
    select_audio_stream,
)


@pytest.mark.parametrize("finder", [find_ffmpeg, find_ffprobe])
def test_an_explicit_media_tool_must_be_executable(tmp_path, finder):
    tool = tmp_path / "tool"
    tool.write_text("not a program\n", encoding="utf-8")
    tool.chmod(0o600)

    with pytest.raises(MediaError, match="not executable"):
        finder(explicit=str(tool))


def test_requested_language_title_wins_over_wrong_tag_and_default_track():
    streams = [
        AudioStream(
            index=1,
            codec="ac3",
            channels=2,
            layout="stereo",
            language="spa",
            title="Lat",
            is_default=True,
            is_forced=True,
            is_hearing_impaired=True,
        ),
        AudioStream(
            index=2,
            codec="aac",
            channels=2,
            layout="stereo",
            language="spa",
            title="Eng",
        ),
    ]

    selection = select_audio_stream(streams, preferred_language="en")

    assert selection.stream.index == 2
    assert "title 'Eng' matches en" in selection.reason


def test_unique_default_track_is_selected_when_language_is_unknown():
    streams = [
        AudioStream(index=1, language="jpn", title="Japanese"),
        AudioStream(index=2, language="eng", title="English", is_default=True),
    ]

    selection = select_audio_stream(streams)

    assert selection.stream.index == 2
    assert selection.reason == "marked default"


def test_equally_plausible_tracks_stop_instead_of_guessing():
    streams = [
        AudioStream(index=1, codec="aac", channels=2),
        AudioStream(index=2, codec="aac", channels=2),
    ]

    with pytest.raises(MediaError, match=r"Cannot safely choose.*--audio-stream"):
        select_audio_stream(streams)


def test_missing_requested_language_stops_with_available_track_details():
    streams = [
        AudioStream(index=1, language="jpn", title="Japanese", is_default=True),
        AudioStream(index=2, language="spa", title="Spanish"),
    ]

    with pytest.raises(MediaError, match=r"No audio stream matches en.*1.*Japanese.*2.*Spanish"):
        select_audio_stream(streams, preferred_language="en")


def test_explicit_stream_index_overrides_incorrect_language_metadata():
    streams = [
        AudioStream(index=1, language="spa", title="Lat", is_default=True),
        AudioStream(index=2, language="spa", title="Eng"),
    ]

    selection = select_audio_stream(streams, preferred_language="ja", requested_index=2)

    assert selection.stream.index == 2
    assert selection.reason == "selected explicitly with --audio-stream 2"


def test_ffprobe_audio_metadata_is_normalized_for_selection(monkeypatch):
    payload = """{
      "streams": [
        {
          "index": 1, "codec_name": "ac3", "channels": 2,
          "channel_layout": "stereo",
          "disposition": {"default": 1, "forced": 1, "hearing_impaired": 1},
          "tags": {"language": "spa", "title": "Lat"}
        },
        {
          "index": 2, "codec_name": "aac", "channels": 2,
          "channel_layout": "stereo",
          "disposition": {"default": 0},
          "tags": {"language": "spa", "title": "Eng"}
        }
      ]
    }"""
    monkeypatch.setattr(
        media_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=payload, stderr=""),
    )

    streams = probe_audio_streams(Path("movie.mkv"), ffprobe_path="/usr/bin/ffprobe")

    assert streams == [
        AudioStream(
            index=1,
            codec="ac3",
            channels=2,
            layout="stereo",
            language="spa",
            title="Lat",
            is_default=True,
            is_forced=True,
            is_hearing_impaired=True,
        ),
        AudioStream(
            index=2,
            codec="aac",
            channels=2,
            layout="stereo",
            language="spa",
            title="Eng",
        ),
    ]


def test_main_dialogue_beats_a_default_commentary_track():
    streams = [
        AudioStream(index=1, language="eng", title="English"),
        AudioStream(
            index=2,
            language="eng",
            title="English Commentary",
            is_default=True,
            is_commentary=True,
        ),
    ]

    selection = select_audio_stream(streams, preferred_language="en")

    assert selection.stream.index == 1


def test_small_metadata_score_gap_still_stops_for_user_choice():
    streams = [
        AudioStream(index=1, language="eng", title="English"),
        AudioStream(index=2, language="eng", title="English", is_forced=True),
    ]

    with pytest.raises(MediaError, match=r"Cannot safely choose.*--audio-stream"):
        select_audio_stream(streams, preferred_language="en")
