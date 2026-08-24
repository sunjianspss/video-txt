from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from video_txt.cli import main
from video_txt.subtitles import parse_srt

SOURCE_SRT = """1
00:00:08,000 --> 00:00:09,000
Before.

2
00:00:10,000 --> 00:00:12,000
Old A.

3
00:00:14,000 --> 00:00:16,000
Old B.

4
00:00:18,000 --> 00:00:19,000
After.
"""

LOCAL_SRT = """1
00:00:00,000 --> 00:00:01,000
Padding before.

2
00:00:01,700 --> 00:00:03,000
New A.

3
00:00:05,000 --> 00:00:07,000
New B.

4
00:00:07,400 --> 00:00:08,700
Padding after.
"""


def install_fake_media_tools(monkeypatch, *, local_srt=LOCAL_SRT):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace() if name in {"whisper", "mlx_whisper"} else real_find_spec(name)
        ),
    )
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    commands: list[list[str]] = []

    def external_tool(command, **_kwargs):
        commands.append(command)
        if "-select_streams" in command:
            media = Path(command[-1])
            stream = 0 if media.suffix == ".wav" else 2
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "streams": [
                            {
                                "index": stream,
                                "codec_name": "pcm_s16le" if stream == 0 else "aac",
                                "channels": 1 if stream == 0 else 2,
                                "tags": {"language": "eng", "title": "English"},
                            }
                        ]
                    }
                ),
                stderr="",
            )
        if command[0] == "/usr/bin/ffmpeg":
            Path(command[-1]).write_bytes(b"wav")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[:3] == [sys.executable, "-m", "whisper"]:
            input_path = Path(command[3])
            output_dir = Path(command[command.index("--output_dir") + 1])
            (output_dir / f"{input_path.stem}.srt").write_text(local_srt, encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected external command: {command}")

    monkeypatch.setattr(subprocess, "run", external_tool)
    return commands


def test_retranscribe_time_range_splices_absolute_cues_and_reports_changes(
    tmp_path, monkeypatch
):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"media")
    source = tmp_path / "movie.srt"
    source.write_text(SOURCE_SRT, encoding="utf-8")
    output = tmp_path / "movie.repaired.srt"
    commands = install_fake_media_tools(monkeypatch)

    assert (
        main(
            [
                "retranscribe-range",
                str(video),
                "--subtitle",
                str(source),
                "--from",
                "00:00:10",
                "--to",
                "00:00:16",
                "--padding",
                "1.5",
                "--language",
                "en",
                "--audio-stream",
                "2",
                "--backend",
                "openai-whisper",
                "--model",
                "tiny",
                "-o",
                str(output),
            ]
        )
        == 0
    )

    assert source.read_text(encoding="utf-8") == SOURCE_SRT
    cues = parse_srt(output)
    assert [(cue.text, cue.timing) for cue in cues] == [
        ("Before.", "00:00:08,000 --> 00:00:09,000"),
        ("New A.", "00:00:10,200 --> 00:00:11,500"),
        ("New B.", "00:00:13,500 --> 00:00:15,500"),
        ("After.", "00:00:18,000 --> 00:00:19,000"),
    ]
    assert [cue.index for cue in cues] == ["1", "2", "3", "4"]
    assert all(
        left.end_seconds <= right.start_seconds
        for left, right in zip(cues, cues[1:], strict=False)
    )

    report_path = tmp_path / "movie.repaired.retranscription-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "video-txt.retranscription-report"
    assert report["version"] == 1
    assert report["range"] == {"start": 10.0, "end": 16.0, "padding": 1.5}
    assert report["extraction"] == {"start": 8.5, "end": 17.5}
    assert report["audio_stream"]["selected"] == 2
    assert [cue["text"] for cue in report["removed_cues"]] == ["Old A.", "Old B."]
    assert [cue["text"] for cue in report["added_cues"]] == ["New A.", "New B."]

    extraction = next(command for command in commands if command[0] == "/usr/bin/ffmpeg")
    assert extraction[extraction.index("-ss") + 1] == "8.500"
    assert extraction[extraction.index("-t") + 1] == "9.000"
    assert extraction[extraction.index("-map") + 1] == "0:2"


def test_retranscribe_cue_range_uses_source_cue_boundaries(tmp_path, monkeypatch):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"media")
    source = tmp_path / "movie.srt"
    source.write_text(SOURCE_SRT, encoding="utf-8")
    output = tmp_path / "by-cue.srt"
    install_fake_media_tools(monkeypatch)

    assert (
        main(
            [
                "retranscribe-range",
                str(video),
                "--subtitle",
                str(source),
                "--from-cue",
                "2",
                "--to-cue",
                "3",
                "--padding",
                "1.5",
                "--language",
                "en",
                "--audio-stream",
                "2",
                "--backend",
                "openai-whisper",
                "-o",
                str(output),
            ]
        )
        == 0
    )

    cues = parse_srt(output)
    assert [cue.text for cue in cues] == ["Before.", "New A.", "New B.", "After."]
    report = json.loads(
        (tmp_path / "by-cue.retranscription-report.json").read_text(encoding="utf-8")
    )
    assert report["range"] == {"start": 10.0, "end": 16.0, "padding": 1.5}
    assert report["requested_range"] == {"from_cue": 2, "to_cue": 3}


def test_retranscribe_dry_run_shows_both_backend_plans_without_writing(
    tmp_path, monkeypatch, capsys
):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"media")
    source = tmp_path / "movie.srt"
    source.write_text(SOURCE_SRT, encoding="utf-8")
    install_fake_media_tools(monkeypatch)

    for backend in ("openai-whisper", "mlx-whisper"):
        output = tmp_path / f"{backend}.srt"
        assert (
            main(
                [
                    "retranscribe-range",
                    str(video),
                    "--subtitle",
                    str(source),
                    "--from",
                    "00:00:10",
                    "--to",
                    "00:00:16",
                    "--audio-stream",
                    "2",
                    "--backend",
                    backend,
                    "-o",
                    str(output),
                    "--dry-run",
                ]
            )
            == 0
        )
        plan = capsys.readouterr().out
        assert "Would replace 2 source cues" in plan
        assert "Core range: 00:00:10,000 --> 00:00:16,000" in plan
        assert "Extraction range: 00:00:08,500 --> 00:00:17,500" in plan
        assert "-ss 8.500" in plan
        assert "-t 9.000" in plan
        assert "-map 0:2" in plan
        expected_launcher = "-m whisper" if backend == "openai-whisper" else "mlx_whisper"
        assert expected_launcher in plan
        assert not output.exists()
        assert not (tmp_path / f"{backend}.retranscription-report.json").exists()


def test_retranscribe_discards_padding_duplicates_that_cross_core_boundaries(
    tmp_path, monkeypatch
):
    local_srt = """1
00:00:00,500 --> 00:00:02,500
Before.

2
00:00:01,700 --> 00:00:03,000
New A.

3
00:00:05,000 --> 00:00:07,000
New B.

4
00:00:07,000 --> 00:00:08,500
After.
"""
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"media")
    source = tmp_path / "movie.srt"
    source.write_text(SOURCE_SRT, encoding="utf-8")
    output = tmp_path / "deduplicated.srt"
    install_fake_media_tools(monkeypatch, local_srt=local_srt)

    assert (
        main(
            [
                "retranscribe-range",
                str(video),
                "--subtitle",
                str(source),
                "--from",
                "00:00:10",
                "--to",
                "00:00:16",
                "--audio-stream",
                "2",
                "--backend",
                "openai-whisper",
                "-o",
                str(output),
            ]
        )
        == 0
    )

    cues = parse_srt(output)
    assert [cue.text for cue in cues] == ["Before.", "New A.", "New B.", "After."]
    assert all(
        left.end_seconds <= right.start_seconds
        for left, right in zip(cues, cues[1:], strict=False)
    )


def test_retranscribe_refuses_to_write_report_over_the_repaired_subtitle(
    tmp_path, monkeypatch
):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"media")
    source = tmp_path / "movie.srt"
    source.write_text(SOURCE_SRT, encoding="utf-8")
    output = tmp_path / "same-path.srt"
    commands = install_fake_media_tools(monkeypatch)

    assert (
        main(
            [
                "retranscribe-range",
                str(video),
                "--subtitle",
                str(source),
                "--from",
                "00:00:10",
                "--to",
                "00:00:16",
                "--audio-stream",
                "2",
                "--backend",
                "openai-whisper",
                "-o",
                str(output),
                "--report",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()
    assert commands == []
