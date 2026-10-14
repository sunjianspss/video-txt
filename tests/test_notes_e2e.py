"""Exercise the installed CLI path with a real, local FFmpeg fixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_real_media_to_readable_notes_and_safe_rerun(tmp_path):
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:s=320x240:r=12:d=20",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:d=20",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:d=20",
            "-filter_complex",
            "[1:a][2:a]amix=inputs=2[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    source, translation = tmp_path / "clip.srt", tmp_path / "clip.zh.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:07,000\nA specific button is blue.\n")
    translation.write_text("1\n00:00:01,000 --> 00:00:07,000\n这一个按钮是蓝色的。\n")
    vault = tmp_path / "vault"
    vault.mkdir()
    argv = [
        sys.executable,
        "-m",
        "video_txt.cli",
        "export-notes",
        str(video),
        "--source",
        str(source),
        "--translation",
        str(translation),
        "--vault",
        str(vault),
        "--title",
        "Real media test",
        "--source-id",
        "real-media",
    ]
    preview = subprocess.run([*argv, "--dry-run"], check=True, capture_output=True, text=True)
    assert json.loads(preview.stdout)["dry_run"]
    assert list(vault.iterdir()) == []
    first = subprocess.run(argv, check=True, capture_output=True, text=True)
    payload = json.loads(first.stdout)
    files = [Path(p) for p in payload["documents"]]
    assert "这一个按钮是蓝色的。" in files[1].read_text()
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]
    subprocess.run(argv, check=True, capture_output=True, text=True)
    assert before == [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]
    files[1].write_text(files[1].read_text() + "\n人工注释\n")
    rejected = subprocess.run(argv, capture_output=True, text=True)
    assert rejected.returncode == 1
    assert "human modification" in rejected.stderr
    assert "人工注释" in files[1].read_text()
