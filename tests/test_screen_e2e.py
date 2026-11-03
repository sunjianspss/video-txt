"""Real CLI -> timestamped OCR -> reviewed plan + speech -> burned video."""

from __future__ import annotations

import json
import subprocess
import sys

from video_txt.media import find_ffmpeg
from video_txt.screen import sha256


def test_real_ocr_resume_speech_merge_and_hard_subtitles(tmp_path):
    ffmpeg = find_ffmpeg(need_subtitles_filter=True)
    source = tmp_path / "clip.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=640x360:r=12:d=6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=6",
            "-vf",
            "drawtext=text='Subscriptions 3048':fontsize=38:fontcolor=black:x=20:y=100:"
            "enable='lt(t,2)',drawtext=text='Subscriptions 3100':fontsize=38:"
            "fontcolor=black:x=20:y=100:enable='gte(t,2)'",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    before_source = sha256(source)
    folder = tmp_path / "screen"
    cli = [sys.executable, "-m", "video_txt.cli", "screen"]
    scan = [*cli, "scan", str(source), "-o", str(folder), "--interval", "1"]
    subprocess.run([*scan, "--dry-run"], check=True, capture_output=True)
    assert not folder.exists()
    subprocess.run(scan, check=True, capture_output=True)
    plan = folder / "plan.json"
    raw = json.loads(plan.read_text())
    assert any("3100" in c["source"] for c in raw["captions"])
    assert any("3048" in v for c in raw["captions"] for v in c["variants"])
    # A known bad draft must fail before rendering or creating an output movie.
    output = tmp_path / "translated.mp4"
    command = [
        *cli,
        "render",
        str(plan),
        "-o",
        str(output),
        "--ffmpeg",
        ffmpeg,
        "--hard-subtitle-layout",
        "normal",
    ]
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode == 1 and "status=ready" in failed.stderr
    assert not output.exists()
    # Human/agent review selects the stable value and a readable 2-second slot.
    caption = next(c for c in raw["captions"] if "3100" in c["source"])
    caption.update(
        start=2, end=4, source="Subscriptions 3100", translation="订阅数为3100", status="ready"
    )
    raw["captions"] = [caption]
    plan.write_text(json.dumps(raw, ensure_ascii=False))
    reviewed = plan.read_bytes()
    frame = folder / "frames" / "000002.jpg"
    before_frame = frame.stat().st_mtime_ns
    subprocess.run(scan, check=True, capture_output=True)
    assert plan.read_bytes() == reviewed and frame.stat().st_mtime_ns == before_frame
    mismatch = subprocess.run([*scan, "--interval", "0.5"], capture_output=True, text=True)
    assert mismatch.returncode == 1 and "different inputs" in mismatch.stderr
    english, chinese = tmp_path / "speech.srt", tmp_path / "speech.zh.srt"
    english.write_text("1\n00:00:04,100 --> 00:00:05,900\nVery convenient.\n")
    chinese.write_text("1\n00:00:04,100 --> 00:00:05,900\n用起来很方便。\n")
    command.extend(["--speech-source", str(english), "--speech-translation", str(chinese)])
    subprocess.run([*command, "--dry-run"], check=True, capture_output=True)
    assert not output.exists() and not output.with_suffix(".screen-render.json").exists()
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads(output.with_suffix(".screen-render.json").read_text())
    assert report["cue_count"] == 2
    assert "订阅数为3100" in report["translation_srt"]
    assert "用起来很方便。" in report["translation_srt"]

    def command_output(path, extra):
        return subprocess.check_output([ffmpeg, "-v", "error", "-i", str(path), *extra])

    audio = ["-map", "0:a:0", "-c:a", "copy", "-f", "hash", "-hash", "sha256", "-"]
    assert command_output(source, audio) == command_output(output, audio)
    # Disable soft subtitles: the changed pixels must be in decoded video frames.
    pixels = [
        "-ss",
        "3",
        "-frames:v",
        "1",
        "-vf",
        "crop=640:80:0:280",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    # With no background band, the initially white crop must gain glyphs.
    # Existing ASS uses a translucent black outline (&H99), which is grey
    # (~153) over white. Detect actual glyph ink, not only opaque black pixels.
    assert sum(p < 230 for p in command_output(source, pixels)) == 0
    assert sum(p < 230 for p in command_output(output, pixels)) > 100
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )
    before_output = sha256(output)
    collision = subprocess.run(command, capture_output=True, text=True)
    assert collision.returncode == 1
    assert sha256(output) == before_output and sha256(source) == before_source
    forbidden = subprocess.run(
        [*command, "-o", str(source), "--overwrite-video"], capture_output=True, text=True
    )
    assert forbidden.returncode == 1 and "cannot replace" in forbidden.stderr
    assert sha256(source) == before_source
