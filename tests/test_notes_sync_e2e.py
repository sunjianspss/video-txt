"""Real CLI/media/filesystem; deterministic model substitute checks CLI isolation."""

import json
import os
import subprocess
import sys


def test_cli_queue_publishes_cited_summary_and_reports_bad_alignment(tmp_path):
    media, vault, state, bin_dir = (tmp_path / p for p in ("media", "vault", "state", "bin"))
    for p in (media, vault, bin_dir):
        p.mkdir()
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
            str(media / "clip.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    source, translated = media / "clip.en.srt", media / "clip.zh.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:07,000\nThis button is blue.\n")
    translated.write_text("1\n00:00:01,000 --> 00:00:07,000\n这个按钮是蓝色的。\n")
    fake = bin_dir / "claude"
    fake.write_text(
        f"#!{sys.executable}\n"
        + r"""import json, re, sys
assert sys.argv[sys.argv.index('--tools') + 1] == ''
assert '--strict-mcp-config' in sys.argv and '--no-session-persistence' in sys.argv
assert sys.argv[sys.argv.index('--setting-sources') + 1] == ''
blocks = list(dict.fromkeys(re.findall(r"\^(p-[a-f0-9]+)", sys.stdin.read())))
print(json.dumps({"question": "讨论什么？", "overview": "这个按钮的颜色。",
 "claims": [{"text": "这个按钮是蓝色的。", "block": b} for b in blocks],
 "analysis": ["不能泛化。"], "questions": ["颜色需核实。"], "related": []}))
"""
    )
    fake.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
    argv = [
        sys.executable,
        "-m",
        "video_txt.cli",
        "sync-notes",
        "--vault",
        str(vault),
        "--source-dir",
        str(media),
        "--state-dir",
        str(state),
        "--min-age",
        "0",
    ]

    def run(*extra):
        result = subprocess.run([*argv, *extra], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    assert run()["counts"] == {"historical": 1}

    def saved():
        return json.loads((state / "state.json").read_text())

    job_id = next(iter(saved()["jobs"]))
    run("--action", "enqueue", "--job-id", job_id)
    assert run()["processed"] == [job_id]
    job = saved()["jobs"][job_id]
    summary = vault / job["summary"]
    assert "#^p-" in summary.read_text() and 'status: "needs-review"' in summary.read_text()
    original = {p: p.read_bytes() for p in vault.rglob("*.md")}
    assert run()["processed"] == []
    assert original == {p: p.read_bytes() for p in original}
    translated.write_text(translated.read_text().replace("00:00:07,000", "00:00:09,000"))
    assert run()["failed"] == [job_id]
    assert "do not align" in saved()["jobs"][job_id]["error"]
    assert original == {p: p.read_bytes() for p in original}
    assert len(list(state.rglob("summaries/*.json"))) == 1
