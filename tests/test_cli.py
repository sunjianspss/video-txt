from __future__ import annotations

import pytest

from video_txt.cli import build_parser, main, resolve_provider_settings

SAMPLE = "1\n00:00:01,000 --> 00:00:03,000\nHello there\n\n2\n00:00:04,000 --> 00:00:06,000\nBye\n"


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("FFMPEG_PATH", "/bin/echo")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text(SAMPLE, encoding="utf-8")
    return video, subtitle


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


def test_legacy_transcribe_flags_still_parse():
    args = parse(
        [
            "transcribe",
            "in.mp4",
            "-o",
            "/out",
            "-m",
            "medium",
            "-l",
            "en",
            "--mode",
            "translate",
            "-f",
            "srt",
            "--dry-run",
        ]
    )
    assert args.whisper_model == "medium"
    assert args.language == "en"
    assert args.mode == "translate"
    assert args.format == "srt"
    assert args.dry_run


def test_legacy_translate_flags_still_parse():
    args = parse(
        [
            "translate",
            "in.srt",
            "-o",
            "out.srt",
            "--model",
            "deepseek-v4-flash",
            "--base-url",
            "https://api.deepseek.com",
            "--api-key-env",
            "DEEPSEEK_API_KEY",
            "--batch-chars",
            "2400",
            "--retries",
            "3",
            "--note",
            "casual",
            "--preserve-term",
            "MCP",
            "--overwrite",
            "--dry-run",
        ]
    )
    assert args.batch_chars == 2400
    assert args.preserve_term == ["MCP"]
    assert args.resume is True


def test_legacy_mux_flags_still_parse():
    args = parse(
        [
            "mux",
            "clip.mp4",
            "clip.srt",
            "--model",
            "deepseek-v4-flash",
            "--base-url",
            "https://api.deepseek.com",
            "--api-key-env",
            "DEEPSEEK_API_KEY",
            "--mux-mode",
            "hard",
            "--hard-subtitle-layout",
            "bottom-box",
            "--hard-subtitle-font-size",
            "36",
            "--hard-subtitle-margin-v",
            "40",
            "--hard-subtitle-box-height",
            "0.3",
            "--hard-subtitle-box-opacity",
            "0.5",
            "--language-code",
            "zho",
            "--default-subtitle",
            "--retranslate",
            "--overwrite-video",
            "--translation-debug-dir",
            "/tmp/dbg",
            "--dry-run",
        ]
    )
    assert args.mux_mode == "hard"
    assert args.hard_subtitle_layout == "bottom-box"
    assert args.hard_subtitle_font_size == 36
    assert str(args.debug_dir) == "/tmp/dbg"


def test_hard_subtitle_metrics_default_to_auto():
    args = parse(["mux", "clip.mp4", "clip.srt"])
    assert args.hard_subtitle_font_size is None
    assert args.hard_subtitle_margin_v is None


def test_provider_preset_fills_base_url_key_and_model():
    args = parse(["translate", "in.srt", "--provider", "deepseek"])
    assert resolve_provider_settings(args) == (
        "https://api.deepseek.com",
        "DEEPSEEK_API_KEY",
        "deepseek-v4-flash",
    )


def test_explicit_flags_win_over_the_provider_preset():
    args = parse(
        [
            "translate",
            "in.srt",
            "--provider",
            "deepseek",
            "--model",
            "custom",
            "--base-url",
            "https://other.test",
            "--api-key-env",
            "OTHER_KEY",
        ]
    )
    assert resolve_provider_settings(args) == ("https://other.test", "OTHER_KEY", "custom")


def test_missing_model_is_reported(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    args = parse(["translate", "in.srt"])
    with pytest.raises(SystemExit, match="Missing model name"):
        resolve_provider_settings(args)


def test_translate_dry_run_reports_the_plan(project, capsys):
    _, subtitle = project
    assert main(["translate", str(subtitle), "--provider", "deepseek", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "clip.zh.srt" in out
    assert "Model: deepseek-v4-flash" in out
    assert "Subtitle blocks: 2 (2 with text)" in out
    assert not (subtitle.parent / "clip.zh.srt").exists()


def test_translate_refuses_to_clobber_an_existing_file(project):
    _, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["translate", str(subtitle), "--provider", "deepseek"])


def test_mux_dry_run_prints_the_ffmpeg_command(project, capsys):
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")
    assert main(["mux", str(video), str(subtitle), "--provider", "deepseek", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Translate: skip, reusing" in out
    assert "-c:s mov_text" in out
    assert "clip.zh-subbed.mp4" in out


def test_run_dry_run_walks_every_stage(project, capsys):
    video, _ = project
    fresh = video.with_name("fresh.mp4")
    fresh.write_bytes(b"not really a video")
    assert main(["run", str(fresh), "--provider", "deepseek", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[1/3] Transcribe: running Whisper (dry run)" in out
    assert "-m whisper" in out or "mlx_whisper" in out
    assert "--output_format srt" in out or "--output-format srt" in out
    assert "[2/3] Translate" in out
    assert "Would translate" in out
    assert "[3/3] Mux" in out
    assert "fresh.zh-subbed.mp4" in out


def test_run_dry_run_reuses_an_existing_transcript(project, capsys):
    video, subtitle = project
    assert main(["run", str(video), "--provider", "deepseek", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"skip, reusing {subtitle}" in out


def test_dub_dry_run_prints_the_mux_command(project, capsys):
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")
    assert main(["dub", str(video), "--provider", "deepseek", "--keep-bgm", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "clip.zh-dubbed.mp4" in out
    assert "amix" in out
    assert "clip.dub-cache" in out


def test_unsupported_soft_container_fails_cleanly(project, capsys):
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")
    code = main(
        [
            "mux",
            str(video),
            str(subtitle),
            "--provider",
            "deepseek",
            "-o",
            str(video.with_suffix(".avi")),
            "--dry-run",
        ]
    )
    assert code == 1
    assert "not supported" in capsys.readouterr().err
