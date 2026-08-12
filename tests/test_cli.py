from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_txt import cli as cli_module
from video_txt.cli import (
    COMMANDS,
    build_parser,
    main,
    parse_clone_references,
    resolve_provider_settings,
)

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
    assert resolve_provider_settings(args, build_parser()) == (
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
    assert resolve_provider_settings(args, build_parser()) == (
        "https://other.test",
        "OTHER_KEY",
        "custom",
    )


def test_missing_model_is_reported(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    args = parse(["translate", "in.srt"])
    with pytest.raises(SystemExit):
        resolve_provider_settings(args, build_parser())

    assert "Missing model name" in capsys.readouterr().err


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


def test_run_reuses_an_existing_translation_without_model_or_key(project, monkeypatch, capsys):
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        cli_module,
        "resolve_api_key",
        lambda *_args, **_kwargs: pytest.fail("a skipped translation must not load a key"),
    )

    assert main(["run", str(video)]) == 0

    assert "Translate: skip, reusing" in capsys.readouterr().out


def test_translation_help_does_not_publish_the_default_credentials_path(capsys):
    with pytest.raises(SystemExit):
        main(["translate", "--help"])

    assert "~/.secrets" not in capsys.readouterr().out


def test_dub_dry_run_prints_the_mux_command(project, capsys):
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")
    assert main(["dub", str(video), "--provider", "deepseek", "--keep-bgm", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "clip.zh-dubbed.mp4" in out
    assert "amix" in out
    assert "clip.dub-cache" in out


def test_dub_speaks_whole_sentences_unless_told_otherwise(project, capsys):
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")

    assert parse(["dub", "clip.mp4"]).voice_unit == "sentence"
    assert main(["dub", str(video), "--provider", "deepseek", "--dry-run"]) == 0
    assert "one per sentence" in capsys.readouterr().out

    argv = ["dub", str(video), "--provider", "deepseek", "--voice-unit", "line", "--dry-run"]
    assert main(argv) == 0
    assert "one per subtitle line" in capsys.readouterr().out


def test_dub_duration_fit_is_off_unless_asked_for():
    assert parse(["dub", "clip.mp4"]).fit_duration is False
    args = parse(["dub", "clip.mp4", "--fit-duration", "--fit-tempo", "1.4", "--fit-rounds", "3"])
    assert (args.fit_duration, args.fit_tempo, args.fit_rounds) == (True, 1.4, 3)


def test_dub_dry_run_reports_the_duration_fit(project, capsys):
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")
    argv = ["dub", str(video), "--provider", "deepseek", "--fit-duration", "--dry-run"]
    assert main(argv) == 0
    assert "Duration-aware rewrite: on" in capsys.readouterr().out


def test_dub_rejects_a_fit_tempo_below_normal_speed(project):
    video, _ = project
    with pytest.raises(SystemExit):
        main(["dub", str(video), "--fit-duration", "--fit-tempo", "0.9", "--dry-run"])


def test_dub_rejects_a_fit_tempo_the_renderer_cannot_reach(project, capsys):
    video, _ = project
    argv = ["dub", str(video), "--fit-duration", "--fit-tempo", "1.6", "--max-atempo", "1.2"]
    with pytest.raises(SystemExit):
        main([*argv, "--dry-run"])
    assert "Lower --fit-tempo or raise --max-atempo" in capsys.readouterr().err


SPEAKER_TURNS = {
    "version": 1,
    "model": "pyannote/speaker-diarization-3.1",
    "turns": [
        {"start": 0.0, "end": 3.5, "speaker": "SPEAKER_00"},
        {"start": 3.5, "end": 6.0, "speaker": "SPEAKER_01"},
    ],
}


@pytest.fixture
def diarized(project):
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")
    (subtitle.parent / "clip.speakers.json").write_text(json.dumps(SPEAKER_TURNS), encoding="utf-8")
    return video


def test_dub_dry_run_gives_each_speaker_a_voice(diarized, capsys):
    assert main(["dub", str(diarized), "--provider", "deepseek", "--diarize", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "[1/4] Diarize: skip, reusing" in out
    assert "[4/4] Dub" in out
    assert (
        "Speakers: 2 (SPEAKER_00 -> zh-CN-XiaoxiaoNeural, SPEAKER_01 -> zh-CN-YunxiNeural)" in out
    )


def test_dub_dry_run_takes_the_voice_named_for_a_speaker(diarized, capsys):
    argv = ["dub", str(diarized), "--provider", "deepseek", "--diarize", "--dry-run"]
    assert main([*argv, "--speaker-voice", "1=zh-CN-YunyangNeural"]) == 0

    assert "SPEAKER_01 -> zh-CN-YunyangNeural" in capsys.readouterr().out


def test_dub_refuses_a_voice_for_a_speaker_nobody_found(diarized, capsys):
    argv = ["dub", str(diarized), "--provider", "deepseek", "--diarize", "--dry-run"]

    assert main([*argv, "--speaker-voice", "SPEAKER_09=zh-CN-YunyangNeural"]) == 1

    assert "SPEAKER_09" in capsys.readouterr().err


def test_dub_without_diarize_stays_a_three_stage_run(project, capsys):
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")

    assert main(["dub", str(video), "--provider", "deepseek", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "[1/3] Transcribe" in out
    assert "Speakers:" not in out


def test_dub_dry_run_reports_the_cloning_model(diarized, capsys):
    argv = ["dub", str(diarized), "--provider", "deepseek", "--tts-engine", "f5-tts", "--dry-run"]
    assert main(argv) == 0

    out = capsys.readouterr().out
    assert "Dub: f5-tts cloning the original voice" in out
    assert "Voice cloning: f5-tts (F5TTS_v1_Base)" in out


def test_cosyvoice_says_which_flag_it_is_missing(diarized, capsys):
    argv = ["dub", str(diarized), "--provider", "deepseek", "--tts-engine", "cosyvoice"]
    assert main([*argv, "--dry-run"]) == 1

    assert "--clone-model" in capsys.readouterr().err


def test_a_voice_per_speaker_needs_diarization(project, capsys):
    video, _ = project
    with pytest.raises(SystemExit):
        main(["dub", str(video), "--speaker-voice", "SPEAKER_01=zh-CN-YunxiNeural", "--dry-run"])

    assert "--speaker-voice needs --diarize" in capsys.readouterr().err


def test_a_reference_clip_named_for_a_speaker_needs_diarization(project, capsys):
    video, _ = project
    argv = ["dub", str(video), "--tts-engine", "f5-tts", "--dry-run"]
    with pytest.raises(SystemExit):
        main([*argv, "--clone-reference", "SPEAKER_01=/voices/guest.wav"])

    assert "--clone-reference /voices/guest.wav" in capsys.readouterr().err


def test_cloning_flags_are_refused_by_the_online_engines(project, capsys):
    video, _ = project
    with pytest.raises(SystemExit):
        main(["dub", str(video), "--clone-model", "F5TTS_v1_Base", "--dry-run"])

    assert "only applies to a cloning engine" in capsys.readouterr().err


def test_a_voice_name_is_refused_by_the_engine_that_cannot_say_it(project, capsys):
    """say has voices of its own; an edge-tts name only fails once synthesis starts."""
    video, _ = project
    with pytest.raises(SystemExit):
        main(["dub", str(video), "--tts-engine", "say", "--voice", "zh-CN-YunxiNeural"])

    assert "say -v '?'" in capsys.readouterr().err


def test_a_speakers_voice_is_refused_by_the_engine_that_cannot_say_it(diarized, capsys):
    """The same name reaches the same engine whichever flag carried it."""
    argv = ["dub", str(diarized), "--tts-engine", "say", "--diarize", "--dry-run"]
    with pytest.raises(SystemExit):
        main([*argv, "--speaker-voice", "SPEAKER_01=zh-CN-YunxiNeural"])

    assert "--speaker-voice zh-CN-YunxiNeural is an edge-tts voice" in capsys.readouterr().err


def test_the_offline_engine_runs_without_being_told_a_voice(project, capsys):
    """The default --voice is an edge-tts name that say answers to with one of its own,
    so complaining about it turns the offline fallback away for a flag nobody passed."""
    video, _ = project
    argv = ["dub", str(video), "--provider", "deepseek", "--dry-run"]

    assert main([*argv, "--tts-engine", "say"]) == 0

    assert "voice Tingting" in capsys.readouterr().out


def test_naming_a_voice_is_refused_when_the_voice_is_being_cloned(project, capsys):
    """Cloning takes the voice from the clip, so --voice would go nowhere."""
    video, _ = project
    argv = ["dub", str(video), "--tts-engine", "f5-tts", "--dry-run"]
    with pytest.raises(SystemExit):
        main([*argv, "--voice", "zh-CN-YunxiNeural"])

    assert "--voice has nothing to name" in capsys.readouterr().err


def test_a_voice_per_speaker_is_refused_when_the_voices_are_being_cloned(diarized, capsys):
    argv = ["dub", str(diarized), "--tts-engine", "f5-tts", "--diarize", "--dry-run"]
    with pytest.raises(SystemExit):
        main([*argv, "--speaker-voice", "SPEAKER_01=zh-CN-YunxiNeural"])

    assert "--speaker-voice has nothing to name" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--rate", "10", "signed percentage"),
        ("--max-atempo", "0.8", "--max-atempo must be 1.0 or greater"),
        ("--bgm-volume", "1.5", "--bgm-volume must be between 0 and 1"),
        ("--tts-concurrency", "0", "--tts-concurrency must be 1 or greater"),
    ],
)
def test_a_setting_out_of_range_is_refused_rather_than_pulled_back_in(
    project, capsys, flag, value, message
):
    video, _ = project
    with pytest.raises(SystemExit):
        main(["dub", str(video), flag, value, "--dry-run"])

    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--batch-chars", "0", "--batch-chars must be 1 or greater"),
        ("--retries", "-1", "--retries must be 0 or greater"),
        ("--concurrency", "0", "--concurrency must be 1 or greater"),
        ("--context-cues", "-1", "--context-cues must be 0 or greater"),
    ],
)
def test_a_translation_setting_out_of_range_is_refused(project, capsys, flag, value, message):
    _, subtitle = project
    with pytest.raises(SystemExit):
        main(["translate", str(subtitle), "--provider", "deepseek", flag, value, "--dry-run"])

    assert message in capsys.readouterr().err


@pytest.mark.parametrize("command", ["run", "dub"])
@pytest.mark.parametrize(
    "flag", ["--subtitle", "--subtitle-output", "--output-dir", "-o", "--ffmpeg"]
)
def test_the_two_pipelines_take_the_same_paths(command, flag):
    """run and dub differ in what they produce, not in how you tell them where to put it."""
    parse([command, "clip.mp4", flag, "somewhere"])


def test_every_subcommand_knows_what_to_run():
    """Arguments are declared apart from the handlers; an unwired one only fails at runtime."""
    for command in COMMANDS:
        assert callable(command.handler), command.name


def test_a_reference_clip_can_be_named_per_speaker_or_on_its_own():
    args = parse(
        [
            "dub",
            "clip.mp4",
            "--tts-engine",
            "f5-tts",
            "--clone-reference",
            "/voices/host.wav",
            "--clone-reference",
            "SPEAKER_01=/voices/guest.wav",
        ]
    )
    references = parse_clone_references(build_parser(), args.clone_references)

    assert references[""] == Path("/voices/host.wav")
    assert references["SPEAKER_01"] == Path("/voices/guest.wav")


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
