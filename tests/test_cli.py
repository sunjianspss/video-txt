from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_txt import cli as cli_module
from video_txt import transcribe as transcribe_module
from video_txt.cli import (
    COMMANDS,
    build_parser,
    build_translate_stage,
    build_translation_config,
    main,
    parse_clone_references,
    resolve_provider_settings,
)
from video_txt.diarize import (
    DiarizeOptions,
    SpeakerTurn,
    diarize_cache_key,
    save_turns,
    speakers_path,
)
from video_txt.translate import TranslationError

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


def test_audit_writes_a_machine_readable_report(tmp_path, capsys):
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:29,980\nThank you.\n",
        encoding="utf-8",
    )

    assert main(["audit", str(subtitle)]) == 1

    report_path = tmp_path / "clip.audit.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "video-txt.subtitle-audit"
    assert payload["version"] == 1
    assert payload["source"] == str(subtitle.resolve())
    assert payload["summary"]["full_window_hallucination"] == 1
    assert payload["findings"][0]["repair"] == "remove"
    output = capsys.readouterr().out
    assert "1 finding" in output
    assert str(report_path) in output


def test_translation_audit_command_reports_glossary_violations_without_rewriting(tmp_path, capsys):
    source = tmp_path / "story.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nWoody is here.\n", encoding="utf-8"
    )
    translation = tmp_path / "story.zh.srt"
    original = "1\n00:00:01,000 --> 00:00:02,000\n伍迪来了。\n"
    translation.write_text(original, encoding="utf-8")
    term_file = tmp_path / "project.terms.json"
    term_file.write_text(
        json.dumps(
            {
                "schema": "video-txt.terminology",
                "version": 1,
                "terms": [
                    {
                        "source": "Woody",
                        "target": "胡迪",
                        "match": "word",
                        "aliases": ["伍迪"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "audit-translation",
            str(source),
            str(translation),
            "--term-file",
            str(term_file),
        ]
    ) == 1

    assert translation.read_text(encoding="utf-8") == original
    report_path = tmp_path / "story.zh.translation-audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "video-txt.translation-audit"
    assert {finding["code"] for finding in report["findings"]} == {
        "glossary_alias",
        "glossary_target_missing",
    }
    assert "2 findings" in capsys.readouterr().out


def test_clean_writes_a_new_subtitle_and_records_safe_repairs(tmp_path, capsys):
    subtitle = tmp_path / "clip.srt"
    original = (
        "1\n00:00:01,000 --> 00:00:03,000\nI was\n\n"
        "2\n00:00:03,000 --> 00:00:03,000\ngoing.\n\n"
        "3\n00:00:04,000 --> 00:00:33,980\nThank you.\n\n"
        "4\n00:00:35,000 --> 00:00:37,000\nNext line.\n"
    )
    subtitle.write_text(original, encoding="utf-8")
    output = tmp_path / "clip.clean.srt"

    assert main(["clean", str(subtitle), "-o", str(output)]) == 0

    assert subtitle.read_text(encoding="utf-8") == original
    cleaned = output.read_text(encoding="utf-8")
    assert "I was going." in cleaned
    assert "Thank you." not in cleaned
    assert "Next line." in cleaned
    payload = json.loads((tmp_path / "clip.clean.audit.json").read_text(encoding="utf-8"))
    assert payload["repair"]["removed_count"] == 1
    assert payload["repair"]["merged_count"] == 1
    assert payload["post_repair"]["is_clean"] is True
    assert f"Cleaned: {output}" in capsys.readouterr().out


def test_clean_refuses_to_replace_its_source_even_with_overwrite(tmp_path):
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text(SAMPLE, encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["clean", str(subtitle), "-o", str(subtitle), "--overwrite"])

    assert subtitle.read_text(encoding="utf-8") == SAMPLE


def test_clean_preserves_an_existing_output_without_overwrite(tmp_path):
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text(SAMPLE, encoding="utf-8")
    output = tmp_path / "clean.srt"
    output.write_text("caller-owned", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["clean", str(subtitle), "-o", str(output)])

    assert output.read_text(encoding="utf-8") == "caller-owned"


def test_clean_dry_run_writes_nothing(tmp_path, capsys):
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:29,980\nThank you.\n",
        encoding="utf-8",
    )
    output = tmp_path / "clean.srt"

    assert main(["clean", str(subtitle), "-o", str(output), "--dry-run"]) == 0

    assert not output.exists()
    assert not (tmp_path / "clean.audit.json").exists()
    assert "Would clean" in capsys.readouterr().out


def test_clean_succeeds_with_review_only_warnings(tmp_path, capsys):
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:12,000\nOne two three four five six seven.\n",
        encoding="utf-8",
    )
    output = tmp_path / "clean.srt"

    assert main(["clean", str(subtitle), "-o", str(output)]) == 0

    assert output.is_file()
    assert "Manual review still needed" in capsys.readouterr().err


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


def test_subtitle_refinement_flag_is_available_to_transcribe_and_pipelines():
    standalone = parse(["transcribe", "in.mp4", "-f", "srt", "--refine-subtitles"])
    pipeline = parse(["run", "in.mp4", "--refine-subtitles"])

    assert standalone.refine_subtitles is True
    assert pipeline.refine_subtitles is True
    assert cli_module.build_transcribe_stage(pipeline).refine_subtitles is True


def test_audio_stream_override_is_available_to_transcribe_run_and_dub():
    standalone = parse(["transcribe", "in.mkv", "--audio-stream", "2"])
    run = parse(["run", "in.mkv", "--audio-stream", "3"])
    dub = parse(["dub", "in.mkv", "--audio-stream", "4"])

    assert standalone.audio_stream == 2
    assert cli_module.build_transcribe_stage(run).audio_stream == 3
    assert cli_module.build_transcribe_stage(dub).audio_stream == 4


def test_transcribe_passes_refinement_to_the_backend(project, monkeypatch):
    video, _ = project
    seen = []

    def capture(options, *, dry_run=False):
        seen.append((options, dry_run))
        return options.output_dir / f"{options.input_path.stem}.srt"

    monkeypatch.setattr(cli_module, "run_transcribe", capture)

    assert main(
        ["transcribe", str(video), "-f", "srt", "--refine-subtitles", "--dry-run"]
    ) == 0
    assert seen[0][0].refine_subtitles is True
    assert seen[0][1] is True


def test_transcribe_refinement_cli_writes_the_complete_artifact_pair(project, monkeypatch):
    video, subtitle = project
    monkeypatch.setattr(transcribe_module, "find_ffmpeg", lambda: Path("/usr/bin/ffmpeg"))
    monkeypatch.setattr(transcribe_module, "resolve_backend", lambda _value: "openai-whisper")

    def whisper(command, **_kwargs):
        if "-select_streams" in command:
            return SimpleNamespace(
                returncode=0,
                stdout='{"streams":[{"index":1,"codec_name":"aac","channels":2}]}',
                stderr="",
            )
        raw_output_dir = Path(command[command.index("--output_dir") + 1])
        (raw_output_dir / "clip.json").write_text(
            '{"language":"en","segments":[{"words":['
            '{"word":" A","start":0.0,"end":0.5},'
            '{"word":" complete.","start":0.5,"end":1.0}]}]}',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(transcribe_module.subprocess, "run", whisper)

    assert main(
        [
            "transcribe",
            str(video),
            "-f",
            "srt",
            "--refine-subtitles",
            "--skip-transcript-check",
        ]
    ) == 0
    assert "A complete." in subtitle.read_text(encoding="utf-8")
    assert (video.parent / "clip.words.json").is_file()


def test_transcribe_refinement_dry_run_shows_word_json_command(
    project, monkeypatch, capsys
):
    video, _ = project
    monkeypatch.setattr(transcribe_module, "resolve_backend", lambda _value: "openai-whisper")

    assert main(
        ["transcribe", str(video), "-f", "srt", "--refine-subtitles", "--dry-run"]
    ) == 0

    command = capsys.readouterr().out
    assert "--output_format json" in command
    assert "--word_timestamps True" in command


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


@pytest.mark.parametrize(
    "argv",
    [
        ["translate", "in.srt"],
        ["run", "clip.mp4"],
        ["dub", "clip.mp4"],
    ],
)
def test_translation_workflows_accept_one_project_terminology_file(argv):
    args = parse([*argv, "--term-file", "project.terms.json", "--dry-run"])

    assert args.term_file == Path("project.terms.json")


def test_translation_config_loads_the_term_file_and_uses_its_source_language(tmp_path):
    term_file = tmp_path / "project.terms.json"
    term_file.write_text(
        json.dumps(
            {
                "schema": "video-txt.terminology",
                "version": 1,
                "source_language": "en",
                "target_language": "zh-CN",
                "terms": [{"source": "Woody", "target": "胡迪", "match": "word"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = parse(
        [
            "translate",
            "in.srt",
            "--provider",
            "deepseek",
            "--term-file",
            str(term_file),
            "--dry-run",
        ]
    )

    config = build_translation_config(
        args,
        build_parser(),
        require_key=False,
        discover_model=False,
    )

    assert config.source_language == "en"
    assert config.terminology is not None
    assert config.terminology.terms[0].target == "胡迪"


def test_pipeline_stage_loads_terminology_before_deciding_to_reuse_a_translation(tmp_path):
    term_file = tmp_path / "project.terms.json"
    term_file.write_text(
        json.dumps(
            {
                "schema": "video-txt.terminology",
                "version": 1,
                "terms": [{"source": "Woody", "target": "胡迪"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = parse(
        [
            "run",
            "clip.mp4",
            "--provider",
            "deepseek",
            "--term-file",
            str(term_file),
            "--dry-run",
        ]
    )

    stage = build_translate_stage(
        args,
        build_parser(),
        require_key=False,
        output_path=None,
    )

    assert stage.config.terminology is not None
    assert stage.config.terminology.terms[0].target == "胡迪"


def test_translation_refuses_a_term_file_for_another_target_language(project, tmp_path, capsys):
    _, subtitle = project
    term_file = tmp_path / "japanese.terms.json"
    term_file.write_text(
        json.dumps(
            {
                "schema": "video-txt.terminology",
                "version": 1,
                "target_language": "Japanese",
                "terms": [{"source": "Woody", "target": "ウッディ"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "translate",
            str(subtitle),
            "--provider",
            "deepseek",
            "--term-file",
            str(term_file),
            "--dry-run",
        ]
    ) == 1

    assert "target language" in capsys.readouterr().err.lower()


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


def test_lmstudio_preset_points_at_the_local_server():
    args = parse(["translate", "in.srt", "--provider", "lmstudio", "--model", "qwen3"])
    assert resolve_provider_settings(args, build_parser()) == (
        "http://localhost:1234/v1",
        "LMSTUDIO_API_KEY",
        "qwen3",
    )


def test_lmstudio_asks_the_server_which_model_is_loaded(monkeypatch):
    monkeypatch.setattr(cli_module, "loaded_local_model", lambda base_url: "loaded-qwen3")
    args = parse(["translate", "in.srt", "--provider", "lmstudio"])
    assert resolve_provider_settings(args, build_parser())[2] == "loaded-qwen3"


def test_lmstudio_with_nothing_loaded_says_so(monkeypatch, capsys):
    def refuse(base_url: str) -> str:
        raise TranslationError("http://localhost:1234 has no chat model loaded.")

    monkeypatch.setattr(cli_module, "loaded_local_model", refuse)
    args = parse(["translate", "in.srt", "--provider", "lmstudio"])
    with pytest.raises(SystemExit):
        resolve_provider_settings(args, build_parser())

    assert "no chat model loaded" in capsys.readouterr().err


def test_lmstudio_dry_run_does_not_contact_the_local_server(project, monkeypatch, capsys):
    _, subtitle = project
    monkeypatch.setattr(
        cli_module,
        "loaded_local_model",
        lambda _base_url: pytest.fail("a dry run must not contact LM Studio"),
    )

    assert main(["translate", str(subtitle), "--provider", "lmstudio", "--dry-run"]) == 0

    assert "Model: <loaded-local-model>" in capsys.readouterr().out


def test_lmstudio_runs_without_an_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)
    args = parse(
        [
            "translate",
            "in.srt",
            "--provider",
            "lmstudio",
            "--model",
            "qwen3",
            "--secrets-file",
            str(tmp_path / "absent"),
        ]
    )
    config = build_translation_config(args, build_parser(), require_key=True)
    assert config.api_key == ""
    assert config.model == "qwen3"


def effort_for(tmp_path, argv: list[str]) -> str:
    args = parse(["translate", "in.srt", *argv, "--secrets-file", str(tmp_path / "absent")])
    return build_translation_config(args, build_parser(), require_key=False).reasoning_effort


def thinking_for(tmp_path, argv: list[str]) -> str:
    args = parse(["translate", "in.srt", *argv, "--secrets-file", str(tmp_path / "absent")])
    return build_translation_config(args, build_parser(), require_key=False).thinking


def test_lmstudio_turns_reasoning_off(tmp_path):
    """Local reasoning models spend minutes deliberating over a subtitle line."""
    assert effort_for(tmp_path, ["--provider", "lmstudio", "--model", "q"]) == "none"


def test_lmstudio_asks_for_smaller_batches_than_a_cloud_provider(tmp_path):
    def batch_chars_for(argv: list[str]) -> int:
        args = parse(["translate", "in.srt", *argv, "--secrets-file", str(tmp_path / "absent")])
        return build_translation_config(args, build_parser(), require_key=False).batch_chars

    local = ["--provider", "lmstudio", "--model", "q"]
    assert batch_chars_for(local) == 800
    assert batch_chars_for(["--provider", "deepseek"]) == 3200
    assert batch_chars_for([*local, "--batch-chars", "2000"]) == 2000


def test_other_providers_say_nothing_about_reasoning(tmp_path):
    assert effort_for(tmp_path, ["--provider", "deepseek"]) == "auto"


def test_deepseek_translation_disables_thinking_by_default(tmp_path):
    assert thinking_for(tmp_path, ["--provider", "deepseek"]) == "disabled"


def test_reasoning_effort_flag_wins_over_the_preset(tmp_path):
    local = ["--provider", "lmstudio", "--model", "q"]
    assert effort_for(tmp_path, [*local, "--reasoning-effort", "high"]) == "high"
    assert effort_for(tmp_path, [*local, "--reasoning-effort", "auto"]) == "auto"


def test_deepseek_maps_reasoning_controls_to_its_thinking_api(tmp_path):
    disabled = ["--provider", "deepseek", "--reasoning-effort", "none"]
    enabled = ["--provider", "deepseek", "--reasoning-effort", "max"]

    assert thinking_for(tmp_path, disabled) == "disabled"
    assert effort_for(tmp_path, disabled) == "auto"
    assert thinking_for(tmp_path, enabled) == "enabled"
    assert effort_for(tmp_path, enabled) == "max"


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


def test_run_places_an_external_subtitle_translation_in_the_output_directory(project, capsys):
    video, subtitle = project
    output_dir = video.parent / "intermediates"

    assert (
        main(
            [
                "run",
                str(video),
                "--subtitle",
                str(subtitle),
                "--output-dir",
                str(output_dir),
                "--provider",
                "deepseek",
                "--dry-run",
            ]
        )
        == 0
    )

    assert f"{output_dir / 'clip.zh.srt'}" in capsys.readouterr().out


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

    assert main(["run", str(video), "--dry-run"]) == 0

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


def test_dub_rejects_an_audio_output_that_would_replace_the_source_video(project, capsys):
    video, _subtitle = project

    code = main(
        [
            "dub",
            str(video),
            "--provider",
            "deepseek",
            "--audio-output",
            str(video),
            "--dry-run",
        ]
    )

    assert code == 1
    assert "--audio-output conflicts with the source video" in capsys.readouterr().err


def test_dub_rejects_an_existing_caller_owned_audio_output(project, capsys):
    video, _subtitle = project
    audio_output = video.with_name("my-recording.wav")
    audio_output.write_bytes(b"caller-owned")

    code = main(
        [
            "dub",
            str(video),
            "--provider",
            "deepseek",
            "--audio-output",
            str(audio_output),
            "--dry-run",
        ]
    )

    assert code == 1
    assert "Audio output already exists" in capsys.readouterr().err
    assert audio_output.read_bytes() == b"caller-owned"


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


def test_dub_dry_run_reports_the_separated_background(project, capsys):
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")
    argv = ["dub", str(video), "--provider", "deepseek", "--separate-bgm", "--dry-run"]

    assert main(argv) == 0

    out = capsys.readouterr().out
    assert "minus its voices" in out
    assert "no_vocals.flac" in out  # mixed from the cached track, not from 0:a


def test_dub_asks_for_a_spoken_translation_and_run_does_not(project, monkeypatch):
    """The dub's lines are read aloud, so only they should be written like speech."""
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")
    asked: list[bool] = []
    original = cli_module.build_translate_stage

    def spy(args, parser, **kwargs):
        asked.append(kwargs.get("spoken", False))
        return original(args, parser, **kwargs)

    monkeypatch.setattr(cli_module, "build_translate_stage", spy)

    assert main(["dub", str(video), "--provider", "deepseek", "--dry-run"]) == 0
    assert main(["run", str(video), "--provider", "deepseek", "--dry-run"]) == 0
    assert asked == [True, False]


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


SPEAKER_TURNS = [
    SpeakerTurn(0.0, 3.5, "SPEAKER_00"),
    SpeakerTurn(3.5, 6.0, "SPEAKER_01"),
]


@pytest.fixture
def diarized(project):
    video, subtitle = project
    (subtitle.parent / "clip.zh.srt").write_text(SAMPLE, encoding="utf-8")
    options = DiarizeOptions()
    save_turns(
        speakers_path(video),
        SPEAKER_TURNS,
        model=options.model,
        cache_key=diarize_cache_key(video, options),
    )
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


def test_the_cloning_engine_is_told_the_language_in_two_letters(project):
    """The container subtitle code is three letters (zho); the IndexTTS lang
    map wants the two-letter family (zh). Mixing them up cost a model load."""
    video, _ = project
    args = parse(["dub", str(video), "--tts-engine", "index-tts"])

    options = cli_module.build_clone_options(args, build_parser())

    assert options.lang == "zh"


def test_a_voice_per_speaker_needs_diarization(project, capsys):
    video, _ = project
    with pytest.raises(SystemExit):
        main(["dub", str(video), "--speaker-voice", "SPEAKER_01=zh-CN-YunxiNeural", "--dry-run"])

    assert "--speaker-voice needs --diarize" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--speakers", "--min-speakers", "--max-speakers"])
def test_speaker_count_constraints_need_diarization(project, capsys, flag):
    video, _ = project

    with pytest.raises(SystemExit):
        main(["dub", str(video), flag, "2", "--dry-run"])

    assert f"{flag} needs --diarize" in capsys.readouterr().err


def test_rediarize_needs_diarization(project, capsys):
    video, _ = project

    with pytest.raises(SystemExit):
        main(["dub", str(video), "--rediarize", "--dry-run"])

    assert "--rediarize needs --diarize" in capsys.readouterr().err


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
