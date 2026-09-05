from __future__ import annotations

import hashlib
import json

import pytest

from video_txt import cli as cli_module
from video_txt.cli import main
from video_txt.project import build_project_state, inspect_project, write_json
from video_txt.terminology import translation_audit_path_for


def test_project_init_refuses_what_the_dub_workflow_would_refuse(tmp_path):
    """`project run` replays a dub project as the dub command. A combination that
    command rejects has to fail here, not one stage and one config file later."""
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"movie-content")
    project_file = tmp_path / "project.video-txt.json"

    with pytest.raises(SystemExit):
        main(
            [
                "project",
                "init",
                str(video),
                "-o",
                str(project_file),
                "--workflow",
                "dub",
                "--tts-engine",
                "index-tts",
                "--voice",
                "zh-CN-YunxiNeural",
            ]
        )

    assert not project_file.exists()


def test_project_init_refuses_what_the_run_workflow_would_refuse(tmp_path):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"movie-content")
    project_file = tmp_path / "project.video-txt.json"

    with pytest.raises(SystemExit):
        main(
            [
                "project",
                "init",
                str(video),
                "-o",
                str(project_file),
                "--hard-subtitle-box-opacity",
                "5",
            ]
        )

    assert not project_file.exists()


def test_project_init_writes_relative_content_fingerprints_without_secrets(
    tmp_path, monkeypatch
):
    project_dir = tmp_path / "toy-story"
    media_dir = project_dir / "media"
    media_dir.mkdir(parents=True)
    video = media_dir / "movie.mkv"
    video.write_bytes(b"movie-content")
    term_file = project_dir / "project.terms.json"
    term_file.write_text(
        json.dumps(
            {
                "schema": "video-txt.terminology",
                "version": 1,
                "terms": [],
            }
        ),
        encoding="utf-8",
    )
    project_file = project_dir / "project.video-txt.json"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-written")

    assert (
        main(
            [
                "project",
                "init",
                str(video),
                "-o",
                str(project_file),
                "--audio-stream",
                "2",
                "--language",
                "en",
                "--provider",
                "deepseek",
                "--term-file",
                str(term_file),
                "--video-output",
                "outputs/movie.zh-subbed.mkv",
            ]
        )
        == 0
    )

    raw = project_file.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["schema"] == "video-txt.project"
    assert payload["version"] == 1
    assert payload["video"] == {
        "path": "media/movie.mkv",
        "sha256": hashlib.sha256(b"movie-content").hexdigest(),
    }
    assert payload["transcription"]["audio_stream"] == 2
    assert payload["transcription"]["language"] == "en"
    assert payload["translation"]["provider"] == "deepseek"
    assert payload["translation"]["model"] == "deepseek-v4-flash"
    assert payload["translation"]["terminology"] == {
        "path": "project.terms.json",
        "sha256": hashlib.sha256(term_file.read_bytes()).hexdigest(),
    }
    assert payload["paths"]["video_output"] == "outputs/movie.zh-subbed.mkv"
    assert "must-not-be-written" not in raw
    assert not (project_dir / ".video-txt" / "state.json").exists()


def test_project_status_resolves_paths_from_the_project_file(tmp_path, monkeypatch, capsys):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    video = project_dir / "movie.mkv"
    video.write_bytes(b"movie")
    project_file = project_dir / "project.video-txt.json"
    assert (
        main(
            [
                "project",
                "init",
                str(video),
                "-o",
                str(project_file),
                "--provider",
                "deepseek",
            ]
        )
        == 0
    )
    capsys.readouterr()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert main(["project", "status", str(project_file)]) == 1

    output = capsys.readouterr().out
    assert f"Video: {video}" in output
    assert "transcribe: stale — artifact missing" in output
    assert "translate: stale — upstream transcribe is stale" in output
    assert "mux: stale — upstream translate is stale" in output


def test_project_status_marks_only_translation_and_downstream_after_terms_change(
    tmp_path, capsys
):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"movie")
    terms = tmp_path / "project.terms.json"
    terms.write_text(
        '{"schema":"video-txt.terminology","version":1,"terms":[]}',
        encoding="utf-8",
    )
    project_file = tmp_path / "project.video-txt.json"
    assert (
        main(
            [
                "project",
                "init",
                str(video),
                "-o",
                str(project_file),
                "--provider",
                "deepseek",
                "--term-file",
                str(terms),
            ]
        )
        == 0
    )
    inspection = inspect_project(project_file)
    inspection.paths["source_subtitle"].write_text("source", encoding="utf-8")
    inspection.paths["translated_subtitle"].write_text("translation", encoding="utf-8")
    inspection.paths["translation_audit"].write_text("{}", encoding="utf-8")
    inspection.paths["video_output"].write_bytes(b"muxed")
    inspection = inspect_project(project_file)
    write_json(
        inspection.state_path,
        build_project_state(
            inspection,
            audio_selection={
                "requested_stream": None,
                "selected_stream": 2,
                "reason": "title 'Eng' matches en",
            },
        ),
    )
    assert main(["project", "status", str(project_file)]) == 0
    capsys.readouterr()

    terms.write_text(
        '{"schema":"video-txt.terminology","version":1,"terms":[],"note":"changed"}',
        encoding="utf-8",
    )

    assert main(["project", "status", str(project_file)]) == 1
    output = capsys.readouterr().out
    assert "Selected audio stream: 2 — title 'Eng' matches en" in output
    assert "transcribe: current" in output
    assert "translate: stale — terminology changed" in output
    assert "mux: stale — upstream translate is stale" in output
    assert "tts: disabled for this workflow" in output


def test_project_status_marks_only_mux_after_subtitle_style_change(tmp_path, capsys):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"movie")
    project_file = tmp_path / "project.video-txt.json"
    assert (
        main(
            [
                "project",
                "init",
                str(video),
                "-o",
                str(project_file),
                "--provider",
                "deepseek",
                "--mux-mode",
                "hard",
            ]
        )
        == 0
    )
    inspection = inspect_project(project_file)
    inspection.paths["source_subtitle"].write_text("source", encoding="utf-8")
    inspection.paths["translated_subtitle"].write_text("translation", encoding="utf-8")
    inspection.paths["translation_audit"].write_text("{}", encoding="utf-8")
    inspection.paths["video_output"].write_bytes(b"muxed")
    inspection = inspect_project(project_file)
    write_json(inspection.state_path, build_project_state(inspection))
    assert main(["project", "status", str(project_file)]) == 0
    capsys.readouterr()

    payload = json.loads(project_file.read_text(encoding="utf-8"))
    payload["mux"]["hard_subtitle_font"] = "Noto Sans CJK SC"
    write_json(project_file, payload)

    assert main(["project", "status", str(project_file)]) == 1
    output = capsys.readouterr().out
    assert "transcribe: current" in output
    assert "translate: current" in output
    assert "mux: stale — settings changed" in output
    assert "tts: disabled for this workflow" in output


def test_project_run_dry_run_reproduces_toy_story_settings_without_writing_state(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("FFMPEG_PATH", "/bin/echo")
    video = tmp_path / "Toy.Story.5.2026.mkv"
    video.write_bytes(b"movie")
    source = tmp_path / "work" / "Toy.Story.5.2026.en.cleaned.srt"
    source.parent.mkdir()
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    terms = tmp_path / "project.terms.json"
    terms.write_text(
        json.dumps(
            {
                "schema": "video-txt.terminology",
                "version": 1,
                "source_language": "en",
                "target_language": "zh-CN",
                "terms": [],
            }
        ),
        encoding="utf-8",
    )
    project_file = tmp_path / "project.video-txt.json"
    assert (
        main(
            [
                "project",
                "init",
                str(video),
                "-o",
                str(project_file),
                "--subtitle",
                str(source),
                "--audio-stream",
                "2",
                "--language",
                "en",
                "--provider",
                "lmstudio",
                "--model",
                "Qwen3.5-35B-A3B",
                "--target-language",
                "zh-CN",
                "--term-file",
                str(terms),
                "--subtitle-output",
                "work/Toy.Story.5.2026.zh-CN.srt",
                "--video-output",
                "outputs/Toy.Story.5.2026.zh-subbed.mkv",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["project", "run", str(project_file), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "Requested audio stream: 2" in output
    assert "Spoken language: en" in output
    assert f"Terminology file: {terms}" in output
    assert "Qwen3.5-35B-A3B" in output
    assert str(tmp_path / "work" / "Toy.Story.5.2026.zh-CN.srt") in output
    assert str(tmp_path / "outputs" / "Toy.Story.5.2026.zh-subbed.mkv") in output
    assert not (tmp_path / ".video-txt" / "state.json").exists()


def test_project_run_preserves_existing_untracked_outputs_without_state(
    tmp_path, capsys
):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"movie")
    source = tmp_path / "movie.srt"
    source.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    translated = tmp_path / "movie.zh.srt"
    translated.write_text("caller-owned", encoding="utf-8")
    project_file = tmp_path / "project.video-txt.json"
    assert (
        main(
            [
                "project",
                "init",
                str(video),
                "-o",
                str(project_file),
                "--subtitle",
                str(source),
                "--provider",
                "deepseek",
            ]
        )
        == 0
    )

    assert main(["project", "run", str(project_file)]) == 1

    assert translated.read_text(encoding="utf-8") == "caller-owned"
    assert "not recorded in project state" in capsys.readouterr().err


def test_successful_project_run_records_artifacts_and_actual_stream(tmp_path, monkeypatch):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"movie")
    project_file = tmp_path / "project.video-txt.json"
    assert (
        main(
            [
                "project",
                "init",
                str(video),
                "-o",
                str(project_file),
                "--provider",
                "deepseek",
                "--audio-stream",
                "2",
            ]
        )
        == 0
    )

    def fake_pipeline(**kwargs):
        source = kwargs["output_dir"] / "movie.srt"
        source.write_text("source", encoding="utf-8")
        translated = kwargs["translate_stage"].output_path
        assert translated is not None
        translated.write_text("translation", encoding="utf-8")
        translation_audit_path_for(translated).write_text("{}", encoding="utf-8")
        output = kwargs["mux_options_for"](translated).video_output
        output.write_bytes(b"muxed")
        return output

    monkeypatch.setattr(cli_module, "run_pipeline", fake_pipeline)

    assert main(["project", "run", str(project_file)]) == 0

    state_path = tmp_path / ".video-txt" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["schema"] == "video-txt.project-state"
    assert state["audio_selection"] == {
        "requested_stream": 2,
        "selected_stream": 2,
        "reason": "selected explicitly with --audio-stream 2",
    }
    assert state["transcription_resolution"]["backend"] in {
        "openai-whisper",
        "mlx-whisper",
    }
    assert state["transcription_resolution"]["model"]
    assert set(state["stages"]) == {"transcribe", "translate", "mux"}
    assert inspect_project(project_file).stale is False

    monkeypatch.setattr(
        cli_module,
        "run_pipeline",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("a current project must not enter the legacy pipeline")
        ),
    )
    assert main(["project", "run", str(project_file)]) == 0


def test_project_init_records_dub_voice_speaker_and_cache_settings(
    tmp_path, monkeypatch, capsys
):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"movie")
    reference = tmp_path / "voices" / "host.wav"
    reference.parent.mkdir()
    reference.write_bytes(b"voice")
    project_file = tmp_path / "dub.video-txt.json"
    monkeypatch.setenv("HF_TOKEN", "must-not-be-written")

    assert (
        main(
            [
                "project",
                "init",
                str(video),
                "-o",
                str(project_file),
                "--workflow",
                "dub",
                "--provider",
                "deepseek",
                "--tts-engine",
                "f5-tts",
                "--voice-unit",
                "line",
                "--rate",
                "+10%",
                "--keep-bgm",
                "--diarize",
                "--speakers",
                "2",
                "--clone-reference",
                f"SPEAKER_01={reference}",
                "--cache-dir",
                "cache/voices",
                "--audio-output",
                "outputs/movie.dub.wav",
            ]
        )
        == 0
    )

    raw = project_file.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["workflow"] == "dub"
    assert payload["tts"]["engine"] == "f5-tts"
    assert payload["tts"]["voice"] == "zh-CN-XiaoxiaoNeural"
    assert payload["tts"]["voice_unit"] == "line"
    assert payload["tts"]["rate"] == "+10%"
    assert payload["tts"]["keep_bgm"] is True
    assert payload["speakers"]["enabled"] is True
    assert payload["speakers"]["count"] == 2
    assert payload["speakers"]["voices"] == []
    assert payload["voice_clone"]["references"] == [
        {"speaker": "SPEAKER_01", "path": "voices/host.wav"}
    ]
    assert payload["paths"]["cache_dir"] == "cache/voices"
    assert payload["paths"]["dub_audio"] == "outputs/movie.dub.wav"
    assert "must-not-be-written" not in raw

    monkeypatch.setenv("FFMPEG_PATH", "/bin/echo")
    capsys.readouterr()
    assert main(["project", "run", str(project_file), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Dub: f5-tts cloning the original voice" in output
    assert str(tmp_path / "cache" / "voices") in output
    assert str(tmp_path / "outputs" / "movie.dub.wav") in output
