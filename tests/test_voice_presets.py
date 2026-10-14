from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_txt import cli
from video_txt.clone import (
    EMOTION_ALPHA,
    CloneJob,
    CloneOptions,
    build_manifest,
    supplied_reference,
)
from video_txt.cli import main
from video_txt.project import inspect_project, project_workflow_config
from video_txt.voices import DEFAULT_VOICE


@pytest.fixture
def media(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture video")
    subtitle = video.with_suffix(".srt")
    subtitle.write_text("1\n00:00:01,000 --> 00:00:06,000\nHello!\n", encoding="utf-8")
    return video, subtitle


@pytest.mark.parametrize("voice", [None, "mambo", "曼波"])
def test_named_voice_reaches_the_cloning_pipeline(media, tmp_path, monkeypatch, voice):
    video, subtitle = media
    observed = []

    def run(workflow):
        observed.append(workflow.dub_options_for(subtitle, subtitle, []))
        return tmp_path / "dubbed.mp4"

    monkeypatch.setattr(cli, "run_workflow", run)
    voice_flags = ["--voice", voice] if voice else []
    assert main([
        "dub", str(video), *voice_flags, "--provider", "lmstudio", "--dry-run",
    ]) == 0
    options = observed[0]
    assert options.engine == "index-tts"
    assert options.clone.repo == Path.home() / "tools" / "index-tts"
    reference = options.clone.references[""]
    assert reference.is_file()
    assert reference.with_suffix(".txt").read_text(encoding="utf-8").strip() == (
        "大家好，欢迎来到我的频道，今天给大家分享一个有趣的内容"
    )
    # Pin the actual sound the user approved, not merely a file bearing its name.
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
        "6031533a08a3c1fc9daef66620db5c287cd48c6c3b44b3912922db1f644715da"
    )


@pytest.mark.parametrize("voice_flags", [[], ["--voice", "mambo"]])
def test_mambo_project_saves_explicit_clone_settings(media, tmp_path, voice_flags):
    video, subtitle = media
    project = tmp_path / "mambo.video-txt.json"
    repo = tmp_path / "custom-index"
    assert main([
        "project", "init", str(video), "-o", str(project), "--workflow", "dub",
        "--subtitle", str(subtitle), *voice_flags, "--clone-repo", str(repo),
        "--provider", "lmstudio",
    ]) == 0
    saved = json.loads(project.read_text(encoding="utf-8"))
    assert saved["tts"]["engine"] == "index-tts"
    assert saved["tts"]["voice"] == DEFAULT_VOICE  # Preset expanded; no new project schema.
    restored = project_workflow_config(inspect_project(project), dry_run=True)
    assert restored.clone_repo == repo
    assert len(restored.clone_references) == 1
    assert Path(restored.clone_references[0]).is_file()
    # Replaying an expanded project must remain valid without the alias.
    cli.build_workflow(restored, cli.build_parser())


@pytest.mark.parametrize("extra, message", [
    (["--tts-engine", "say"], "index-tts"),
    (["--tts-engine", "cosyvoice"], "index-tts"),
    (["--clone-reference", "/another.wav"], "--clone-reference"),
    (["--speaker-voice", "HUMAN=another"], "--speaker-voice"),
    (["--diarize"], "--diarize"),
])
def test_mambo_conflicts_fail_before_running(media, monkeypatch, capsys, extra, message):
    video, _ = media

    def unexpected(*args):
        pytest.fail("Conflicting voice options must fail before running the workflow")

    monkeypatch.setattr(cli, "run_workflow", unexpected)
    with pytest.raises(SystemExit):
        main(["dub", str(video), "--voice", "mambo", *extra])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("flags", [
    ["--voice", "zh-CN-XiaoxiaoNeural"],
    ["--tts-engine", "edge-tts"],
    ["--diarize"],
])
def test_explicit_edge_voice_or_multispeaker_mode_still_uses_xiaoxiao(flags):
    args = cli.build_parser().parse_args(["dub", "clip.mp4", *flags])
    cli.validate_voice_options(cli.build_parser(), args)
    assert args.tts_engine == "edge-tts"
    assert args.voice == DEFAULT_VOICE
    assert args.clone_references == []


def test_missing_mambo_sample_fails_before_running(media, tmp_path, monkeypatch, capsys):
    video, _ = media
    monkeypatch.setattr(cli, "MAMBO_REFERENCE", tmp_path / "missing.wav")
    with pytest.raises(SystemExit):
        main(["dub", str(video), "--voice", "mambo", "--dry-run"])
    assert "reference audio or transcript is missing" in capsys.readouterr().err


def test_mambo_borrows_its_delivery_from_the_emotion_clip(media, tmp_path, monkeypatch):
    video, subtitle = media
    observed = []

    def run(workflow):
        observed.append(workflow.dub_options_for(subtitle, subtitle, []))
        return tmp_path / "dubbed.mp4"

    monkeypatch.setattr(cli, "run_workflow", run)
    assert main(["dub", str(video), "--provider", "lmstudio", "--dry-run"]) == 0
    reference_path = observed[0].clone.references[""]
    reference = supplied_reference("", reference_path)
    assert reference.emotion_audio == reference_path.with_suffix(".emo.wav")
    assert hashlib.sha256(reference.emotion_audio.read_bytes()).hexdigest() == (
        "173dfbc8a0e68850a5b0189433a32793f8e91a8cff61702c2ff4be3afd64fdf6"
    )
    job = CloneJob(text="你好", output=tmp_path / "out.wav", reference=reference)
    manifest = build_manifest([job], CloneOptions(engine="index-tts", model="m"))
    assert manifest["jobs"][0]["emotion_audio"] == str(reference.emotion_audio)
    assert manifest["jobs"][0]["emotion_alpha"] == EMOTION_ALPHA


def test_emotion_clip_changes_the_cache_fingerprint_only_when_present(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"voice")
    plain = supplied_reference("", audio)
    assert plain.emotion_audio is None
    (tmp_path / "voice.emo.wav").write_bytes(b"drawl")
    styled = supplied_reference("", audio)
    assert styled.emotion_audio == tmp_path / "voice.emo.wav"
    assert styled.fingerprint != plain.fingerprint
    (tmp_path / "voice.emo.wav").write_bytes(b"other")
    assert supplied_reference("", audio).fingerprint != styled.fingerprint


def test_missing_mambo_emotion_clip_fails_before_running(media, tmp_path, monkeypatch, capsys):
    video, _ = media
    for suffix in (".wav", ".txt"):
        (tmp_path / f"mambo{suffix}").write_bytes(b"x")
    monkeypatch.setattr(cli, "MAMBO_REFERENCE", tmp_path / "mambo.wav")
    with pytest.raises(SystemExit):
        main(["dub", str(video), "--voice", "mambo", "--dry-run"])
    assert "delivery reference is missing" in capsys.readouterr().err
