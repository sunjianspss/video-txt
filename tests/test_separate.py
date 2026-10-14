from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from video_txt import separate as separate_module
from video_txt.separate import (
    SeparateError,
    ensure_instrumental,
    extract_audio_command,
    separate_command,
    separated_bgm_path,
    separated_voice_path,
)


def test_the_finished_background_is_reused_as_is(tmp_path, monkeypatch):
    """Separation costs minutes; a rerun must not pay them, or even need demucs installed."""
    monkeypatch.setattr(
        separate_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("the cached track leaves nothing to run"),
    )
    monkeypatch.setattr(
        separate_module,
        "require_demucs",
        lambda: pytest.fail("reuse must not depend on demucs being installed"),
    )
    target = separated_bgm_path(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"flac")

    assert ensure_instrumental(Path("/v/a.mp4"), cache_dir=tmp_path, ffmpeg_path="ffmpeg") == target


def test_a_shared_cache_never_reuses_the_background_from_another_video(tmp_path, monkeypatch):
    monkeypatch.setattr(separate_module, "require_demucs", lambda: None)
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first video")
    second.write_bytes(b"second video")
    splits = 0

    def fake_run(command, **_kwargs):
        nonlocal splits
        if "demucs.separate" in command:
            splits += 1
            stem = tmp_path / "cache" / "bgm" / "htdemucs" / "source" / "no_vocals.wav"
            stem.parent.mkdir(parents=True, exist_ok=True)
            stem.write_bytes(b"wav")
        else:
            Path(command[-1]).write_bytes(b"data")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(separate_module.subprocess, "run", fake_run)

    ensure_instrumental(first, cache_dir=tmp_path / "cache", ffmpeg_path="ffmpeg")
    ensure_instrumental(second, cache_dir=tmp_path / "cache", ffmpeg_path="ffmpeg")

    assert splits == 2


def test_a_missing_demucs_is_reported_with_the_install_command(tmp_path, monkeypatch):
    monkeypatch.setattr(separate_module, "find_spec", lambda _name: None)

    with pytest.raises(SeparateError, match="uv sync --extra separate"):
        ensure_instrumental(Path("/v/a.mp4"), cache_dir=tmp_path, ffmpeg_path="ffmpeg")


def test_demucs_is_asked_for_two_stems_of_the_extracted_audio():
    command = separate_command(Path("/cache/bgm/source.wav"), output_dir=Path("/cache/bgm"))

    assert command[command.index("--two-stems") + 1] == "vocals"
    assert command[command.index("-n") + 1] == "htdemucs"
    assert command[-1] == "/cache/bgm/source.wav"


def test_the_audio_is_extracted_as_plain_stereo_wav():
    command = extract_audio_command(
        Path("/v/a.mkv"), target=Path("/cache/bgm/source.wav"), ffmpeg_path="ffmpeg"
    )

    assert "-vn" in command
    assert command[command.index("-ac") + 1] == "2"
    assert command[-1] == "/cache/bgm/source.wav"


def test_separation_keeps_both_stems_compressed_and_neither_wav(tmp_path, monkeypatch):
    """The WAV stems run to hundreds of megabytes; the FLACs are what gets read again.

    The vocal one is kept for cutting clone references from -- demucs writes it
    either way, and it holds the speaker without the music.
    """
    monkeypatch.setattr(separate_module, "require_demucs", lambda: None)

    def fake_run(command, **_kwargs):
        if "demucs.separate" in command:
            stem = tmp_path / "bgm" / "htdemucs" / "source" / "no_vocals.wav"
            stem.parent.mkdir(parents=True, exist_ok=True)
            stem.write_bytes(b"wav")
            (stem.parent / "vocals.wav").write_bytes(b"wav")
        else:
            Path(command[-1]).write_bytes(b"data")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(separate_module.subprocess, "run", fake_run)

    target = ensure_instrumental(Path("/v/a.mp4"), cache_dir=tmp_path, ffmpeg_path="ffmpeg")

    assert target == separated_bgm_path(tmp_path)
    assert target.is_file()
    assert separated_voice_path(tmp_path).is_file()
    assert not (tmp_path / "bgm" / "htdemucs").exists()
    assert not (tmp_path / "bgm" / "source.wav").exists()


def test_a_lost_voice_stem_does_not_cost_the_background(tmp_path, monkeypatch, capsys):
    """The vocal stem is a bonus. Failing to keep it must not fail the separation."""
    monkeypatch.setattr(separate_module, "require_demucs", lambda: None)

    def fake_run(command, **_kwargs):
        if "demucs.separate" in command:
            stem = tmp_path / "bgm" / "htdemucs" / "source" / "no_vocals.wav"
            stem.parent.mkdir(parents=True, exist_ok=True)
            stem.write_bytes(b"wav")
            (stem.parent / "vocals.wav").write_bytes(b"wav")
        elif Path(command[command.index("-i") + 1]).name == "vocals.wav":
            return SimpleNamespace(returncode=1, stdout="", stderr="flac: no space left")
        else:
            Path(command[-1]).write_bytes(b"data")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(separate_module.subprocess, "run", fake_run)

    target = ensure_instrumental(Path("/v/a.mp4"), cache_dir=tmp_path, ffmpeg_path="ffmpeg")

    assert target.is_file()
    assert not separated_voice_path(tmp_path).exists()
    assert "original mix" in capsys.readouterr().err


def test_a_failed_split_names_the_step_that_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(separate_module, "require_demucs", lambda: None)

    def fake_run(command, **_kwargs):
        if "demucs.separate" in command:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        Path(command[-1]).write_bytes(b"data")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(separate_module.subprocess, "run", fake_run)

    with pytest.raises(SeparateError, match="Demucs"):
        ensure_instrumental(Path("/v/a.mp4"), cache_dir=tmp_path, ffmpeg_path="ffmpeg")
