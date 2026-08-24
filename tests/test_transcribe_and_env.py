from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_txt import media as media_module
from video_txt import transcribe as transcribe_module
from video_txt.env import (
    CredentialError,
    parse_secrets_text,
    read_secrets_file,
    resolve_api_key,
    resolve_optional_key,
)
from video_txt.subtitles import parse_srt
from video_txt.transcribe import (
    TranscribeError,
    TranscribeOptions,
    build_command,
    expected_output_path,
    resolve_model,
    run_transcribe,
)


def make_options(**overrides) -> TranscribeOptions:
    defaults = dict(input_path=Path("/media/talk.mp4"), output_dir=Path("/media/out"))
    return TranscribeOptions(**{**defaults, **overrides})


def test_openai_whisper_command_uses_underscore_flags():
    command = build_command(
        make_options(output_format="srt", language="en", initial_prompt="Claude Code, MCP"),
        backend="openai-whisper",
        model="turbo",
    )
    assert command[:4] == [sys.executable, "-m", "whisper", "/media/talk.mp4"]
    assert command[command.index("--output_format") + 1] == "srt"
    assert command[command.index("--output_dir") + 1] == "/media/out"
    assert command[command.index("--language") + 1] == "en"
    assert command[command.index("--initial_prompt") + 1] == "Claude Code, MCP"


def test_openai_refinement_command_requests_word_timestamp_json():
    command = build_command(
        make_options(output_format="json", refine_subtitles=True),
        backend="openai-whisper",
        model="turbo",
    )

    assert command[command.index("--output_format") + 1] == "json"
    assert command[command.index("--word_timestamps") + 1] == "True"


def test_mlx_whisper_command_uses_hyphen_flags_and_pins_the_output_name():
    command = build_command(
        make_options(output_format="srt", extra_args=["--word-timestamps", "True"]),
        backend="mlx-whisper",
        model="mlx-community/whisper-large-v3-turbo",
    )
    assert command[command.index("--output-format") + 1] == "srt"
    assert command[command.index("--output-dir") + 1] == "/media/out"
    assert command[command.index("--output-name") + 1] == "talk"
    assert command[-2:] == ["--word-timestamps", "True"]


def test_mlx_refinement_command_requests_word_timestamp_json():
    command = build_command(
        make_options(output_format="json", refine_subtitles=True),
        backend="mlx-whisper",
        model="mlx-community/whisper-large-v3-turbo",
    )

    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--word-timestamps") + 1] == "True"


def test_dry_run_extracts_the_requested_audio_stream_before_whisper(
    tmp_path, monkeypatch, capsys
):
    video = tmp_path / "talk.mkv"
    video.write_bytes(b"media")
    monkeypatch.setattr(transcribe_module, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(transcribe_module, "resolve_backend", lambda _value: "openai-whisper")
    monkeypatch.setattr(
        media_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                '{"streams":['
                '{"index":1,"codec_name":"ac3","channels":2,'
                '"tags":{"language":"spa","title":"Lat"}},'
                '{"index":2,"codec_name":"aac","channels":2,'
                '"tags":{"language":"spa","title":"Eng"}}]}'
            ),
            stderr="",
        ),
    )

    output = run_transcribe(
        TranscribeOptions(
            input_path=video,
            output_dir=tmp_path / "out",
            output_format="srt",
            language="en",
            audio_stream=2,
        ),
        dry_run=True,
    )

    lines = capsys.readouterr().out.splitlines()
    assert "Audio stream: 2" in lines[0]
    assert "selected explicitly" in lines[0]
    assert "-map 0:2" in lines[1]
    assert "-ac 1" in lines[1]
    assert "-ar 16000" in lines[1]
    assert "talk.wav" in lines[1]
    assert "talk.wav" in lines[2]
    assert output == tmp_path / "out" / "talk.srt"


def test_dry_run_automatically_selects_language_title_from_multiple_tracks(
    tmp_path, monkeypatch, capsys
):
    video = tmp_path / "talk.mkv"
    video.write_bytes(b"media")
    monkeypatch.setattr(transcribe_module, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(transcribe_module, "resolve_backend", lambda _value: "openai-whisper")
    monkeypatch.setattr(
        media_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                '{"streams":['
                '{"index":1,"disposition":{"default":1},'
                '"tags":{"language":"spa","title":"Lat"}},'
                '{"index":2,"tags":{"language":"spa","title":"Eng"}}]}'
            ),
            stderr="",
        ),
    )

    run_transcribe(
        TranscribeOptions(
            input_path=video,
            output_dir=tmp_path / "out",
            output_format="srt",
            language="en",
        ),
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert "Audio stream: 2 (Eng, spa)" in output
    assert "title 'Eng' matches en" in output
    assert "-map 0:2" in output


def test_selected_stream_is_extracted_to_temporary_audio_for_whisper(
    tmp_path, monkeypatch
):
    video = tmp_path / "talk.mkv"
    video.write_bytes(b"media")
    output_dir = tmp_path / "out"
    commands: list[list[str]] = []
    extracted_paths: list[Path] = []
    monkeypatch.setattr(transcribe_module, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(transcribe_module, "resolve_backend", lambda _value: "openai-whisper")
    monkeypatch.setattr(media_module, "find_ffprobe", lambda _path=None: "/usr/bin/ffprobe")

    def external_tool(command, **_kwargs):
        commands.append(command)
        if "-select_streams" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    '{"streams":['
                    '{"index":1,"tags":{"language":"spa","title":"Lat"}},'
                    '{"index":2,"tags":{"language":"spa","title":"Eng"}}]}'
                ),
                stderr="",
            )
        if command[0] == "/usr/bin/ffmpeg":
            extracted = Path(command[-1])
            extracted_paths.append(extracted)
            extracted.write_bytes(b"wav")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(media_module.subprocess, "run", external_tool)

    output = run_transcribe(
        TranscribeOptions(
            input_path=video,
            output_dir=output_dir,
            output_format="srt",
            language="en",
        )
    )

    extraction = next(command for command in commands if command[0] == "/usr/bin/ffmpeg")
    whisper = next(command for command in commands if "whisper" in command)
    assert extraction[extraction.index("-map") + 1] == "0:2"
    assert Path(whisper[3]).name == "talk.wav"
    assert output == output_dir / "talk.srt"
    assert extracted_paths and not extracted_paths[0].exists()


def test_missing_explicit_audio_stream_is_a_transcription_error(tmp_path, monkeypatch):
    video = tmp_path / "talk.mkv"
    video.write_bytes(b"media")
    monkeypatch.setattr(transcribe_module, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(transcribe_module, "resolve_backend", lambda _value: "openai-whisper")
    monkeypatch.setattr(media_module, "find_ffprobe", lambda _path=None: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        media_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"streams":[{"index":1,"tags":{"title":"English"}}]}',
            stderr="",
        ),
    )

    with pytest.raises(TranscribeError, match=r"Audio stream 9.*Available indexes: 1"):
        run_transcribe(
            TranscribeOptions(
                input_path=video,
                output_dir=tmp_path / "out",
                audio_stream=9,
            ),
            dry_run=True,
        )


def test_mlx_whisper_receives_the_same_selected_temporary_audio(tmp_path, monkeypatch, capsys):
    video = tmp_path / "talk.mkv"
    video.write_bytes(b"media")
    monkeypatch.setattr(transcribe_module, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(transcribe_module, "resolve_backend", lambda _value: "mlx-whisper")
    monkeypatch.setattr(transcribe_module, "mlx_launcher", lambda: ["/usr/bin/mlx_whisper"])
    monkeypatch.setattr(media_module, "find_ffprobe", lambda _path=None: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        media_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                '{"streams":['
                '{"index":1,"tags":{"language":"spa","title":"Lat"}},'
                '{"index":2,"tags":{"language":"eng","title":"English"}}]}'
            ),
            stderr="",
        ),
    )

    run_transcribe(
        TranscribeOptions(
            input_path=video,
            output_dir=tmp_path / "out",
            output_format="srt",
            language="en",
            backend="mlx-whisper",
        ),
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert "/usr/bin/mlx_whisper" in output
    assert "talk.wav" in output
    assert "--output-name talk" in output
    assert "-map 0:2" in output


def test_resolve_model_defaults_per_mode():
    assert resolve_model(mode="transcribe", model=None, backend="openai-whisper") == "turbo"
    assert resolve_model(mode="translate", model=None, backend="openai-whisper") == "medium"


def test_resolve_model_maps_aliases_for_mlx():
    assert resolve_model(mode="transcribe", model=None, backend="mlx-whisper") == (
        "mlx-community/whisper-large-v3-turbo"
    )
    assert resolve_model(mode="transcribe", model="small", backend="mlx-whisper") == (
        "mlx-community/whisper-small"
    )
    assert resolve_model(mode="transcribe", model="org/custom", backend="mlx-whisper") == (
        "org/custom"
    )


def test_resolve_model_rejects_turbo_for_translation():
    with pytest.raises(TranscribeError, match="does not support translation"):
        resolve_model(mode="translate", model="turbo", backend="openai-whisper")


def test_expected_output_path():
    assert expected_output_path(make_options(output_format="srt")) == Path("/media/out/talk.srt")
    assert expected_output_path(make_options(output_format="all")) == Path("/media/out/talk.txt")


def test_refined_transcription_writes_srt_and_normalized_words_json(tmp_path, monkeypatch):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"media")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(transcribe_module, "find_ffmpeg", lambda: Path("/usr/bin/ffmpeg"))
    monkeypatch.setattr(transcribe_module, "resolve_backend", lambda _requested: "openai-whisper")

    def whisper(command, **_kwargs):
        if "-select_streams" in command:
            return SimpleNamespace(
                returncode=0,
                stdout='{"streams":[{"index":1,"codec_name":"aac","channels":2}]}',
                stderr="",
            )
        raw_output_dir = Path(command[command.index("--output_dir") + 1])
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        (raw_output_dir / "talk.json").write_text(
            '{"language":"en","segments":[{"words":['
            '{"word":" Hello","start":0.0,"end":0.5},'
            '{"word":" world.","start":0.5,"end":1.0}]}]}',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(transcribe_module.subprocess, "run", whisper)

    output = run_transcribe(
        TranscribeOptions(
            input_path=video,
            output_dir=output_dir,
            output_format="srt",
            backend="openai-whisper",
            refine_subtitles=True,
        )
    )

    assert output == output_dir / "talk.srt"
    assert parse_srt(output)[0].text == "Hello world."
    assert (output_dir / "talk.words.json").is_file()
    assert not (output_dir / "talk.json").exists()


def test_refinement_rejects_non_srt_output_before_starting_whisper(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"media")

    with pytest.raises(TranscribeError, match="requires --format srt"):
        run_transcribe(
            TranscribeOptions(
                input_path=video,
                output_dir=tmp_path,
                output_format="txt",
                refine_subtitles=True,
            )
        )


def test_refinement_rejects_raw_arguments_that_override_its_required_json(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"media")

    with pytest.raises(TranscribeError, match="conflicts with --refine-subtitles"):
        run_transcribe(
            TranscribeOptions(
                input_path=video,
                output_dir=tmp_path,
                output_format="srt",
                refine_subtitles=True,
                extra_args=["--output_format", "srt"],
            )
        )


def test_parse_secrets_text_handles_export_quotes_and_comments():
    values = parse_secrets_text(
        "\n".join(
            [
                "# a comment",
                "export DEEPSEEK_API_KEY=sk-plain",
                'OPENAI_API_KEY="sk-quoted"',
                "OPENAI_MODEL='deepseek-v4-flash'",
                "TRAILING=value # inline note",
                "not an assignment",
            ]
        )
    )
    assert values == {
        "DEEPSEEK_API_KEY": "sk-plain",
        "OPENAI_API_KEY": "sk-quoted",
        "OPENAI_MODEL": "deepseek-v4-flash",
        "TRAILING": "value",
    }


def test_the_shell_wins_over_the_secrets_file(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets"
    secrets.write_text("export A_KEY=from-file\nexport B_KEY=from-file\n", encoding="utf-8")
    secrets.chmod(0o600)
    monkeypatch.setenv("A_KEY", "from-shell")
    monkeypatch.delenv("B_KEY", raising=False)

    assert resolve_api_key("A_KEY", secrets_file=secrets) == "from-shell"
    assert resolve_api_key("B_KEY", secrets_file=secrets) == "from-file"


def test_a_key_that_was_read_is_not_handed_to_every_subprocess(tmp_path, monkeypatch):
    """Only the lookup that asked for a key gets it; ffmpeg and whisper do not."""
    secrets = tmp_path / "secrets"
    secrets.write_text("export WANTED_KEY=v\nexport UNRELATED_KEY=v\n", encoding="utf-8")
    secrets.chmod(0o600)
    monkeypatch.delenv("WANTED_KEY", raising=False)
    monkeypatch.delenv("UNRELATED_KEY", raising=False)

    assert resolve_api_key("WANTED_KEY", secrets_file=secrets) == "v"
    assert "WANTED_KEY" not in os.environ
    assert "UNRELATED_KEY" not in os.environ


def test_read_secrets_file_tolerates_a_missing_file(tmp_path):
    assert read_secrets_file(tmp_path / "nope") == {}


def test_an_optional_key_that_is_nowhere_comes_back_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert resolve_optional_key("HF_TOKEN", secrets_file=tmp_path / "nope") == ""


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions only")
def test_read_secrets_file_refuses_group_readable_permissions(tmp_path):
    secrets = tmp_path / "secrets"
    secrets.write_text("export PERM_TEST_KEY=v\n", encoding="utf-8")
    secrets.chmod(0o644)

    with pytest.raises(CredentialError, match=r"mode 644.*chmod 600"):
        read_secrets_file(secrets)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions only")
def test_read_secrets_file_stays_quiet_for_owner_only_permissions(tmp_path, capsys):
    secrets = tmp_path / "secrets"
    secrets.write_text("export PERM_TEST_KEY=v\n", encoding="utf-8")
    secrets.chmod(0o600)

    read_secrets_file(secrets)
    assert capsys.readouterr().err == ""


def test_resolve_api_key_explains_how_to_set_it(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(CredentialError, match="export MISSING_KEY"):
        resolve_api_key("MISSING_KEY", secrets_file=tmp_path / "nope")
