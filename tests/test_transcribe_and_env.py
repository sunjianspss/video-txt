from __future__ import annotations

import sys
from pathlib import Path

import pytest

from video_txt.env import load_secrets_file, parse_secrets_text, resolve_api_key
from video_txt.transcribe import (
    TranscribeError,
    TranscribeOptions,
    build_command,
    expected_output_path,
    resolve_model,
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


def test_load_secrets_file_fills_only_missing_variables(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets"
    secrets.write_text("export A_KEY=from-file\nexport B_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("A_KEY", "from-shell")
    monkeypatch.delenv("B_KEY", raising=False)

    loaded = load_secrets_file(secrets)
    assert loaded == ["B_KEY"]
    assert resolve_api_key("A_KEY", secrets_file=secrets) == "from-shell"
    assert resolve_api_key("B_KEY", secrets_file=secrets) == "from-file"


def test_load_secrets_file_tolerates_a_missing_file(tmp_path):
    assert load_secrets_file(tmp_path / "nope") == []


def test_resolve_api_key_explains_how_to_set_it(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(SystemExit, match="export MISSING_KEY"):
        resolve_api_key("MISSING_KEY", secrets_file=tmp_path / "nope")
