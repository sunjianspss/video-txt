from __future__ import annotations

from pathlib import Path

import pytest

from video_txt import transcribe
from video_txt.refine import TranscriptDocument, Word
from video_txt.subtitles import parse_srt, write_srt, SubtitleCue

PROMPT = transcribe.DEFAULT_INITIAL_PROMPT
ECHO = "Claude Code, Anthropic's coding agent, and the Claude Code, Anthropic's coding agent"


def options(**kwargs):
    base = dict(input_path=Path("talk.mp4"), output_dir=Path("."), output_format="srt")
    return transcribe.TranscribeOptions(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, PROMPT),
        ({"language": "en"}, PROMPT),
        ({"language": "English"}, PROMPT),
        ({"language": "zh"}, None),
        ({"language": "Chinese"}, None),
        ({"output_format": "txt"}, PROMPT),
        ({"output_format": "vtt"}, None),
        ({"output_format": "json", "refine_subtitles": True}, PROMPT),
        ({"initial_prompt": "MCP"}, "MCP"),
        ({"initial_prompt": ""}, None),
        ({"initial_prompt": "MCP", "language": "zh"}, "MCP"),
    ],
)
def test_effective_initial_prompt(kwargs, expected):
    assert transcribe.effective_initial_prompt(options(**kwargs)) == expected


@pytest.mark.parametrize("backend", ["mlx-whisper", "openai-whisper"])
def test_default_prompt_reaches_whisper_command(backend):
    command = transcribe.build_command(options(), backend=backend, model="turbo")
    flag = "--initial-prompt" if backend == "mlx-whisper" else "--initial_prompt"
    assert command[command.index(flag) + 1] == PROMPT
    quiet = transcribe.build_command(options(initial_prompt=""), backend=backend, model="turbo")
    assert flag not in quiet


def test_echo_dropped_from_subtitles_but_speech_kept(tmp_path):
    path = tmp_path / "a.srt"
    cue = lambda i, t: SubtitleCue(str(i), f"00:00:0{i},000 --> 00:00:0{i},900", t.split("\n"))
    write_srt(
        path,
        [
            cue(1, "Claude Code is a coding agent."),
            cue(2, ECHO),
            cue(3, "Claude Code"),
            cue(4, "We talk about the models."),
        ],
    )
    transcribe._drop_prompt_echo_subtitles(path, PROMPT)
    kept = parse_srt(path)
    assert [c.index for c in kept] == ["1", "2", "3"]
    assert [c.text for c in kept] == [
        "Claude Code is a coding agent.",
        "Claude Code",
        "We talk about the models.",
    ]


def test_echo_dropped_from_text(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text(f"Hello there.\n{ECHO}\nClaude Code\n")
    transcribe._drop_prompt_echo_text(path, PROMPT)
    assert path.read_text() == "Hello there.\nClaude Code\n"


def test_echo_dropped_from_words_keeps_neighbours():
    spoken = "I use Claude Code daily".split()
    echo = ECHO.replace(",", "").split()
    texts = spoken[:1] + echo + spoken[1:]
    words = [Word(text=f" {t}", start=float(i), end=i + 0.5) for i, t in enumerate(texts)]
    document = TranscriptDocument(words=words, language="en")
    cleaned = transcribe._drop_prompt_echo_words(document, PROMPT)
    assert [w.text.strip() for w in cleaned.words] == ["I", "use", "Claude", "Code", "daily"]
