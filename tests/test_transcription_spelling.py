from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_txt import transcribe
from video_txt.refine import (
    TranscriptDocument,
    Word,
    load_word_document,
    word_document_matches_subtitle,
)
from video_txt.subtitles import SubtitleCue, parse_srt, write_srt


@pytest.mark.parametrize("backend", ["openai-whisper", "mlx-whisper"])
@pytest.mark.parametrize("refined", [False, True])
def test_transcribe_corrects_product_spelling_before_consumers(
    tmp_path, monkeypatch, backend, refined
):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"media")
    monkeypatch.setattr(transcribe, "find_ffmpeg", lambda: Path("/usr/bin/ffmpeg"))
    monkeypatch.setattr(transcribe, "probe_audio_streams", lambda *a, **kw: [])
    monkeypatch.setattr(transcribe, "resolve_backend", lambda requested: backend)

    def whisper(command, **kwargs):
        flag = "--output-dir" if backend == "mlx-whisper" else "--output_dir"
        directory = Path(command[command.index(flag) + 1])
        if refined:
            payload = {
                "language": "en",
                "segments": [
                    {
                        "words": [
                            {"word": " Claud", "start": 0.0, "end": 0.5},
                            {"word": " Code.", "start": 0.5, "end": 1.0},
                        ]
                    }
                ],
            }
            (directory / "talk.json").write_text(json.dumps(payload))
        else:
            (directory / "talk.srt").write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nClaud\nCode.\n\n"
                "2\n00:00:02,000 --> 00:00:04,000\n"
                "Claud Shannon uses cloud code, Quad Code and Claud Codes.\n"
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(transcribe.subprocess, "run", whisper)
    output = transcribe.run_transcribe(
        transcribe.TranscribeOptions(
            input_path=video,
            output_dir=tmp_path,
            language="en",
            output_format="srt",
            backend=backend,
            refine_subtitles=refined,
        )
    )
    cues = parse_srt(output)
    assert " ".join(cues[0].text.split()) == "Claude Code."
    assert cues[0].start_seconds == 0.0
    assert cues[0].end_seconds == 1.0
    if refined:
        word_path = tmp_path / "talk.words.json"
        document = load_word_document(word_path)
        assert [word.text for word in document.words] == ["Claude", "Code."]
        assert [(word.start, word.end) for word in document.words] == [(0, 0.5), (0.5, 1)]
        assert word_document_matches_subtitle(word_path, output)
    else:
        assert cues[1].text == "Claud Shannon uses cloud code, Quad Code and Claud Codes."
        assert cues[1].timing == "00:00:02,000 --> 00:00:04,000"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Claud Shannon.",
        "cloud code",
        "Quad Code",
        "McClaud Code",
        "Claud Codebase",
        "Claud Codes",
    ],
)
def test_unrelated_subtitle_bytes_are_untouched(tmp_path, text):
    path = tmp_path / "raw.srt"
    path.write_text(text)
    transcribe._correct_subtitle_spelling(path)
    assert path.read_text() == text


def test_word_correction_preserves_internal_newlines_and_ignores_long_pauses():
    document = TranscriptDocument(
        words=[
            Word("use\nclaud", 0, 0.5),
            Word("code.", 0.5, 1),
            Word("Claud", 2, 2.5),
            Word("Code", 4, 4.5),
        ],
        language="en",
    )
    corrected = transcribe._correct_word_spelling(document)
    assert [word.text for word in corrected.words] == ["use\nClaude", "Code.", "Claud", "Code"]
    assert [(w.start, w.end) for w in corrected.words] == [(w.start, w.end) for w in document.words]


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (["I use Claud", "Code every day."], ["I use Claude", "Code every day."]),
        (["<u>Claud</u> Code", "Claud <u>Code</u>"], ["<u>Claude</u> Code", "Claude <u>Code</u>"]),
        (
            ["I use <u>Claud</u>", "<u>code</u> every day."],
            ["I use <u>Claude</u>", "<u>Code</u> every day."],
        ),
        (
            ["<i>Cl<u>aud</u></i> Code", "Keep cloud code."],
            ["<i>Cl<u>aude</u></i> Code", "Keep cloud code."],
        ),
    ],
)
def test_subtitle_correction_preserves_blocks_and_markup(tmp_path, texts, expected):
    path = tmp_path / "split.srt"
    original = [
        SubtitleCue("7", "00:00:00,000 --> 00:00:01,000", [texts[0]]),
        SubtitleCue("8", "00:00:01,100 --> 00:00:02,000", [texts[1]]),
    ]
    write_srt(path, original)
    transcribe._correct_subtitle_spelling(path)
    corrected = parse_srt(path)
    assert [cue.text for cue in corrected] == expected
    assert [(cue.index, cue.timing) for cue in corrected] == [
        (cue.index, cue.timing) for cue in original
    ]
    once = path.read_bytes()
    transcribe._correct_subtitle_spelling(path)
    assert path.read_bytes() == once


@pytest.mark.parametrize("second_start", ["00:00:00,500", "00:00:02,000"])
def test_subtitle_correction_does_not_join_overlaps_or_long_pauses(tmp_path, second_start):
    path = tmp_path / "separate.srt"
    write_srt(
        path,
        [
            SubtitleCue("1", "00:00:00,000 --> 00:00:01,000", ["Claud"]),
            SubtitleCue("2", f"{second_start} --> 00:00:03,000", ["Code is a word."]),
        ],
    )
    original = path.read_bytes()
    transcribe._correct_subtitle_spelling(path)
    assert path.read_bytes() == original


def test_subtitle_correction_includes_exact_join_gap(tmp_path):
    path = tmp_path / "boundary.srt"
    write_srt(
        path,
        [
            SubtitleCue("1", "00:00:01,000 --> 00:00:02,300", ["Claud"]),
            SubtitleCue("2", "00:00:03,100 --> 00:00:04,000", ["Code"]),
        ],
    )
    transcribe._correct_subtitle_spelling(path)
    assert [cue.text for cue in parse_srt(path)] == ["Claude", "Code"]
