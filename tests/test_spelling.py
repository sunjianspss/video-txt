from __future__ import annotations

import pytest

from video_txt import transcribe
from video_txt.refine import TranscriptDocument, Word
from video_txt.spelling import correct_texts
from video_txt.subtitles import SubtitleCue, parse_srt, write_srt

# Enough to show the talk is about Claude.
ABOUT_CLAUDE = ["Claude Code is built by Anthropic.", "Claude Opus is the big one."]


def fix(*texts, evidence=True):
    lines = [*(ABOUT_CLAUDE if evidence else []), *texts]
    return correct_texts(lines, ["\n"] * (len(lines) - 1))[len(lines) - len(texts) :]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("So quad could edit files.", "So Claude could edit files."),
        ("Quad starts another quad.", "Claude starts another Claude."),
        ("I use Quad Code daily.", "I use Claude Code daily."),
        ("I use cloud code daily.", "I use Claude Code daily."),
        ("Try Cloud Opus and cloud sonnet.", "Try Claude Opus and Claude Sonnet."),
        ("It's a quad-powered thing, said Quad.", "It's a quad-powered thing, said Claude."),
    ],
)
def test_misheard_claude_is_corrected_in_a_talk_about_claude(text, expected):
    assert fix(text) == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "Move it to the cloud.",  # cloud alone is a real word
        "The cloud is somebody else's computer.",
        "A quad-core chip.",
        "A quad core chip and a quad bike.",
        "Quad copters and quad GPUs.",
        "Google Cloud Code lives in your IDE.",
        "Use google  cloud code for that.",
    ],
)
def test_real_words_and_other_products_are_left_alone(text):
    assert fix(text) == [text]


def test_nothing_changes_when_the_text_is_not_about_claude():
    texts = ["I use Quad Code.", "Cloud Code is an extension.", "Quad did it."]
    assert fix(*texts, evidence=False) == texts


def test_a_talk_about_claude_shannon_is_not_about_claude():
    texts = ["Claude Shannon wrote it.", "Claude loved juggling.", "Claude was brilliant.", "A quad."]
    assert correct_texts(texts, ["\n"] * 3) == texts


def test_anthropic_alone_counts_as_evidence():
    mentions = ["Anthropic said so.", "Anthropic again.", "Anthropic once more."]
    assert correct_texts([*mentions, "Quad is here."], ["\n"] * 3)[-1] == "Claude is here."


def test_claud_code_is_always_corrected_even_without_evidence():
    assert fix("Claud Code is here.", evidence=False) == ["Claude Code is here."]


def test_phrase_across_blocks_and_markup():
    assert fix("I use Quad", "Code.") == ["I use Claude", "Code."]
    assert fix("<u>Quad</u> Code", "so <i>Cl</i>oud code") == ["<u>Claude</u> Code", "so <i>Cl</i>aude Code"]
    assert fix("<b>Quad</b> Code", "ok") == ["<b>Claude</b> Code", "ok"]


def test_phrase_does_not_span_a_pause():
    texts = [*ABOUT_CLAUDE, "Cloud", "Code is nice."]
    out = correct_texts(texts, ["\n", "\n", "\0"])
    assert out[-2:] == ["Cloud", "Code is nice."]


def test_correction_is_idempotent():
    once = fix("Quad Code and Quad.")
    assert fix(*once, evidence=True) == once


def test_subtitle_file_and_word_document(tmp_path):
    path = tmp_path / "talk.srt"
    cues = [
        SubtitleCue(str(n), f"00:00:0{n},000 --> 00:00:0{n},900", [text])
        for n, text in enumerate([*ABOUT_CLAUDE, "Quad could do that.", "Cloud Code is good."], 1)
    ]
    write_srt(path, cues)
    transcribe._correct_subtitle_spelling(path)
    corrected = parse_srt(path)
    assert [c.text for c in corrected][2:] == ["Claude could do that.", "Claude Code is good."]
    assert [(c.index, c.timing) for c in corrected] == [(c.index, c.timing) for c in cues]

    words = [
        Word(text, i * 0.5, i * 0.5 + 0.4)
        for i, text in enumerate(["Claude", "Code", "by", "Anthropic.", "Claude", "Opus.", "Quad", "Code", "wins."])
    ]
    document = transcribe._correct_word_spelling(TranscriptDocument(words=words, language="en"))
    assert [w.text for w in document.words][6:] == ["Claude", "Code", "wins."]
    assert [(w.start, w.end) for w in document.words] == [(w.start, w.end) for w in words]
