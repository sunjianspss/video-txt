from __future__ import annotations

import json

import pytest

from video_txt.refine import (
    RefineError,
    RefinePolicy,
    TranscriptDocument,
    Word,
    document_from_whisper_result,
    load_word_document,
    refine_transcript,
    word_document_matches_subtitle,
    write_word_document,
)


def test_refine_transcript_splits_at_sentence_boundaries_with_word_timings():
    document = TranscriptDocument(
        language="en",
        words=[
            Word("Welcome", 0.0, 0.7),
            Word("to", 0.7, 0.9),
            Word("the", 0.9, 1.1),
            Word("show.", 1.1, 1.6),
            Word("Today", 2.0, 2.4),
            Word("we", 2.4, 2.6),
            Word("discuss", 2.6, 3.1),
            Word("memory.", 3.1, 3.6),
        ],
    )

    result = refine_transcript(document, RefinePolicy())

    assert [cue.text for cue in result.cues] == [
        "Welcome to the show.",
        "Today we discuss memory.",
    ]
    assert result.cues[0].time_range == pytest.approx((0.0, 1.6))
    assert result.cues[1].time_range == pytest.approx((2.0, 3.6))
    assert result.source_words == 8
    assert result.cue_count == 2


def test_refine_transcript_caps_a_long_sentence_by_duration():
    document = TranscriptDocument(
        words=[
            Word("One", 0.0, 1.0),
            Word("two", 1.0, 2.0),
            Word("three", 2.0, 3.0),
            Word("four", 3.0, 4.0),
            Word("five", 4.0, 5.0),
            Word("six", 5.0, 6.0),
            Word("seven", 6.0, 7.0),
            Word("eight.", 7.0, 8.0),
        ]
    )

    result = refine_transcript(document, RefinePolicy(max_duration=4.0))

    assert [cue.text for cue in result.cues] == [
        "One two three four",
        "five six seven eight.",
    ]
    assert all(cue.duration <= 4.0 for cue in result.cues)


def test_refine_transcript_limits_each_cue_to_two_readable_lines():
    document = TranscriptDocument(
        words=[
            Word("alpha", 0.0, 0.5),
            Word("beta", 0.5, 1.0),
            Word("gamma", 1.0, 1.5),
            Word("delta.", 1.5, 2.0),
        ]
    )

    result = refine_transcript(
        document,
        RefinePolicy(max_line_width=10, max_lines=2),
    )

    assert [cue.text for cue in result.cues] == ["alpha beta\ngamma", "delta."]
    assert all(len(line) <= 10 for cue in result.cues for line in cue.text_lines)


def test_refine_transcript_splits_on_a_long_speech_pause():
    document = TranscriptDocument(
        words=[
            Word("Well,", 0.0, 0.4),
            Word("this", 2.0, 2.4),
            Word("works.", 2.4, 3.0),
        ]
    )

    result = refine_transcript(document, RefinePolicy(pause_split=0.8))

    assert [cue.text for cue in result.cues] == ["Well,", "this works."]


def test_duration_split_prefers_the_last_clause_boundary():
    document = TranscriptDocument(
        words=[
            Word("A", 0.0, 1.0),
            Word("short", 1.0, 2.0),
            Word("clause,", 2.0, 3.0),
            Word("followed", 3.0, 4.0),
            Word("by", 4.0, 5.0),
            Word("more", 5.0, 6.0),
            Word("words", 6.0, 7.0),
            Word("here.", 7.0, 8.0),
        ]
    )

    result = refine_transcript(document, RefinePolicy(max_duration=6.0))

    assert [cue.text for cue in result.cues] == [
        "A short clause,",
        "followed by more words here.",
    ]


def test_whisper_result_is_normalized_to_the_internal_word_document():
    document = document_from_whisper_result(
        {
            "language": "en",
            "segments": [
                {
                    "words": [
                        {"word": " Hello", "start": 0.1, "end": 0.5, "probability": 0.9},
                        {"word": " world.", "start": 0.5, "end": 1.0},
                    ]
                }
            ],
        }
    )

    assert document == TranscriptDocument(
        language="en",
        words=[
            Word("Hello", 0.1, 0.5, 0.9),
            Word("world.", 0.5, 1.0),
        ],
    )


def test_internal_word_document_round_trips_as_versioned_json(tmp_path):
    document = TranscriptDocument(
        language="zh",
        words=[Word("你好", 0.1, 0.8, 0.95), Word("。", 0.8, 1.0)],
    )
    output = tmp_path / "clip.words.json"

    write_word_document(
        output,
        document,
        backend="openai-whisper",
        model="turbo",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "video-txt.word-timestamps"
    assert payload["version"] == 1
    assert payload["backend"] == "openai-whisper"
    assert payload["model"] == "turbo"
    assert load_word_document(output) == document


def test_word_document_is_bound_to_the_srt_it_completes(tmp_path):
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text("refined subtitle\n", encoding="utf-8")
    output = tmp_path / "clip.words.json"
    write_word_document(
        output,
        TranscriptDocument(words=[Word("refined", 0.0, 1.0)]),
        backend="openai-whisper",
        model="turbo",
        subtitle_path=subtitle,
    )

    assert word_document_matches_subtitle(output, subtitle)

    subtitle.write_text("changed subtitle\n", encoding="utf-8")
    assert not word_document_matches_subtitle(output, subtitle)


def test_refinement_does_not_put_spaces_before_separate_punctuation_tokens():
    document = TranscriptDocument(
        words=[
            Word("Hello", 0.0, 0.3),
            Word(",", 0.3, 0.4),
            Word("world", 0.4, 0.8),
            Word("!", 0.8, 1.0),
        ]
    )

    result = refine_transcript(document)

    assert result.cues[0].text == "Hello, world!"


@pytest.mark.parametrize(
    "policy",
    [
        RefinePolicy(max_duration=0),
        RefinePolicy(max_line_width=0),
        RefinePolicy(max_line_width=1),
        RefinePolicy(max_lines=0),
        RefinePolicy(pause_split=-1),
    ],
)
def test_refinement_rejects_invalid_policy_values(policy):
    with pytest.raises(RefineError, match="policy"):
        refine_transcript(TranscriptDocument(words=[Word("Hi.", 0.0, 1.0)]), policy)


@pytest.mark.parametrize(
    "words",
    [
        [],
        [Word("bad", -1.0, 1.0)],
        [Word("bad", 2.0, 1.0)],
        [Word("later", 2.0, 3.0), Word("earlier", 1.0, 2.0)],
        [Word("bad", float("nan"), 1.0)],
    ],
)
def test_transcript_document_rejects_unusable_word_timings(words):
    with pytest.raises(RefineError, match="word timestamp"):
        TranscriptDocument(words=words)
