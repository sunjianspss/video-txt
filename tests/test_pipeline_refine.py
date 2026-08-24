from __future__ import annotations

import os

import pytest

from video_txt import pipeline as pipeline_module
from video_txt.pipeline import (
    TranscribeStage,
    TranslateStage,
    ensure_source_subtitle,
    ensure_translated_subtitle,
)
from video_txt.refine import TranscriptDocument, Word, write_word_document
from video_txt.terminology import Term, Terminology
from video_txt.transcribe import TranscribeError
from video_txt.translate import TranslationConfig, TranslationError


def test_pipeline_does_not_reuse_an_unrefined_srt_when_refinement_is_requested(
    tmp_path, monkeypatch
):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"media")
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text("existing but unrefined", encoding="utf-8")
    seen = []

    def transcribe(options, *, dry_run=False):
        seen.append((options, dry_run))
        return subtitle

    monkeypatch.setattr(pipeline_module, "run_transcribe", transcribe)

    output = ensure_source_subtitle(
        video,
        subtitle=None,
        output_dir=tmp_path,
        stage=TranscribeStage(refine_subtitles=True, skip_transcript_check=True),
    )

    assert output == subtitle
    assert seen[0][0].refine_subtitles is True


def test_pipeline_reuses_a_refined_srt_with_its_word_document(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"media")
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text("refined", encoding="utf-8")
    write_word_document(
        tmp_path / "clip.words.json",
        TranscriptDocument(words=[Word("refined", 0.0, 1.0)]),
        backend="openai-whisper",
        model="turbo",
        subtitle_path=subtitle,
    )
    calls = []
    monkeypatch.setattr(
        pipeline_module,
        "run_transcribe",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    output = ensure_source_subtitle(
        video,
        subtitle=None,
        output_dir=tmp_path,
        stage=TranscribeStage(refine_subtitles=True, skip_transcript_check=True),
    )

    assert output == subtitle
    assert calls == []


def test_pipeline_rejects_refinement_of_an_external_srt_without_word_timings(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"media")
    subtitle = tmp_path / "external.srt"
    subtitle.write_text("subtitle", encoding="utf-8")

    with pytest.raises(TranscribeError, match="word timestamps"):
        ensure_source_subtitle(
            video,
            subtitle=subtitle,
            output_dir=tmp_path,
            stage=TranscribeStage(refine_subtitles=True, skip_transcript_check=True),
        )


def test_pipeline_regenerates_a_refinement_with_a_corrupt_completion_marker(
    tmp_path, monkeypatch
):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"media")
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text("old", encoding="utf-8")
    (tmp_path / "clip.words.json").write_text("{}", encoding="utf-8")
    calls = []

    def transcribe(options, *, dry_run=False):
        calls.append((options, dry_run))
        return subtitle

    monkeypatch.setattr(pipeline_module, "run_transcribe", transcribe)

    ensure_source_subtitle(
        video,
        subtitle=None,
        output_dir=tmp_path,
        stage=TranscribeStage(refine_subtitles=True, skip_transcript_check=True),
    )

    assert len(calls) == 1


def test_pipeline_preserves_a_refined_srt_that_was_edited_by_hand(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"media")
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text("generated", encoding="utf-8")
    word_document = tmp_path / "clip.words.json"
    write_word_document(
        word_document,
        TranscriptDocument(words=[Word("generated", 0.0, 1.0)]),
        backend="openai-whisper",
        model="turbo",
        subtitle_path=subtitle,
    )
    subtitle.write_text("careful manual edit", encoding="utf-8")

    with pytest.raises(TranscribeError, match="--retranscribe"):
        ensure_source_subtitle(
            video,
            subtitle=None,
            output_dir=tmp_path,
            stage=TranscribeStage(refine_subtitles=True, skip_transcript_check=True),
        )

    assert subtitle.read_text(encoding="utf-8") == "careful manual edit"


def test_a_newly_refined_source_invalidates_an_older_translation(tmp_path, monkeypatch):
    source = tmp_path / "clip.srt"
    source.write_text("new refined source", encoding="utf-8")
    translated = tmp_path / "clip.zh.srt"
    translated.write_text("old translation", encoding="utf-8")
    os.utime(translated, ns=(1_000_000_000, 1_000_000_000))
    os.utime(source, ns=(2_000_000_000, 2_000_000_000))
    calls = []

    def translate(**kwargs):
        calls.append(kwargs)
        return translated

    monkeypatch.setattr(pipeline_module, "translate_subtitle_file", translate)

    output = ensure_translated_subtitle(
        source,
        stage=TranslateStage(
            config=TranslationConfig(
                base_url="https://example.test",
                api_key="key",
                model="model",
            ),
            output_path=translated,
        ),
    )

    assert output == translated
    assert len(calls) == 1


def test_a_reused_translation_is_audited_with_the_current_terminology(tmp_path):
    source = tmp_path / "story.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nWoody is here.\n", encoding="utf-8"
    )
    translated = tmp_path / "story.zh.srt"
    translated.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n伍迪来了。\n", encoding="utf-8"
    )
    terminology = Terminology(
        source_language="en",
        target_language="Simplified Chinese",
        terms=(Term("Woody", "胡迪", "word", ("伍迪",)),),
    )

    with pytest.raises(TranslationError, match="--retranslate"):
        ensure_translated_subtitle(
            source,
            stage=TranslateStage(
                config=TranslationConfig(
                    base_url="",
                    api_key="",
                    model="",
                    terminology=terminology,
                ),
                output_path=translated,
            ),
        )

    assert (tmp_path / "story.zh.translation-audit.json").is_file()
