from __future__ import annotations

import os

import pytest

from video_txt.pipeline import TranslateStage, ensure_translated_subtitle
from video_txt.project import (
    evaluate_stages,
    fingerprint_payload,
    sha256_file,
    stage_record,
)
from video_txt.translate import TranslationConfig


def base_inputs():
    return {
        "transcribe": {
            "video_sha256": "video-a",
            "settings": {"audio_stream": 2, "language": "en"},
        },
        "translate": {
            "source_sha256": "source-a",
            "terminology_sha256": "terms-a",
            "settings": {"model": "translator"},
        },
        "mux": {
            "video_sha256": "video-a",
            "translation_sha256": "zh-a",
            "settings": {"font": "PingFang SC"},
        },
        "tts": {
            "translation_sha256": "zh-a",
            "settings": {"voice": "Alice"},
        },
    }


def compare_to_recorded(desired, recorded_inputs):
    return evaluate_stages(
        desired,
        recorded={name: stage_record(inputs) for name, inputs in recorded_inputs.items()},
        artifacts={name: True for name in desired},
        enabled=set(desired),
    )


def test_terminology_change_invalidates_translation_and_downstream_only():
    desired = base_inputs()
    desired["translate"]["terminology_sha256"] = "terms-b"
    recorded_inputs = {
        **desired,
        "translate": {
            **desired["translate"],
            "terminology_sha256": "terms-a",
        },
    }
    recorded = {name: stage_record(inputs) for name, inputs in recorded_inputs.items()}

    statuses = evaluate_stages(
        desired,
        recorded=recorded,
        artifacts={name: True for name in desired},
        enabled=set(desired),
    )

    assert statuses["transcribe"].stale is False
    assert statuses["translate"].stale is True
    assert "terminology" in statuses["translate"].reason
    assert statuses["mux"].stale is True
    assert statuses["tts"].stale is True


@pytest.mark.parametrize(
    ("stage", "key", "value", "expected_stale"),
    [
        ("transcribe", "settings", {"audio_stream": 1, "language": "en"}, set(base_inputs())),
        ("translate", "source_sha256", "source-b", {"translate", "mux", "tts"}),
        ("mux", "settings", {"font": "Noto Sans CJK SC"}, {"mux"}),
        ("tts", "settings", {"voice": "Bob"}, {"tts"}),
    ],
)
def test_stage_changes_have_the_narrow_expected_blast_radius(
    stage, key, value, expected_stale
):
    recorded = base_inputs()
    desired = base_inputs()
    desired[stage][key] = value

    statuses = compare_to_recorded(desired, recorded)

    assert {name for name, status in statuses.items() if status.stale} == expected_stale


def test_stage_fingerprints_are_independent_of_json_key_order():
    assert fingerprint_payload({"language": "en", "model": "turbo"}) == fingerprint_payload(
        {"model": "turbo", "language": "en"}
    )


def test_content_fingerprint_reads_the_whole_file_and_ignores_mtime(tmp_path):
    artifact = tmp_path / "large.bin"
    artifact.write_bytes(b"a" * 1_100_000 + b"middle-a" + b"z" * 1_100_000)
    first = sha256_file(artifact)
    stat = artifact.stat()
    os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert sha256_file(artifact) == first

    artifact.write_bytes(b"a" * 1_100_000 + b"middle-b" + b"z" * 1_100_000)
    assert sha256_file(artifact) != first


def test_project_policy_can_reuse_identical_translation_despite_source_mtime(
    tmp_path, monkeypatch
):
    source = tmp_path / "movie.srt"
    translated = tmp_path / "movie.zh.srt"
    sample = "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
    source.write_text(sample, encoding="utf-8")
    translated.write_text(sample, encoding="utf-8")
    translated.with_name("movie.zh.translation-audit.json").write_text(
        "{}", encoding="utf-8"
    )
    stat = translated.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    monkeypatch.setattr(
        "video_txt.pipeline.translate_subtitle_file",
        lambda **_kwargs: pytest.fail("content-identical project input must be reused"),
    )

    result = ensure_translated_subtitle(
        source,
        stage=TranslateStage(
            config=TranslationConfig(
                base_url="",
                api_key="",
                model="",
                target_language="Simplified Chinese",
            ),
            output_path=translated,
            reuse_if_exists=True,
        ),
        dry_run=True,
    )

    assert result == translated
