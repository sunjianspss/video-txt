from __future__ import annotations

import json

import pytest

from video_txt.subtitles import parse_srt_text
from video_txt.terminology import (
    Term,
    Terminology,
    TerminologyError,
    audit_translation,
    enforce_terminology,
    load_terminology,
    write_translation_audit_report,
)


def test_project_terminology_file_loads_the_approved_mapping_contract(tmp_path):
    path = tmp_path / "project.terms.json"
    path.write_text(
        json.dumps(
            {
                "schema": "video-txt.terminology",
                "version": 1,
                "source_language": "en",
                "target_language": "Simplified Chinese",
                "terms": [
                    {
                        "source": "Woody",
                        "target": "胡迪",
                        "match": "word",
                        "aliases": ["伍迪", "乌迪"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    terminology = load_terminology(path)

    assert terminology.source_language == "en"
    assert terminology.target_language == "Simplified Chinese"
    assert terminology.terms[0].source == "Woody"
    assert terminology.terms[0].target == "胡迪"
    assert terminology.terms[0].match == "word"
    assert terminology.terms[0].aliases == ("伍迪", "乌迪")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema": "other"}, "schema"),
        ({"version": 2}, "version"),
        ({"terms": [{"source": "", "target": "胡迪"}]}, "non-empty"),
        (
            {"terms": [{"source": "Woody", "target": "胡迪", "match": "fuzzy"}]},
            "match",
        ),
        (
            {
                "terms": [
                    {"source": "Woody", "target": "胡迪"},
                    {"source": "woody", "target": "伍迪"},
                ]
            },
            "duplicate",
        ),
        (
            {
                "terms": [
                    {"source": "Woody", "target": "胡迪", "aliases": ["伍迪"]},
                    {"source": "Buzz", "target": "巴斯", "aliases": ["伍迪"]},
                ]
            },
            "ambiguous",
        ),
    ],
)
def test_project_terminology_rejects_an_ambiguous_contract(tmp_path, change, message):
    payload = {
        "schema": "video-txt.terminology",
        "version": 1,
        "terms": [{"source": "Woody", "target": "胡迪"}],
    }
    payload.update(change)
    path = tmp_path / "bad.terms.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(TerminologyError, match=message):
        load_terminology(path)


def test_enforcement_only_normalizes_approved_terms_in_the_matching_source_cue():
    source = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\nWoody is here.\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nThe Woodyard is open.\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nBuzz Lightyear!\n"
    )
    translated = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\n伍迪来了。\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n伍迪工坊开门了。\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nBuzz Lightyear！\n"
    )
    terminology = Terminology(
        source_language="en",
        target_language="Simplified Chinese",
        terms=(
            Term("Woody", "胡迪", "word", ("伍迪",)),
            Term("Buzz Lightyear", "巴斯光年", "phrase"),
        ),
    )

    result = enforce_terminology(source, translated, terminology)

    assert [cue.text for cue in result.cues] == ["胡迪来了。", "伍迪工坊开门了。", "巴斯光年！"]
    assert [cue.timing for cue in result.cues] == [cue.timing for cue in translated]
    assert [change.position for change in result.changes] == [1, 3]


def test_translation_audit_reports_structure_terms_and_suspicious_text():
    source = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\nWoody is here.\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nThis line remains entirely untranslated.\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nHi\n\n"
        "4\n00:00:07,000 --> 00:00:08,000\nBye\n"
    )
    translated = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\n伍迪在这里。\n\n"
        "9\n00:00:03,500 --> 00:00:04,000\nThis line remains entirely untranslated.\n\n"
        f"3\n00:00:05,000 --> 00:00:06,000\n{'很' * 100}\n"
    )
    terminology = Terminology(
        source_language="en",
        target_language="Simplified Chinese",
        terms=(Term("Woody", "胡迪", "word", ("伍迪",)),),
    )

    report = audit_translation(source, translated, terminology)

    codes = {finding.code for finding in report.findings}
    assert codes == {
        "cue_count_mismatch",
        "cue_index_mismatch",
        "cue_timing_mismatch",
        "glossary_alias",
        "glossary_target_missing",
        "source_text_unchanged",
        "translation_unusually_long",
    }
    assert report.has_errors is True
    assert report.summary["error"] == 5
    assert report.summary["warning"] == 2

    fixed = enforce_terminology(source[:1], translated[:1], terminology)
    clean = audit_translation(source[:1], fixed.cues, terminology)
    assert clean.is_clean is True


def test_translation_audit_report_preserves_an_existing_file_without_permission(tmp_path):
    source_path = tmp_path / "source.srt"
    translation_path = tmp_path / "source.zh.srt"
    report_path = tmp_path / "kept.json"
    report_path.write_text("caller-owned", encoding="utf-8")
    source = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\nHello there.\n")
    translation = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\n你好。\n")

    with pytest.raises(TerminologyError, match="already exists"):
        write_translation_audit_report(
            source_path=source_path,
            translation_path=translation_path,
            terminology=None,
            audit=audit_translation(source, translation),
            report_path=report_path,
        )

    assert report_path.read_text(encoding="utf-8") == "caller-owned"


def test_terms_may_move_by_one_cue_when_a_sentence_is_reflowed():
    source = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\nWe tried all summer\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nto meet the Jordan twins.\n"
    )
    translated = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\n我们整个夏天都想认识乔丹家的双胞胎\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n并和她们成为朋友。\n"
    )
    terminology = Terminology(
        source_language="en",
        target_language="Simplified Chinese",
        terms=(Term("Jordan twins", "乔丹双胞胎", "phrase", ("乔丹家的双胞胎",)),),
    )

    enforced = enforce_terminology(source, translated, terminology)
    report = audit_translation(source, enforced.cues, terminology)

    assert enforced.cues[0].text == "我们整个夏天都想认识乔丹双胞胎"
    assert report.is_clean is True


def test_latin_word_residue_is_replaced_when_it_touches_cjk_text():
    source = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\nIs that Blaze?\n")
    translated = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\n是那个Blaze吗？\n")
    terminology = Terminology(
        source_language="en",
        target_language="Simplified Chinese",
        terms=(Term("Blaze", "布蕾兹", "word"),),
    )

    result = enforce_terminology(source, translated, terminology)

    assert result.cues[0].text == "是那个布蕾兹吗？"
    assert audit_translation(source, result.cues, terminology).is_clean is True


def test_translation_audit_warns_about_a_multiword_source_fragment_left_behind():
    source = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\nInstall the package and continue.\n"
    )
    translated = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\n安装后执行 install the package，然后继续。\n"
    )

    report = audit_translation(source, translated)

    assert [finding.code for finding in report.findings] == ["source_text_residue"]
    assert report.summary["warning"] == 1
