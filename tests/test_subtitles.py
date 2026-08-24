from __future__ import annotations

import pytest

from video_txt import subtitles as subtitles_module
from video_txt.subtitles import (
    SubtitleCue,
    SubtitleFormatError,
    build_ass_subtitle,
    chunk_cues,
    language_code,
    language_suffix,
    parse_srt,
    parse_srt_text,
    resolve_style_metrics,
    seconds_to_srt_time,
    serialize_srt,
    srt_time_to_ass,
    srt_time_to_seconds,
    translated_subtitle_path,
    write_srt,
)

SAMPLE = """1
00:00:01,000 --> 00:00:03,500
Hello there
second line

2
00:00:04,000 --> 00:00:06,000
Another cue
"""


def test_parse_srt_text_reads_index_timing_and_lines():
    cues = parse_srt_text(SAMPLE)
    assert [cue.index for cue in cues] == ["1", "2"]
    assert cues[0].text_lines == ["Hello there", "second line"]
    assert cues[0].timing == "00:00:01,000 --> 00:00:03,500"
    assert cues[1].text == "Another cue"


def test_parse_srt_handles_bom_and_crlf(tmp_path):
    path = tmp_path / "sample.srt"
    path.write_text(SAMPLE.replace("\n", "\r\n"), encoding="utf-8-sig")
    cues = parse_srt(path)
    assert len(cues) == 2
    assert cues[0].text_lines == ["Hello there", "second line"]


def test_parse_srt_text_keeps_empty_cue_instead_of_failing():
    cues = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\n\n2\n00:00:03,000 --> 00:00:04,000\nhi"
    )
    assert len(cues) == 2
    assert cues[0].is_empty
    assert cues[0].text_lines == []
    assert not cues[1].is_empty


def test_parse_srt_text_accepts_blocks_without_an_index():
    cues = parse_srt_text("00:00:01,000 --> 00:00:02,000\nno index here")
    assert cues[0].index == "1"
    assert cues[0].text == "no index here"


def test_parse_srt_text_rejects_blocks_without_a_time_range():
    with pytest.raises(SubtitleFormatError, match="no '-->' time range"):
        parse_srt_text("1\nnot a timing line\nsome text")


def test_parse_srt_text_rejects_empty_input():
    with pytest.raises(SubtitleFormatError, match="empty"):
        parse_srt_text("   \n\n  ")


def test_serialize_srt_round_trips_including_empty_cues():
    cues = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\n\n2\n00:00:03,000 --> 00:00:04,000\nhi"
    )
    assert parse_srt_text(serialize_srt(cues)) == cues


def test_write_srt_preserves_the_old_file_when_the_new_write_is_interrupted(tmp_path, monkeypatch):
    target = tmp_path / "out.srt"
    target.write_text("caller-owned subtitle\n", encoding="utf-8")
    cues = parse_srt_text(SAMPLE)

    def interrupted(_fd):
        raise OSError("disk write interrupted")

    monkeypatch.setattr(subtitles_module.os, "fsync", interrupted)

    with pytest.raises(OSError, match="interrupted"):
        write_srt(target, cues)

    assert target.read_text(encoding="utf-8") == "caller-owned subtitle\n"
    assert list(tmp_path.glob(".out.srt.*.tmp")) == []


def test_time_conversions():
    assert srt_time_to_seconds("00:00:01,500") == pytest.approx(1.5)
    assert srt_time_to_seconds("01:02:03,004") == pytest.approx(3723.004)
    assert seconds_to_srt_time(1.5) == "00:00:01,500"
    assert seconds_to_srt_time(-3) == "00:00:00,000"
    assert srt_time_to_ass("00:01:02,345") == "0:01:02.34"
    assert srt_time_to_ass("1:02:03.4") == "1:02:03.40"


def test_srt_time_to_ass_rejects_garbage():
    with pytest.raises(SubtitleFormatError):
        srt_time_to_ass("not a timestamp")


def test_cue_time_range_and_duration():
    cue = SubtitleCue("1", "00:00:02,000 --> 00:00:05,000", ["x"])
    assert cue.start_seconds == pytest.approx(2.0)
    assert cue.end_seconds == pytest.approx(5.0)
    assert cue.duration == pytest.approx(3.0)


def test_chunk_cues_splits_on_character_budget():
    cues = [SubtitleCue(str(i), "00:00:00,000 --> 00:00:01,000", ["x" * 40]) for i in range(5)]
    batches = chunk_cues(cues, 100)
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert sum(len(batch) for batch in batches) == 5


def test_chunk_cues_never_drops_an_oversized_cue():
    cues = [SubtitleCue("1", "00:00:00,000 --> 00:00:01,000", ["x" * 500])]
    assert chunk_cues(cues, 10) == [cues]


def test_language_suffix_and_output_path(tmp_path):
    assert language_suffix("Simplified Chinese") == "zh"
    assert language_suffix("中文") == "zh"
    assert language_suffix("English") == "en"
    assert language_suffix("Japanese") == "ja"
    assert language_suffix("日本語") == "ja"
    assert language_suffix("Klingon") == "translated"
    assert translated_subtitle_path(tmp_path / "a.srt", "zh").name == "a.zh.srt"
    assert translated_subtitle_path(tmp_path / "a.srt", "ja").name == "a.ja.srt"


def test_a_subtitle_track_never_claims_a_language_it_is_not():
    assert language_code("Japanese") == "jpn"
    assert language_code("繁体中文") == "zho"
    assert language_code("Simplified Chinese") == "zho"
    assert language_code("Klingon") == "und"


def test_traditional_and_simplified_chinese_land_in_different_files():
    assert language_suffix("Traditional Chinese") == "zh-hant"
    assert language_suffix("Simplified Chinese") == "zh"


def test_resolve_style_metrics_scales_with_video_height():
    assert resolve_style_metrics(video_height=1080, font_size=None, margin_v=None) == (49, 54)
    assert resolve_style_metrics(video_height=720, font_size=None, margin_v=None) == (32, 36)
    assert resolve_style_metrics(video_height=1080, font_size=36, margin_v=0) == (36, 0)


def test_build_ass_subtitle_uses_real_resolution_and_layout():
    cues = parse_srt_text(SAMPLE)
    ass = build_ass_subtitle(
        cues=cues,
        video_width=1920,
        video_height=1080,
        layout="top",
        font="PingFang SC",
        font_size=48,
        margin_v=60,
    )
    assert "PlayResX: 1920" in ass
    assert "PlayResY: 1080" in ass
    assert "Style: Default,PingFang SC,48," in ass
    assert ",1,4,0,8,77,77,60,1" in ass
    assert ass.count("Dialogue:") == 2
    assert r"Hello there\Nsecond line" in ass


def test_build_ass_subtitle_bottom_alignment_and_skips_empty_cues():
    cues = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\n\n2\n00:00:03,000 --> 00:00:04,000\nhi"
    )
    ass = build_ass_subtitle(
        cues=cues,
        video_width=1280,
        video_height=720,
        layout="normal",
        font="PingFang SC",
        font_size=32,
        margin_v=36,
    )
    assert ass.count("Dialogue:") == 1
    assert ",2,51,51,36,1" in ass


def test_build_ass_subtitle_sanitizes_font_and_escapes_braces():
    cues = [SubtitleCue("1", "00:00:01,000 --> 00:00:02,000", ["{weird} <i>tag</i> &amp; more"])]
    ass = build_ass_subtitle(
        cues=cues,
        video_width=1920,
        video_height=1080,
        layout="normal",
        font="Bad,Font",
        font_size=40,
        margin_v=50,
    )
    assert "Style: Default,Bad Font,40," in ass
    assert r"\{weird\} tag & more" in ass


def test_ass_text_keeps_literal_comparison_operators():
    cues = [
        SubtitleCue(
            "1",
            "00:00:01,000 --> 00:00:02,000",
            ["Use x < y and z > 2, but remove <i>markup</i>."],
        )
    ]

    ass = build_ass_subtitle(
        cues=cues,
        video_width=1280,
        video_height=720,
        layout="normal",
        font="PingFang SC",
        font_size=32,
        margin_v=36,
    )

    assert "x < y and z > 2" in ass
    assert "<i>" not in ass
