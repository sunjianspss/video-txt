from __future__ import annotations

from video_txt.quality import audit_transcript, repair_transcript
from video_txt.subtitles import SubtitleCue, parse_srt_text, seconds_to_srt_time


def test_audit_flags_a_short_phrase_that_fills_a_whisper_window():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:29,980\nThank you.\n\n"
        "2\n00:00:31,000 --> 00:00:33,000\nLet's go.\n"
    )

    report = audit_transcript(cues, language="en")

    assert report.cue_count == 2
    assert report.is_clean is False
    assert [finding.code for finding in report.findings] == ["full_window_hallucination"]
    finding = report.findings[0]
    assert finding.cue_positions == (1,)
    assert finding.cue_indexes == ("1",)
    assert finding.repair == "remove"


def test_repair_removes_full_window_hallucinations_without_mutating_input():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello.\n\n"
        "2\n00:00:03,000 --> 00:00:32,980\nThank you.\n\n"
        "3\n00:00:34,000 --> 00:00:36,000\nGoodbye.\n"
    )
    original = [(cue.index, cue.timing, list(cue.text_lines)) for cue in cues]

    result = repair_transcript(cues, audit_transcript(cues))

    assert [cue.index for cue in result.cues] == ["1", "2"]
    assert [cue.text for cue in result.cues] == ["Hello.", "Goodbye."]
    assert result.removed_count == 1
    assert result.merged_count == 0
    assert [action.code for action in result.actions] == ["remove_full_window_hallucination"]
    assert [(cue.index, cue.timing, cue.text_lines) for cue in cues] == original


def test_zero_duration_fragment_is_reported_and_merged_into_previous_cue():
    cues = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:03,000\nI was\n\n"
        "2\n00:00:03,000 --> 00:00:03,000\ngoing.\n\n"
        "3\n00:00:04,000 --> 00:00:05,000\nNext line.\n"
    )

    report = audit_transcript(cues)
    finding = next(finding for finding in report.findings if finding.code == "zero_duration")
    result = repair_transcript(cues, report)

    assert finding.repair == "merge_previous"
    assert [cue.text for cue in result.cues] == ["I was going.", "Next line."]
    assert [cue.index for cue in result.cues] == ["1", "2"]
    assert result.merged_count == 1
    assert [action.code for action in result.actions] == ["merge_zero_duration"]


def test_empty_cue_is_removed_safely():
    cues = [
        SubtitleCue("1", "00:00:01,000 --> 00:00:02,000", ["Hello"]),
        SubtitleCue("2", "00:00:03,000 --> 00:00:04,000", []),
    ]

    report = audit_transcript(cues)
    result = repair_transcript(cues, report)

    assert [finding.code for finding in report.findings] == ["empty_cue"]
    assert [cue.text for cue in result.cues] == ["Hello"]
    assert result.removed_count == 1
    assert [action.code for action in result.actions] == ["remove_empty_cue"]


def test_reversed_timing_is_reported_but_never_guessed_at():
    cue = SubtitleCue("1", "00:00:05,000 --> 00:00:04,000", ["Hello"])

    report = audit_transcript([cue])
    result = repair_transcript([cue], report)

    assert [(finding.code, finding.severity, finding.repair) for finding in report.findings] == [
        ("reversed_timing", "error", None)
    ]
    assert result.actions == ()
    assert [finding.code for finding in result.unresolved] == ["reversed_timing"]
    assert result.cues[0].timing == cue.timing


def test_overlapping_cues_are_reported_without_changing_them():
    cues = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:04,000\nFirst speaker.\n\n"
        "2\n00:00:03,500 --> 00:00:05,000\nSecond speaker.\n"
    )

    report = audit_transcript(cues)
    overlap = next(finding for finding in report.findings if finding.code == "overlap")
    result = repair_transcript(cues, report)

    assert overlap.cue_positions == (1, 2)
    assert overlap.cue_indexes == ("1", "2")
    assert overlap.repair is None
    assert report.has_errors is False
    assert result.actions == ()
    assert [cue.timing for cue in result.cues] == [cue.timing for cue in cues]


def test_overlong_cue_is_reported_for_manual_review():
    cue = SubtitleCue(
        "1",
        "00:00:01,000 --> 00:00:13,000",
        ["One two three four five six seven."],
    )

    report = audit_transcript([cue])

    assert [finding.code for finding in report.findings] == ["overlong_duration"]
    assert report.findings[0].repair is None


def test_more_than_two_lines_is_reported_as_a_layout_problem():
    cue = SubtitleCue(
        "1",
        "00:00:01,000 --> 00:00:03,000",
        ["First line", "Second line", "Third line"],
    )

    report = audit_transcript([cue])

    assert [finding.code for finding in report.findings] == ["too_many_lines"]
    assert "3 lines" in report.findings[0].message


def test_wide_cjk_line_uses_display_width_not_character_count():
    cue = SubtitleCue(
        "1",
        "00:00:01,000 --> 00:00:03,000",
        ["这是一行需要按照双倍显示宽度检查的中文字幕内容"],
    )

    report = audit_transcript([cue])

    assert [finding.code for finding in report.findings] == ["line_too_wide"]
    assert "display columns" in report.findings[0].message


def test_non_sequential_indexes_are_reported_and_renumbered():
    cues = parse_srt_text(
        "4\n00:00:01,000 --> 00:00:02,000\nHello.\n\n"
        "9\n00:00:03,000 --> 00:00:04,000\nGoodbye.\n"
    )

    report = audit_transcript(cues)
    result = repair_transcript(cues, report)

    assert [finding.code for finding in report.findings] == ["non_sequential_indexes"]
    assert report.findings[0].repair == "renumber"
    assert [cue.index for cue in result.cues] == ["1", "2"]
    assert [action.code for action in result.actions] == ["renumber_cues"]


def test_existing_repeat_and_language_checks_are_structured_findings():
    cues = [
        SubtitleCue(
            str(position),
            f"{seconds_to_srt_time(position * 6)} --> "
            f"{seconds_to_srt_time(position * 6 + 6)}",
            ["This English sentence keeps repeating every single time."],
        )
        for position in range(1, 13)
    ]

    report = audit_transcript(cues, language="ja")

    codes = [finding.code for finding in report.findings]
    assert "repeated_text" in codes
    assert "wrong_script" in codes


def test_media_coverage_check_is_a_structured_finding():
    cue = SubtitleCue("1", "00:00:00,000 --> 00:00:25,000", ["Hello"])

    report = audit_transcript([cue], media_duration=2900.0)

    assert "low_coverage" in [finding.code for finding in report.findings]


def test_cues_past_the_end_of_the_media_are_a_warning_not_a_blocker():
    """A subtitle from another cut is worth saying; it is not a broken transcript."""
    cues = [
        SubtitleCue("1", "00:24:10,000 --> 00:24:14,000", ["Hello there."]),
        SubtitleCue("2", "00:48:21,000 --> 00:48:25,000", ["Goodbye now."]),
    ]

    report = audit_transcript(cues, media_duration=2900.0)

    overrun = [finding for finding in report.findings if finding.code == "past_media_end"]
    assert len(overrun) == 1
    assert overrun[0].severity == "warning"
    assert overrun[0].cue_indexes == ("2",)
    assert not report.has_errors


def test_full_window_dialogue_is_not_removed_when_it_is_not_a_short_phrase():
    cue = SubtitleCue(
        "1",
        "00:00:00,000 --> 00:00:29,980",
        ["One two three four five six seven."],
    )

    report = audit_transcript([cue])
    result = repair_transcript([cue], report)

    assert "full_window_hallucination" not in [finding.code for finding in report.findings]
    assert result.removed_count == 0
    assert [fixed.text for fixed in result.cues] == [cue.text]


def test_first_zero_duration_cue_is_left_for_manual_review():
    cue = SubtitleCue("1", "00:00:01,000 --> 00:00:01,000", ["Hello"])

    report = audit_transcript([cue])
    result = repair_transcript([cue], report)

    finding = next(finding for finding in report.findings if finding.code == "zero_duration")
    assert finding.repair is None
    assert result.actions == ()
    assert [fixed.text for fixed in result.cues] == ["Hello"]


def test_zero_duration_fragment_is_not_merged_across_a_removed_cue():
    cues = parse_srt_text(
        "1\n00:00:00,000 --> 00:00:29,980\nThank you.\n\n"
        "2\n00:00:29,980 --> 00:00:29,980\nFragment.\n"
    )

    report = audit_transcript(cues)
    result = repair_transcript(cues, report)

    zero = next(finding for finding in report.findings if finding.code == "zero_duration")
    assert zero.repair is None
    assert [cue.text for cue in result.cues] == ["Fragment."]
    assert [action.code for action in result.actions] == ["remove_full_window_hallucination"]
    assert "zero_duration" in [finding.code for finding in result.unresolved]
