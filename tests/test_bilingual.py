from __future__ import annotations

import pytest

from video_txt.bilingual import (
    BilingualError,
    bilingual_cues,
    bilingual_subtitle_path,
    merge_subtitles,
    secondary_line_counts,
)
from video_txt.cli import main
from video_txt.mux import MuxOptions, prepare_styled_subtitle
from video_txt.subtitles import (
    SubtitleFormatError,
    build_ass_subtitle,
    parse_srt,
    parse_srt_text,
    write_srt,
)

SOURCE = (
    "1\n00:00:01,000 --> 00:00:02,000\nWhere is Jessie?\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nWe reviewed every figure\nand the board asked for more.\n"
)
TRANSLATION = (
    "1\n00:00:01,000 --> 00:00:02,000\n翠丝在哪里？\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\n我们复核了每个数字，\n董事会要求补充。\n"
)


def merged(source: str = SOURCE, translation: str = TRANSLATION, **kwargs):
    return merge_subtitles(parse_srt_text(source), parse_srt_text(translation), **kwargs)


def test_the_translation_reads_first_and_the_original_sits_under_it():
    result = merged()

    assert [cue.text_lines for cue in bilingual_cues(result)] == [
        ["翠丝在哪里？", "Where is Jessie?"],
        [
            "我们复核了每个数字，董事会要求补充。",
            "We reviewed every figure and the board asked for more.",
        ],
    ]
    assert secondary_line_counts(result) == [1, 1]


def test_the_original_can_read_first_instead():
    result = bilingual_cues(merged(order="source-first"))

    assert result[0].text_lines == ["Where is Jessie?", "翠丝在哪里？"]


def test_each_language_is_flattened_onto_one_line():
    """A source broken over two lines and a translation broken over two would
    stack four deep and take half the frame. The break was a decision about one
    language's line width and does not survive being put beside another."""
    result = bilingual_cues(merged())

    assert all(len(cue.text_lines) == 2 for cue in result)
    # Joined the way each language writes it: a space in English, none in Chinese.
    assert result[1].text_lines[0] == "我们复核了每个数字，董事会要求补充。"
    assert "figure and the board" in result[1].text_lines[1]


def test_timings_and_indexes_come_from_the_pair_unchanged():
    result = bilingual_cues(merged())

    assert [cue.index for cue in result] == ["1", "2"]
    assert result[0].timing == "00:00:01,000 --> 00:00:02,000"


def test_two_subtitles_that_are_not_the_same_subtitle_are_refused():
    with pytest.raises(BilingualError, match="audit-translation"):
        merged(translation="1\n00:00:01,000 --> 00:00:02,000\n只有一句\n")


def test_a_cue_timed_differently_in_the_two_files_is_refused():
    shifted = TRANSLATION.replace("00:00:03,000 --> 00:00:04,000", "00:00:03,500 --> 00:00:04,000")

    with pytest.raises(BilingualError, match="timed differently"):
        merged(translation=shifted)


def test_an_unknown_order_is_refused():
    with pytest.raises(BilingualError, match="Unknown bilingual order"):
        merged(order="upside-down")


def test_a_cue_with_only_one_language_keeps_one_line_and_no_second_style():
    result = merge_subtitles(
        parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\nOnly English.\n"),
        parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\n \n"),
    )

    assert bilingual_cues(result)[0].text_lines == ["Only English."]
    assert secondary_line_counts(result) == [0]


def test_the_burned_original_is_smaller_and_dimmer_than_the_line_above_it():
    cues = bilingual_cues(merged())

    ass = build_ass_subtitle(
        cues=cues,
        video_width=1920,
        video_height=1080,
        layout="normal",
        font="PingFang SC",
        font_size=50,
        margin_v=54,
        secondary_lines=secondary_line_counts(merged()),
    )

    dialogue = [line for line in ass.splitlines() if line.startswith("Dialogue")]
    # 50 * 0.72 rounds to 36; the override opens on the second line only.
    assert r"翠丝在哪里？\N{\fs36\alpha&H50&}Where is Jessie?" in dialogue[0]
    assert "{" not in dialogue[0].split(r"\N")[0]


def test_one_language_is_still_styled_as_one_language():
    ass = build_ass_subtitle(
        cues=parse_srt_text(TRANSLATION),
        video_width=1920,
        video_height=1080,
        layout="normal",
        font="PingFang SC",
        font_size=50,
        margin_v=54,
    )

    assert "\\alpha" not in ass


def test_a_secondary_count_per_cue_is_required_when_given():
    with pytest.raises(SubtitleFormatError, match="secondary-line counts"):
        build_ass_subtitle(
            cues=parse_srt_text(TRANSLATION),
            video_width=1920,
            video_height=1080,
            layout="normal",
            font="PingFang SC",
            font_size=50,
            margin_v=54,
            secondary_lines=[1],
        )


def test_hard_subtitles_find_the_secondary_line_in_a_bilingual_file(tmp_path):
    """The burn reads a flat .srt, so which line is the original has to be
    derivable from it: a bilingual cue is exactly [translation, original]."""
    subtitle = tmp_path / "clip.zh.bilingual.srt"
    write_srt(subtitle, bilingual_cues(merged()))
    options = MuxOptions(
        video_input=tmp_path / "clip.mp4",
        subtitle_input=subtitle,
        video_output=tmp_path / "out.mp4",
        mux_mode="hard",
        video_size=(1920, 1080),
        bilingual=True,
    )

    written = prepare_styled_subtitle(
        options, ffmpeg_path="ffmpeg", output_path=tmp_path / "clip.ass"
    )

    assert written.read_text(encoding="utf-8").count(r"\alpha&H50&") == 2


def test_bilingual_command_writes_the_merged_subtitle(tmp_path, capsys):
    source = tmp_path / "clip.srt"
    source.write_text(SOURCE, encoding="utf-8")
    translation = tmp_path / "clip.zh.srt"
    translation.write_text(TRANSLATION, encoding="utf-8")

    assert main(["bilingual", str(source), str(translation)]) == 0

    output = tmp_path / "clip.zh.bilingual.srt"
    assert parse_srt(output)[0].text_lines == ["翠丝在哪里？", "Where is Jessie?"]
    assert translation.read_text(encoding="utf-8") == TRANSLATION
    assert "2 cue(s) in both languages" in capsys.readouterr().out


def test_bilingual_command_refuses_to_overwrite_an_input(tmp_path):
    source = tmp_path / "clip.srt"
    source.write_text(SOURCE, encoding="utf-8")
    translation = tmp_path / "clip.zh.srt"
    translation.write_text(TRANSLATION, encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["bilingual", str(source), str(translation), "-o", str(translation)])

    assert translation.read_text(encoding="utf-8") == TRANSLATION


def test_bilingual_subtitle_path_sits_beside_the_translation(tmp_path):
    assert (
        bilingual_subtitle_path(tmp_path / "clip.zh.srt") == tmp_path / "clip.zh.bilingual.srt"
    )
