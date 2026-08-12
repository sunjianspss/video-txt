from __future__ import annotations

from pathlib import Path

import pytest

from video_txt import cli as cli_module
from video_txt.cli import main
from video_txt.pipeline import TranscribeStage, ensure_source_subtitle
from video_txt.quality import check_transcript, stuck_runs, transcript_problems
from video_txt.subtitles import SubtitleCue, parse_srt_text, seconds_to_srt_time
from video_txt.transcribe import TranscribeError


def cues(entries: list[tuple[str, float, float]]) -> list[SubtitleCue]:
    blocks = [
        f"{index}\n{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n{text}\n"
        for index, (text, start, end) in enumerate(entries, start=1)
    ]
    return parse_srt_text("\n".join(blocks))


def every_30_seconds(text: str, count: int, *, first: float = 0.0) -> list[SubtitleCue]:
    """The shape Whisper leaves behind: one line per decoding window, nothing else."""
    return cues([(text, first + step * 30.0, first + step * 30.0 + 29.98) for step in range(count)])


JAPANESE = [
    ("なぜ", 30.0, 35.2),
    ("誰だ", 35.2, 40.1),
    ("伏せろ！", 40.1, 42.0),
    ("かんちゃん逃げて", 42.0, 45.0),
]


def test_a_line_that_repeats_for_minutes_is_called_out():
    problems = transcript_problems(every_30_seconds("Let's go.", 10))
    assert len(problems) == 1
    assert '"Let\'s go." repeats 10 times in a row' in problems[0]
    assert "(0:00:00 -> 0:04:59)" in problems[0]


def test_a_word_said_a_few_times_in_a_row_is_left_alone():
    shouting = cues([("はい", start, start + 1.0) for start in (1.0, 2.5, 4.0, 5.5, 7.0)])
    assert transcript_problems(shouting) == []


def test_a_short_repeat_is_not_enough_to_worry():
    assert transcript_problems(every_30_seconds("Thank you.", 3)) == []


def test_an_interrupted_loop_is_still_two_loops():
    spinning = [
        *every_30_seconds("Let's go.", 5),
        *cues([("犯人はこの中にいる", 200.0, 203.0)]),
        *every_30_seconds("Let's go.", 6, first=210.0),
    ]
    runs = stuck_runs(spinning)
    assert [run.count for run in runs] == [6, 5]


def test_real_dialogue_raises_nothing():
    assert transcript_problems(cues(JAPANESE), language="ja") == []


def test_asking_for_japanese_and_getting_english_is_called_out():
    english = cues([("Who is it?", 30.0, 35.0), ("Get down!", 35.0, 40.0)])
    problems = transcript_problems(english, language="日本語")
    assert problems == [
        "--language asked for Japanese, but only 0% of the lines contain Japanese characters"
    ]


def test_a_latin_script_language_cannot_be_checked_this_way():
    english = cues([("Who is it?", 30.0, 35.0), ("Get down!", 35.0, 40.0)])
    assert transcript_problems(english, language="French") == []


def test_a_few_english_lines_among_japanese_are_tolerated():
    mixed = [*cues(JAPANESE), *cues([("OK", 46.0, 47.0)])]
    assert transcript_problems(mixed, language="ja") == []


def test_the_report_says_what_to_do_about_it(tmp_path: Path):
    subtitle = tmp_path / "clip.srt"
    subtitle.write_text(
        "".join(
            f"{index}\n{seconds_to_srt_time(index * 30.0)} --> "
            f"{seconds_to_srt_time(index * 30.0 + 29.98)}\nLet's go.\n\n"
            for index in range(10)
        ),
        encoding="utf-8",
    )
    report = check_transcript(subtitle, language="ja")
    assert report is not None
    assert "--retranscribe --language ja" in report
    assert "--condition_on_previous_text" in report
    assert "--skip-transcript-check" in report


def test_a_transcript_that_cannot_be_read_is_not_second_guessed(tmp_path: Path):
    assert check_transcript(tmp_path / "missing.srt") is None
    unreadable = tmp_path / "clip.srt"
    unreadable.write_text("this is not a subtitle file", encoding="utf-8")
    assert check_transcript(unreadable) is None


def broken_subtitle(tmp_path: Path) -> Path:
    path = tmp_path / "clip.en.srt"
    path.write_text(
        "".join(
            f"{index}\n{seconds_to_srt_time(index * 30.0)} --> "
            f"{seconds_to_srt_time(index * 30.0 + 29.98)}\nThank you.\n\n"
            for index in range(10)
        ),
        encoding="utf-8",
    )
    return path


def test_the_pipeline_stops_before_paying_for_a_broken_transcript(tmp_path: Path, capsys):
    subtitle = broken_subtitle(tmp_path)
    with pytest.raises(TranscribeError) as stop:
        ensure_source_subtitle(
            tmp_path / "clip.mp4",
            subtitle=subtitle,
            output_dir=tmp_path,
            stage=TranscribeStage(language="ja"),
        )
    message = str(stop.value)
    assert "'Thank you.' repeats 10 times in a row" in message
    assert "Stopping before the steps that cost time." in message


def test_the_pipeline_can_be_told_to_use_a_broken_transcript_anyway(tmp_path: Path):
    subtitle = broken_subtitle(tmp_path)
    assert (
        ensure_source_subtitle(
            tmp_path / "clip.mp4",
            subtitle=subtitle,
            output_dir=tmp_path,
            stage=TranscribeStage(language="ja", skip_transcript_check=True),
        )
        == subtitle
    )


def test_transcribing_on_its_own_keeps_the_file_but_reports_a_failure(
    tmp_path: Path, monkeypatch, capsys
):
    """The transcript is on disk to look at; the exit code is what stops a && chain."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    subtitle = broken_subtitle(tmp_path)
    monkeypatch.setattr(cli_module, "run_transcribe", lambda options, dry_run=False: subtitle)

    assert main(["transcribe", str(video), "-f", "srt", "--language", "ja"]) == 1
    captured = capsys.readouterr()
    assert "'Thank you.' repeats 10 times in a row" in captured.err
    assert f"Done: {subtitle}" in captured.out
    assert subtitle.is_file()


def test_a_transcript_that_reads_fine_finishes_quietly(tmp_path: Path, monkeypatch, capsys):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    subtitle = tmp_path / "clip.en.srt"
    subtitle.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nHello there\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nGood to see you\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_module, "run_transcribe", lambda options, dry_run=False: subtitle)

    assert main(["transcribe", str(video), "-f", "srt"]) == 0
    assert capsys.readouterr().err == ""


def test_a_dry_run_does_not_need_a_transcript_to_check(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    assert (
        ensure_source_subtitle(
            video,
            subtitle=None,
            output_dir=tmp_path,
            stage=TranscribeStage(),
            dry_run=True,
        )
        == tmp_path / "clip.srt"
    )
