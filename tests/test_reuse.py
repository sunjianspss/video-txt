from __future__ import annotations

import pytest

from video_txt.reuse import (
    PreviousTranslation,
    ReuseError,
    match_translations,
    plan_reuse,
)
from video_txt.subtitles import SubtitleCue, write_srt


def cue(index: int, start: float, text: str, *, length: float = 2.0) -> SubtitleCue:
    def clock(seconds: float) -> str:
        return (
            f"{int(seconds // 3600):02d}:{int(seconds // 60) % 60:02d}:"
            f"{int(seconds) % 60:02d},{round(seconds % 1 * 1000):03d}"
        )

    return SubtitleCue(str(index), f"{clock(start)} --> {clock(start + length)}", [text])


BEFORE = [
    cue(1, 0.0, "Welcome to the show."),
    cue(2, 4.0, "Mumble mumble."),
    cue(3, 8.0, "Thanks for having me."),
]
BEFORE_ZH = [
    cue(1, 0.0, "欢迎收看。"),
    cue(2, 4.0, "咕哝咕哝。"),
    cue(3, 8.0, "谢谢邀请。"),
]


def test_only_the_repaired_line_is_left_for_the_model():
    repaired = [
        cue(1, 0.0, "Welcome to the show."),
        cue(2, 4.0, "What the microphone actually caught."),
        cue(3, 8.0, "Thanks for having me."),
    ]

    matched = match_translations(
        repaired, previous_source=BEFORE, previous_translation=BEFORE_ZH
    )

    assert matched == {"1": "欢迎收看。", "3": "谢谢邀请。"}


def test_a_line_that_only_moved_or_was_respaced_keeps_its_translation():
    """retranscribe-range renumbers and retimes; the words are what identify a line."""
    moved = [cue(1, 61.0, "  Welcome   to the SHOW.  ")]

    matched = match_translations(moved, previous_source=BEFORE, previous_translation=BEFORE_ZH)

    assert matched == {"1": "欢迎收看。"}


def test_a_line_said_twice_takes_the_translation_from_its_own_moment():
    previous = [cue(1, 10.0, "Yes."), cue(2, 600.0, "Yes.")]
    previous_zh = [cue(1, 10.0, "好的。"), cue(2, 600.0, "是。")]

    matched = match_translations(
        [cue(1, 598.0, "Yes.")], previous_source=previous, previous_translation=previous_zh
    )

    assert matched == {"1": "是。"}


def test_a_held_pause_is_carried_over_like_any_other_line():
    """'...' costs an API call of its own when it is not carried over."""
    previous = [cue(1, 0.0, "..."), cue(2, 4.0, "Welcome to the show.")]
    previous_zh = [cue(1, 0.0, "……"), cue(2, 4.0, "欢迎收看。")]

    matched = match_translations(
        [cue(1, 0.0, "..."), cue(2, 4.0, "Welcome to the show.")],
        previous_source=previous,
        previous_translation=previous_zh,
    )

    assert matched == {"1": "……", "2": "欢迎收看。"}


def test_cues_with_no_text_at_all_are_left_out_of_the_matching():
    """An empty cue never reaches the model, so it has no translation to carry."""
    previous = [cue(1, 0.0, ""), cue(2, 4.0, "Welcome to the show.")]
    previous_zh = [cue(1, 0.0, ""), cue(2, 4.0, "欢迎收看。")]

    matched = match_translations(
        [cue(1, 0.0, ""), cue(2, 4.0, "Welcome to the show.")],
        previous_source=previous,
        previous_translation=previous_zh,
    )

    assert matched == {"2": "欢迎收看。"}


def test_a_previous_pair_that_does_not_line_up_is_refused():
    """Nothing sensible can be carried over when the two files disagree on cue count."""
    with pytest.raises(ReuseError, match="do not line up"):
        match_translations(BEFORE, previous_source=BEFORE, previous_translation=BEFORE_ZH[:2])


def test_the_plan_counts_what_is_carried_over_and_what_is_left(tmp_path):
    source = tmp_path / "before.srt"
    translation = tmp_path / "before.zh.srt"
    write_srt(source, BEFORE)
    write_srt(translation, BEFORE_ZH)
    repaired = [BEFORE[0], cue(2, 4.0, "Something else entirely."), BEFORE[2]]

    plan = plan_reuse(repaired, PreviousTranslation(source=source, translation=translation))

    assert plan.reused == 2
    assert plan.fresh == 1
    assert plan.translatable == 3
