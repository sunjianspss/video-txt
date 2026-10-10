from __future__ import annotations

import json

import pytest

from video_txt.cli import main
from video_txt.revise import (
    Revision,
    RevisionError,
    RevisionSet,
    apply_revisions,
    load_revisions,
    revised_subtitle_path,
)
from video_txt.subtitles import parse_srt, parse_srt_text

SOURCE = (
    "1\n00:00:01,000 --> 00:00:02,000\nYou can take today to figure out how to\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nOh.\n\n"
    "3\n00:00:05,000 --> 00:00:06,000\nWhere is Jessie?\n\n"
    "4\n00:00:07,000 --> 00:00:08,000\nOh.\n"
)
TRANSLATION = (
    "1\n00:00:01,000 --> 00:00:02,000\n你今天可以去探险\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\n哦。\n\n"
    "3\n00:00:05,000 --> 00:00:06,000\n杰西在哪里？\n\n"
    "4\n00:00:07,000 --> 00:00:08,000\n哦。\n"
)


def revision_set(*revisions: Revision) -> RevisionSet:
    return RevisionSet(source_language="en", target_language="zh-CN", revisions=revisions)


def apply(source: str, translation: str, *revisions: Revision):
    return apply_revisions(
        parse_srt_text(source), parse_srt_text(translation), revision_set(*revisions)
    )


def test_a_revision_lands_on_its_line_however_the_cues_are_numbered():
    """The whole point of anchoring on the source text: `clean` removes a cue and
    every number below it shifts, but the line a person corrected is still that line."""
    renumbered_source = "\n\n".join(SOURCE.strip().split("\n\n")[1:]).replace("2\n0", "1\n0")
    renumbered_translation = "\n\n".join(TRANSLATION.strip().split("\n\n")[1:]).replace(
        "2\n0", "1\n0"
    )

    result = apply(
        renumbered_source,
        renumbered_translation,
        Revision(source="Where is Jessie?", text="翠丝在哪里？"),
    )

    assert [cue.text for cue in result.cues] == ["哦。", "翠丝在哪里？", "哦。"]
    assert result.changed_count == 1
    # The cue is the second one now, not the third it was when the fix was written.
    assert result.applied[0].position == 2


def test_an_anchor_that_matches_nothing_stops_the_run():
    """Silently doing nothing is the failure this format exists to prevent: the
    run would report success with every corrected line back to the model's wording."""
    with pytest.raises(RevisionError, match="not in the source subtitle"):
        apply(SOURCE, TRANSLATION, Revision(source="A line nobody says", text="没人说过"))


def test_a_line_said_twice_has_to_say_which_one():
    with pytest.raises(RevisionError, match='said 2 times.*Add "at"'):
        apply(SOURCE, TRANSLATION, Revision(source="Oh.", text="哎呀。"))


def test_a_timestamp_picks_the_repeat_that_was_meant():
    result = apply(
        SOURCE, TRANSLATION, Revision(source="Oh.", text="哎呀。", at=7.0)
    )

    assert [cue.text for cue in result.cues][3] == "哎呀。"
    assert result.applied[0].cue_index == "4"
    assert result.applied[0].drift == 0.0


def test_the_nearest_repeat_wins_even_when_the_timing_moved():
    """Re-transcribing shifts a line by a second or two. The anchor still resolves,
    and the drift is reported rather than treated as a different line."""
    result = apply(SOURCE, TRANSLATION, Revision(source="Oh.", text="哎呀。", at=9.5))

    assert result.applied[0].cue_index == "4"
    assert result.applied[0].drift == pytest.approx(2.5)
    assert result.drifted == ()


def test_a_line_that_moved_by_minutes_is_reported_as_drifted():
    result = apply(SOURCE, TRANSLATION, Revision(source="Oh.", text="哎呀。", at=600.0))

    assert result.applied[0].cue_index == "4"
    assert [item.cue_index for item in result.drifted] == ["4"]


def test_two_revisions_cannot_claim_the_same_cue():
    with pytest.raises(RevisionError, match="both correct cue 3"):
        apply(
            SOURCE,
            TRANSLATION,
            Revision(source="Where is Jessie?", text="翠丝在哪里？"),
            Revision(source="where  is   jessie?", text="翠丝人呢？"),
        )


def test_applying_the_same_revisions_twice_changes_nothing_the_second_time():
    """Revisions are reapplied after every translation, so running them over a
    subtitle that already reads that way has to be safe and has to say so."""
    once = apply(SOURCE, TRANSLATION, Revision(source="Where is Jessie?", text="翠丝在哪里？"))
    twice = apply_revisions(
        parse_srt_text(SOURCE),
        once.cues,
        revision_set(Revision(source="Where is Jessie?", text="翠丝在哪里？")),
    )

    assert twice.changed_count == 0
    assert [item.cue_index for item in twice.already_current] == ["3"]
    assert [cue.text for cue in twice.cues] == [cue.text for cue in once.cues]


def test_a_revision_may_set_a_cue_to_two_lines():
    result = apply(
        SOURCE,
        TRANSLATION,
        Revision(source="Where is Jessie?", text="翠丝\n在哪里？"),
    )

    assert result.cues[2].text_lines == ["翠丝", "在哪里？"]


def test_a_translation_that_is_not_the_same_subtitle_is_refused():
    with pytest.raises(RevisionError, match="audit-translation"):
        apply(
            SOURCE,
            "1\n00:00:01,000 --> 00:00:02,000\n只有一句\n",
            Revision(source="Oh.", text="哎呀。", at=3.0),
        )


def write_revisions(tmp_path, revisions: list[dict], **overrides):
    path = tmp_path / "revisions.json"
    payload = {
        "schema": "video-txt.revisions",
        "version": 1,
        "source_language": "en",
        "target_language": "zh-CN",
        "revisions": revisions,
        **overrides,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_reads_anchors_notes_and_timestamps(tmp_path):
    path = write_revisions(
        tmp_path,
        [
            {"source": "Oh.", "text": "哎呀。", "at": "00:00:07,000", "note": "语气"},
            {"source": "Where is Jessie?", "text": "翠丝在哪里？"},
        ],
    )

    loaded = load_revisions(path)

    assert loaded.source_language == "en"
    assert loaded.target_language == "zh-CN"
    assert loaded.revisions[0] == Revision(
        source="Oh.", text="哎呀。", at=7.0, note="语气"
    )
    assert loaded.revisions[1].at is None


@pytest.mark.parametrize(
    ("revisions", "overrides", "message"),
    [
        ([], {"schema": "something-else"}, "Revision schema"),
        ([], {"version": 2}, "Revision version"),
        ([{"text": "缺原文"}], {}, "'source' must be a non-empty string"),
        ([{"source": "Oh.", "text": "  "}], {}, "'text' must be a non-empty string"),
        ([{"source": "Oh.", "text": "哎呀。", "at": "7s"}], {}, "not an SRT timestamp"),
        ([{"source": "Oh.", "text": "哎呀。", "note": 3}], {}, "'note' must be a string"),
        (
            [{"source": "Oh.", "text": "哎呀。"}, {"source": "oh.", "text": "哦哦。"}],
            {},
            "repeats an earlier anchor",
        ),
    ],
)
def test_a_malformed_revision_file_is_rejected_with_the_reason(
    tmp_path, revisions, overrides, message
):
    path = write_revisions(tmp_path, revisions, **overrides)

    with pytest.raises(RevisionError, match=message):
        load_revisions(path)


def test_revised_subtitle_path_sits_beside_the_translation(tmp_path):
    assert revised_subtitle_path(tmp_path / "clip.zh.srt") == tmp_path / "clip.zh.revised.srt"


def test_revise_command_writes_a_corrected_subtitle_and_a_report(tmp_path, capsys):
    source = tmp_path / "clip.srt"
    source.write_text(SOURCE, encoding="utf-8")
    translation = tmp_path / "clip.zh.srt"
    translation.write_text(TRANSLATION, encoding="utf-8")
    revisions = write_revisions(
        tmp_path,
        [
            {"source": "Where is Jessie?", "text": "翠丝在哪里？", "note": "角色名"},
            {"source": "Oh.", "text": "哎呀。", "at": "00:00:07,000"},
        ],
    )
    output = tmp_path / "clip.zh.revised.srt"

    assert main(["revise", str(source), str(translation), "--revisions", str(revisions)]) == 0

    assert [cue.text for cue in parse_srt(output)] == [
        "你今天可以去探险",
        "哦。",
        "翠丝在哪里？",
        "哎呀。",
    ]
    # The translation it was built from is left exactly as it was.
    assert translation.read_text(encoding="utf-8") == TRANSLATION
    report = json.loads((tmp_path / "clip.zh.revised.revision-report.json").read_text("utf-8"))
    assert report["changed_count"] == 2
    assert report["revisions"][0]["note"] == "角色名"
    assert "2 of 2 line(s) changed" in capsys.readouterr().out


def test_revise_command_refuses_to_overwrite_an_input(tmp_path):
    source = tmp_path / "clip.srt"
    source.write_text(SOURCE, encoding="utf-8")
    translation = tmp_path / "clip.zh.srt"
    translation.write_text(TRANSLATION, encoding="utf-8")
    revisions = write_revisions(
        tmp_path, [{"source": "Oh.", "text": "哎呀。", "at": "00:00:03,000"}]
    )

    with pytest.raises(SystemExit):
        main(
            [
                "revise",
                str(source),
                str(translation),
                "--revisions",
                str(revisions),
                "-o",
                str(translation),
            ]
        )

    assert translation.read_text(encoding="utf-8") == TRANSLATION


def test_revise_command_reports_a_dead_anchor_instead_of_writing_a_wrong_line(tmp_path, capsys):
    source = tmp_path / "clip.srt"
    source.write_text(SOURCE, encoding="utf-8")
    translation = tmp_path / "clip.zh.srt"
    translation.write_text(TRANSLATION, encoding="utf-8")
    revisions = write_revisions(tmp_path, [{"source": "A line nobody says", "text": "没人说过"}])

    assert main(["revise", str(source), str(translation), "--revisions", str(revisions)]) == 1

    assert "not in the source subtitle" in capsys.readouterr().err
    assert not (tmp_path / "clip.zh.revised.srt").exists()
