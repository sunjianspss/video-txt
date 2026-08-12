from __future__ import annotations

from types import SimpleNamespace

import pytest

from video_txt import diarize as diarize_module
from video_txt.diarize import (
    DiarizeError,
    DiarizeOptions,
    SpeakerTurn,
    assign_speakers,
    collect_turns,
    describe_speakers,
    ensure_speakers,
    load_turns,
    save_turns,
    speaker_order,
    speakers_path,
)
from video_txt.subtitles import parse_srt_text

CONVERSATION = (
    "1\n00:00:00,000 --> 00:00:02,000\nWelcome to the show.\n\n"
    "2\n00:00:02,000 --> 00:00:04,000\nGlad to be here.\n\n"
    "3\n00:00:04,000 --> 00:00:06,000\nLet us start with memory.\n\n"
    "4\n00:00:30,000 --> 00:00:32,000\nAnd that is the whole idea.\n"
)
TURNS = [
    SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
    SpeakerTurn(2.0, 4.0, "SPEAKER_01"),
    SpeakerTurn(4.0, 6.0, "SPEAKER_00"),
]


def numbered(text: str) -> list[tuple[int, object]]:
    return list(enumerate(parse_srt_text(text), start=1))


def fake_annotation(turns: list[tuple[float, float, str]]) -> object:
    def itertracks(yield_label: bool = False):
        for start, end, label in turns:
            yield SimpleNamespace(start=start, end=end), None, label

    return SimpleNamespace(itertracks=itertracks)


def test_collect_turns_reads_a_plain_annotation():
    turns = collect_turns(fake_annotation([(0.0, 1.0, "SPEAKER_00")]))

    assert turns == [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]


def test_collect_turns_unwraps_the_result_object_pyannote_4_returns():
    wrapped = SimpleNamespace(speaker_diarization=fake_annotation([(0.0, 1.0, "SPEAKER_00")]))

    assert collect_turns(wrapped) == [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]


def test_an_unknown_result_shape_is_reported_rather_than_guessed():
    with pytest.raises(DiarizeError, match="not supported"):
        collect_turns(object())


def test_a_brief_pause_does_not_end_a_turn():
    turns = collect_turns(fake_annotation([(0.0, 2.0, "SPEAKER_00"), (2.3, 5.0, "SPEAKER_00")]))

    assert turns == [SpeakerTurn(0.0, 5.0, "SPEAKER_00")]


def test_the_other_speaker_in_between_keeps_two_turns_apart():
    turns = collect_turns(
        fake_annotation(
            [(0.0, 2.0, "SPEAKER_00"), (2.1, 2.9, "SPEAKER_01"), (3.0, 5.0, "SPEAKER_00")]
        )
    )

    assert [turn.speaker for turn in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


def test_a_sliver_of_a_turn_is_not_worth_attributing():
    assert collect_turns(fake_annotation([(1.0, 1.05, "SPEAKER_01")])) == []


def test_a_line_goes_to_whoever_covers_most_of_it():
    assigned = assign_speakers(numbered(CONVERSATION), TURNS)

    assert assigned[1] == "SPEAKER_00"
    assert assigned[2] == "SPEAKER_01"
    assert assigned[3] == "SPEAKER_00"


def test_a_line_nobody_was_speaking_over_stays_with_the_last_voice():
    # The closing line at 30s is past every turn: silence in the diarization is
    # a missed breath far more often than it is a new person.
    assert assign_speakers(numbered(CONVERSATION), TURNS)[4] == "SPEAKER_00"


def test_lines_before_the_first_turn_go_to_the_main_speaker():
    late = [SpeakerTurn(20.0, 30.0, "SPEAKER_01"), SpeakerTurn(30.0, 60.0, "SPEAKER_00")]

    assert assign_speakers(numbered(CONVERSATION), late)[1] == "SPEAKER_00"


def test_without_turns_nothing_is_attributed():
    assert assign_speakers(numbered(CONVERSATION), []) == {}


def test_speakers_are_ordered_by_how_long_they_hold_the_floor():
    assert speaker_order(TURNS) == ["SPEAKER_00", "SPEAKER_01"]
    assert describe_speakers(TURNS) == "SPEAKER_00 67%, SPEAKER_01 33%"


def test_turns_survive_the_round_trip_to_disk(tmp_path):
    path = tmp_path / "clip.speakers.json"
    save_turns(path, TURNS, model="pyannote/x")

    assert load_turns(path, model="pyannote/x") == TURNS


def test_turns_from_another_model_are_not_reused(tmp_path):
    path = tmp_path / "clip.speakers.json"
    save_turns(path, TURNS, model="pyannote/x")

    assert load_turns(path, model="pyannote/y") is None
    assert load_turns(tmp_path / "missing.json", model="pyannote/x") is None


def test_a_damaged_file_is_ignored_rather_than_crashing(tmp_path):
    path = tmp_path / "clip.speakers.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_turns(path, model="pyannote/x") is None


def test_ensure_speakers_reuses_the_turns_instead_of_loading_the_model(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    save_turns(speakers_path(video), TURNS, model="pyannote/x")
    monkeypatch.setattr(
        diarize_module,
        "diarize_media",
        lambda *_args, **_kwargs: pytest.fail("the turns were already on disk"),
    )

    assert ensure_speakers(video, options=DiarizeOptions(model="pyannote/x")) == TURNS


def test_rediarize_runs_again_over_the_saved_turns(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    save_turns(speakers_path(video), TURNS, model="pyannote/x")
    fresh = [SpeakerTurn(0.0, 9.0, "SPEAKER_00")]
    monkeypatch.setattr(diarize_module, "diarize_media", lambda *_args, **_kwargs: fresh)

    options = DiarizeOptions(model="pyannote/x", rediarize=True)

    assert ensure_speakers(video, options=options) == fresh
    assert load_turns(speakers_path(video), model="pyannote/x") == fresh


def test_a_dry_run_neither_loads_the_model_nor_writes_anything(tmp_path, monkeypatch, capsys):
    video = tmp_path / "clip.mp4"
    monkeypatch.setattr(
        diarize_module,
        "diarize_media",
        lambda *_args, **_kwargs: pytest.fail("a dry run must not load the model"),
    )

    assert ensure_speakers(video, options=DiarizeOptions(), dry_run=True) == []
    assert "would run" in capsys.readouterr().out
    assert not speakers_path(video).exists()


def test_a_video_with_no_speech_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(diarize_module, "diarize_media", lambda *_args, **_kwargs: [])

    with pytest.raises(DiarizeError, match="no speech"):
        ensure_speakers(tmp_path / "clip.mp4", options=DiarizeOptions())
