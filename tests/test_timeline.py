from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_txt import timeline as timeline_module
from video_txt.subtitles import SubtitleCue, parse_srt_text
from video_txt.timeline import (
    Segment,
    build_segments,
    decode_segment,
    group_cues_into_sentences,
    render_audio_track,
    speaking_budgets,
    spoken_duration,
)

SAMPLE_RATE = 48000

TIMED = (
    "1\n00:00:01,000 --> 00:00:03,000\nfirst\n\n"
    "2\n00:00:05,000 --> 00:00:06,000\nsecond\n\n"
    "3\n00:00:10,000 --> 00:00:12,000\nthird\n"
)
SPLIT_SENTENCE = (
    "1\n00:00:00,000 --> 00:00:02,000\n我们先看第一点，\n\n"
    "2\n00:00:02,000 --> 00:00:04,000\n也就是记忆的写入。\n\n"
    "3\n00:00:04,000 --> 00:00:06,000\n第二点是检索。\n"
)
SPLIT_SENTENCE_SPOKEN = ["我们先看第一点，也就是记忆的写入。", "第二点是检索。"]
RUN_ON = (
    "1\n00:00:00,000 --> 00:00:02,000\n一二三四五\n\n"
    "2\n00:00:02,000 --> 00:00:04,000\n六七八九十\n\n"
    "3\n00:00:04,000 --> 00:00:06,000\n收尾。\n"
)


def numbered(text: str) -> list[tuple[int, SubtitleCue]]:
    return list(enumerate(parse_srt_text(text), start=1))


def test_build_segments_borrows_the_gap_before_the_next_line():
    cues = numbered(TIMED)
    paths = [Path(f"/cache/{index}.mp3") for index, _ in cues]
    segments = build_segments(cues, paths)

    assert [segment.start for segment in segments] == [1.0, 5.0, 10.0]
    assert segments[0].slot == 3.95
    assert segments[1].slot == 4.95
    assert segments[2].slot == 4.0


def test_sentences_absorb_the_lines_they_were_split_across():
    units = group_cues_into_sentences(numbered(SPLIT_SENTENCE))

    assert [cue.text for _, cue in units] == SPLIT_SENTENCE_SPOKEN
    assert units[0][1].timing == "00:00:00,000 --> 00:00:04,000"
    assert [position for position, _ in units] == [1, 3]


def test_a_sentence_that_fits_one_line_is_left_alone():
    cues = numbered(SPLIT_SENTENCE)

    assert group_cues_into_sentences(cues)[1][1] is cues[2][1]


def test_a_long_pause_ends_the_sentence_even_without_punctuation():
    paused = (
        "1\n00:00:00,000 --> 00:00:02,000\n先说结论\n\n2\n00:00:05,000 --> 00:00:07,000\n再说细节\n"
    )

    assert len(group_cues_into_sentences(numbered(paused))) == 2


def test_a_run_on_sentence_is_cut_before_the_clip_grows_unwieldy():
    assert len(group_cues_into_sentences(numbered(RUN_ON))) == 1
    assert len(group_cues_into_sentences(numbered(RUN_ON), max_chars=5)) == 3


def test_a_sentence_never_carries_two_speakers():
    cues = numbered(RUN_ON)
    assert len(group_cues_into_sentences(cues)) == 1

    units = group_cues_into_sentences(cues, speakers={1: "A", 2: "A", 3: "B"})

    assert [cue.text for _, cue in units] == ["一二三四五六七八九十", "收尾。"]


def test_the_speaking_budget_splits_where_the_clips_do():
    speakers = {1: "A", 2: "A", 3: "B"}

    budgets = speaking_budgets(numbered(RUN_ON), voice_unit="sentence", speakers=speakers)

    assert len(budgets) == 3
    # Lines 1 and 2 share the first clip's four seconds; line 3 keeps its own.
    assert sum(budgets[:2]) == pytest.approx(4.0)


def test_merged_latin_text_keeps_the_spaces_between_words():
    latin = (
        "1\n00:00:00,000 --> 00:00:02,000\nwe start\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nwith memory.\n"
    )
    units = group_cues_into_sentences(numbered(latin))

    assert [cue.text for _, cue in units] == ["we start with memory."]


def test_a_line_inside_a_sentence_gets_its_share_of_the_sentence_budget():
    cues = numbered(SPLIT_SENTENCE)

    per_line = speaking_budgets(cues, voice_unit="line")
    shared = speaking_budgets(cues, voice_unit="sentence")

    # Lines 1 and 2 are one sentence: the wordier line takes time from the sparser
    # one instead of being judged against a span it happens to share.
    assert sum(shared[:2]) == pytest.approx(sum(per_line[:2]))
    assert shared[1] > per_line[1]
    assert shared[0] < per_line[0]


def test_a_merged_sentence_claims_the_whole_span_it_covers():
    units = group_cues_into_sentences(numbered(SPLIT_SENTENCE))
    segments = build_segments(units, [Path("/cache/a.mp3"), Path("/cache/b.mp3")])

    assert [segment.start for segment in segments] == [0.0, 4.0]
    assert segments[0].slot == pytest.approx(4.0)


def fake_decoder(monkeypatch, *, seconds: float):
    """Stand in for ffmpeg: hand back `seconds` of PCM, then echo whatever it is given."""
    commands: list[list[str]] = []

    def fake_run(command, input=None, **_kwargs):
        commands.append(command)
        pcm = input if input is not None else bytes(int(seconds * SAMPLE_RATE) * 2)
        return SimpleNamespace(returncode=0, stdout=pcm, stderr=b"")

    monkeypatch.setattr(timeline_module.subprocess, "run", fake_run)
    return commands


def decode_one(slot: float, *, max_atempo: float = 1.35) -> tuple[bytes, float]:
    return decode_segment(
        Segment(
            cue=SubtitleCue("1", "00:00:00,000 --> 00:00:01,000", ["一"]),
            audio_path=Path("/cache/a.mp3"),
            start=0.0,
            slot=slot,
        ),
        ffmpeg_path="ffmpeg",
        sample_rate=SAMPLE_RATE,
        max_atempo=max_atempo,
    )


def test_decode_segment_drops_the_silence_the_engine_pads_clips_with(monkeypatch):
    commands = fake_decoder(monkeypatch, seconds=1.0)

    decode_one(slot=5.0)

    assert len(commands) == 1
    assert any("silenceremove" in str(part) for part in commands[0])


def test_duration_fitting_measures_the_same_trimmed_audio_as_rendering(monkeypatch):
    commands = fake_decoder(monkeypatch, seconds=1.25)

    duration = spoken_duration(Path("/cache/a.mp3"), ffmpeg_path="ffmpeg", sample_rate=SAMPLE_RATE)

    assert duration == pytest.approx(1.25)
    assert any("silenceremove" in str(part) for part in commands[0])


def test_decode_segment_speeds_a_clip_up_to_reach_its_slot(monkeypatch):
    commands = fake_decoder(monkeypatch, seconds=1.5)

    _, tempo = decode_one(slot=1.0, max_atempo=2.0)

    assert tempo == 1.5
    assert "atempo=1.5000" in commands[1]


def test_decode_segment_never_speeds_a_clip_past_the_ceiling(monkeypatch):
    fake_decoder(monkeypatch, seconds=4.0)

    _, tempo = decode_one(slot=1.0, max_atempo=1.5)

    assert tempo == 1.5


def test_decode_segment_measures_the_audio_it_produced_not_the_file(monkeypatch):
    """The trim happens before the fit, so a padded clip must not be sped up for padding."""
    fake_decoder(monkeypatch, seconds=1.0)

    _, tempo = decode_one(slot=1.0)

    assert tempo == 1.0


def test_render_audio_track_streams_gaps_clips_overlaps_and_tail_padding(tmp_path, monkeypatch):
    clip_a = b"\x11\x11" * 50  # 0.5 s at 100 Hz
    clip_b = b"\x22\x22" * 30  # 0.3 s at 100 Hz
    clips = {Path("/cache/a.mp3"): (clip_a, 1.0), Path("/cache/b.mp3"): (clip_b, 1.2)}
    monkeypatch.setattr(
        timeline_module, "decode_segment", lambda segment, **_kwargs: clips[segment.audio_path]
    )

    segments = [
        Segment(
            cue=SubtitleCue("1", "00:00:01,000 --> 00:00:02,000", ["一"]),
            audio_path=Path("/cache/a.mp3"),
            start=1.0,
            slot=1.0,
        ),
        # Starts at 1.2 s, but the first clip plays until 1.5 s, so it gets pushed.
        Segment(
            cue=SubtitleCue("2", "00:00:01,200 --> 00:00:02,000", ["二"]),
            audio_path=Path("/cache/b.mp3"),
            start=1.2,
            slot=0.8,
        ),
    ]
    output = tmp_path / "track.wav"
    placed = render_audio_track(
        segments,
        ffmpeg_path="ffmpeg",
        sample_rate=100,
        max_atempo=1.35,
        total_duration=3.0,
        output_path=output,
    )

    with wave.open(str(output)) as handle:
        assert handle.getframerate() == 100
        assert handle.getnframes() == 400  # video length + 1 s of padding
        data = handle.readframes(400)
    assert data[:200] == bytes(200)  # silence until 1.0 s
    assert data[200:300] == clip_a
    assert data[300:360] == clip_b  # pushed from 1.2 s to 1.5 s
    assert data[360:] == bytes(len(data) - 360)
    assert [item.start for item in placed] == [1.0, 1.5]
    assert placed[0].tempo == 1.0
    assert placed[1].tempo == 1.2
