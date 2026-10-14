from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_txt import transcribe
from video_txt.chunking import (
    Chunk,
    clamp_cues,
    clamp_payload,
    merge_whisper_payloads,
    parse_silences,
    collapse_repeats,
    plan_chunks,
    renumber,
    repeated_runs,
    sentences,
    shift_cues,
)
from video_txt.subtitles import SubtitleCue, parse_srt


def test_parse_silences_pairs_start_and_end():
    log = (
        "[silencedetect @ 0x1] silence_start: 118.5\n"
        "[silencedetect @ 0x1] silence_end: 121.5 | silence_duration: 3\n"
        "[silencedetect @ 0x1] silence_start: 300\n"  # still silent at EOF, never ends
    )
    assert parse_silences(log) == [(118.5, 121.5)]


def test_plan_chunks_cuts_in_the_middle_of_the_nearest_silence():
    chunks = plan_chunks(500, [(118, 122), (238, 242), (358, 362)])
    assert chunks == [Chunk(0, 120), Chunk(120, 240), Chunk(240, 360), Chunk(360, 500)]


def test_plan_chunks_falls_back_to_fixed_cuts_without_silence():
    chunks = plan_chunks(500, [])
    assert [c.start for c in chunks] == [0, 120, 240, 360]
    assert chunks[-1].end == 500


def test_plan_chunks_covers_the_audio_exactly_once():
    chunks = plan_chunks(1000, [(100, 101), (250, 251), (700, 703)])
    assert chunks[0].start == 0 and chunks[-1].end == 1000
    assert all(a.end == b.start for a, b in zip(chunks, chunks[1:]))
    assert all(c.duration > 0 for c in chunks)


def test_plan_chunks_short_audio_is_one_piece():
    assert plan_chunks(130, [(60, 62)]) == [Chunk(0, 130)]
    assert plan_chunks(0, []) == []


def test_shift_and_renumber_cues():
    cue = SubtitleCue("1", "00:00:01,000 --> 00:00:02,500", ["hi"])
    moved = shift_cues([cue], 120)[0]
    assert moved.timing == "00:02:01,000 --> 00:02:02,500"
    assert moved.text == "hi"
    assert [c.index for c in renumber([moved, moved])] == ["1", "2"]


def test_merge_whisper_payloads_offsets_words_and_keeps_language():
    first = {"language": "en", "segments": [{"start": 0, "end": 1, "words": [
        {"word": " A", "start": 0.0, "end": 0.5, "probability": 0.9}]}]}
    second = {"language": "en", "segments": [{"start": 1, "end": 2, "words": [
        {"word": " B", "start": 1.0, "end": 1.5}]}]}
    merged = merge_whisper_payloads([(0.0, first), (120.0, second)])
    assert merged["language"] == "en"
    words = [w for s in merged["segments"] for w in s["words"]]
    assert [(w["word"], w["start"], w["end"]) for w in words] == [
        (" A", 0.0, 0.5),
        (" B", 121.0, 121.5),
    ]
    assert words[0]["probability"] == 0.9


def test_clamp_cues_drops_the_hallucinated_tail_of_a_piece():
    cues = [
        SubtitleCue("1", "00:00:01,000 --> 00:00:02,000", ["real"]),
        SubtitleCue("2", "00:01:59,000 --> 00:02:02,000", ["runs past the end"]),
        SubtitleCue("3", "00:01:59,500 --> 00:01:59,900", ["!"]),
        SubtitleCue("4", "00:02:00,500 --> 00:02:01,500", ["starts after the end"]),
        SubtitleCue("5", "00:00:03,000 --> 00:00:02,000", ["backwards"]),
    ]
    kept = clamp_cues(cues, 120.0)
    assert [(c.text, c.timing) for c in kept] == [
        ("real", "00:00:01,000 --> 00:00:02,000"),
        ("runs past the end", "00:01:59,000 --> 00:02:00,000"),
        ("backwards", "00:00:03,000 --> 00:00:03,000"),
    ]


def test_clamp_payload_drops_words_past_the_end():
    payload = {"language": "en", "segments": [
        {"start": 0, "end": 5, "words": [{"word": " a", "start": 0.0, "end": 1.0}]},
        {"start": 119, "end": 125, "words": [
            {"word": " b", "start": 119.0, "end": 125.0},
            {"word": " c", "start": 121.0, "end": 122.0}]},
        {"start": 130, "end": 131, "words": [{"word": " d", "start": 130.0, "end": 131.0}]},
    ]}
    clamped = clamp_payload(payload, 120.0)
    words = [w for s in clamped["segments"] for w in s["words"]]
    assert [(w["word"].strip(), w["end"]) for w in words] == [("a", 1.0), ("b", 120.0)]


def test_repeated_runs_finds_loops_not_ordinary_repetition():
    assert repeated_runs(["a", "Yeah.", "yeah. ", "YEAH.", "yeah.", "b"]) == [(1, 4)]
    assert repeated_runs(["no", "no", "no", "go"]) == []  # three is just emphatic
    assert repeated_runs(["", "", "", "", ""]) == []  # silence is not a loop


def test_a_loop_inside_one_segment_is_still_seen():
    segment = "It does the thing. " * 5
    assert repeated_runs(sentences(["Before.", segment, "After."])) == [(1, 5)]
    assert repeated_runs(["Before.", segment, "After."]) == []  # per segment it is invisible


def test_collapse_repeats_keeps_the_first_line_of_each_loop():
    items = list("abbbbbc")
    assert collapse_repeats(items, items) == list("abc")
    assert collapse_repeats(list("abc"), list("abc")) == list("abc")


# --- the whole path, with ffmpeg and Whisper faked ---------------------------------


class Fake:
    """Stands in for ffmpeg and Whisper; records what it was asked to do."""

    def __init__(self, *, language="en", silences="", empty_pieces=(), loops=(), stubborn=False):
        self.loops = set(loops)  # pieces on which Whisper loops...
        self.stubborn = stubborn  # ...even without context
        self.language = language
        self.silences = silences
        self.empty_pieces = set(empty_pieces)
        self.start = None
        self.whisper = []  # (command, start of the piece it ran on)

    def __call__(self, command, **_kwargs):
        if command[0] == "/usr/bin/ffmpeg":
            if "-af" in command:
                return SimpleNamespace(returncode=0, stdout="", stderr=self.silences)
            self.start = float(command[command.index("-ss") + 1])
            Path(command[-1]).write_bytes(b"wav")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        self.whisper.append((command, self.start))
        flag = "--output_dir"
        directory = Path(command[command.index(flag) + 1])
        stem = Path(command[3]).stem
        fmt = command[command.index("--output_format") + 1]
        looping = self.start in self.loops and (
            self.stubborn or "--condition_on_previous_text" not in command
        )
        if fmt == "json":
            words = "--word_timestamps" in command
            segment = {"start": 1.0, "end": 2.0, "text": "x"}
            if words:
                segment["words"] = [
                    {"word": " Claude", "start": 1.0, "end": 1.5},
                    {"word": " Code.", "start": 1.5, "end": 2.0},
                ]
            segments = [segment]
            if looping:
                segments = [{**segment, "text": "Yeah."} for _ in range(6)]
            payload = {"language": self.language, "segments": segments}
            (directory / f"{stem}.json").write_text(json.dumps(payload))
        elif fmt == "srt":
            body = (
                ""
                if self.start in self.empty_pieces
                else f"1\n00:00:01,000 --> 00:00:02,000\nat {(self.start or 0):.0f}\n"
            )
            if looping:
                body = "".join(
                    f"{n}\n00:00:0{n},000 --> 00:00:0{n},900\nYeah.\n\n" for n in range(1, 7)
                )
            (directory / f"{stem}.srt").write_text(body)
        else:
            (directory / f"{stem}.txt").write_text(f"line at {(self.start or 0):.0f}\n")
        return SimpleNamespace(returncode=0)


SILENCES = (
    "silence_start: 118\nsilence_end: 122\n"
    "silence_start: 238\nsilence_end: 242\n"
    "silence_start: 358\nsilence_end: 362\n"
)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"media")
    monkeypatch.setattr(transcribe, "find_ffmpeg", lambda: Path("/usr/bin/ffmpeg"))
    monkeypatch.setattr(transcribe, "probe_audio_streams", lambda *a, **kw: [])
    monkeypatch.setattr(transcribe, "resolve_backend", lambda requested: "openai-whisper")
    monkeypatch.setattr(transcribe, "probe_duration", lambda *a, **kw: 500.0)

    def run(fake, **kwargs):
        monkeypatch.setattr(transcribe.subprocess, "run", fake)
        options = dict(input_path=video, output_dir=tmp_path, output_format="srt")
        return transcribe.run_transcribe(transcribe.TranscribeOptions(**{**options, **kwargs}))

    return run


def prompts(fake):
    return [c[c.index("--initial_prompt") + 1] if "--initial_prompt" in c else None
            for c, _ in fake.whisper]


def test_long_audio_is_transcribed_in_pieces_each_with_the_prompt(rig):
    fake = Fake(silences=SILENCES)
    output = rig(fake)

    cues = parse_srt(output)
    assert [c.text for c in cues] == ["at 0", "at 120", "at 240", "at 360"]
    assert [c.timing.split(" --> ")[0] for c in cues] == [
        "00:00:01,000", "00:02:01,000", "00:04:01,000", "00:06:01,000"]
    assert [c.index for c in cues] == ["1", "2", "3", "4"]
    # One language probe, then one run per piece, every one primed.
    assert len(fake.whisper) == 5
    assert set(prompts(fake)[1:]) == {transcribe.DEFAULT_INITIAL_PROMPT}
    # The language found by the probe is pinned for every piece.
    assert all("--language" in c and c[c.index("--language") + 1] == "en"
               for c, _ in fake.whisper[1:])


def test_pieces_with_nothing_audible_are_skipped_not_fatal(rig):
    fake = Fake(silences=SILENCES, empty_pieces={120.0})
    cues = parse_srt(rig(fake))
    assert [c.text for c in cues] == ["at 0", "at 240", "at 360"]
    assert [c.index for c in cues] == ["1", "2", "3"]


def test_txt_output_is_joined_in_order(rig, tmp_path):
    output = rig(Fake(silences=SILENCES), output_format="txt")
    assert output.read_text().splitlines() == [
        "line at 0", "line at 120", "line at 240", "line at 360"]


def test_explicit_language_skips_the_probe(rig):
    fake = Fake(silences=SILENCES)
    rig(fake, language="en")
    assert len(fake.whisper) == 4


def test_refined_transcription_merges_word_timestamps_across_pieces(rig, tmp_path):
    output = rig(Fake(silences=SILENCES), refine_subtitles=True)
    words = json.loads((tmp_path / "talk.words.json").read_text())["words"]
    assert [w["start"] for w in words if w["text"] == "Claude"] == [1.0, 121.0, 241.0, 361.0]
    assert [c.text for c in parse_srt(output)][0].startswith("Claude Code")


def test_short_audio_is_one_run(rig, monkeypatch):
    monkeypatch.setattr(transcribe, "probe_duration", lambda *a, **kw: 200.0)
    fake = Fake()
    rig(fake)
    assert len(fake.whisper) == 1
    assert prompts(fake) == [transcribe.DEFAULT_INITIAL_PROMPT]


def test_detected_non_english_runs_once_with_the_language_pinned(rig):
    fake = Fake(language="zh", silences=SILENCES)
    rig(fake)
    # The probe, then a single normal run: no chunks, and no English prompt.
    assert len(fake.whisper) == 2
    assert prompts(fake)[-1] is None
    last = fake.whisper[-1][0]
    assert last[last.index("--language") + 1] == "zh"


def test_formats_the_echo_filter_cannot_clean_are_not_chunked(rig):
    fake = Fake()
    rig(fake, output_format="vtt")
    assert len(fake.whisper) == 1


def test_a_looping_piece_is_run_again_without_context(rig):
    fake = Fake(silences=SILENCES, loops={120.0})
    cues = parse_srt(rig(fake))
    assert [c.text for c in cues] == ["at 0", "at 120", "at 240", "at 360"]
    on_piece = [c for c, start in fake.whisper if start == 120.0]
    assert len(on_piece) == 2
    assert "--condition_on_previous_text" not in on_piece[0]
    assert on_piece[1][on_piece[1].index("--condition_on_previous_text") + 1] == "False"
    # The other pieces are not slowed down.
    assert sum(1 for c, start in fake.whisper if start == 240.0) == 1


def test_a_loop_that_persists_is_cut_to_one_line(rig, capsys):
    fake = Fake(silences=SILENCES, loops={120.0}, stubborn=True)
    cues = parse_srt(rig(fake))
    assert [c.text for c in cues] == ["at 0", "Yeah.", "at 240", "at 360"]
    assert "unreliable" in capsys.readouterr().out


def test_looping_refined_piece_is_retried_too(rig, tmp_path):
    fake = Fake(silences=SILENCES, loops={120.0})
    rig(fake, refine_subtitles=True)
    on_piece = [c for c, start in fake.whisper if start == 120.0]
    assert len(on_piece) == 2


def test_a_user_set_condition_flag_is_not_second_guessed(rig):
    fake = Fake(silences=SILENCES, loops={120.0}, stubborn=True)
    rig(fake, extra_args=["--condition_on_previous_text", "True"])
    assert sum(1 for _, start in fake.whisper if start == 120.0) == 1
