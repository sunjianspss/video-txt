from __future__ import annotations

from pathlib import Path

import pytest

from video_txt.music import (
    CLEAN_VOICE_SHARE,
    MEASURE_INTERVAL,
    MusicPiece,
    clock,
    extract_command,
    find_music,
    is_sung,
    piece_filename,
    runs_of,
    sung_spans,
)
from video_txt.separate import STEM_MODELS, separate_command, stem_paths
from video_txt.subtitles import parse_srt_text


def envelope(*sections: tuple[float, float]) -> list[float]:
    """Build a loudness envelope from (seconds, LUFS) sections."""
    return [
        level for seconds, level in sections for _ in range(round(seconds / MEASURE_INTERVAL))
    ]


LOUD, SILENT = -20.0, -80.0


def test_a_rest_inside_a_piece_does_not_end_it():
    assert runs_of([True, True, False, True, True], bridge=1) == [(0, 5)]
    assert runs_of([True, True, False, False, True], bridge=1) == [(0, 2), (4, 5)]


def test_music_is_where_the_instrumental_stays_up_long_enough():
    instrumental = envelope((30, LOUD), (30, SILENT), (30, LOUD))
    voice = envelope((90, SILENT))

    pieces = find_music(instrumental, voice, min_duration=20.0)

    assert [(p.start, p.end) for p in pieces] == [(0.0, 30.0), (60.0, 90.0)]
    assert all(p.kind == "clean" for p in pieces)


def test_a_sting_too_short_to_play_is_not_a_piece():
    instrumental = envelope((5, LOUD), (30, SILENT), (30, LOUD))

    pieces = find_music(instrumental, envelope((65, SILENT)), min_duration=20.0)

    assert [p.start for p in pieces] == [35.0]


def test_a_piece_reports_how_much_of_it_has_a_voice_over_it():
    instrumental = envelope((40, LOUD))
    voice = envelope((10, LOUD), (30, SILENT))

    piece = find_music(instrumental, voice, min_duration=20.0)[0]

    assert piece.voice_share == pytest.approx(0.25)
    assert piece.kind == "under-dialogue"


def test_clean_keeps_only_what_nobody_talks_over():
    """A separator leaves its worst artefacts where the dialogue was loudest, so
    these are the pieces worth dropping into somebody else's video."""
    instrumental = envelope((60, LOUD))
    voice = envelope((25, SILENT), (10, LOUD), (25, SILENT))

    every = find_music(instrumental, voice, min_duration=20.0)
    clean = find_music(instrumental, voice, min_duration=20.0, clean_only=True)

    assert [(p.start, p.end) for p in every] == [(0.0, 60.0)]
    assert [(p.start, p.end) for p in clean] == [(0.0, 25.0), (35.0, 60.0)]
    assert all(p.voice_share <= CLEAN_VOICE_SHARE for p in clean)


def test_clean_does_not_bridge_across_somebody_talking():
    """The wide bridge that joins a bar of rest would swallow a spoken line and
    hand back a piece that breaks the promise its name makes."""
    instrumental = envelope((60, LOUD))
    voice = envelope((25, SILENT), (2, LOUD), (33, SILENT))

    clean = find_music(instrumental, voice, min_duration=10.0, clean_only=True)

    assert len(clean) == 2
    assert all(p.voice_share == 0.0 for p in clean)


def test_no_piece_is_ever_called_a_song_from_the_audio():
    """Measured on a real film, the first stretch this would have called a song
    was a seventy-second conversation. A separator cannot hear the difference."""
    instrumental = envelope((60, LOUD))
    voice = envelope((60, LOUD))

    assert {p.kind for p in find_music(instrumental, voice, min_duration=20.0)} == {
        "under-dialogue"
    }


def test_songs_come_from_the_lines_a_person_marked_as_sung():
    cues = parse_srt_text(
        "1\n00:00:10,000 --> 00:00:13,000\n♪ There's going to be ♪\n\n"
        "2\n00:00:13,000 --> 00:00:16,000\n♪ a revival tonight ♪\n\n"
        "3\n00:01:00,000 --> 00:01:02,000\nPlain dialogue.\n\n"
        "4\n00:02:00,000 --> 00:02:03,000\n♪ A different song ♪\n"
    )

    spans = sung_spans(cues)

    assert [(s.start, s.end) for s in spans] == [(10.0, 16.0), (120.0, 123.0)]
    assert all(s.kind == "song" for s in spans)


def test_a_note_on_its_own_marks_music_playing_not_somebody_singing():
    """Measured across one season: 72 of the 128 short marked cues carry no words.
    They mean a sting or a radio in the background, and cutting them out gives two
    seconds of nothing."""
    lyric, marker = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:03,000\n♪ Love is all we need ♪\n\n"
        "2\n00:00:05,000 --> 00:00:07,000\n♪♪\n"
    )

    assert is_sung(lyric)
    assert not is_sung(marker)


def test_one_song_stays_one_song_across_the_gaps_between_its_lines():
    """Sung lines are subtitled sparsely. Measured on a real episode the gaps run
    three to four seconds, which the audio bridge would cut into six fragments."""
    cues = parse_srt_text(
        "1\n00:32:45,000 --> 00:32:54,000\n♪ There's going to be a revival ♪\n\n"
        "2\n00:32:57,000 --> 00:33:05,000\n♪ Everybody gonna jump and shout ♪\n\n"
        "3\n00:33:08,000 --> 00:33:16,000\n♪ All your sisters and your brothers ♪\n"
    )

    spans = sung_spans(cues)

    assert len(spans) == 1
    assert spans[0].duration == pytest.approx(31.0)


def test_a_subtitle_with_no_marks_says_nothing_about_songs():
    """Whisper never writes them, so an empty answer means the subtitle is silent
    on the question -- not that the film has no songs."""
    assert sung_spans(parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\nHello.\n")) == []


def test_a_song_is_not_handed_back_again_inside_the_music_around_it():
    """Measured on a real episode: the song the subtitle placed at 32:45 sat
    inside a 130-second stretch the audio found, and both were written out --
    the same music twice, 45 MB and 26 MB of it."""
    instrumental = envelope((120, LOUD))
    voice = envelope((120, SILENT))

    without = find_music(instrumental, voice, min_duration=20.0)
    around = find_music(instrumental, voice, min_duration=20.0, exclude=[(40.0, 80.0)])

    assert [(p.start, p.end) for p in without] == [(0.0, 120.0)]
    assert [(p.start, p.end) for p in around] == [(0.0, 40.0), (80.0, 120.0)]


def test_what_is_left_beside_a_song_keeps_its_own_measurements():
    """Masking before the runs are grouped, rather than trimming afterwards, is
    what keeps each remaining piece measured over itself."""
    instrumental = envelope((120, LOUD))
    voice = envelope((40, SILENT), (40, LOUD), (40, SILENT))

    pieces = find_music(instrumental, voice, min_duration=20.0, exclude=[(40.0, 80.0)])

    assert all(piece.voice_share == 0.0 for piece in pieces)


def test_a_piece_is_named_for_when_it_starts_and_what_it_is():
    piece = MusicPiece(start=3723.0, end=3800.0, voice_share=0.0, peak_lufs=-20.0)

    assert clock(3723.0) == "01h02m03s"
    assert piece_filename(Path("/v/Movie.mkv"), 7, piece) == (
        "Movie.music-07.clean.01h02m03s.flac"
    )


def test_a_cut_piece_is_faded_at_both_ends_and_levelled_for_playback():
    piece = MusicPiece(start=12.0, end=42.0, voice_share=0.0, peak_lufs=-20.0)

    command = extract_command(
        Path("/cache/no_vocals.flac"),
        piece,
        output=Path("/out/piece.flac"),
        ffmpeg_path="ffmpeg",
    )

    filters = command[command.index("-af") + 1]
    assert filters.startswith("afade=t=in:st=0:d=0.05")
    assert "afade=t=out:st=29.950" in filters
    assert "loudnorm" in filters
    assert command[command.index("-ss") + 1] == "12.000"
    assert command[command.index("-t") + 1] == "30.000"


def test_raw_levels_leaves_the_piece_at_the_level_it_was_played_at():
    piece = MusicPiece(start=0.0, end=10.0, voice_share=0.0, peak_lufs=-20.0)

    command = extract_command(
        Path("/a.flac"),
        piece,
        output=Path("/b.flac"),
        ffmpeg_path="ffmpeg",
        normalize=False,
    )

    assert "loudnorm" not in command[command.index("-af") + 1]


def test_the_dub_still_asks_for_two_stems_and_music_asks_for_all_of_them():
    two = separate_command(Path("/c/source.wav"), output_dir=Path("/c"))
    every = separate_command(
        Path("/c/source.wav"), output_dir=Path("/c"), model="htdemucs_6s", two_stems=None
    )

    assert two[two.index("--two-stems") + 1] == "vocals"
    assert "--two-stems" not in every
    assert every[every.index("-n") + 1] == "htdemucs_6s"


def test_every_stem_the_model_makes_has_a_place_in_the_cache():
    paths = stem_paths(Path("/cache"), "htdemucs_6s")

    assert set(paths) == set(STEM_MODELS["htdemucs_6s"])
    assert paths["piano"] == Path("/cache/stems/htdemucs_6s/piano.flac")
