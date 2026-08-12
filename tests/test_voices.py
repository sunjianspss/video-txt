from __future__ import annotations

from video_txt.clone import Reference
from video_txt.voices import (
    DEFAULT_SAY_VOICE,
    DEFAULT_VOICE,
    SPEAKER_VOICE_POOL,
    VoiceChoice,
    VoicePlan,
    assign_voices,
    resolve_voice_name,
    speaker_name_problem,
)

TWO_SPEAKERS = ["SPEAKER_00", "SPEAKER_01"]


def voices_for(speakers: list[str], *, engine: str = "edge-tts", **named: str) -> dict[str, str]:
    return assign_voices(speakers, engine=engine, voice=DEFAULT_VOICE, named=named)


def test_an_engine_falls_back_to_a_voice_it_actually_has():
    assert resolve_voice_name("edge-tts", DEFAULT_VOICE) == DEFAULT_VOICE
    assert resolve_voice_name("say", DEFAULT_VOICE) == DEFAULT_SAY_VOICE
    assert resolve_voice_name("say", "Meijia") == "Meijia"


def test_a_line_nobody_was_attributed_is_read_by_whoever_talks_most():
    plan = VoicePlan(
        default=VoiceChoice(name="zh-CN-XiaoxiaoNeural"),
        by_position={2: VoiceChoice(name="zh-CN-YunxiNeural")},
    )

    assert plan.for_position(2).name == "zh-CN-YunxiNeural"
    assert plan.for_position(99) is plan.default


def test_a_cloned_clip_is_cached_against_the_reference_it_came_from(tmp_path):
    first = VoiceChoice(
        name="main@F5TTS_v1_Base",
        reference=Reference(speaker="", audio=tmp_path / "a.wav", text="hello there"),
    )
    second = VoiceChoice(
        name="main@F5TTS_v1_Base",
        reference=Reference(speaker="", audio=tmp_path / "b.wav", text="hello there"),
    )
    reworded = VoiceChoice(
        name="main@F5TTS_v1_Base",
        reference=Reference(speaker="", audio=tmp_path / "a.wav", text="something else"),
    )

    assert first.identity != second.identity
    assert first.identity != reworded.identity


def test_the_speaker_who_talks_most_keeps_the_voice_that_was_chosen():
    names = voices_for(["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])

    assert names["SPEAKER_00"] == DEFAULT_VOICE
    assert len(set(names.values())) == 3


def test_a_voice_named_for_one_speaker_wins_over_the_pool():
    names = voices_for(TWO_SPEAKERS, SPEAKER_01="zh-CN-YunyangNeural")

    assert names["SPEAKER_01"] == "zh-CN-YunyangNeural"


def test_a_voice_named_for_one_speaker_is_not_handed_out_again():
    """Naming a pool voice for one speaker must not leave two speakers sounding alike."""
    speakers = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]

    names = assign_voices(
        speakers,
        engine="edge-tts",
        voice=DEFAULT_VOICE,
        named={"SPEAKER_01": SPEAKER_VOICE_POOL[0]},
    )

    assert names["SPEAKER_01"] == SPEAKER_VOICE_POOL[0]
    assert len(set(names.values())) == 3


def test_an_engine_with_one_chinese_voice_hands_it_to_everybody():
    """say has a single Mandarin voice, so there is no pool to hand out."""
    names = voices_for(TWO_SPEAKERS, engine="say")

    assert set(names.values()) == {DEFAULT_SAY_VOICE}


def test_names_that_all_match_a_speaker_raise_nothing():
    assert (
        speaker_name_problem(TWO_SPEAKERS, voices={"SPEAKER_01"}, references={"SPEAKER_00"}) is None
    )


def test_a_voice_named_for_a_speaker_nobody_found_is_refused():
    """Unchecked, the name matches no one and every line keeps the voice it had."""
    problem = speaker_name_problem(TWO_SPEAKERS, voices={"SPEAKER_09"}, references={})

    assert problem is not None
    assert "--speaker-voice names SPEAKER_09" in problem


def test_a_voice_that_never_says_whose_it_is_asks_for_a_name():
    problem = speaker_name_problem(TWO_SPEAKERS, voices={""}, references={})

    assert problem is not None
    assert "which of the 2 speakers" in problem


def test_a_reference_clip_named_for_a_speaker_nobody_found_is_refused():
    problem = speaker_name_problem(TWO_SPEAKERS, voices={}, references={"SPEAKER_09"})

    assert problem is not None
    assert "--clone-reference names SPEAKER_09" in problem


def test_a_lone_speaker_needs_no_name_in_front_of_their_reference_clip():
    assert speaker_name_problem(["SPEAKER_00"], voices={}, references={""}) is None


def test_a_lone_speaker_needs_no_name_in_front_of_their_voice():
    """The bare form is allowed through the name check, so it has to be honoured."""
    assert speaker_name_problem(["SPEAKER_00"], voices={""}, references={}) is None

    names = assign_voices(
        ["SPEAKER_00"], engine="edge-tts", voice=DEFAULT_VOICE, named={"": "zh-CN-YunxiNeural"}
    )

    assert names == {"SPEAKER_00": "zh-CN-YunxiNeural"}


def test_a_voice_named_for_the_speaker_beats_the_bare_one():
    names = assign_voices(
        ["SPEAKER_00"],
        engine="edge-tts",
        voice=DEFAULT_VOICE,
        named={"": "zh-CN-YunxiNeural", "SPEAKER_00": "zh-CN-YunyangNeural"},
    )

    assert names == {"SPEAKER_00": "zh-CN-YunyangNeural"}


def test_a_bare_voice_is_left_alone_when_there_is_more_than_one_speaker():
    """With a crowd the bare form is refused outright, so it must not be guessed at."""
    names = voices_for(TWO_SPEAKERS, **{"": "zh-CN-YunxiNeural"})

    assert names["SPEAKER_00"] == DEFAULT_VOICE
