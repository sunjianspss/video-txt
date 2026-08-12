from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from video_txt import dub as dub_module
from video_txt import fit as fit_module
from video_txt.clone import CloneOptions, Reference
from video_txt.diarize import SpeakerTurn
from video_txt.dub import (
    SHORTEN_CACHE_NAME,
    DubError,
    DubOptions,
    build_dub_mux_command,
    build_tts_command,
    build_voice_plan,
    default_audio_output,
    default_dubbed_output,
    fit_subtitle_to_timeline,
    measure_units,
    partial_clip_path,
    report_cache,
    run_dub,
    segment_filename,
    source_texts_by_position,
    synthesize_segments,
)
from video_txt.fit import FitOptions
from video_txt.subtitles import SubtitleCue, parse_srt_text
from video_txt.timeline import PlacedSegment
from video_txt.translate import TranslationConfig
from video_txt.voices import VoiceChoice, VoicePlan

TIMED = (
    "1\n00:00:01,000 --> 00:00:03,000\nfirst\n\n"
    "2\n00:00:05,000 --> 00:00:06,000\nsecond\n\n"
    "3\n00:00:10,000 --> 00:00:12,000\nthird\n"
)
BACK_TO_BACK = (
    "1\n00:00:00,000 --> 00:00:01,000\n这是一句很长很长的中文台词啊。\n\n"
    "2\n00:00:01,000 --> 00:00:03,000\n短句。\n"
)
SPLIT_SENTENCE = (
    "1\n00:00:00,000 --> 00:00:02,000\n我们先看第一点，\n\n"
    "2\n00:00:02,000 --> 00:00:04,000\n也就是记忆的写入。\n\n"
    "3\n00:00:04,000 --> 00:00:06,000\n第二点是检索。\n"
)
SPLIT_SENTENCE_SPOKEN = ["我们先看第一点，也就是记忆的写入。", "第二点是检索。"]


def numbered(text: str) -> list[tuple[int, SubtitleCue]]:
    return list(enumerate(parse_srt_text(text), start=1))


def make_options(**overrides) -> DubOptions:
    defaults = dict(
        video_input=Path("/videos/clip.mp4"),
        subtitle_input=Path("/videos/clip.zh.srt"),
        video_output=Path("/videos/clip.zh-dubbed.mp4"),
    )
    return DubOptions(**{**defaults, **overrides})


def test_default_dubbed_output_names_the_dub():
    assert default_dubbed_output(Path("/v/a.mp4")).name == "a.zh-dubbed.mp4"
    assert default_dubbed_output(Path("/v/a.webm")).name == "a.zh-dubbed.mp4"


def test_the_dub_is_named_after_the_language_it_is_spoken_in():
    """A Japanese dub called .zh-dubbed.mp4 overwrites the Chinese one made before it."""
    japanese = default_dubbed_output(Path("/v/a.mp4"), target_language="Japanese")
    unknown = default_dubbed_output(Path("/v/a.mp4"), target_language="Klingon")

    assert japanese.name == "a.ja-dubbed.mp4"
    assert unknown.name == "a.translated-dubbed.mp4"


def test_edge_tts_command_shape():
    command = build_tts_command(
        make_options(voice="zh-CN-YunxiNeural", rate="+10%"),
        launcher=["/bin/edge-tts"],
        text="你好",
        output_path=Path("/cache/a.mp3"),
    )
    assert command[command.index("--voice") + 1] == "zh-CN-YunxiNeural"
    assert "--rate=+10%" in command
    assert "--text=你好" in command
    assert command[command.index("--write-media") + 1] == "/cache/a.mp3"


def test_say_command_falls_back_to_a_mac_voice():
    command = build_tts_command(
        make_options(engine="say"),
        launcher=["/usr/bin/say"],
        text="你好",
        output_path=Path("/cache/a.aiff"),
    )
    assert command == ["/usr/bin/say", "-v", "Tingting", "-o", "/cache/a.aiff", "--", "你好"]


@pytest.mark.parametrize("engine", ["edge-tts", "say"])
def test_a_line_that_opens_with_a_dash_is_spoken_not_read_as_a_flag(engine):
    """Dialogue is often dashed. Handed over loose, it is parsed as an option."""
    command = build_tts_command(
        make_options(engine=engine),
        launcher=["/bin/tts"],
        text="--这是台词",
        output_path=Path("/cache/a.mp3"),
    )

    assert command[-1] != "--这是台词" or command[-2] == "--"
    assert "--这是台词" in " ".join(command)


def test_segment_filename_is_stable_and_text_sensitive():
    first = segment_filename("edge-tts", "你好", "v", "+0%")
    assert first == segment_filename("edge-tts", "你好", "v", "+0%")
    assert first != segment_filename("edge-tts", "你好啊", "v", "+0%")
    assert first != segment_filename("edge-tts", "你好", "v", "+10%")
    assert first != segment_filename("edge-tts", "你好", "other", "+0%")
    assert first.endswith(".mp3")
    assert segment_filename("say", "x", "v", "+0%").endswith(".aiff")
    assert segment_filename("f5-tts", "x", "v", "+0%").endswith(".wav")


def test_two_speakers_never_share_a_cached_clip():
    plan = VoicePlan(
        default=VoiceChoice(name="zh-CN-XiaoxiaoNeural"),
        by_position={2: VoiceChoice(name="zh-CN-YunxiNeural")},
    )
    names = [plan.for_position(position).identity for position in (1, 2)]

    assert names[0] != names[1]
    assert segment_filename("edge-tts", "你好", names[0], "+0%") != segment_filename(
        "edge-tts", "你好", names[1], "+0%"
    )


def test_where_a_line_sits_does_not_change_the_clip_it_gets():
    """An insertion above must not throw away every clip below it."""
    assert segment_filename("edge-tts", "你好", "v", "+0%") == segment_filename(
        "edge-tts", "你好", "v", "+0%"
    )


def test_a_line_said_twice_is_synthesized_once(tmp_path, monkeypatch):
    """Two workers racing to write the one clip they share would clobber each other."""
    asked: list[list[str]] = []

    def fake_online(pending, _options):
        asked.append([text for text, _, _ in pending])
        for _, _, path in pending:
            path.write_bytes(b"mp3")

    monkeypatch.setattr(dub_module, "synthesize_online", fake_online)

    paths = synthesize_segments(
        [(1, "谢谢"), (2, "不客气"), (3, "谢谢")], make_options(), tmp_path / "cache"
    )

    assert asked == [["谢谢", "不客气"]]
    assert paths[0] == paths[2]


def test_source_texts_line_up_with_the_translation_by_position(tmp_path):
    translated = parse_srt_text(TIMED)
    source = tmp_path / "clip.srt"
    source.write_text(TIMED, encoding="utf-8")
    assert source_texts_by_position(source, translated) == {1: "first", 2: "second", 3: "third"}
    assert source_texts_by_position(None, translated) == {}

    mismatched = tmp_path / "short.srt"
    mismatched.write_text("1\n00:00:01,000 --> 00:00:02,000\nonly\n", encoding="utf-8")
    assert source_texts_by_position(mismatched, translated) == {}


def test_fit_subtitle_to_timeline_rewrites_the_overrunning_lines(tmp_path, monkeypatch):
    subtitle = tmp_path / "clip.zh.srt"
    subtitle.write_text(BACK_TO_BACK, encoding="utf-8")
    source = tmp_path / "clip.srt"
    source.write_text(
        TIMED.replace("3\n00:00:10,000 --> 00:00:12,000\nthird\n", ""), encoding="utf-8"
    )
    seen: list[str | None] = []

    def fake_synthesize(lines, _options, _cache_dir, **_kwargs):
        return [Path(f"/cache/{position}-{len(text)}.mp3") for position, text in lines]

    def fake_probe(paths, **_kwargs):
        # Four characters per second, encoded in the clip name by fake_synthesize.
        return [int(path.stem.split("-")[1]) / 4 for path in paths]

    def fake_shorten(requests, **_kwargs):
        seen.extend(request.source_text for request in requests)
        return {request.position: "短" * request.max_chars for request in requests}

    monkeypatch.setattr(dub_module, "find_ffprobe", lambda *_a, **_k: "ffprobe")
    monkeypatch.setattr(dub_module, "synthesize_segments", fake_synthesize)
    monkeypatch.setattr(dub_module, "probe_durations", fake_probe)
    monkeypatch.setattr(fit_module, "request_shorter_texts", fake_shorten)

    options = make_options(subtitle_input=subtitle, source_subtitle=source)
    cues = list(enumerate(parse_srt_text(BACK_TO_BACK), start=1))
    fit = FitOptions(
        translation=TranslationConfig(base_url="https://x.test", api_key="k", model="m")
    )
    updated, fitted_path = fit_subtitle_to_timeline(
        cues, options, fit=fit, cache_dir=tmp_path / "cache", ffmpeg_path="ffmpeg"
    )

    # Two rounds, both rewriting from the English line, each capped at a 35% cut.
    assert seen == ["first", "first"]
    assert [cue.text for _, cue in updated] == ["短短短短短短", "短句。"]
    assert fitted_path == tmp_path / "clip.zh.fitted.srt"
    assert "短短短短短短" in fitted_path.read_text(encoding="utf-8")
    assert subtitle.read_text(encoding="utf-8") == BACK_TO_BACK


def timed_clips(monkeypatch, seconds: list[float]) -> list[list[str]]:
    """Stand in for synthesis and ffprobe; record the text of every clip asked for."""
    spoken: list[list[str]] = []

    def fake_synthesize(lines, _options, _cache_dir, **_kwargs):
        spoken.append([text for _, text in lines])
        return [Path(f"/cache/{position}.mp3") for position, _ in lines]

    monkeypatch.setattr(dub_module, "find_ffprobe", lambda *_a, **_k: "ffprobe")
    monkeypatch.setattr(dub_module, "synthesize_segments", fake_synthesize)
    monkeypatch.setattr(dub_module, "probe_durations", lambda paths, **_k: seconds[: len(paths)])
    return spoken


def test_a_sentences_seconds_are_shared_out_between_the_lines_it_holds(tmp_path, monkeypatch):
    timed_clips(monkeypatch, [3.4, 1.5])
    cues = numbered(SPLIT_SENTENCE)

    measured = measure_units(
        cues,
        {position: cue.text for position, cue in cues},
        make_options(),
        cache_dir=tmp_path,
        ffmpeg_path="ffmpeg",
    )

    # Lines 1 and 2 were spoken as one 3.4 s clip; line 3 had a clip to itself.
    assert sum(measured[position] for position in (1, 2)) == pytest.approx(3.4)
    assert measured[1] < measured[2]
    assert measured[3] == pytest.approx(1.5)


def test_speaking_line_by_line_still_times_every_line_on_its_own(tmp_path, monkeypatch):
    timed_clips(monkeypatch, [1.0, 2.0, 3.0])
    cues = numbered(SPLIT_SENTENCE)

    measured = measure_units(
        cues,
        {position: cue.text for position, cue in cues},
        make_options(voice_unit="line"),
        cache_dir=tmp_path,
        ffmpeg_path="ffmpeg",
    )

    assert measured == {1: 1.0, 2: 2.0, 3: 3.0}


def test_the_fit_times_the_clips_the_dub_will_actually_speak(tmp_path, monkeypatch):
    """Timing line by line bills each line for a lead-in the merged clip never pays,
    and leaves behind a full set of clips that never reach the video."""
    subtitle = tmp_path / "clip.zh.srt"
    subtitle.write_text(SPLIT_SENTENCE, encoding="utf-8")
    spoken = timed_clips(monkeypatch, [1.7, 1.7])
    monkeypatch.setattr(
        fit_module,
        "request_shorter_texts",
        lambda *_args, **_kwargs: pytest.fail("every sentence fits its own span"),
    )

    fit = FitOptions(
        translation=TranslationConfig(base_url="https://x.test", api_key="k", model="m")
    )
    fit_subtitle_to_timeline(
        numbered(SPLIT_SENTENCE),
        make_options(subtitle_input=subtitle),
        fit=fit,
        cache_dir=tmp_path / "cache",
        ffmpeg_path="ffmpeg",
    )

    assert spoken == [SPLIT_SENTENCE_SPOKEN]


def test_fit_subtitle_to_timeline_keeps_the_original_file_when_nothing_overruns(
    tmp_path, monkeypatch
):
    subtitle = tmp_path / "clip.zh.srt"
    subtitle.write_text(BACK_TO_BACK, encoding="utf-8")

    monkeypatch.setattr(dub_module, "find_ffprobe", lambda *_a, **_k: "ffprobe")
    monkeypatch.setattr(
        dub_module,
        "synthesize_segments",
        lambda lines, *_args, **_kwargs: [Path(f"/cache/{p}.mp3") for p, _ in lines],
    )
    monkeypatch.setattr(dub_module, "probe_durations", lambda paths, **_kwargs: [0.2] * len(paths))
    monkeypatch.setattr(
        fit_module,
        "request_shorter_texts",
        lambda *_args, **_kwargs: pytest.fail("no line overruns its slot"),
    )

    options = make_options(subtitle_input=subtitle)
    cues = list(enumerate(parse_srt_text(BACK_TO_BACK), start=1))
    fit = FitOptions(
        translation=TranslationConfig(base_url="https://x.test", api_key="k", model="m")
    )
    updated, fitted_path = fit_subtitle_to_timeline(
        cues, options, fit=fit, cache_dir=tmp_path / "cache", ffmpeg_path="ffmpeg"
    )

    assert updated == cues
    assert fitted_path == subtitle
    assert not (tmp_path / "clip.zh.fitted.srt").exists()


def dub_with_fakes(monkeypatch, tmp_path, **overrides) -> Path:
    """Run the whole dub with ffmpeg stubbed out; hand back the intermediate track."""
    subtitle = tmp_path / "clip.zh.srt"
    subtitle.write_text(BACK_TO_BACK, encoding="utf-8")

    def fake_render(segments, *, output_path, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"wav")
        return [PlacedSegment(start=0.0, duration=1.0, tempo=1.0, overflow=0.0) for _ in segments]

    monkeypatch.setattr(dub_module, "find_ffmpeg", lambda **_kwargs: "ffmpeg")
    monkeypatch.setattr(
        dub_module,
        "synthesize_segments",
        lambda lines, *_args, **_kwargs: [Path(f"/cache/{position}.mp3") for position, _ in lines],
    )
    monkeypatch.setattr(dub_module, "probe_duration", lambda *_args, **_kwargs: 10.0)
    monkeypatch.setattr(dub_module, "render_audio_track", fake_render)
    monkeypatch.setattr(
        dub_module.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0)
    )

    options = make_options(
        **{
            "subtitle_input": subtitle,
            "video_output": tmp_path / "clip.zh-dubbed.mp4",
            **overrides,
        }
    )
    run_dub(options)
    return options.audio_output or default_audio_output(options.video_output)


def test_the_intermediate_voice_track_goes_away_once_it_is_in_the_video(monkeypatch, tmp_path):
    assert not dub_with_fakes(monkeypatch, tmp_path).exists()


def test_the_intermediate_voice_track_can_be_kept(monkeypatch, tmp_path):
    assert dub_with_fakes(monkeypatch, tmp_path, keep_audio=True).is_file()


def test_a_voice_track_the_caller_named_is_never_removed(monkeypatch, tmp_path):
    chosen = tmp_path / "mine.wav"

    assert dub_with_fakes(monkeypatch, tmp_path, audio_output=chosen).is_file()


def test_a_folder_named_for_the_output_is_made_before_ffmpeg_needs_it(monkeypatch, tmp_path):
    """Rendering the voice track elsewhere leaves nobody else to create the folder,
    and ffmpeg only reports it as 'No such file or directory' once the work is done."""
    destination = tmp_path / "dubs" / "final"

    dub_with_fakes(
        monkeypatch,
        tmp_path,
        video_output=destination / "clip.zh-dubbed.mp4",
        audio_output=tmp_path / "mine.wav",
    )

    assert destination.is_dir()


def stocked_cache(tmp_path) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("clip-aaa.mp3", "clip-bbb.mp3", "clip-ccc.aiff"):
        (cache / name).write_bytes(b"x" * 1024)
    (cache / SHORTEN_CACHE_NAME).write_text("{}", encoding="utf-8")
    return cache, cache / "clip-aaa.mp3"


def test_pruning_clears_the_clips_this_run_did_not_use(tmp_path):
    cache, used = stocked_cache(tmp_path)

    report_cache(cache, [used], prune=True)

    assert sorted(path.name for path in cache.iterdir()) == sorted([used.name, SHORTEN_CACHE_NAME])


def test_a_cache_full_of_old_clips_is_pointed_out_rather_than_emptied(tmp_path, capsys):
    cache, used = stocked_cache(tmp_path)

    report_cache(cache, [used], prune=False)

    assert "--prune-cache" in capsys.readouterr().out
    assert (cache / "clip-bbb.mp3").is_file()


def test_a_cache_holding_only_what_was_used_says_nothing(tmp_path, capsys):
    cache, used = stocked_cache(tmp_path)
    clips = sorted(path for path in cache.iterdir() if path.suffix in {".mp3", ".aiff"})

    report_cache(cache, clips, prune=False)

    assert capsys.readouterr().out == ""
    assert used.is_file()


def test_pruning_leaves_recordings_the_dub_did_not_make(tmp_path):
    """--cache-dir can be pointed at a folder that already holds the user's own audio."""
    cache, used = stocked_cache(tmp_path)
    theirs = cache / "interview-master.wav"
    theirs.write_bytes(b"x" * 1024)

    report_cache(cache, [used], prune=True)

    assert theirs.is_file()


def test_pruning_sweeps_up_the_clips_an_interrupted_run_half_wrote(tmp_path):
    cache, used = stocked_cache(tmp_path)
    half_written = partial_clip_path(cache / "clip-ddd.mp3")
    half_written.write_bytes(b"x" * 1024)

    report_cache(cache, [used], prune=True)

    assert not half_written.exists()


def test_every_speaker_is_cloned_from_a_clip_of_their_own(tmp_path, monkeypatch):
    asked: dict[str, object] = {}

    def fake_build_references(_video, *, wanted, **_kwargs):
        asked["wanted"] = wanted
        return {
            speaker: Reference(speaker=speaker, audio=tmp_path / f"{speaker}.wav", text=speaker)
            for speaker in wanted
        }

    monkeypatch.setattr(dub_module, "build_references", fake_build_references)
    options = make_options(
        engine="f5-tts",
        clone=CloneOptions(engine="f5-tts"),
        turns=[SpeakerTurn(0.0, 5.0, "SPEAKER_00"), SpeakerTurn(5.0, 7.0, "SPEAKER_01")],
    )

    plan = build_voice_plan(
        options,
        speakers={1: "SPEAKER_00", 2: "SPEAKER_01"},
        cache_dir=tmp_path,
        ffmpeg_path="ffmpeg",
    )

    assert asked["wanted"] == ["SPEAKER_00", "SPEAKER_01"]
    assert plan.for_position(1).name == "SPEAKER_00@F5TTS_v1_Base"
    assert plan.for_position(2).reference.audio.name == "SPEAKER_01.wav"
    # A line diarization never attributed is read by whoever talks most.
    assert plan.for_position(99) is plan.default


def cloning_plan(tmp_path) -> VoicePlan:
    reference = Reference(speaker="", audio=tmp_path / "ref.wav", text="hi")
    return VoicePlan(default=VoiceChoice(name="main@F5TTS_v1_Base", reference=reference))


def test_cloned_clips_are_synthesized_in_one_batch(tmp_path, monkeypatch):
    batches: list[list[object]] = []

    def fake_synthesize_clips(jobs, **_kwargs):
        batches.append(jobs)
        for job in jobs:
            job.output.write_bytes(b"RIFF")

    monkeypatch.setattr(dub_module, "synthesize_clips", fake_synthesize_clips)
    options = make_options(engine="f5-tts", clone=CloneOptions(engine="f5-tts"))
    plan = cloning_plan(tmp_path)
    lines = [(1, "你好"), (2, "再见")]

    paths = synthesize_segments(lines, options, tmp_path / "cache", plan=plan)

    assert [path.suffix for path in paths] == [".wav", ".wav"]
    assert len(batches) == 1
    assert [job.text for job in batches[0]] == ["你好", "再见"]

    # The model is expensive to load, so a rerun must not start it for nothing.
    synthesize_segments(lines, options, tmp_path / "cache", plan=plan)
    assert len(batches) == 1


def test_a_voice_with_nothing_to_clone_from_stops_before_the_model_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dub_module,
        "synthesize_clips",
        lambda *_args, **_kwargs: pytest.fail("there is no reference to clone"),
    )
    options = make_options(engine="f5-tts", clone=CloneOptions(engine="f5-tts"))
    plan = VoicePlan(default=VoiceChoice(name="main@F5TTS_v1_Base"))

    with pytest.raises(DubError, match="No reference clip"):
        synthesize_segments([(1, "你好")], options, tmp_path / "cache", plan=plan)


def test_dub_mux_command_uses_the_fitted_subtitle_when_one_was_written():
    command = build_dub_mux_command(
        make_options(soft_subtitle=True),
        ffmpeg_path="/bin/ffmpeg",
        audio_path=Path("/videos/clip.dub.wav"),
        keep_bgm=False,
        subtitle_path=Path("/videos/clip.zh.fitted.srt"),
    )
    assert "/videos/clip.zh.fitted.srt" in command
    assert "/videos/clip.zh.srt" not in command


def test_dub_mux_command_replaces_the_audio_track_by_default():
    command = build_dub_mux_command(
        make_options(),
        ffmpeg_path="/bin/ffmpeg",
        audio_path=Path("/videos/clip.dub.wav"),
        keep_bgm=False,
    )
    assert "-filter_complex" not in command
    assert command[command.index("-map") + 1] == "0:v:0"
    assert "1:a:0" in command
    assert command[command.index("-c:a") + 1] == "aac"
    assert "-shortest" in command
    assert command[command.index("-movflags") + 1] == "+faststart"


def test_dub_mux_command_mixes_the_original_audio_when_keeping_bgm():
    command = build_dub_mux_command(
        make_options(bgm_volume=0.2),
        ffmpeg_path="/bin/ffmpeg",
        audio_path=Path("/videos/clip.dub.wav"),
        keep_bgm=True,
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "volume=0.200" in graph
    assert "amix=inputs=2:duration=first:normalize=0[aout]" in graph
    assert "[aout]" in command


def test_dub_mux_command_can_add_the_subtitle_track():
    command = build_dub_mux_command(
        make_options(soft_subtitle=True),
        ffmpeg_path="/bin/ffmpeg",
        audio_path=Path("/videos/clip.dub.wav"),
        keep_bgm=False,
    )
    assert command.count("-i") == 3
    assert command[command.index("-c:s") + 1] == "mov_text"
    assert "2:0" in command
