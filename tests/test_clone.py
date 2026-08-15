from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_txt import clone as clone_module
from video_txt import clone_worker
from video_txt.clone import (
    MANIFEST_NAME,
    CloneError,
    CloneJob,
    CloneOptions,
    Reference,
    build_extract_command,
    build_manifest,
    build_references,
    build_worker_command,
    pick_reference_window,
    reference_runs,
    resolve_model,
    speed_from_rate,
    synthesize_clips,
    window_of,
)
from video_txt.subtitles import SubtitleCue, parse_srt_text

INTERVIEW = (
    "1\n00:00:00,000 --> 00:00:02,000\nWelcome to the show.\n\n"
    "2\n00:00:02,000 --> 00:00:07,000\nToday we look at how agents remember things.\n\n"
    "3\n00:00:08,000 --> 00:00:11,000\nThanks for having me.\n\n"
    "4\n00:00:11,000 --> 00:00:25,000\nI have been at this for a decade and it is still hard.\n"
)
SPEAKERS = {1: "SPEAKER_00", 2: "SPEAKER_00", 3: "SPEAKER_01", 4: "SPEAKER_01"}


def numbered(text: str) -> list[tuple[int, SubtitleCue]]:
    return list(enumerate(parse_srt_text(text), start=1))


def test_a_run_holds_only_the_lines_one_speaker_says_in_a_row():
    runs = reference_runs(numbered(INTERVIEW), speaker="SPEAKER_00", speakers=SPEAKERS)

    assert [[cue.index for cue in run] for run in runs] == [["1", "2"]]


def test_a_pause_splits_one_speaker_into_two_runs():
    paused = (
        "1\n00:00:00,000 --> 00:00:02,000\nFirst thought.\n\n"
        "2\n00:00:05,000 --> 00:00:07,000\nSecond thought.\n"
    )
    runs = reference_runs(numbered(paused), speaker="", speakers={})

    assert len(runs) == 2


def test_the_window_stops_before_the_clip_grows_too_long_to_clone_from():
    cues = [cue for _, cue in numbered(INTERVIEW)]

    # 8s to 25s would be 17 seconds, so the second line is left out.
    start, end, text = window_of(cues[2:])

    assert (start, end) == (8.0, 11.0)
    assert text == "Thanks for having me."


def test_the_window_takes_as_much_speech_as_it_can_hold():
    start, end, text = window_of([cue for _, cue in numbered(INTERVIEW)][:2])

    assert (start, end) == (0.0, 7.0)
    assert text.startswith("Welcome to the show. Today")


def test_a_clip_of_usable_length_beats_a_wordier_one_that_is_too_short():
    cues = numbered(INTERVIEW)

    chosen = pick_reference_window(cues, speaker="SPEAKER_01", speakers=SPEAKERS)

    # The 14s line says more, but only the 3s one can be cloned from at all.
    assert chosen == (8.0, 11.0, "Thanks for having me.")


def test_reference_selection_refuses_every_window_outside_three_to_twelve_seconds():
    unusable = (
        "1\n00:00:00,000 --> 00:00:02,000\ntoo short\n\n"
        "2\n00:00:03,000 --> 00:00:16,000\ntoo long\n"
    )

    assert pick_reference_window(numbered(unusable), speaker="", speakers={}) is None


def test_reference_selection_can_skip_an_invalid_leading_cue():
    later_window = (
        "1\n00:00:00,000 --> 00:00:14,000\ntoo long\n\n"
        "2\n00:00:14,000 --> 00:00:18,000\nusable reference\n"
    )

    assert pick_reference_window(numbered(later_window), speaker="", speakers={}) == (
        14.0,
        18.0,
        "usable reference",
    )


def test_a_speaker_who_never_speaks_has_nothing_to_clone():
    assert (
        pick_reference_window(numbered(INTERVIEW), speaker="SPEAKER_09", speakers=SPEAKERS) is None
    )


def test_the_reference_clip_is_cut_as_mono_audio_at_the_window():
    command = build_extract_command(
        Path("/videos/clip.mp4"),
        start=8.0,
        end=10.5,
        output=Path("/cache/reference/SPEAKER_01-8000.wav"),
        ffmpeg_path="/bin/ffmpeg",
    )

    assert command[command.index("-ss") + 1] == "8.000"
    assert command[command.index("-t") + 1] == "2.500"
    assert command[command.index("-ac") + 1] == "1"
    assert "-vn" in command
    assert command[-1] == "/cache/reference/SPEAKER_01-8000.wav"


def test_references_are_cut_once_per_speaker_and_kept_beside_their_transcript(
    tmp_path, monkeypatch
):
    cut: list[list[str]] = []

    def fake_run(command, **_kwargs):
        cut.append(command)
        Path(command[-1]).write_bytes(b"RIFF")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(clone_module.subprocess, "run", fake_run)

    references = build_references(
        tmp_path / "clip.mp4",
        cues=numbered(INTERVIEW),
        speakers=SPEAKERS,
        wanted=["SPEAKER_00", "SPEAKER_01"],
        options=CloneOptions(engine="f5-tts"),
        cache_dir=tmp_path / "cache",
        ffmpeg_path="ffmpeg",
    )

    assert len(cut) == 2
    assert references["SPEAKER_00"].text.startswith("Welcome to the show.")
    assert references["SPEAKER_01"].audio.name == "SPEAKER_01-8000.wav"
    transcript = references["SPEAKER_01"].audio.with_suffix(".txt")
    assert transcript.read_text(encoding="utf-8").strip() == "Thanks for having me."


def test_a_supplied_clip_is_used_as_is_along_with_the_text_beside_it(tmp_path, monkeypatch):
    monkeypatch.setattr(
        clone_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("the caller already has a clip"),
    )
    audio = tmp_path / "mine.wav"
    audio.write_bytes(b"RIFF")
    audio.with_suffix(".txt").write_text("what I say in it", encoding="utf-8")

    references = build_references(
        tmp_path / "clip.mp4",
        cues=[],
        speakers={},
        wanted=[""],
        options=CloneOptions(engine="f5-tts", references={"": audio}),
        cache_dir=tmp_path / "cache",
        ffmpeg_path="ffmpeg",
    )

    assert references[""] == Reference(speaker="", audio=audio, text="what I say in it")


def test_nothing_to_clone_from_points_at_the_flag_that_fixes_it(tmp_path):
    with pytest.raises(CloneError, match="--clone-reference"):
        build_references(
            tmp_path / "clip.mp4",
            cues=[],
            speakers={},
            wanted=["SPEAKER_00"],
            options=CloneOptions(engine="f5-tts"),
            cache_dir=tmp_path / "cache",
            ffmpeg_path="ffmpeg",
        )


def test_cosyvoice_without_a_model_says_what_to_pass():
    with pytest.raises(CloneError, match="--clone-model"):
        resolve_model(CloneOptions(engine="cosyvoice"))

    assert resolve_model(CloneOptions(engine="f5-tts")) == "F5TTS_v1_Base"
    assert resolve_model(CloneOptions(engine="cosyvoice", model="/m")) == "/m"


def test_indextts_finds_its_weights_inside_the_checkout(tmp_path):
    repo = tmp_path / "index-tts"
    checkpoints = repo / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    assert resolve_model(CloneOptions(engine="index-tts", repo=repo)) == str(checkpoints)
    assert resolve_model(CloneOptions(engine="index-tts", model="/elsewhere")) == "/elsewhere"


def test_indextts_asks_only_for_the_setup_step_that_is_missing(tmp_path):
    """Whoever already cloned the repo must not be told to clone it again: that
    reads as 'your checkout is wrong' when all that is missing is a download."""
    with pytest.raises(CloneError, match="git clone") as no_repo:
        resolve_model(CloneOptions(engine="index-tts"))
    assert "--clone-repo" in str(no_repo.value)
    assert "hf download" in str(no_repo.value)

    with pytest.raises(CloneError, match="does not exist") as missing:
        resolve_model(CloneOptions(engine="index-tts", repo=tmp_path / "nowhere"))
    assert "git clone" in str(missing.value)
    assert "hf download" not in str(missing.value)

    repo = tmp_path / "index-tts"
    repo.mkdir()
    with pytest.raises(CloneError, match="hf download") as no_weights:
        resolve_model(CloneOptions(engine="index-tts", repo=repo))
    # The download command names the directory it has to land in, so it can be
    # pasted as printed, and nothing suggests cloning the checkout again.
    assert str(repo / "checkpoints") in str(no_weights.value)
    assert "git clone" not in str(no_weights.value)


def test_the_speaking_rate_carries_over_to_the_cloning_model():
    assert speed_from_rate("+0%") == 1.0
    assert speed_from_rate("+10%") == pytest.approx(1.1)
    assert speed_from_rate("-25%") == pytest.approx(0.75)
    assert speed_from_rate("nonsense") == 1.0


def test_the_worker_runs_on_the_interpreter_that_has_the_model():
    command = build_worker_command(
        CloneOptions(engine="f5-tts", python="/venvs/f5/bin/python"), Path("/cache/jobs.json")
    )

    assert command[0] == "/venvs/f5/bin/python"
    assert command[1].endswith("clone_worker.py")
    assert command[2] == "/cache/jobs.json"


def test_the_worker_finds_the_virtualenv_inside_the_checkout(tmp_path):
    """IndexTTS pins its own Python and torch, so its checkout carries a venv
    this package can never share. Nobody should have to spell that path out."""
    repo = tmp_path / "index-tts"
    python = repo / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    with_venv = CloneOptions(engine="index-tts", repo=repo)
    assert build_worker_command(with_venv, Path("/j"))[0] == str(python)

    bare = CloneOptions(engine="index-tts", repo=tmp_path / "bare")
    assert build_worker_command(bare, Path("/j"))[0] == sys.executable

    explicit = CloneOptions(engine="index-tts", repo=repo, python="/mine/python")
    assert build_worker_command(explicit, Path("/j"))[0] == "/mine/python"


def test_the_manifest_carries_every_job_with_the_clip_it_clones(tmp_path):
    reference = Reference(speaker="SPEAKER_00", audio=tmp_path / "ref.wav", text="hello")
    manifest = build_manifest(
        [CloneJob(text="你好", output=tmp_path / "a.wav", reference=reference)],
        CloneOptions(engine="f5-tts", device="mps", speed=1.1),
    )

    assert manifest["engine"] == "f5-tts"
    assert manifest["model"] == "F5TTS_v1_Base"
    assert manifest["device"] == "mps"
    assert manifest["speed"] == 1.1
    assert manifest["jobs"] == [
        {
            "text": "你好",
            "output": str(tmp_path / "a.wav"),
            "reference_audio": str(tmp_path / "ref.wav"),
            "reference_text": "hello",
            "speed": None,
        }
    ]


def test_the_manifest_says_what_language_indextts_is_speaking(tmp_path):
    reference = Reference(speaker="", audio=tmp_path / "ref.wav")
    jobs = [
        CloneJob(text="你好", output=tmp_path / "a.wav", reference=reference),
        CloneJob(text="再见", output=tmp_path / "b.wav", reference=reference, speed=1.3),
    ]

    manifest = build_manifest(jobs, CloneOptions(engine="index-tts", model="/ckpt", lang="zh"))

    assert manifest["lang"] == "ZH"
    # A re-spoken clip carries its own speed; the rest keep the manifest-wide one.
    assert [job["speed"] for job in manifest["jobs"]] == [None, 1.3]

    unknown = build_manifest(jobs, CloneOptions(engine="index-tts", model="/ckpt", lang="fr"))
    assert unknown["lang"] is None


def test_a_cosyvoice_checkout_is_put_on_the_workers_import_path(tmp_path):
    repo = tmp_path / "CosyVoice"
    (repo / "third_party" / "Matcha-TTS").mkdir(parents=True)
    manifest = build_manifest([], CloneOptions(engine="cosyvoice", model="/m", repo=repo))

    assert manifest["sys_path"] == [str(repo), str(repo / "third_party" / "Matcha-TTS")]


def test_a_respoken_clip_keeps_its_own_speed_in_the_worker():
    assert clone_worker.job_speed({"speed": 1.3}, 1.0) == 1.3
    assert clone_worker.job_speed({"speed": None}, 1.1) == 1.1
    assert clone_worker.job_speed({}, 1.1) == 1.1


def worker_calls(monkeypatch, *, returncode: int, writes: bool) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if writes:
            manifest = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            for job in manifest["jobs"]:
                Path(job["output"]).write_bytes(b"RIFF")
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(clone_module.subprocess, "run", fake_run)
    return commands


def test_a_conversation_is_synthesized_one_speaker_at_a_time(tmp_path, monkeypatch):
    """The model caches the encoding of the last reference clip it saw, so
    interleaved speakers would recompute it on nearly every line."""
    worker_calls(monkeypatch, returncode=0, writes=True)
    host = Reference(speaker="HOST", audio=tmp_path / "host.wav", text="hi")
    guest = Reference(speaker="GUEST", audio=tmp_path / "guest.wav", text="yo")
    jobs = [
        CloneJob(text="line 1", output=tmp_path / "1.wav", reference=host),
        CloneJob(text="line 2", output=tmp_path / "2.wav", reference=guest),
        CloneJob(text="line 3", output=tmp_path / "3.wav", reference=host),
        CloneJob(text="line 4", output=tmp_path / "4.wav", reference=guest),
    ]

    synthesize_clips(jobs, options=CloneOptions(engine="f5-tts"), cache_dir=tmp_path)

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [job["reference_audio"] for job in manifest["jobs"]] == [
        str(guest.audio),
        str(guest.audio),
        str(host.audio),
        str(host.audio),
    ]


def test_the_whole_batch_goes_to_one_worker_run(tmp_path, monkeypatch):
    commands = worker_calls(monkeypatch, returncode=0, writes=True)
    reference = Reference(speaker="", audio=tmp_path / "ref.wav", text="hi")
    jobs = [
        CloneJob(text=f"line {index}", output=tmp_path / f"{index}.wav", reference=reference)
        for index in range(3)
    ]

    synthesize_clips(jobs, options=CloneOptions(engine="f5-tts"), cache_dir=tmp_path)

    assert len(commands) == 1
    assert (tmp_path / MANIFEST_NAME).is_file()
    assert all(job.output.is_file() for job in jobs)


def test_a_worker_that_writes_nothing_is_not_reported_as_success(tmp_path, monkeypatch):
    worker_calls(monkeypatch, returncode=0, writes=False)
    reference = Reference(speaker="", audio=tmp_path / "ref.wav", text="hi")
    jobs = [CloneJob(text="line", output=tmp_path / "a.wav", reference=reference)]

    with pytest.raises(CloneError, match="wrote no audio"):
        synthesize_clips(jobs, options=CloneOptions(engine="f5-tts"), cache_dir=tmp_path)


def test_a_failed_worker_says_how_much_of_the_batch_survived(tmp_path, monkeypatch):
    worker_calls(monkeypatch, returncode=1, writes=True)
    reference = Reference(speaker="", audio=tmp_path / "ref.wav", text="hi")
    jobs = [CloneJob(text="line", output=tmp_path / "a.wav", reference=reference)]

    with pytest.raises(CloneError, match="1 of 1 clips are cached"):
        synthesize_clips(jobs, options=CloneOptions(engine="f5-tts"), cache_dir=tmp_path)


def test_the_worker_skips_clips_an_earlier_run_finished(tmp_path):
    done = tmp_path / "done.wav"
    done.write_bytes(b"RIFF")
    jobs = [
        {"output": str(done), "reference_audio": "", "reference_text": "", "text": "a"},
        {
            "output": str(tmp_path / "todo.wav"),
            "reference_audio": "",
            "reference_text": "",
            "text": "b",
        },
    ]

    assert clone_worker.pending_jobs(jobs) == [jobs[1]]


def test_the_worker_checks_the_reference_exists_before_loading_a_model(tmp_path):
    jobs = [{"reference_audio": str(tmp_path / "gone.wav"), "reference_text": "x", "text": "a"}]

    with pytest.raises(clone_worker.WorkerError, match="Reference audio not found"):
        clone_worker.check_references(jobs, engine="f5-tts")


def test_the_worker_refuses_an_engine_it_does_not_know():
    with pytest.raises(clone_worker.WorkerError, match="Unknown cloning engine"):
        clone_worker.build_engine({"engine": "espeak"})


def test_the_worker_turns_a_failure_into_an_exit_code_not_a_traceback(tmp_path, capsys):
    manifest = tmp_path / "jobs.json"
    job = {
        "text": "你好",
        "output": str(tmp_path / "a.wav"),
        "reference_audio": str(tmp_path / "gone.wav"),
        "reference_text": "hi",
    }
    manifest.write_text(
        json.dumps({"engine": "f5-tts", "model": "x", "jobs": [job]}), encoding="utf-8"
    )

    assert clone_worker.main([str(manifest)]) == 1
    assert "Reference audio not found" in capsys.readouterr().err
    assert clone_worker.main([]) == 2
