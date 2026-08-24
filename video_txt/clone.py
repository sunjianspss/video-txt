"""Speaking the translation in the original speaker's own voice.

edge-tts hands back a stranger reading the subtitles. CosyVoice and F5-TTS can
say the same words in the voice of whoever said them first, given a few seconds
of that person as a reference clip — which the source video already contains,
and the source subtitle already tells us where.

Both models take tens of seconds to load and hold gigabytes of weights, so
clips are not synthesized one subprocess at a time the way the online engines
are. A whole run goes to a single worker process as a manifest. That worker
imports nothing from this package: the models often need a virtualenv of their
own, and --clone-python points the same script at that interpreter.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .diarize import DEFAULT_SPEAKER
from .subtitles import SubtitleCue, join_text_parts

CLONE_ENGINES = ("cosyvoice", "f5-tts", "index-tts")
DEFAULT_F5_MODEL = "F5TTS_v1_Base"
# The layout the IndexTTS README tells everyone to download into.
INDEX_CHECKPOINTS_DIR = "checkpoints"
# The languages IndexTTS-2.5 takes as an explicit lang= hint, keyed by the
# filename suffix language_suffix() gives a target language.
INDEX_LANGS = {"zh": "ZH", "zh-hant": "ZH", "en": "EN", "ja": "JA", "es": "ES"}
MANIFEST_NAME = "clone-jobs.json"
REFERENCE_DIR_NAME = "reference"
# What both models ask for: long enough to carry a voice, short enough to stay
# one continuous stretch of speech.
REFERENCE_MIN_SECONDS = 3.0
REFERENCE_MAX_SECONDS = 12.0
# Lines further apart than this are two separate stretches of talking.
REFERENCE_JOIN_GAP = 0.6
REFERENCE_SAMPLE_RATE = 24000
RATE_PATTERN = re.compile(r"^([+-]\d+(?:\.\d+)?)%$")


class CloneError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reference:
    """The clip a voice is cloned from, and the words spoken in it."""

    speaker: str
    audio: Path
    text: str = ""

    @property
    def label(self) -> str:
        return self.speaker or "main"

    @cached_property
    def fingerprint(self) -> str:
        """Audio and transcript identity used to isolate synthesized-clip caches."""
        digest = hashlib.sha1()
        resolved = self.audio.expanduser().resolve()
        try:
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            # Dry runs and unit-level plans can name a future reference clip. Its
            # absolute path still prevents unrelated same-named clips from colliding.
            digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.text.encode("utf-8"))
        return digest.hexdigest()


@dataclass
class CloneOptions:
    engine: str
    model: str | None = None
    python: str | None = None
    repo: Path | None = None
    device: str | None = None
    speed: float = 1.0
    lang: str | None = None
    references: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class CloneJob:
    text: str
    output: Path
    reference: Reference
    # Speed multiplier for this one clip; None keeps the manifest-wide speed.
    speed: float | None = None


def is_rate(value: str) -> bool:
    """Whether --rate reads as the signed percentage every engine here expects."""
    return RATE_PATTERN.match(value.strip()) is not None


def speed_from_rate(rate: str) -> float:
    """Read --rate, which edge-tts spells as a percentage, as a speed multiplier."""
    match = RATE_PATTERN.match(rate.strip())
    if not match:
        return 1.0
    return max(0.2, 1.0 + float(match.group(1)) / 100)


INDEX_CLONE_STEPS = (
    "  git clone https://github.com/index-tts/index-tts.git\n  cd index-tts && uv sync\n"
)


def index_download_step(checkpoints: Path) -> str:
    return (
        f"  uvx --from huggingface-hub hf download IndexTeam/IndexTTS-2.5 --local-dir {checkpoints}"
    )


def resolve_index_model(options: CloneOptions) -> str:
    """The checkpoints directory inside the IndexTTS checkout.

    Three ways to not have one, and telling somebody who already cloned the
    repo to clone it again is how a missing download reads as a missing
    checkout. So each case says only the step that is actually left.
    """
    if options.repo is None:
        raise CloneError(
            "IndexTTS needs its checkout and the weights downloaded into it:\n"
            + INDEX_CLONE_STEPS
            + index_download_step(Path(INDEX_CHECKPOINTS_DIR))
            + "\nThen point at the checkout with: --clone-repo /path/to/index-tts"
        )

    repo = options.repo.expanduser().resolve()
    if not repo.is_dir():
        raise CloneError(
            f"--clone-repo points at {repo}, which does not exist. Clone IndexTTS "
            "there first:\n" + INDEX_CLONE_STEPS.rstrip()
        )

    checkpoints = repo / INDEX_CHECKPOINTS_DIR
    if not (checkpoints / "config.yaml").is_file():
        raise CloneError(
            f"The IndexTTS checkout at {repo} has no weights in it yet "
            f"({checkpoints / 'config.yaml'} is missing). Download them:\n"
            + index_download_step(checkpoints)
        )
    return str(checkpoints)


def resolve_model(options: CloneOptions) -> str:
    if options.model:
        return options.model
    if options.engine == "f5-tts":
        return DEFAULT_F5_MODEL
    if options.engine == "index-tts":
        return resolve_index_model(options)
    raise CloneError(
        "CosyVoice needs the model directory it was downloaded to.\n"
        "Pass it with: --clone-model /path/to/CosyVoice2-0.5B\n"
        "Or use --tts-engine f5-tts, which downloads its own weights."
    )


def worker_path() -> Path:
    return Path(__file__).with_name("clone_worker.py")


def repo_python(options: CloneOptions) -> Path | None:
    """The interpreter of the checkout's own virtualenv, when it has one.

    IndexTTS pins Python 3.11 and its own torch, so its checkout carries a
    virtualenv this package can never share. Finding it saves everybody the
    --clone-python flag."""
    if options.repo is None:
        return None
    candidate = options.repo.expanduser().resolve() / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def sys_path_entries(options: CloneOptions) -> list[str]:
    """Where a CosyVoice checkout keeps the packages it expects on sys.path."""
    if options.repo is None:
        return []
    repo = options.repo.expanduser().resolve()
    entries = [repo, repo / "third_party" / "Matcha-TTS"]
    return [str(entry) for entry in entries if entry.is_dir()]


def build_worker_command(options: CloneOptions, manifest: Path) -> list[str]:
    python = options.python or repo_python(options) or sys.executable
    return [str(python), str(worker_path()), str(manifest)]


def build_manifest(jobs: list[CloneJob], options: CloneOptions) -> dict[str, object]:
    return {
        "engine": options.engine,
        "model": resolve_model(options),
        "device": options.device,
        "speed": options.speed,
        "lang": INDEX_LANGS.get((options.lang or "").lower()),
        "sys_path": sys_path_entries(options),
        "jobs": [
            {
                "text": job.text,
                "output": str(job.output),
                "reference_audio": str(job.reference.audio),
                "reference_text": job.reference.text,
                "speed": job.speed,
            }
            for job in jobs
        ],
    }


def synthesize_clips(jobs: list[CloneJob], *, options: CloneOptions, cache_dir: Path) -> None:
    """Hand the whole batch to one worker process, then check what came back."""
    if not jobs:
        return

    # Clips from one reference sit together: the model caches the speaker
    # encoding of the last reference it saw, and a conversation would
    # otherwise recompute it on nearly every line.
    ordered = sorted(jobs, key=lambda job: str(job.reference.audio))

    manifest = cache_dir / MANIFEST_NAME
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(build_manifest(ordered, options), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    command = build_worker_command(options, manifest)
    completed = subprocess.run(command)
    missing = [job for job in jobs if not job.output.is_file() or job.output.stat().st_size == 0]
    if completed.returncode != 0:
        done = len(jobs) - len(missing)
        finished = f" {done} of {len(jobs)} clips are cached and will be reused." if done else ""
        raise CloneError(
            f"{options.engine} failed with exit code {completed.returncode}.{finished}"
        )
    if missing:
        raise CloneError(
            f"{options.engine} reported success but wrote no audio for "
            f"{len(missing)} of {len(jobs)} clips, starting with {missing[0].output.name}."
        )


def reference_runs(
    cues: list[tuple[int, SubtitleCue]], *, speaker: str, speakers: dict[int, str]
) -> list[list[SubtitleCue]]:
    """Stretches where this speaker talks without being interrupted."""
    runs: list[list[SubtitleCue]] = []
    current: list[SubtitleCue] = []

    for position, cue in cues:
        if cue.is_empty or speakers.get(position, DEFAULT_SPEAKER) != speaker:
            if current:
                runs.append(current)
                current = []
            continue
        if current and cue.time_range[0] - current[-1].time_range[1] > REFERENCE_JOIN_GAP:
            runs.append(current)
            current = []
        current.append(cue)

    if current:
        runs.append(current)
    return runs


def window_of(run: list[SubtitleCue]) -> tuple[float, float, str]:
    """Lines off the front of a run, up to as much as a reference clip may hold."""
    chosen = [run[0]]
    start = run[0].time_range[0]
    for cue in run[1:]:
        if cue.time_range[1] - start > REFERENCE_MAX_SECONDS:
            break
        chosen.append(cue)
    text = join_text_parts([cue.text.replace("\n", " ").strip() for cue in chosen])
    return start, chosen[-1].time_range[1], text


def window_rank(window: tuple[float, float, str]) -> tuple[int, float, int]:
    start, end, text = window
    duration = end - start
    slack = max(REFERENCE_MIN_SECONDS - duration, duration - REFERENCE_MAX_SECONDS, 0.0)
    # Length first, because a clip outside the range clones badly however much
    # it says; among clips that fit, the one with the most speech in it wins.
    return (slack == 0, -slack, len(text))


def pick_reference_window(
    cues: list[tuple[int, SubtitleCue]], *, speaker: str, speakers: dict[int, str]
) -> tuple[float, float, str] | None:
    windows = [
        window_of(run[offset:])
        for run in reference_runs(cues, speaker=speaker, speakers=speakers)
        for offset in range(len(run))
    ]
    windows = [
        window
        for window in windows
        if REFERENCE_MIN_SECONDS <= window[1] - window[0] <= REFERENCE_MAX_SECONDS
    ]
    if not windows:
        return None
    return max(windows, key=window_rank)


def build_extract_command(
    video: Path, *, start: float, end: float, output: Path, ffmpeg_path: str
) -> list[str]:
    return [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{max(end - start, 0.1):.3f}",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(REFERENCE_SAMPLE_RATE),
        str(output),
    ]


def extract_reference(
    video: Path, *, start: float, end: float, output: Path, ffmpeg_path: str
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 0:
        return output
    command = build_extract_command(
        video, start=start, end=end, output=output, ffmpeg_path=ffmpeg_path
    )
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not output.is_file():
        detail = (completed.stderr or completed.stdout or "").strip()
        raise CloneError(f"Could not cut the reference clip out of {video.name}: {detail}")
    return output


def reference_text_path(audio: Path) -> Path:
    return audio.with_suffix(".txt")


def supplied_reference(speaker: str, audio: Path) -> Reference:
    """A clip the caller pointed at, plus the transcript they may have left beside it."""
    resolved = audio.expanduser().resolve()
    if not resolved.is_file():
        raise CloneError(f"Reference audio not found: {resolved}")
    sidecar = reference_text_path(resolved)
    text = sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else ""
    return Reference(speaker=speaker, audio=resolved, text=text)


def build_references(
    video: Path,
    *,
    cues: list[tuple[int, SubtitleCue]],
    speakers: dict[int, str],
    wanted: list[str],
    options: CloneOptions,
    cache_dir: Path,
    ffmpeg_path: str,
) -> dict[str, Reference]:
    """One reference clip per speaker, cut from the video unless one was supplied."""
    references: dict[str, Reference] = {}
    for speaker in wanted:
        supplied = options.references.get(speaker)
        if supplied is None and len(wanted) == 1:
            supplied = options.references.get(DEFAULT_SPEAKER)
        if supplied is not None:
            reference = supplied_reference(speaker, supplied)
            print(f"  {reference.label}: {reference.audio}")
            references[speaker] = reference
            continue

        window = pick_reference_window(cues, speaker=speaker, speakers=speakers)
        if window is None:
            raise CloneError(
                f"No continuous {REFERENCE_MIN_SECONDS:.0f}-{REFERENCE_MAX_SECONDS:.0f} "
                f"second speech window from {speaker or 'the speaker'} could be found in "
                "the source subtitle. Point at a clip yourself with --clone-reference "
                f"{speaker + '=' if speaker else ''}/path/to/voice.wav"
            )

        start, end, text = window
        label = speaker or "main"
        audio = cache_dir / REFERENCE_DIR_NAME / f"{label}-{round(start * 1000)}.wav"
        extract_reference(video, start=start, end=end, output=audio, ffmpeg_path=ffmpeg_path)
        reference_text_path(audio).write_text(text + "\n", encoding="utf-8")
        print(f"  {label}: {end - start:.1f}s from {start:.1f}s -> {audio.name}")
        references[speaker] = Reference(speaker=speaker, audio=audio, text=text)

    return references
