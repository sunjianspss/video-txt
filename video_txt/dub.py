"""Speaking a translated subtitle over the video it came from.

This is the stage that turns text into sound and puts the sound back in the
file: it decides what goes into each clip, keeps a cache of the clips, has the
lines rewritten when they take longer to say than they have room for, and hands
the finished track to ffmpeg.

Two neighbours carry the parts that stand on their own: `voices` works out who
speaks in which voice, and `timeline` places the clips on the subtitle's clock.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

from .clone import (
    CLONE_ENGINES,
    CloneJob,
    CloneOptions,
    build_references,
    resolve_model,
    speed_from_rate,
    synthesize_clips,
)
from .constants import DEFAULT_LANGUAGE_CODE, DEFAULT_TARGET_LANGUAGE
from .diarize import DEFAULT_SPEAKER, SpeakerTurn, assign_speakers, speaker_order
from .fit import (
    FitItem,
    FitOptions,
    ShortenCache,
    fit_cues_to_slots,
)
from .media import (
    find_ffmpeg,
    find_ffprobe,
    format_command,
    probe_duration,
    run_ffprobe,
    temporary_output_path,
)
from .mux import container_arguments, subtitle_track_arguments
from .parallel import map_in_parallel
from .separate import ensure_instrumental, require_demucs, separated_bgm_path
from .subtitles import SubtitleCue, language_suffix, parse_srt, write_srt
from .timeline import (
    build_segments,
    compute_slots,
    group_cues_into_sentences,
    merge_cues,
    render_audio_track,
    report,
    share_out,
    speaking_budgets,
    split_into_sentences,
    spoken_duration,
)
from .translate import cleanup_debug_dir, default_debug_dir
from .voices import (
    DEFAULT_VOICE,
    VoiceChoice,
    VoicePlan,
    assign_voices,
    resolve_voice_name,
    speaker_name_problem,
)

ENGINES = ("edge-tts", "say", *CLONE_ENGINES)
SHORTEN_CACHE_NAME = "shortened.jsonl"
CLIP_PREFIX = "clip-"
CLIP_SUFFIXES = (".mp3", ".aiff", ".wav")
PARTIAL_SUFFIX = ".part"

# Scripts where one character already is a word, so a one-character line is
# speech rather than a stray mark: Chinese, Japanese kana, Korean hangul.
SYLLABIC_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
WORD_CHARACTER_PATTERN = re.compile(r"\w")

# Engines that can be asked for a faster reading of one line. The others fall
# back to stretching the waveform, which is what the re-speak exists to avoid.
# edge-tts takes a rate flag; the cloning engines take a speed multiplier per
# clip (IndexTTS spells it duration_factor, natively duration-controlled).
RESPEAK_ENGINES = ("edge-tts", *CLONE_ENGINES)
# A stretch this small is inaudible; a new clip would buy nothing.
RESPEAK_TOLERANCE = 1.05
# Rates come in these steps so a rerun lands on the same cached clips.
RESPEAK_STEP = 0.05
# Faster than this stops sounding like speech; atempo covers what is left.
MAX_RESPEAK_SPEED = 1.6

# EBU R128 for online speech: every dub comes out equally loud, however the
# engine levelled its clips.
LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
# loudnorm works internally at 192 kHz and would hand that to the encoder.
OUTPUT_SAMPLE_RATE = "48000"
# The whole original mix has to sit far under the dub or its speech fights the
# new one; a separated background has no speech left and keeps its own balance.
DEFAULT_BGM_VOLUME = 0.15
DEFAULT_SEPARATED_BGM_VOLUME = 1.0


class DubError(RuntimeError):
    pass


@dataclass
class DubOptions:
    video_input: Path
    subtitle_input: Path
    video_output: Path
    source_subtitle: Path | None = None
    audio_output: Path | None = None
    cache_dir: Path | None = None
    engine: str = "edge-tts"
    voice: str = DEFAULT_VOICE
    voice_unit: str = "sentence"
    rate: str = "+0%"
    max_atempo: float = 1.35
    keep_bgm: bool = False
    separate_bgm: bool = False
    keep_audio: bool = False
    prune_cache: bool = False
    bgm_volume: float | None = None
    soft_subtitle: bool = False
    language_code: str = DEFAULT_LANGUAGE_CODE
    concurrency: int = 4
    sample_rate: int = 48000
    overwrite: bool = False
    ffmpeg_path: str | None = None
    fit: FitOptions | None = None
    # Who speaks when, from diarization, and the voice each of them gets.
    turns: list[SpeakerTurn] = field(default_factory=list)
    speaker_voices: dict[str, str] = field(default_factory=dict)
    clone: CloneOptions | None = None


def default_audio_output(video_output: Path) -> Path:
    return video_output.with_name(f"{video_output.stem}.dub.wav")


def default_dubbed_output(video: Path, *, target_language: str = DEFAULT_TARGET_LANGUAGE) -> Path:
    suffix = video.suffix.lower() if video.suffix.lower() in {".mp4", ".mov", ".mkv"} else ".mp4"
    return video.with_name(f"{video.stem}.{language_suffix(target_language)}-dubbed{suffix}")


def default_cache_dir(video: Path) -> Path:
    return video.with_name(f"{video.stem}.dub-cache")


def fitted_subtitle_path(subtitle: Path) -> Path:
    return subtitle.with_name(f"{subtitle.stem}.fitted.srt")


def tts_launcher(engine: str) -> list[str]:
    if engine == "say":
        say = shutil.which("say")
        if not say:
            raise DubError("The 'say' engine needs macOS. Use --tts-engine edge-tts instead.")
        return [say]

    console_script = shutil.which("edge-tts")
    if console_script:
        return [console_script]
    try:
        __import__("edge_tts")
    except ImportError as exc:
        raise DubError(
            "Missing dependency: edge-tts.\n"
            "Install it with: uv add --optional dub edge-tts\n"
            "Or use the offline fallback: --tts-engine say"
        ) from exc
    return [sys.executable, "-m", "edge_tts"]


def build_tts_command(
    options: DubOptions,
    *,
    launcher: list[str],
    text: str,
    output_path: Path,
    voice: str | None = None,
    rate: str | None = None,
) -> list[str]:
    name = resolve_voice_name(options.engine, voice or options.voice)
    # A translated line may well open with a dash. Both engines would read that
    # as the start of a flag unless the text is handed over as one glued token.
    if options.engine == "say":
        return [*launcher, "-v", name, "-o", str(output_path), "--", text]
    return [
        *launcher,
        "--voice",
        name,
        f"--rate={rate or options.rate}",
        f"--text={text}",
        "--write-media",
        str(output_path),
    ]


def clip_suffix(engine: str) -> str:
    if engine == "say":
        return "aiff"
    return "wav" if engine in CLONE_ENGINES else "mp3"


def partial_clip_path(clip: Path) -> Path:
    """Where a clip lives while it is still being written."""
    return clip.with_suffix(clip.suffix + PARTIAL_SUFFIX)


def is_clip(path: Path) -> bool:
    """Whether this file is one the dub named itself, and so may delete.

    --cache-dir can be pointed anywhere. Going by the extension alone puts every
    other recording in that directory within reach of --prune-cache.
    """
    if not path.name.startswith(CLIP_PREFIX) or not path.is_file():
        return False
    return path.name.removesuffix(PARTIAL_SUFFIX).endswith(CLIP_SUFFIXES)


def has_speakable_text(text: str) -> bool:
    """Whether a TTS engine can make any sound out of this line.

    Real subtitles carry lines with no words in them -- "..." for a pause,
    musical notes around a song. edge-tts answers those with a NoAudioReceived
    error that would stop the whole run. Such a line still shows on screen as a
    subtitle; it just must not become a voice clip.

    A lone letter or digit is the other kind of line worth passing over: 'x' and
    '0' are what Whisper writes down when it hears a door or a breath, and a
    voice saying "ex" over that moment is worse than silence. Counting down
    survives -- '3, 2, 1' has three of them -- and so does a single Chinese,
    Japanese or Korean character, which is a whole word.
    """
    if SYLLABIC_PATTERN.search(text):
        return True
    return len(WORD_CHARACTER_PATTERN.findall(text)) > 1


def segment_filename(engine: str, text: str, voice: str, rate: str) -> str:
    """Name a clip after what is in it, and nothing else.

    Where a line sits in the subtitle is not part of how it sounds. Numbering the
    clips by position threw the whole cache away whenever a rewrite or a merge
    shifted the lines below it, and kept a second copy of every line that gets
    said twice.
    """
    digest = hashlib.sha1(f"{engine}|{voice}|{rate}|{text}".encode()).hexdigest()[:12]
    return f"clip-{digest}.{clip_suffix(engine)}"


def synthesize_locally(
    pending: list[tuple[str, VoiceChoice, Path, str]], options: DubOptions, *, cache_dir: Path
) -> None:
    """Hand the batch to a cloning model, which is loaded once for all of it.

    All of it includes the re-spoken clips: every job carries its own speed, so
    mixed rates cost one model load, not one per rate."""
    if options.clone is None:
        raise DubError(f"--tts-engine {options.engine} needs voice cloning options.")
    unreferenced = [voice.name for _, voice, _, _ in pending if voice.reference is None]
    if unreferenced:
        raise DubError(f"No reference clip to clone {unreferenced[0]} from.")
    jobs = [
        CloneJob(text=text, output=path, reference=voice.reference, speed=speed_from_rate(rate))
        for text, voice, path, rate in pending
        if voice.reference is not None
    ]
    synthesize_clips(jobs, options=options.clone, cache_dir=cache_dir)


def synthesize_online(
    pending: list[tuple[str, VoiceChoice, Path, str]], options: DubOptions
) -> None:
    launcher = tts_launcher(options.engine)
    done = 0
    progress_lock = threading.Lock()

    def synthesize(job: tuple[str, VoiceChoice, Path, str]) -> None:
        nonlocal done
        text, voice, path, rate = job
        temp_path = partial_clip_path(path)
        command = build_tts_command(
            options,
            launcher=launcher,
            text=text,
            output_path=temp_path,
            voice=voice.name,
            rate=rate,
        )
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not temp_path.is_file():
            temp_path.unlink(missing_ok=True)
            detail = (completed.stderr or completed.stdout or "").strip()
            raise DubError(f"Text-to-speech failed for {text[:40]!r}: {detail}")
        temp_path.replace(path)
        with progress_lock:
            done += 1
            current = done
        if current % 10 == 0 or current == len(pending):
            print(f"  {current}/{len(pending)} clips done")

    map_in_parallel(pending, synthesize, workers=options.concurrency)


def synthesize_segments(
    lines: list[tuple[int, str]],
    options: DubOptions,
    cache_dir: Path,
    *,
    plan: VoicePlan | None = None,
    rate: str | None = None,
    rates: list[str] | None = None,
) -> list[Path]:
    speaking = plan or VoicePlan(
        default=VoiceChoice(name=resolve_voice_name(options.engine, options.voice))
    )
    line_rates = rates or [rate or options.rate] * len(lines)
    cache_dir.mkdir(parents=True, exist_ok=True)
    voices = [speaking.for_position(position) for position, _ in lines]
    paths = [
        cache_dir / segment_filename(options.engine, text, voice.identity, line_rate)
        for (_, text), voice, line_rate in zip(lines, voices, line_rates, strict=True)
    ]

    # Keyed by path so a line said twice is synthesized once: two workers racing
    # to write the one clip they now share would clobber each other.
    wanted: dict[Path, tuple[str, VoiceChoice, Path, str]] = {}
    for (_, text), voice, path, line_rate in zip(lines, voices, paths, line_rates, strict=True):
        if path.is_file() and path.stat().st_size > 0:
            continue
        wanted.setdefault(path, (text, voice, path, line_rate))

    pending = list(wanted.values())
    cached = sum(1 for path in paths if path not in wanted)
    if cached:
        print(f"Reusing {cached} cached voice clip{'s' if cached > 1 else ''}.")
    if not pending:
        return paths

    listed = ", ".join(sorted({voice.name for _, voice, _, _ in pending}))
    print(f"Synthesizing {len(pending)} voice clips with {options.engine} ({listed})...")
    if options.engine in CLONE_ENGINES:
        synthesize_locally(pending, options, cache_dir=cache_dir)
    else:
        synthesize_online(pending, options)
    return paths


def probe_durations(
    paths: list[Path], *, ffmpeg_path: str, sample_rate: int, workers: int
) -> list[float]:
    return map_in_parallel(
        paths,
        lambda path: spoken_duration(path, ffmpeg_path=ffmpeg_path, sample_rate=sample_rate),
        workers=workers,
    )


def respoken_rate(natural: float, slot: float, *, base_rate: str) -> str | None:
    """The rate to say an overrunning clip again at, or None to leave it alone.

    The clip was spoken at base_rate and still needs natural/slot more speed, so
    the two multiply. The result is rounded up to the next step — up, because a
    clip that lands a hair short of its slot fits, and one a hair over is back
    to being stretched.
    """
    if slot <= 0 or natural <= slot * RESPEAK_TOLERANCE:
        return None
    needed = speed_from_rate(base_rate) * (natural / slot)
    total = min(math.ceil(needed / RESPEAK_STEP - 1e-9) * RESPEAK_STEP, MAX_RESPEAK_SPEED)
    if total <= speed_from_rate(base_rate) + 1e-9:
        return None
    return f"{round((total - 1.0) * 100):+d}%"


def respeak_overruns(
    cues: list[tuple[int, SubtitleCue]],
    paths: list[Path],
    options: DubOptions,
    cache_dir: Path,
    *,
    plan: VoicePlan | None,
    ffmpeg_path: str,
) -> tuple[list[Path], int]:
    """Have the engine say the clips that overrun their slot again, faster.

    A clip that does not fit gets sped up one way or the other. Asked to speak
    faster, the engine delivers naturally quick speech; stretching the recorded
    waveform afterwards is where the mechanical sound comes from. So atempo is
    kept for the leftovers the rate cap puts out of reach.
    """
    slots = compute_slots(cues)
    durations = probe_durations(
        paths, ffmpeg_path=ffmpeg_path, sample_rate=options.sample_rate, workers=options.concurrency
    )
    needed: dict[int, str] = {}
    for index, (natural, slot) in enumerate(zip(durations, slots, strict=True)):
        rate = respoken_rate(natural, slot, base_rate=options.rate)
        if rate is not None:
            needed[index] = rate
    if not needed:
        return paths, 0

    print(f"Speaking {len(needed)} overrunning clip(s) again at a faster rate...")
    lines = [(cues[index][0], cues[index][1].text) for index in needed]
    faster = synthesize_segments(lines, options, cache_dir, plan=plan, rates=list(needed.values()))
    respoken = list(paths)
    for index, path in zip(needed, faster, strict=True):
        respoken[index] = path
    return respoken, len(needed)


def reference_lines(options: DubOptions) -> list[tuple[int, SubtitleCue]]:
    """Source-language cues: what the original voices actually say, and when."""
    if options.source_subtitle is None or not options.source_subtitle.is_file():
        return []
    return [
        (position, cue)
        for position, cue in enumerate(parse_srt(options.source_subtitle), start=1)
        if not cue.is_empty
    ]


def build_voice_plan(
    options: DubOptions,
    *,
    speakers: dict[int, str],
    cache_dir: Path,
    ffmpeg_path: str,
) -> VoicePlan:
    order = speaker_order(options.turns) or [DEFAULT_SPEAKER]

    if options.engine in CLONE_ENGINES:
        sources = reference_lines(options)
        print(f"Cloning {len(order)} voice{'s' if len(order) > 1 else ''} from the original audio:")
        references = build_references(
            options.video_input,
            cues=sources,
            speakers=assign_speakers(sources, options.turns),
            wanted=order,
            options=options.clone or CloneOptions(engine=options.engine),
            cache_dir=cache_dir,
            ffmpeg_path=ffmpeg_path,
        )
        model = resolve_model(options.clone or CloneOptions(engine=options.engine))
        choices = {
            speaker: VoiceChoice(
                name=f"{references[speaker].label}@{model}", reference=references[speaker]
            )
            for speaker in order
        }
    else:
        names = assign_voices(
            order, engine=options.engine, voice=options.voice, named=options.speaker_voices
        )
        choices = {speaker: VoiceChoice(name=names[speaker]) for speaker in order}
        if len(order) > 1:
            for speaker in order:
                print(f"  {speaker}: {names[speaker]}")

    return VoicePlan(
        default=choices[order[0]],
        by_position={
            position: choices[speaker]
            for position, speaker in speakers.items()
            if speaker in choices
        },
    )


def source_texts_by_position(
    source_subtitle: Path | None, translated: list[SubtitleCue]
) -> dict[int, str]:
    """Original lines keyed by cue position, so a rewrite can go back to the source."""
    if source_subtitle is None or not source_subtitle.is_file():
        return {}
    source = parse_srt(source_subtitle)
    if len(source) != len(translated):
        return {}
    return {position: cue.text for position, cue in enumerate(source, start=1) if not cue.is_empty}


def measure_units(
    cues: list[tuple[int, SubtitleCue]],
    texts: dict[int, str],
    options: DubOptions,
    *,
    cache_dir: Path,
    ffmpeg_path: str,
    plan: VoicePlan | None = None,
    speakers: dict[int, str] | None = None,
) -> dict[int, float]:
    """Time the clips this run will really speak, then share each one out.

    Timing line by line while the dub speaks sentences charges every line a
    lead-in and a sentence-final fall that the merged clip never pays, so lines
    with room to spare come back over budget and get rewritten for nothing. It
    also synthesizes a second full set of clips that nobody ever hears.
    """
    current = [(position, cue.with_text(texts[position])) for position, cue in cues]
    groups = (
        split_into_sentences(current, speakers=speakers)
        if options.voice_unit == "sentence"
        else [[entry] for entry in current]
    )
    units = [merge_cues(group) for group in groups]
    lines = [(position, cue.text) for position, cue in units]
    seconds = probe_durations(
        synthesize_segments(lines, options, cache_dir, plan=plan),
        ffmpeg_path=ffmpeg_path,
        sample_rate=options.sample_rate,
        workers=options.concurrency,
    )

    measured: dict[int, float] = {}
    for group, total in zip(groups, seconds, strict=True):
        measured.update(share_out(group, total))
    return measured


def fit_subtitle_to_timeline(
    cues: list[tuple[int, SubtitleCue]],
    options: DubOptions,
    *,
    fit: FitOptions,
    cache_dir: Path,
    ffmpeg_path: str,
    plan: VoicePlan | None = None,
    speakers: dict[int, str] | None = None,
) -> tuple[list[tuple[int, SubtitleCue]], Path]:
    def measure(texts: dict[int, str]) -> dict[int, float]:
        return measure_units(
            cues,
            texts,
            options,
            cache_dir=cache_dir,
            ffmpeg_path=ffmpeg_path,
            plan=plan,
            speakers=speakers,
        )

    all_cues = parse_srt(options.subtitle_input)
    sources = source_texts_by_position(options.source_subtitle, all_cues)
    budgets = speaking_budgets(cues, voice_unit=options.voice_unit, speakers=speakers)
    items = [
        FitItem(position=position, text=cue.text, slot=slot, source_text=sources.get(position))
        for (position, cue), slot in zip(cues, budgets, strict=True)
    ]
    namespace = json.dumps(
        {
            "model": fit.translation.model,
            "target_language": fit.translation.target_language,
            "preserve_terms": fit.translation.preserve_terms,
            "note": fit.translation.note or "",
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    print("Fitting the Chinese lines to the timeline...")
    fit_debug_dir = default_debug_dir(options.subtitle_input)
    outcome = fit_cues_to_slots(
        items,
        measure=measure,
        options=fit,
        cache=ShortenCache(cache_dir / SHORTEN_CACHE_NAME, namespace=namespace),
        debug_dir=fit_debug_dir,
    )
    removed_snapshots = cleanup_debug_dir(fit_debug_dir)
    if removed_snapshots:
        print(
            f"  Removed {removed_snapshots} debug snapshot(s); "
            "every rewrite succeeded after retries."
        )

    if not outcome.shortened:
        print(
            f"  Nothing worth rewriting; the tightest line needs "
            f"{outcome.tempo_before:.2f}x speed-up."
        )
        return cues, options.subtitle_input

    print(
        f"  Rewrote {len(outcome.shortened)} line(s); the tightest line now needs "
        f"{outcome.tempo_after:.2f}x speed-up, down from {outcome.tempo_before:.2f}x."
    )
    fitted_cues = [
        cue.with_text(outcome.shortened[position]) if position in outcome.shortened else cue
        for position, cue in enumerate(all_cues, start=1)
    ]
    fitted_path = fitted_subtitle_path(options.subtitle_input)
    write_srt(fitted_path, fitted_cues)
    print(f"  Wrote the fitted subtitles to: {fitted_path}")

    updated = [
        (position, cue) for position, cue in enumerate(fitted_cues, start=1) if not cue.is_empty
    ]
    return updated, fitted_path


def unused_clips(cache_dir: Path, keep: list[Path]) -> list[Path]:
    """Clips from earlier settings, plus whatever an interrupted run left half-written."""
    kept = {path.resolve() for path in keep}
    return [clip for clip in cache_dir.iterdir() if is_clip(clip) and clip.resolve() not in kept]


def report_cache(cache_dir: Path, used: list[Path], *, prune: bool) -> None:
    """Clear, or at least own up to, the clips left behind by earlier settings.

    Every change of voice or speaking rate resynthesizes the whole video, and the
    old clips stay behind in case that setting comes back. A few rounds of tuning
    leave far more clips in the cache than any one run needs.
    """
    if not cache_dir.is_dir():
        return
    extra = unused_clips(cache_dir, used)
    if not extra:
        return

    megabytes = sum(clip.stat().st_size for clip in extra) / 1_048_576
    if prune:
        for clip in extra:
            clip.unlink(missing_ok=True)
        print(f"Pruned {len(extra)} unused voice clips ({megabytes:.0f} MB) from the cache.")
    elif len(extra) > len(used):
        print(
            f"The clip cache holds {len(extra)} clips this run did not use "
            f"({megabytes:.0f} MB). Pass --prune-cache to clear them."
        )


def has_audio_stream(video: Path, *, ffprobe_path: str) -> bool:
    output = run_ffprobe(
        ffprobe_path,
        ["-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(video)],
    )
    return bool(output.strip())


def resolved_bgm_volume(options: DubOptions, *, separated: bool) -> float:
    """How loud the background plays, when the flag was left to its default.

    The two backgrounds need very different defaults: the whole original mix
    still has its speech in it and must sit far under the dub, while a
    separated background is the original balance minus the voice and can keep
    the volume its producer already chose.
    """
    if options.bgm_volume is not None:
        return min(max(options.bgm_volume, 0.0), 1.0)
    return DEFAULT_SEPARATED_BGM_VOLUME if separated else DEFAULT_BGM_VOLUME


def build_dub_mux_command(
    options: DubOptions,
    *,
    ffmpeg_path: str,
    audio_path: Path,
    keep_bgm: bool,
    subtitle_path: Path | None = None,
    bgm_path: Path | None = None,
) -> list[str]:
    command = [
        ffmpeg_path,
        "-y" if options.overwrite else "-n",
        "-i",
        str(options.video_input),
        "-i",
        str(audio_path),
    ]
    if bgm_path is not None:
        command.extend(["-i", str(bgm_path)])
    if options.soft_subtitle:
        command.extend(["-i", str(subtitle_path or options.subtitle_input)])

    if keep_bgm:
        background = "2:a" if bgm_path is not None else "0:a"
        volume = resolved_bgm_volume(options, separated=bgm_path is not None)
        command.extend(
            [
                "-filter_complex",
                f"[{background}]volume={volume:.3f}[bg];[1:a]volume=1.0[vo];"
                "[bg][vo]amix=inputs=2:duration=longest:normalize=0[mix];"
                f"[mix]{LOUDNORM_FILTER}[aout]",
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
            ]
        )
    else:
        command.extend(["-map", "0:v:0", "-map", "1:a:0", "-filter:a", LOUDNORM_FILTER])

    if options.soft_subtitle:
        subtitle_input = 3 if bgm_path is not None else 2
        command.extend(
            subtitle_track_arguments(
                options.video_output,
                stream=f"{subtitle_input}:0",
                language_code=options.language_code,
            )
        )

    command.extend(
        ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", OUTPUT_SAMPLE_RATE, "-shortest"]
    )
    command.extend(container_arguments(options.video_output))
    command.append(str(options.video_output))
    return command


def check_speaker_names(options: DubOptions) -> None:
    problem = speaker_name_problem(
        speaker_order(options.turns),
        voices=options.speaker_voices,
        references=options.clone.references if options.clone else {},
    )
    if problem:
        raise DubError(problem)


def validate_audio_output(options: DubOptions, audio_output: Path) -> None:
    """Keep the intermediate voice track away from every input and final output."""
    candidate = audio_output.expanduser().resolve()
    protected = (
        ("the source video", options.video_input),
        ("the translated subtitle", options.subtitle_input),
        ("the source subtitle", options.source_subtitle),
        ("the output video", options.video_output),
    )
    for label, path in protected:
        if path is not None and candidate == path.expanduser().resolve():
            raise DubError(f"--audio-output conflicts with {label}: {candidate}")
    if options.audio_output is not None and candidate.exists():
        raise DubError(
            f"Audio output already exists: {candidate}. Choose another --audio-output path."
        )


def run_dub(options: DubOptions, *, dry_run: bool = False) -> Path:
    audio_output = options.audio_output or default_audio_output(options.video_output)
    validate_audio_output(options, audio_output)
    ffmpeg_path = find_ffmpeg(explicit=options.ffmpeg_path)
    cache_dir = options.cache_dir or default_cache_dir(options.video_input)
    if options.turns:
        check_speaker_names(options)

    if dry_run:
        print(f"Voice clip cache: {cache_dir}")
        if options.voice_unit == "sentence":
            print("Voice clips: one per sentence, merged from the subtitle lines")
        else:
            print("Voice clips: one per subtitle line")
        if options.engine in RESPEAK_ENGINES:
            print("Overrunning clips: spoken again at a faster rate before any stretching")
        if options.separate_bgm:
            print(f"Background: original audio minus its voices, cached at {cache_dir / 'bgm'}")
        cloning = options.engine in CLONE_ENGINES
        if cloning:
            clone = options.clone or CloneOptions(engine=options.engine)
            print(f"Voice cloning: {options.engine} ({resolve_model(clone)})")
        if options.turns:
            found = speaker_order(options.turns)
            if cloning:
                print(f"Speakers: {len(found)}, each cloned from its own reference clip")
            else:
                voices = assign_voices(
                    found,
                    engine=options.engine,
                    voice=options.voice,
                    named=options.speaker_voices,
                )
                listed = ", ".join(f"{speaker} -> {voices[speaker]}" for speaker in found)
                print(f"Speakers: {len(found)} ({listed})")
        if options.fit is not None:
            print(
                f"Duration-aware rewrite: on, up to {options.fit.rounds} round(s) "
                f"for lines that need more than {options.fit.tempo:.2f}x speed-up"
            )
        kept = options.audio_output is not None or options.keep_audio
        print(f"Dub audio track: {audio_output}{'' if kept else ' (removed after muxing)'}")
        print("Mux command:")
        print(
            format_command(
                build_dub_mux_command(
                    options,
                    ffmpeg_path=ffmpeg_path,
                    audio_path=audio_output,
                    keep_bgm=options.keep_bgm or options.separate_bgm,
                    bgm_path=separated_bgm_path(cache_dir) if options.separate_bgm else None,
                )
            )
        )
        return options.video_output

    if options.video_output.exists() and not options.overwrite:
        raise DubError(
            f"Output video already exists: {options.video_output}. "
            "Pass --overwrite-video to replace it."
        )
    # Before any of the work, so a folder that has to be made is not discovered
    # by ffmpeg at the very end of a run that has already synthesized everything.
    options.video_output.parent.mkdir(parents=True, exist_ok=True)
    if options.separate_bgm:
        # The separation itself can wait, but a missing dependency must not
        # surface after minutes of synthesis have already been paid for.
        require_demucs()

    cues = [
        (position, cue)
        for position, cue in enumerate(parse_srt(options.subtitle_input), start=1)
        if not cue.is_empty
    ]
    unvoiced = sum(1 for _, cue in cues if not has_speakable_text(cue.text))
    if unvoiced:
        print(
            f"Leaving {unvoiced} line(s) with nothing to say unvoiced, "
            "e.g. '...' held pauses and stray single characters."
        )
        cues = [(position, cue) for position, cue in cues if has_speakable_text(cue.text)]
    if not cues:
        raise DubError(f"No spoken lines found in {options.subtitle_input}")

    speakers = assign_speakers(cues, options.turns)
    plan = build_voice_plan(
        options, speakers=speakers, cache_dir=cache_dir, ffmpeg_path=ffmpeg_path
    )

    subtitle_for_mux = options.subtitle_input
    if options.fit is not None:
        cues, subtitle_for_mux = fit_subtitle_to_timeline(
            cues,
            options,
            fit=options.fit,
            cache_dir=cache_dir,
            ffmpeg_path=ffmpeg_path,
            plan=plan,
            speakers=speakers,
        )

    if options.voice_unit == "sentence":
        units = group_cues_into_sentences(cues, speakers=speakers)
        if len(units) < len(cues):
            print(f"Speaking {len(units)} sentences merged from {len(cues)} subtitle lines.")
            cues = units

    base_paths = synthesize_segments(
        [(position, cue.text) for position, cue in cues], options, cache_dir, plan=plan
    )
    paths, respoken = base_paths, 0
    if options.engine in RESPEAK_ENGINES:
        paths, respoken = respeak_overruns(
            cues, base_paths, options, cache_dir, plan=plan, ffmpeg_path=ffmpeg_path
        )
    segments = build_segments(cues, paths)
    total_duration = probe_duration(options.video_input, ffmpeg_path=ffmpeg_path)

    print("Aligning voice clips to the subtitle timeline...")
    placed = render_audio_track(
        segments,
        ffmpeg_path=ffmpeg_path,
        sample_rate=options.sample_rate,
        max_atempo=options.max_atempo,
        total_duration=total_duration,
        output_path=audio_output,
    )
    report(placed, segments, respoken=respoken)

    keep_bgm = options.keep_bgm or options.separate_bgm
    if keep_bgm and not has_audio_stream(
        options.video_input, ffprobe_path=find_ffprobe(ffmpeg_path)
    ):
        flag = "--separate-bgm" if options.separate_bgm else "--keep-bgm"
        print(f"Source video has no audio track, so {flag} has nothing to mix.", file=sys.stderr)
        keep_bgm = False

    bgm_path = None
    if options.separate_bgm and keep_bgm:
        bgm_path = ensure_instrumental(
            options.video_input, cache_dir=cache_dir, ffmpeg_path=ffmpeg_path
        )

    print("Muxing the dubbed voice track into the video...")
    temporary_video = temporary_output_path(options.video_output)
    try:
        command = build_dub_mux_command(
            replace(options, video_output=temporary_video, overwrite=True),
            ffmpeg_path=ffmpeg_path,
            audio_path=audio_output,
            keep_bgm=keep_bgm,
            subtitle_path=subtitle_for_mux,
            bgm_path=bgm_path,
        )
        completed = subprocess.run(command)
        if completed.returncode != 0:
            raise DubError(f"ffmpeg failed with exit code {completed.returncode}.")
        if not temporary_video.is_file() or temporary_video.stat().st_size == 0:
            raise DubError("ffmpeg reported success but did not write the dubbed video.")
        temporary_video.replace(options.video_output)
    finally:
        temporary_video.unlink(missing_ok=True)

    # The track is inside the video now. Only clean up the file we chose ourselves;
    # a path the caller named is theirs to keep.
    if options.audio_output is None and not options.keep_audio:
        size = audio_output.stat().st_size if audio_output.is_file() else 0
        audio_output.unlink(missing_ok=True)
        if size:
            print(f"Removed the {size / 1_048_576:.0f} MB intermediate voice track.")

    # The base-rate clips a re-speak replaced still time the next run's
    # measurements, so they count as used and survive --prune-cache.
    report_cache(cache_dir, [*base_paths, *paths], prune=options.prune_cache)
    return options.video_output
