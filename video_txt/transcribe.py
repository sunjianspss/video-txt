from __future__ import annotations

import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from .chunking import (
    MIN_CHUNKED_DURATION,
    SILENCE_FILTER,
    clamp_cues,
    clamp_payload,
    collapse_repeats,
    merge_whisper_payloads,
    parse_silences,
    plan_chunks,
    renumber,
    repeated_runs,
    sentences,
    shift_cues,
)
from .media import (
    AudioStreamSelection,
    MediaError,
    find_ffmpeg,
    format_command,
    probe_audio_streams,
    probe_duration,
    select_audio_stream,
)
from .refine import (
    RefineError,
    TranscriptDocument,
    document_from_whisper_result,
    refine_transcript,
    write_word_document,
)
from .subtitles import SubtitleCue, parse_srt, write_srt

BACKENDS = ("auto", "openai-whisper", "mlx-whisper")
OUTPUT_FORMATS = ("txt", "vtt", "srt", "tsv", "json", "all")
REFINEMENT_RESERVED_FLAGS = {
    "--output-dir",
    "--output-format",
    "--output-name",
    "--word-timestamps",
}

# Only confirmed, unambiguous phrase spellings belong here. In particular,
# Cloud Code / Quad Code and the standalone name Claud are not safe aliases.
_CLAUD_CODE = re.compile(r"(?<!\w)Claud(\s+)Code(?!\w)", re.IGNORECASE)
_INLINE_TAG = re.compile(r"</?(?:u|b|i|font)(?:\s+[^<>]*?)?\s*>", re.IGNORECASE)
_SPELLING_JOIN_GAP = 0.8


def _correct_spelling(text: str) -> str:
    return _CLAUD_CODE.sub(lambda match: f"Claude{match[1]}Code", text)


def _correct_subtitle_spelling(path: Path) -> None:
    plain = _INLINE_TAG.sub("", path.read_text(encoding="utf-8"))
    if not re.search(r"(?<!\w)Claud(?!\w)", plain, re.IGNORECASE) or not re.search(
        r"(?<!\w)Code(?!\w)", plain, re.IGNORECASE
    ):
        return
    cues = parse_srt(path)

    # Match visible text, retaining a map back to each original character. This
    # lets a phrase span adjacent cues or inline tags without moving either.
    visible: list[str] = []
    locations: list[tuple[int, int] | None] = []
    for index, cue in enumerate(cues):
        if index:
            gap = round(cue.start_seconds - cues[index - 1].end_seconds, 3)
            visible.append("\n" if 0 <= gap <= _SPELLING_JOIN_GAP else "\0")
            locations.append(None)
        text = cue.text
        cursor = 0
        for tag in _INLINE_TAG.finditer(text):
            visible.extend(text[cursor:tag.start()])
            locations.extend((index, offset) for offset in range(cursor, tag.start()))
            cursor = tag.end()
        visible.extend(text[cursor:])
        locations.extend((index, offset) for offset in range(cursor, len(text)))

    edits: dict[int, dict[int, str]] = {}
    for match in _CLAUD_CODE.finditer("".join(visible)):
        # Replace individual letters rather than a raw substring, so even tags
        # inside a word survive. The extra e inherits the d's formatting.
        replacements = [
            (match.start(), ("C", "l", "a", "u", "de")),
            (match.end() - 4, tuple("Code")),
        ]
        for start, letters in replacements:
            for position, letter in enumerate(letters, start=start):
                location = locations[position]
                assert location is not None  # Matches only letters, never a cue separator.
                cue_index, offset = location
                edits.setdefault(cue_index, {})[offset] = letter
    corrected = list(cues)
    for index, replacements in edits.items():
        corrected[index] = cues[index].with_text(
            "".join(replacements.get(offset, char) for offset, char in enumerate(cues[index].text))
        )
    count = sum(old.text != new.text for old, new in zip(cues, corrected, strict=True))
    if count:
        write_srt(path, corrected)
        print(f"Corrected Claud Code spelling in {count} subtitle cue(s).")


def _correct_word_spelling(document: TranscriptDocument) -> TranscriptDocument:
    words = [replace(word, text=_correct_spelling(word.text)) for word in document.words]
    for index in range(len(words) - 1):
        left, right = words[index : index + 2]
        # Do not join unrelated words separated by a long pause.
        if right.start - left.end > _SPELLING_JOIN_GAP:
            continue
        combined = f"{left.text}\n{right.text}"
        corrected = _correct_spelling(combined)
        if corrected != combined:
            # Each word was already corrected individually, so this match must
            # span the join: Claud gains one letter, Code keeps its length.
            boundary = len(left.text) + 1
            words[index] = replace(left, text=corrected[:boundary])
            words[index + 1] = replace(right, text=corrected[boundary + 1:])
    count = sum(old.text != new.text for old, new in zip(document.words, words, strict=True))
    if count:
        print(f"Corrected Claud Code spelling in {count} timestamped word(s).")
    return replace(document, words=words)


# Whisper spells a rare proper noun as whatever common word it sounds like
# (Claude -> Quad / Cloud), and once it has, it keeps doing so for the rest of
# the file. Priming it with the right spelling fixes the root cause; replacing
# the wrong words afterwards cannot, because "quad" and "cloud" are real words.
DEFAULT_INITIAL_PROMPT = "A talk about Claude Code, Anthropic's coding agent, and the Claude models."
_PROMPT_ECHO_NGRAM = 4
_PROMPT_ECHO_MIN_TOKENS = 4
_ENGLISH = {"en", "english"}
_PROMPT_TOKEN = re.compile(r"[\w']+")


def effective_initial_prompt(options: TranscribeOptions) -> str | None:
    """The prompt to send to Whisper.

    An explicit --initial-prompt always wins, and an empty one turns the default
    off. The default only applies where it is safe: English or auto-detected
    speech (a Chinese prompt-free run was unchanged in testing), and output
    formats whose prompt echo ``_drop_prompt_echo`` can remove again.
    """
    if options.initial_prompt is not None:
        return options.initial_prompt or None
    if options.language and options.language.lower() not in _ENGLISH:
        return None
    if not options.refine_subtitles and options.output_format not in {"srt", "txt"}:
        return None
    return DEFAULT_INITIAL_PROMPT


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _PROMPT_TOKEN.findall(text)]


def _echo_model(prompt: str) -> tuple[set[str], set[tuple[str, ...]]]:
    tokens = _tokens(prompt)
    grams = {
        tuple(tokens[i : i + _PROMPT_ECHO_NGRAM])
        for i in range(len(tokens) - _PROMPT_ECHO_NGRAM + 1)
    }
    return set(tokens), grams


def _is_prompt_echo(tokens: list[str], vocabulary: set[str], grams: set[tuple[str, ...]]) -> bool:
    """True for text made only of prompt words that repeats a stretch of the prompt.

    On silence or music Whisper tends to write the prompt itself, sometimes in a
    loop. Real speech does not consist solely of prompt words around a verbatim
    four-word run of it.
    """
    if len(tokens) < _PROMPT_ECHO_MIN_TOKENS or not all(t in vocabulary for t in tokens):
        return False
    return any(
        tuple(tokens[i : i + _PROMPT_ECHO_NGRAM]) in grams
        for i in range(len(tokens) - _PROMPT_ECHO_NGRAM + 1)
    )


def _drop_prompt_echo_lines(lines: list[str], prompt: str) -> list[str]:
    vocabulary, grams = _echo_model(prompt)
    return [line for line in lines if not _is_prompt_echo(_tokens(line), vocabulary, grams)]


def _read_cues(path: Path) -> list[SubtitleCue]:
    """Cues of a Whisper SRT; a piece with nothing audible leaves an empty file."""
    if not path.read_text(encoding="utf-8-sig").strip():
        return []
    return parse_srt(path)


def _drop_prompt_echo_subtitles(path: Path, prompt: str) -> None:
    cues = _read_cues(path)
    vocabulary, grams = _echo_model(prompt)
    kept = [c for c in cues if not _is_prompt_echo(_tokens(c.text), vocabulary, grams)]
    if len(kept) == len(cues):
        return
    write_srt(
        path,
        [SubtitleCue(index=str(n), timing=c.timing, text_lines=c.text_lines) for n, c in enumerate(kept, 1)],
    )
    print(f"Dropped {len(cues) - len(kept)} subtitle cue(s) that only repeated the prompt.")


def _drop_prompt_echo_text(path: Path, prompt: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = _drop_prompt_echo_lines(lines, prompt)
    if len(kept) != len(lines):
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"Dropped {len(lines) - len(kept)} line(s) that only repeated the prompt.")


def _drop_prompt_echo_words(document: TranscriptDocument, prompt: str) -> TranscriptDocument:
    vocabulary, grams = _echo_model(prompt)
    words = document.words
    drop: set[int] = set()
    start = 0
    while start < len(words):
        # A run is a maximal stretch of words that all belong to the prompt.
        if not all(t in vocabulary for t in _tokens(words[start].text)) or not _tokens(words[start].text):
            start += 1
            continue
        end = start
        run_tokens: list[str] = []
        while end < len(words):
            tokens = _tokens(words[end].text)
            if not tokens or not all(t in vocabulary for t in tokens):
                break
            run_tokens.extend(tokens)
            end += 1
        if _is_prompt_echo(run_tokens, vocabulary, grams):
            drop.update(range(start, end))
        start = end
    if not drop:
        return document
    print(f"Dropped {len(drop)} word(s) that only repeated the prompt.")
    return replace(document, words=[w for i, w in enumerate(words) if i not in drop])


MLX_MODEL_ALIASES = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large": "mlx-community/whisper-large-v3",
    "large-v2": "mlx-community/whisper-large-v2",
    "large-v3": "mlx-community/whisper-large-v3",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


class TranscribeError(RuntimeError):
    pass


@dataclass
class TranscribeOptions:
    input_path: Path
    output_dir: Path
    mode: str = "transcribe"
    model: str | None = None
    output_format: str = "txt"
    language: str | None = None
    backend: str = "auto"
    device: str | None = None
    initial_prompt: str | None = None
    extra_args: list[str] = field(default_factory=list)
    refine_subtitles: bool = False
    audio_stream: int | None = None


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def resolve_backend(requested: str) -> str:
    if requested not in BACKENDS:
        raise TranscribeError(f"Unknown transcription backend: {requested}")

    if requested == "mlx-whisper":
        if not module_available("mlx_whisper"):
            raise TranscribeError(
                "Missing dependency: mlx-whisper.\nInstall it with: uv add mlx-whisper"
            )
        return "mlx-whisper"

    if requested == "openai-whisper":
        if not module_available("whisper"):
            raise TranscribeError(
                "Missing dependency: openai-whisper.\nInstall it with: uv add openai-whisper"
            )
        return "openai-whisper"

    if platform.machine() == "arm64" and module_available("mlx_whisper"):
        return "mlx-whisper"
    if module_available("whisper"):
        return "openai-whisper"
    raise TranscribeError(
        "No transcription backend available.\n"
        "Install one with: uv add openai-whisper\n"
        "On Apple Silicon, 'uv add mlx-whisper' is several times faster."
    )


def resolve_model(*, mode: str, model: str | None, backend: str) -> str:
    resolved = model or ("medium" if mode == "translate" else "turbo")
    if mode == "translate" and resolved == "turbo":
        raise TranscribeError(
            "The turbo model does not support translation.\n"
            "Use --model medium, --model large, or omit --model in translate mode."
        )
    if backend == "mlx-whisper" and "/" not in resolved:
        return MLX_MODEL_ALIASES.get(resolved, f"mlx-community/whisper-{resolved}")
    return resolved


def mlx_launcher() -> list[str]:
    console_script = shutil.which("mlx_whisper")
    if console_script:
        return [console_script]
    return [sys.executable, "-m", "mlx_whisper.cli"]


def build_command(options: TranscribeOptions, *, backend: str, model: str) -> list[str]:
    prompt = effective_initial_prompt(options)
    if backend == "mlx-whisper":
        command = [
            *mlx_launcher(),
            str(options.input_path),
            "--model",
            model,
            "--task",
            options.mode,
            "--output-format",
            options.output_format,
            "--output-dir",
            str(options.output_dir),
            "--output-name",
            options.input_path.stem,
        ]
        if options.language:
            command.extend(["--language", options.language])
        if prompt:
            command.extend(["--initial-prompt", prompt])
        if options.refine_subtitles:
            command.extend(["--word-timestamps", "True"])
        if options.device:
            print(
                f"Note: --device {options.device} is ignored by the mlx-whisper backend "
                "(it always runs on the Apple Metal GPU).",
                file=sys.stderr,
            )
        return [*command, *options.extra_args]

    command = [
        sys.executable,
        "-m",
        "whisper",
        str(options.input_path),
        "--model",
        model,
        "--task",
        options.mode,
        "--output_format",
        options.output_format,
        "--output_dir",
        str(options.output_dir),
    ]
    if options.language:
        command.extend(["--language", options.language])
    if options.device:
        command.extend(["--device", options.device])
    if prompt:
        command.extend(["--initial_prompt", prompt])
    if options.refine_subtitles:
        command.extend(["--word_timestamps", "True"])
    return [*command, *options.extra_args]


def expected_output_path(options: TranscribeOptions) -> Path:
    suffix = "txt" if options.output_format == "all" else options.output_format
    return options.output_dir / f"{options.input_path.stem}.{suffix}"


def audio_extract_command(
    input_path: Path,
    output_path: Path,
    *,
    stream_index: int,
    ffmpeg_path: str,
) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def describe_audio_selection(selection: AudioStreamSelection) -> str:
    stream = selection.stream
    metadata = ", ".join(
        value for value in (stream.title, stream.language, stream.layout) if value
    )
    details = f" ({metadata})" if metadata else ""
    return f"Audio stream: {stream.index}{details} — {selection.reason}"


def run_transcribe(options: TranscribeOptions, *, dry_run: bool = False) -> Path:
    if not options.input_path.is_file():
        raise TranscribeError(f"Input file not found: {options.input_path}")
    if options.refine_subtitles and options.output_format != "srt":
        raise TranscribeError("--refine-subtitles requires --format srt.")
    conflict = refinement_argument_conflict(options.extra_args)
    if options.refine_subtitles and conflict:
        raise TranscribeError(
            f"--whisper-arg {conflict} conflicts with --refine-subtitles; "
            "the refinement mode controls its JSON output and word timestamps."
        )

    ffmpeg_path = str(find_ffmpeg())
    backend = resolve_backend(options.backend)
    model = resolve_model(mode=options.mode, model=options.model, backend=backend)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = expected_output_path(options)

    try:
        streams = probe_audio_streams(options.input_path, ffmpeg_path=ffmpeg_path)
    except MediaError as exc:
        if not dry_run or options.audio_stream is not None:
            raise TranscribeError(str(exc)) from exc
        streams = []
        print(f"Audio stream: not inspected during dry run ({exc})")
    try:
        selection = (
            select_audio_stream(
                streams,
                preferred_language=options.language,
                requested_index=options.audio_stream,
            )
            if streams
            else None
        )
    except MediaError as exc:
        raise TranscribeError(str(exc)) from exc
    extracted_audio: Path | None = None
    if options.audio_stream is not None or len(streams) > 1:
        assert selection is not None
        extracted_audio = (
            options.output_dir
            / f".{options.input_path.stem}.audio-stream-{selection.stream.index}"
            / f"{options.input_path.stem}.wav"
        )

    if dry_run:
        command_options = (
            replace(options, output_format="json") if options.refine_subtitles else options
        )
        if extracted_audio is not None:
            assert selection is not None
            print(describe_audio_selection(selection))
            print(
                format_command(
                    audio_extract_command(
                        options.input_path,
                        extracted_audio,
                        stream_index=selection.stream.index,
                        ffmpeg_path=ffmpeg_path,
                    )
                )
            )
            command_options = replace(command_options, input_path=extracted_audio)
        command = build_command(command_options, backend=backend, model=model)
        print(format_command(command))
        return output_path

    print(f"Input: {options.input_path}")
    print(f"Output dir: {options.output_dir}")
    print(f"Backend: {backend}")
    print(f"Mode: {options.mode}")
    print(f"Model: {model}")
    if options.language:
        print(f"Language: {options.language}")
    if extracted_audio is not None:
        assert selection is not None
        print(describe_audio_selection(selection))
    print()

    if extracted_audio is None:
        return _run_whisper(
            options,
            backend=backend,
            model=model,
            output_path=output_path,
        )

    with tempfile.TemporaryDirectory(
        dir=options.output_dir,
        # selection cannot be None when extraction was requested.
        prefix=f".{options.input_path.stem}.audio-stream-{selection.stream.index}-",
    ) as temporary_dir:
        audio_path = Path(temporary_dir) / f"{options.input_path.stem}.wav"
        command = audio_extract_command(
            options.input_path,
            audio_path,
            stream_index=selection.stream.index,
            ffmpeg_path=ffmpeg_path,
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not audio_path.is_file():
            detail = completed.stderr.strip() or f"exit code {completed.returncode}"
            raise TranscribeError(
                f"Could not extract audio stream {selection.stream.index}: {detail}"
            )
        return _run_whisper(
            replace(options, input_path=audio_path),
            backend=backend,
            model=model,
            output_path=output_path,
        )


def _detect_silences(input_path: Path, ffmpeg_path: str) -> list[tuple[float, float]]:
    completed = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-nostats", "-i", str(input_path), "-vn",
         "-af", SILENCE_FILTER, "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Without silences the chunks are simply cut at fixed intervals.
    return parse_silences(completed.stderr) if completed.returncode == 0 else []


def _extract_chunk(input_path: Path, output_path: Path, chunk, ffmpeg_path: str) -> Path:
    completed = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{chunk.start:.3f}", "-t", f"{chunk.duration:.3f}", "-i", str(input_path),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not output_path.is_file():
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise TranscribeError(f"Could not cut audio {chunk.start:.0f}s-{chunk.end:.0f}s: {detail}")
    return output_path


def _whisper_piece(
    options: TranscribeOptions,
    piece: Path,
    directory: Path,
    *,
    output_format: str,
    backend: str,
    model: str,
) -> Path:
    piece_options = replace(options, input_path=piece, output_dir=directory, output_format=output_format)
    completed = subprocess.run(build_command(piece_options, backend=backend, model=model))
    if completed.returncode != 0:
        raise TranscribeError(f"Transcription failed with exit code {completed.returncode}.")
    return expected_output_path(piece_options)


def _clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def _piece_texts(kind: str, parsed: object) -> list[str]:
    if kind == "json":
        return [str(segment.get("text", "")) for segment in parsed["segments"]]  # type: ignore[index]
    if kind == "srt":
        return [cue.text for cue in parsed]  # type: ignore[attr-defined]
    return list(parsed)  # type: ignore[call-overload]


def _collapse(kind: str, parsed: object) -> object:
    texts = _piece_texts(kind, parsed)
    if kind == "json":
        return {**parsed, "segments": collapse_repeats(parsed["segments"], texts)}  # type: ignore[index]
    return collapse_repeats(parsed, texts)  # type: ignore[arg-type]


def _sets_condition(options: TranscribeOptions) -> bool:
    return any(arg.replace("_", "-").startswith("--condition-on-previous-text") for arg in options.extra_args)


def _without_context(options: TranscribeOptions, backend: str) -> TranscribeOptions:
    flag = "--condition-on-previous-text" if backend == "mlx-whisper" else "--condition_on_previous_text"
    return replace(options, extra_args=[*options.extra_args, flag, "False"])


def _run_chunked(
    options: TranscribeOptions,
    *,
    kind: str | None,
    backend: str,
    model: str,
) -> tuple[TranscribeOptions, object | None]:
    """Transcribe a long file in pieces, each started with the prompt.

    ``kind`` is what to build: "json" (merged Whisper JSON), "srt" (cues) or
    "txt" (lines); anything else is not chunked. Returns the options, with the
    language pinned when it was detected here, and the result, or None when the
    file should simply be transcribed in one go.
    """
    if kind is None or effective_initial_prompt(options) is None:
        return options, None
    ffmpeg_path = str(find_ffmpeg())
    try:
        duration = probe_duration(options.input_path, ffmpeg_path=ffmpeg_path)
    except MediaError:
        return options, None
    if duration < MIN_CHUNKED_DURATION:
        return options, None
    chunks = plan_chunks(duration, _detect_silences(options.input_path, ffmpeg_path))
    if len(chunks) < 2:
        return options, None

    with tempfile.TemporaryDirectory(dir=options.output_dir, prefix=".video-txt-chunks-") as name:
        directory = Path(name)
        piece = directory / "piece.wav"
        if options.language is None:
            # Detect the language once, so that no single piece, say one of
            # music, decides it differently from the rest.
            _extract_chunk(options.input_path, piece, chunks[0], ffmpeg_path)
            probe_options = replace(options, refine_subtitles=False)
            probe = _whisper_piece(
                probe_options, piece, directory, output_format="json", backend=backend, model=model
            )
            language = json.loads(probe.read_text(encoding="utf-8")).get("language")
            if language:
                options = replace(options, language=str(language))
            # A language that does not take the default prompt is run normally.
            if effective_initial_prompt(options) is None:
                return options, None

        print(f"Transcribing in {len(chunks)} pieces so the prompt applies throughout.")
        parts: list[tuple[float, object]] = []
        for number, chunk in enumerate(chunks, start=1):
            print(f"Piece {number}/{len(chunks)}: {_clock(chunk.start)}-{_clock(chunk.end)}")
            _extract_chunk(options.input_path, piece, chunk, ffmpeg_path)

            def decode(piece_options: TranscribeOptions) -> object:
                result = _whisper_piece(
                    piece_options, piece, directory,
                    output_format=kind, backend=backend, model=model,
                )
                if kind == "json":
                    return clamp_payload(json.loads(result.read_text(encoding="utf-8")), chunk.duration)
                if kind == "srt":
                    return clamp_cues(_read_cues(result), chunk.duration)
                return result.read_text(encoding="utf-8").splitlines()

            parsed = decode(options)
            if repeated_runs(sentences(_piece_texts(kind, parsed))) and not _sets_condition(options):
                # Whisper looped. Without the previous text to lean on it cannot.
                print("  Whisper repeated itself here; running the piece again without context.")
                parsed = decode(_without_context(options, backend))
            if repeated_runs(sentences(_piece_texts(kind, parsed))):
                print(
                    f"  Warning: still repeating after {_clock(chunk.start)}; "
                    "that stretch is unreliable."
                )
                parsed = _collapse(kind, parsed)
            if kind == "srt":
                parsed = shift_cues(parsed, chunk.start)
            parts.append((chunk.start, parsed))
    if kind == "json":
        return options, merge_whisper_payloads(parts)
    if kind == "srt":
        return options, renumber([cue for _, cues in parts for cue in cues])
    return options, [line for _, lines in parts for line in lines]


def _run_whisper(
    options: TranscribeOptions,
    *,
    backend: str,
    model: str,
    output_path: Path,
) -> Path:
    if options.refine_subtitles:
        return _run_refined_transcription(
            options,
            backend=backend,
            model=model,
            output_path=output_path,
        )

    kind = options.output_format if options.output_format in {"srt", "txt"} else None
    options, chunked = _run_chunked(options, kind=kind, backend=backend, model=model)
    if chunked is None:
        command = build_command(options, backend=backend, model=model)
        completed = subprocess.run(command)
        if completed.returncode != 0:
            raise TranscribeError(f"Transcription failed with exit code {completed.returncode}.")
    elif kind == "srt":
        write_srt(output_path, chunked)
    else:
        output_path.write_text("\n".join(chunked) + "\n", encoding="utf-8")
    prompt = effective_initial_prompt(options)
    for suffix in (".srt", ".txt"):
        if options.output_format not in {suffix[1:], "all"}:
            continue
        sibling = output_path.with_suffix(suffix)
        if not sibling.is_file():
            continue
        if prompt:
            (_drop_prompt_echo_subtitles if suffix == ".srt" else _drop_prompt_echo_text)(sibling, prompt)
        if suffix == ".srt" and sibling.is_file():
            _correct_subtitle_spelling(sibling)
    return output_path


def refinement_argument_conflict(extra_args: list[str]) -> str | None:
    for argument in extra_args:
        flag = argument.partition("=")[0].replace("_", "-")
        if flag in REFINEMENT_RESERVED_FLAGS:
            return argument
    return None


def _run_refined_transcription(
    options: TranscribeOptions,
    *,
    backend: str,
    model: str,
    output_path: Path,
) -> Path:
    with tempfile.TemporaryDirectory(
        dir=options.output_dir,
        prefix=".video-txt-transcribe-",
    ) as temporary_dir:
        raw_options = replace(
            options,
            output_dir=Path(temporary_dir),
            output_format="json",
        )
        raw_options, merged = _run_chunked(raw_options, kind="json", backend=backend, model=model)
        options = replace(options, language=raw_options.language)
        if merged is None:
            command = build_command(raw_options, backend=backend, model=model)
            completed = subprocess.run(command)
            if completed.returncode != 0:
                raise TranscribeError(
                    f"Transcription failed with exit code {completed.returncode}."
                )

        raw_path = expected_output_path(raw_options)
        try:
            payload = (
                merged
                if merged is not None
                else json.loads(raw_path.read_text(encoding="utf-8"))
            )
            if not isinstance(payload, Mapping):
                raise RefineError("Whisper JSON root must be an object.")
            document = document_from_whisper_result(payload)
            prompt = effective_initial_prompt(options)
            if prompt:
                document = _drop_prompt_echo_words(document, prompt)
            document = _correct_word_spelling(document)
            word_path = options.output_dir / f"{options.input_path.stem}.words.json"
            result = refine_transcript(document)
            write_srt(output_path, result.cues)
            # This is the completion marker used by the pipeline's reuse check,
            # so write it only after the refined subtitle is safely on disk.
            write_word_document(
                word_path,
                document,
                backend=backend,
                model=model,
                subtitle_path=output_path,
            )
        except (OSError, json.JSONDecodeError, RefineError) as exc:
            raise TranscribeError(f"Cannot refine Whisper word timestamps: {exc}") from exc

    longest = max((cue.duration for cue in result.cues), default=0.0)
    print(
        f"Refined subtitles: {result.source_words} words -> {result.cue_count} cues "
        f"(longest {longest:.1f}s)."
    )
    print(f"Word timestamps: {word_path}")
    return output_path
