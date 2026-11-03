"""Music extraction workflow; dry runs describe work without touching audio."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .artifacts import write_json
from .media import find_ffmpeg, probe_sample_rate
from .music import (
    PITCHED_STEMS,
    MusicError,
    MusicPiece,
    extract_command,
    find_music,
    loudness_envelope,
    piece_filename,
    sung_spans,
)
from .separate import (
    ensure_instrumental,
    ensure_stems,
    mix_instrumental,
    separated_bgm_path,
    separated_voice_path,
)
from .subtitles import parse_srt


@dataclass(frozen=True)
class MusicOptions:
    media: Path
    output_dir: Path | None
    cache_dir: Path | None
    subtitle: Path | None
    min_duration: float
    from_time: float | None
    to_time: float | None
    min_musicality: float
    model: str
    stems: bool
    source: str
    music_floor: float
    voice_floor: float
    clean: bool
    audio_format: str
    raw_levels: bool
    overwrite: bool
    dry_run: bool


def default_music_dir(media: Path) -> Path:
    return media.with_name(f"{media.stem}.music")


def run_music(args: MusicOptions) -> int:
    media = args.media
    output_dir = args.output_dir or default_music_dir(media)
    cache_dir = args.cache_dir or media.with_name(f"{media.stem}.dub-cache")
    if args.min_duration <= 0:
        raise MusicError("--min-duration must be greater than 0")
    named = args.from_time is not None or args.to_time is not None
    if named and (args.from_time is None or args.to_time is None):
        raise MusicError("--from and --to must be passed together.")

    if named and not 0 <= args.from_time < args.to_time:
        raise MusicError("--from and --to must define a positive, increasing time range.")

    print(f"Media: {media}")
    print(f"Separated audio cache: {cache_dir}")
    print(f"Music: {output_dir}")

    gated = args.min_musicality > 0
    if gated and args.model != "htdemucs_6s":
        raise MusicError(
            "--min-musicality needs --model htdemucs_6s: the test asks whether two pitched "
            "instruments sound together, and only that model separates guitar and piano. "
            "Pass --min-musicality 0 to keep everything the loudness found instead."
        )

    if args.dry_run:
        print("Would reuse or separate the soundtrack, find music, then export (dry run).")
        return 0

    ffmpeg_path = find_ffmpeg()
    stems: dict[str, Path] = {}
    if gated or args.stems:
        stems = ensure_stems(media, model=args.model, cache_dir=cache_dir, ffmpeg_path=ffmpeg_path)

    # The dub's own cache first: a film already separated for --separate-bgm costs
    # nothing here. Failing that the instrumental is mixed back from the stems,
    # which saves separating the same soundtrack a second time.
    instrumental = separated_bgm_path(cache_dir)
    voice = separated_voice_path(cache_dir)
    if instrumental.is_file() and voice.is_file():
        print(f"Reusing the separated background track: {instrumental.name}")
    elif stems:
        instrumental = mix_instrumental(stems, target=instrumental, ffmpeg_path=ffmpeg_path)
        voice = stems["vocals"]
    else:
        instrumental = ensure_instrumental(media, cache_dir=cache_dir, ffmpeg_path=ffmpeg_path)
        voice = separated_voice_path(cache_dir)
    if not voice.is_file():
        raise MusicError(
            f"The separated voice track is missing: {voice}. It is what says which "
            f"stretches nobody talks over. Delete {separated_bgm_path(cache_dir).parent} "
            "and run again to rebuild both."
        )

    # Each piece travels with the file it gets cut out of.
    found: list[tuple[MusicPiece, Path]] = []
    if named:
        piece = MusicPiece(
            start=args.from_time,
            end=args.to_time,
            voice_share=0.0,
            peak_lufs=0.0,
            labelled="named",
        )
        found.append((piece, media if args.source == "mix" else instrumental))
    else:
        songs: list[MusicPiece] = []
        if args.subtitle:
            songs = sung_spans(parse_srt(args.subtitle))
            print(f"Songs marked in the subtitle: {len(songs)}")

        print("Measuring the soundtrack...")
        pitched = [
            loudness_envelope(stems[name], ffmpeg_path=ffmpeg_path)
            for name in PITCHED_STEMS
            if name in stems
        ]
        # The songs are found first and masked out of the search, so a song does
        # not come back a second time inside the longer stretch around it.
        pieces = find_music(
            loudness_envelope(instrumental, ffmpeg_path=ffmpeg_path),
            loudness_envelope(voice, ffmpeg_path=ffmpeg_path),
            music_floor=args.music_floor,
            voice_floor=args.voice_floor,
            min_duration=args.min_duration,
            clean_only=args.clean,
            exclude=[(song.start, song.end) for song in songs],
            pitched=pitched,
            min_musicality=args.min_musicality,
        )
        found = [(piece, instrumental) for piece in pieces]
        # A song is cut from the soundtrack itself: without its singing it is not
        # the song, and the singing is the half separation takes out. A person
        # marked these, so they skip the music test -- but a one-second scrap of
        # one is still not worth writing out.
        found += [(song, media) for song in songs if song.duration >= args.min_duration]
        found.sort(key=lambda pair: pair[0].start)

    print(f"Pieces found: {len(found)}")
    for number, (piece, _source) in enumerate(found, start=1):
        print(
            f"  {number:02d}  {piece.kind:15s} {piece.clock_range} "
            f"({piece.duration:6.1f}s, voice {piece.voice_share:.0%})"
        )
    if not found:
        print("Nothing long enough to write out. Lower --min-duration to widen the search.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, object]] = []
    rates: dict[Path, int | None] = {}
    for number, (piece, source) in enumerate(found, start=1):
        target = output_dir / piece_filename(media, number, piece, audio_format=args.audio_format)
        if target.exists() and not args.overwrite:
            raise MusicError(
                f"Music file already exists: {target}. Pass --overwrite to replace it."
            )
        if source not in rates:
            rates[source] = probe_sample_rate(source, ffmpeg_path=ffmpeg_path)
        completed = subprocess.run(
            extract_command(
                source,
                piece,
                output=target,
                ffmpeg_path=ffmpeg_path,
                normalize=not args.raw_levels,
                sample_rate=rates[source],
                audio_format=args.audio_format,
            ),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not target.is_file():
            raise MusicError(f"Could not cut {target.name}: {(completed.stderr or '').strip()}")
        written.append({**piece.to_dict(), "file": target.name})

    report = output_dir / f"{media.stem}.music.json"
    write_json(
        report,
        {
            "schema": "video-txt.music",
            "version": 1,
            "media": str(media),
            "instrumental": str(instrumental),
            "stems": {name: str(path) for name, path in stems.items()},
            "format": args.audio_format,
            "normalized": not args.raw_levels,
            "piece_count": len(written),
            "pieces": written,
        },
    )
    print(f"Wrote {len(written)} music file(s) to: {output_dir}")
    print(f"Whole-film instrumental: {instrumental}")
    if stems:
        print(f"Stems: {', '.join(sorted(stems))} in {next(iter(stems.values())).parent}")
    print(f"Report: {report}")
    return 0
