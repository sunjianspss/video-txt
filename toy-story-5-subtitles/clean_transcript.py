from __future__ import annotations

import argparse
from pathlib import Path

from video_txt.subtitles import SubtitleCue, parse_srt, write_srt


def is_full_window_hallucination(cue: SubtitleCue) -> bool:
    """Match Whisper's characteristic one-short-phrase-per-window failure."""
    return cue.duration >= 29.5 and len(cue.text.split()) <= 6


def clean(cues: list[SubtitleCue]) -> tuple[list[SubtitleCue], list[SubtitleCue], int]:
    cleaned: list[SubtitleCue] = []
    removed: list[SubtitleCue] = []
    merged_zero_duration = 0

    for cue in cues:
        cue = cue.with_text(cue.text.strip())
        if is_full_window_hallucination(cue):
            removed.append(cue)
            continue

        if cue.duration <= 0 and cleaned:
            previous = cleaned[-1]
            separator = "" if previous.text.endswith(("-", "…")) else " "
            previous.text_lines = [f"{previous.text.rstrip()}{separator}{cue.text.lstrip()}"]
            merged_zero_duration += 1
            continue

        cleaned.append(cue)

    for index, cue in enumerate(cleaned, start=1):
        cue.index = str(index)
    return cleaned, removed, merged_zero_duration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    cues = parse_srt(args.source)
    cleaned, removed, merged = clean(cues)
    write_srt(args.output, cleaned)

    print(f"Input cues: {len(cues)}")
    print(f"Output cues: {len(cleaned)}")
    print(f"Removed full-window hallucinations: {len(removed)}")
    for cue in removed:
        print(f"  {cue.timing}  {cue.text}")
    print(f"Merged zero-duration fragments: {merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
