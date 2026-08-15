---
name: verify
description: Build a throwaway video + subtitle fixture and drive the video-txt CLI end to end, so a dub change is observed running rather than unit-tested.
---

# Verifying a video-txt change

The surface is the CLI. `.venv/bin/video-txt` is already installed in the repo
venv (`uv sync` if it is missing). ffmpeg/ffprobe come from Homebrew.

## Fixture

Twenty seconds of colour plus two mixed sine tones stands in for a video with
music under the dialogue — enough for `--separate-bgm` to have something to
separate:

```bash
ffmpeg -y -v error \
  -f lavfi -i "color=c=navy:s=320x240:r=12:d=20" \
  -f lavfi -i "sine=frequency=220:d=20" -f lavfi -i "sine=frequency=330:d=20" \
  -filter_complex "[1:a][2:a]amix=inputs=2[a]" \
  -map 0:v -map "[a]" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest clip.mp4
```

## Skipping the paid translation step

`dub` translates before it speaks. Write the translated subtitle yourself and
it is reused as is — no API key, no network:

```bash
video-txt dub clip.mp4 --subtitle clip.srt --voice-unit line -o out.mp4
#            reads clip.srt (source) and clip.zh.srt (translation) beside it
```

Only pass `--provider deepseek` when the translation prompt itself is what
changed; two subtitle blocks is enough to compare against a
`video-txt translate` run, which does not set `spoken=True`.

`--voice-unit line` keeps clips 1:1 with subtitle lines. The default merges
sentences and makes counts harder to read.

## Driving the paths that matter

- **Overrun / re-speak**: slots have to be genuinely tight. `compute_slots`
  gives a line everything up to the *next remaining* cue, so a generous gap —
  or a dropped `...` line — silently hands it seconds of room and nothing
  overruns. Four back-to-back 2-second cues with 8 / 14 / 43 Chinese
  characters gives one clean fit, one re-speak, one re-speak that still needs
  atempo.
- **Unvoiced lines**: a cue of `...` or `♪♪` should be dropped with a count
  printed, never sent to edge-tts.
- **`--separate-bgm`**: demucs is in the venv already; a 20-second clip
  separates in seconds. Rerun to see `Reusing the separated background track`.
- **Dry run first**: `--dry-run` prints the whole ffmpeg mux command, which is
  where the loudnorm chain, the `-ar 48000`, the bgm input index and the
  `volume=` value are all visible without rendering anything.

## Measuring the audio claims

```bash
# loudnorm reached its target (expect I ≈ -16 LUFS)
ffmpeg -nostats -i out.mp4 -af ebur128=peak=true -f null - 2>&1 | grep -A2 "Integrated"

# background survived the mix: measure a stretch with no speech in it
ffmpeg -hide_banner -nostats -ss 19 -t 1 -i out.mp4 -af astats -f null - 2>&1 | grep "RMS level"
```

For the clip-edge declick, render with `--keep-dub-audio` and read the wav: at
a cue's start time the first sample is 0 and the envelope climbs over exactly
8 ms (`EDGE_FADE_SECONDS`) before levelling off.

## Gotchas

- Piping the CLI through `grep` block-buffers its stdout, so subprocess output
  (demucs progress, ffmpeg banners, stderr errors) interleaves out of order.
  Order in a real terminal is fine — do not report it as a finding.
- ffmpeg writes its banner and progress to stderr; filter with
  `grep -viE "^\[|^ +|^frame|^Output|^Stream|^Input|ffmpeg version"`.
- Cache lives in `<video>.dub-cache/`. Delete it to force resynthesis; leave
  it to check that a rerun is a full cache hit.
