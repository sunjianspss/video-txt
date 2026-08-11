#!/usr/bin/env -S uv run --python 3.12 python

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from video_txt.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["transcribe", *sys.argv[1:]]))
