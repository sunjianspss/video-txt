from __future__ import annotations

import pytest

from video_txt import transcribe
from video_txt.media import ProbeError


@pytest.fixture(autouse=True)
def unreadable_media_is_not_chunked(monkeypatch):
    """Test media are placeholder bytes that ffprobe could not read.

    Long audio is transcribed in pieces, which starts by reading its duration;
    tests that fake out Whisper do not expect that extra ffprobe call. The
    tests of chunking itself patch the duration back in.
    """

    def unreadable(path, **_kwargs):
        raise ProbeError(f"not real media: {path}")

    monkeypatch.setattr(transcribe, "probe_duration", unreadable)
