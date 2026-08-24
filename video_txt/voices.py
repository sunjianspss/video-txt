"""Deciding who speaks each line, and in whose voice.

A dub with one voice needs none of this. With diarization in play there is a
speaker per line, a voice per speaker, and a reference clip per cloned voice —
and a name in any of those that matches nobody is worth stopping for, because
the run would otherwise sound wrong and still report success.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .clone import Reference
from .diarize import DEFAULT_SPEAKER

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_SAY_VOICE = "Tingting"
# Second and later speakers get one of these, alternating between a female and
# a male voice so a conversation stays easy to follow.
SPEAKER_VOICE_POOL = (
    "zh-CN-YunxiNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunjianNeural",
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunyangNeural",
)


@dataclass
class VoiceChoice:
    """The voice one line is spoken in, and the clip it was cloned from."""

    name: str
    reference: Reference | None = None

    @property
    def identity(self) -> str:
        """What makes this voice different from another, for the clip cache."""
        if self.reference is None:
            return self.name
        return f"{self.name}|{self.reference.fingerprint}"


@dataclass
class VoicePlan:
    """Who speaks which line, once every speaker has been given a voice."""

    default: VoiceChoice
    by_position: dict[int, VoiceChoice] = field(default_factory=dict)

    def for_position(self, position: int) -> VoiceChoice:
        return self.by_position.get(position, self.default)


def resolve_voice_name(engine: str, voice: str) -> str:
    """The voice the engine really speaks in, once its own default is applied."""
    if engine == "say" and voice == DEFAULT_VOICE:
        return DEFAULT_SAY_VOICE
    return voice


def assign_voices(
    speakers: list[str], *, engine: str, voice: str, named: dict[str, str]
) -> dict[str, str]:
    """Give the speaker with the most airtime the chosen voice, the rest one each."""
    # A lone speaker needs no SPEAKER= in front of their voice, and the check that
    # rejects unknown names lets the bare form through for exactly that reason.
    if len(speakers) == 1 and speakers[0] not in named and DEFAULT_SPEAKER in named:
        named = {**named, speakers[0]: named[DEFAULT_SPEAKER]}
    chosen_voice = resolve_voice_name(engine, voice)
    # Only edge-tts is guaranteed to have the whole pool; elsewhere the extra
    # speakers keep the one voice we know exists until --speaker-voice says more.
    taken = {chosen_voice, *named.values()}
    available = SPEAKER_VOICE_POOL if engine == "edge-tts" else ()
    pool = [name for name in available if name not in taken]
    names: dict[str, str] = {}
    drawn = 0
    for index, speaker in enumerate(speakers):
        chosen = named.get(speaker)
        if chosen is not None:
            names[speaker] = chosen
        elif index == 0 or not pool:
            names[speaker] = chosen_voice
        else:
            names[speaker] = pool[drawn % len(pool)]
            drawn += 1
    return names


def speaker_name_problem(
    speakers: list[str], *, voices: Iterable[str], references: Iterable[str]
) -> str | None:
    """What is wrong with a voice or a clip named for somebody nobody found.

    Left unchecked the name simply matches no one: every line keeps the voice it
    would have had anyway, nothing is printed, and the run reports success. A
    made-up name also claims a voice out of the pool, so the mistake quietly
    changes what a different speaker sounds like.
    """
    known = set(speakers)
    if len(speakers) == 1:
        # With one speaker the SPEAKER= prefix is unnecessary, not wrong.
        known.add(DEFAULT_SPEAKER)
    listed = ", ".join(speakers)

    for flag, asked in (("--speaker-voice", set(voices)), ("--clone-reference", set(references))):
        unknown = sorted(asked - known)
        if not unknown:
            continue
        if DEFAULT_SPEAKER in unknown:
            return (
                f"{flag} does not say which of the {len(speakers)} speakers it is for. "
                f"Put the name in front of it, for example {flag} {speakers[0]}=..."
            )
        return (
            f"{flag} names {unknown[0]}, who is not one of the speakers that were "
            f"found ({listed}). Use the names from the diarization report."
        )
    return None
