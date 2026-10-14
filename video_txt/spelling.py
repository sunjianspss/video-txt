"""Put right the spelling of Claude when Whisper has heard something else.

Whisper writes the rare name as a common word it sounds like: Claud, Quad,
Cloud. ``Claud Code`` is only ever that mistake, so it is always corrected.
``Quad`` and ``Cloud`` are real words, and Google has a product called Cloud
Code, so they are only corrected when the text itself shows it is about Claude:
enough mentions of Claude Code, Claude Opus and the like, or Anthropic.

Matching is on the visible text, which may run across subtitle blocks and
inline tags; the changes are made letter by letter on the original text so that
tags, block boundaries and timings stay exactly where they were.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_INLINE_TAG = re.compile(r"</?(?:u|b|i|font)(?:\s+[^<>]*?)?\s*>", re.IGNORECASE)

# Always safe: Claud is not a word, and "Claud Code" is the product.
_CLAUD_CODE = re.compile(r"(?<!\w)(?P<name>Claud)(?P<sep>\s+)(?P<tail>Code)(?!\w)", re.IGNORECASE)

_PRODUCT_WORDS = {"code": "Code", "opus": "Opus", "sonnet": "Sonnet", "haiku": "Haiku"}
# Quad / Cloud in front of a Claude product word: "Quad Code", "Cloud Opus".
_MISHEARD_PRODUCT = re.compile(
    r"(?<![\w-])(?P<name>Quad|Cloud)(?P<sep>\s+)(?P<tail>Code|Opus|Sonnet|Haiku)(?![\w-])",
    re.IGNORECASE,
)
_MISHEARD_ALONE = re.compile(r"(?<![\w-])(?P<name>Quad)(?![\w-])", re.IGNORECASE)
# Quad that really is quad.
_REAL_QUAD_AFTER = re.compile(
    r"\s+(?:core|cores|bike|bikes|copter|copters|drone|drones|rotor|rotors|gpu|gpus"
    r"|damage|city|cities|processor|processors)(?!\w)",
    re.IGNORECASE,
)
_GOOGLE_BEFORE = re.compile(r"google\s+$", re.IGNORECASE)

# Signs that the text is about Claude, as opposed to a talk with quad bikes or
# Google Cloud Code in it.
_EVIDENCE = re.compile(
    r"(?<![\w-])(?:Claude\s+(?:Code|Opus|Sonnet|Haiku|\d)|Anthropic)(?![\w-])", re.IGNORECASE
)
MIN_EVIDENCE = 3

CLAUDE = "Claude"


@dataclass(frozen=True)
class _Edit:
    start: int
    end: int
    new: str


def _edits(visible: str) -> list[_Edit]:
    edits: list[_Edit] = []
    claud_matches = list(_CLAUD_CODE.finditer(visible))
    for match in claud_matches:
        edits.append(_Edit(match.start("name"), match.end("name"), CLAUDE))
        edits.append(_Edit(match.start("tail"), match.end("tail"), "Code"))

    evidence = len(_EVIDENCE.findall(visible)) + len(claud_matches)
    if evidence < MIN_EVIDENCE:
        return edits

    taken: list[tuple[int, int]] = []
    for match in _MISHEARD_PRODUCT.finditer(visible):
        if _GOOGLE_BEFORE.search(visible[max(0, match.start() - 8) : match.start()]):
            continue
        edits.append(_Edit(match.start("name"), match.end("name"), CLAUDE))
        edits.append(
            _Edit(match.start("tail"), match.end("tail"), _PRODUCT_WORDS[match["tail"].lower()])
        )
        taken.append((match.start("name"), match.end("name")))
    for match in _MISHEARD_ALONE.finditer(visible):
        if (match.start("name"), match.end("name")) in taken:
            continue
        if _REAL_QUAD_AFTER.match(visible, match.end()):
            continue
        edits.append(_Edit(match.start("name"), match.end("name"), CLAUDE))
    return edits


def correct_texts(texts: Sequence[str], joins: Sequence[str]) -> list[str]:
    """``texts`` with Claude's spelling put right.

    ``joins[i]`` sits between ``texts[i]`` and ``texts[i + 1]``: a newline when
    a phrase may run on from one into the next, anything else when it may not.
    """
    visible: list[str] = []
    locations: list[tuple[int, int] | None] = []
    for index, text in enumerate(texts):
        if index:
            visible.append(joins[index - 1])
            locations.append(None)
        cursor = 0
        for tag in _INLINE_TAG.finditer(text):
            visible.extend(text[cursor : tag.start()])
            locations.extend((index, offset) for offset in range(cursor, tag.start()))
            cursor = tag.end()
        visible.extend(text[cursor:])
        locations.extend((index, offset) for offset in range(cursor, len(text)))

    replacements: dict[int, dict[int, str]] = {}
    for edit in _edits("".join(visible)):
        old = "".join(visible[edit.start : edit.end])
        if old == edit.new:
            continue
        # Give each old letter its new letter; any extra letters go on the last
        # one, so Claud -> Claude puts the e in the d's formatting.
        keep = len(old) - 1
        for position in range(edit.start, edit.end):
            location = locations[position]
            assert location is not None  # An edit covers letters, never a separator.
            index, offset = location
            step = position - edit.start
            letters = edit.new[step] if step < keep else edit.new[keep:]
            replacements.setdefault(index, {})[offset] = letters
    return [
        "".join(replacements.get(index, {}).get(offset, char) for offset, char in enumerate(text))
        for index, text in enumerate(texts)
    ]
