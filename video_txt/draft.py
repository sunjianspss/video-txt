"""Proposing the terminology file before the first line is translated.

Names are what a translation gets wrong most and what costs most to fix. A model
translating episode by episode has no memory across them, so the same character
comes back under a different name every episode -- and the repair is retranslating
the whole season. Writing the glossary first is far cheaper, and the only reason
it gets skipped is that reading a season's subtitles looking for proper nouns is
tedious work nobody wants to start.

That reading is mechanical, so this module does it: it finds the words the
subtitles treat as names and hands back a glossary with every `target` left
blank. Filling those in is the judgement call, and it stays with a person.

The test for a proper noun here is not a list of stopwords. It is how the
subtitle itself writes the word: a name is capitalized in the middle of a
sentence and is almost never seen in lower case, while `well`, `so` and `no` are
capitalized only where a sentence happens to start. Both halves matter -- `I`
passes the first test and fails nothing else, which is why the handful of words
English always capitalizes are named outright.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .subtitles import SubtitleCue
from .terminology import Terminology

# Words a subtitle capitalizes for reasons that have nothing to do with naming
# anything: what English always capitalizes, the titles that sit in front of a
# name, and the exclamations every script is full of. Short, and the same list
# whatever the film is -- everything else is decided from the text.
ALWAYS_CAPITALIZED = frozenset({"i", "i'm", "i'll", "i've", "i'd", "ok", "okay", "tv", "ai"})
HONORIFICS = frozenset({"mr", "mrs", "ms", "dr", "st", "prof", "sgt", "lt", "capt", "jr", "sr"})
EXCLAMATIONS = frozenset({"god", "jesus", "christ"})
NEVER_PROPOSED = ALWAYS_CAPITALIZED | HONORIFICS | EXCLAMATIONS

TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'’-]*")
POSSESSIVE_PATTERN = re.compile(r"['’]s$", re.IGNORECASE)
# Dashes and quotes cling to a word without belonging to it. An internal hyphen
# does belong, so only the edges are trimmed: "Jean-Luc" survives, "I--" does not.
EDGE_PUNCTUATION_PATTERN = re.compile(r"^[-'’]+|[-'’]+$")
# The full stop in "Mr." is an abbreviation, not the end of a sentence. Left in,
# it splits "Mr. Cousineau" apart and costs the name its mid-sentence evidence.
TITLE_PERIOD_PATTERN = re.compile(
    rf"\b({'|'.join(sorted(HONORIFICS))})\.", re.IGNORECASE
)
# A sentence ends at a full stop, and a subtitle also starts one at a speaker
# label ("Sasha: Oh my God") and at the dash that marks the next person talking.
# Both look like mid-sentence to a plain full-stop split, which is how "Oh" and
# "Hey" come to look like names.
# A doubled dash is a line cut off mid-word -- "Who is that-- Oh, that's him" --
# and it hangs off the previous word with no space in front of it, which is how
# the interjection after it comes to look like it sits mid-sentence.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…:])\s+|\s*[-–—]{2,}\s*|\s+[-–—]\s*")
# Subtitle markup, sound effects and the dash that opens a line of dialogue.
NOISE_PATTERN = re.compile(r"</?[^<>]{1,40}>|^[-–—]\s*|\[[^\]]*\]|\([^)]*\)", re.MULTILINE)

# A name has to earn its place mid-sentence more than once; one appearance is as
# likely to be a sentence that started with an ordinary word.
MIN_MIDSENTENCE = 2
# How often a candidate may be seen in lower case before it is an ordinary word
# that some sentences happen to begin with.
MAX_LOWERCASE_SHARE = 0.2
DEFAULT_MIN_COUNT = 3
DEFAULT_MAX_TERMS = 60
MAX_EXAMPLES = 2
EXAMPLE_CHARS = 90
# Tokens this short collide by accident; "Ryan" is a signal, "Al" is noise.
MIN_COLLISION_TOKEN = 4
# Two spellings this close are usually one name the transcript could not settle
# on -- Whisper writes Jessie and Jesse for the same character. Below this length
# a single edit is a different word, not a slip.
MIN_VARIANT_LENGTH = 5


@dataclass(frozen=True)
class Candidate:
    """A word the subtitles treat as a name, and what makes us think so."""

    source: str
    count: int
    match: str
    examples: tuple[str, ...] = ()
    collides_with: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()

    @property
    def warning(self) -> str | None:
        notes: list[str] = []
        if self.collides_with:
            notes.append(
                f"Shares a word with {', '.join(self.collides_with)}, already in the glossary. "
                "Give this name its own entry; without one the older entry pulls it in."
            )
        if self.variants:
            notes.append(
                f"The subtitles also spell this {', '.join(self.variants)}. If those are the "
                "same name, keep one as the target and put the rest in aliases."
            )
        return " ".join(notes) or None

    def to_dict(self) -> dict[str, object]:
        entry: dict[str, object] = {
            "source": self.source,
            # Left blank on purpose: an unfilled draft must not load as a
            # glossary, and load_terminology refuses an empty target.
            "target": "",
            "match": self.match,
            "count": self.count,
            "examples": list(self.examples),
        }
        if self.variants:
            entry["aliases"] = list(self.variants)
        if self.warning:
            entry["draft_warning"] = self.warning
        return entry


@dataclass(frozen=True)
class Draft:
    candidates: tuple[Candidate, ...]
    scanned_cues: int
    already_covered: tuple[str, ...] = ()

    @property
    def colliding(self) -> tuple[Candidate, ...]:
        return tuple(item for item in self.candidates if item.collides_with)


def _cleaned(text: str) -> str:
    without_noise = NOISE_PATTERN.sub(" ", text)
    return TITLE_PERIOD_PATTERN.sub(r"\1", without_noise).replace("\n", " ")


def _sentences(cues: list[SubtitleCue]) -> list[str]:
    """The subtitle as running prose, so a sentence split across cues stays one.

    Which token starts a sentence is the whole question here, and a cue boundary
    is not an answer: subtitles break mid-clause all the time.
    """
    joined = " ".join(_cleaned(cue.text) for cue in cues if not cue.is_empty)
    return [sentence for sentence in SENTENCE_SPLIT.split(joined) if sentence.strip()]


def _is_capitalized(token: str) -> bool:
    return token[:1].isupper()


def _base(token: str) -> str:
    """The name without the possessive or the dashes the sentence hung on it."""
    return EDGE_PUNCTUATION_PATTERN.sub("", POSSESSIVE_PATTERN.sub("", token))


def _within_one_edit(left: str, right: str) -> bool:
    """Whether two spellings differ by a single insertion, deletion or swap."""
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    index = 0
    while index < len(left) and left[index] == right[index]:
        index += 1
    if index == len(left):
        return len(right) - len(left) <= 1
    if len(left) == len(right):
        return left[index + 1 :] == right[index + 1 :]
    return left[index:] == right[index + 1 :]


def _name_like(
    capitalized: Counter[str], lowercase: Counter[str], midsentence: Counter[str]
) -> set[str]:
    """Words the subtitle writes the way it writes names."""
    return {
        key
        for key, seen in capitalized.items()
        if key not in NEVER_PROPOSED
        and len(key) >= 2
        and midsentence[key] >= MIN_MIDSENTENCE
        and lowercase[key] <= seen * MAX_LOWERCASE_SHARE
    }


def _phrase_joinable(capitalized: Counter[str], lowercase: Counter[str]) -> set[str]:
    """Words allowed to sit inside a name without being one on their own.

    A surname is often only ever seen behind a given name, so it never earns the
    mid-sentence evidence a standalone name needs. It still belongs in the phrase.
    """
    return {
        key
        for key, seen in capitalized.items()
        if key not in NEVER_PROPOSED
        and len(key) >= 2
        and seen >= MIN_MIDSENTENCE
        and lowercase[key] <= seen * MAX_LOWERCASE_SHARE
    }


def scan_candidates(
    cues: list[SubtitleCue],
    *,
    min_count: int = DEFAULT_MIN_COUNT,
    max_terms: int = DEFAULT_MAX_TERMS,
) -> list[Candidate]:
    """Names the subtitles keep using, most frequent first."""
    sentences = _sentences(cues)
    capitalized: Counter[str] = Counter()
    lowercase: Counter[str] = Counter()
    midsentence: Counter[str] = Counter()
    tokenized: list[list[tuple[str, int, int]]] = []

    for sentence in sentences:
        tokens = [
            (_base(match.group(0)), match.start(), match.end())
            for match in TOKEN_PATTERN.finditer(sentence)
        ]
        tokenized.append(tokens)
        for position, (token, _start, _end) in enumerate(tokens):
            key = token.casefold()
            if _is_capitalized(token):
                capitalized[key] += 1
                if position > 0:
                    midsentence[key] += 1
            else:
                lowercase[key] += 1

    names = _name_like(capitalized, lowercase, midsentence)
    if not names:
        return []
    joinable = _phrase_joinable(capitalized, lowercase)

    # Counted by lower-cased key so one name is one candidate however the line
    # happened to capitalize it, with the spelling it wears most often kept.
    counts: Counter[str] = Counter()
    kinds: dict[str, str] = {}
    surfaces: dict[str, Counter[str]] = {}
    examples: dict[str, list[str]] = {}

    def record(words: list[str], sentence: str) -> None:
        surface = " ".join(words)
        key = surface.casefold()
        counts[key] += 1
        kinds[key] = "phrase" if len(words) > 1 else "word"
        surfaces.setdefault(key, Counter())[surface] += 1
        examples.setdefault(key, []).append(sentence)

    for sentence, tokens in zip(sentences, tokenized, strict=True):
        run: list[str] = []
        previous_end = -1
        for token, start, end in [*tokens, ("", -1, -1)]:
            # Only whitespace may separate the words of one name. "Oh, Lilypan"
            # is a greeting and a name, not a two-word name, and the comma is the
            # only thing that says so.
            touching = previous_end >= 0 and not sentence[previous_end:start].strip()
            if token and _is_capitalized(token) and token.casefold() in joinable:
                if not touching:
                    if run and any(word.casefold() in names for word in run):
                        record(run, sentence)
                    run = []
                run.append(token)
                previous_end = end
                continue
            # A run is a name only if something in it stands on its own as one;
            # otherwise it is two sentence-initial words that happened to meet.
            if run and any(word.casefold() in names for word in run):
                record(run, sentence)
            run = []
            previous_end = -1

    ranked = sorted(
        (key for key, count in counts.items() if count >= min_count),
        key=lambda key: (-counts[key], key),
    )[:max_terms]

    chosen = {key: surfaces[key].most_common(1)[0][0] for key in ranked}
    return [
        Candidate(
            source=chosen[key],
            count=counts[key],
            match=kinds[key],
            examples=_examples(examples, key),
            variants=_spelling_variants(key, chosen),
        )
        for key in ranked
    ]


def _spelling_variants(key: str, chosen: dict[str, str]) -> tuple[str, ...]:
    if len(key) < MIN_VARIANT_LENGTH:
        return ()
    return tuple(
        surface
        for other, surface in chosen.items()
        if other != key and len(other) >= MIN_VARIANT_LENGTH and _within_one_edit(key, other)
    )


def _examples(examples: dict[str, list[str]], key: str) -> tuple[str, ...]:
    chosen: list[str] = []
    for sentence in examples.get(key, []):
        trimmed = " ".join(sentence.split())[:EXAMPLE_CHARS]
        if trimmed and trimmed not in chosen:
            chosen.append(trimmed)
        if len(chosen) == MAX_EXAMPLES:
            break
    return tuple(chosen)


def _known_forms(existing: Terminology) -> set[str]:
    forms: set[str] = set()
    for term in existing.terms:
        forms.add(term.source.casefold())
        forms.update(alias.casefold() for alias in term.aliases)
    return forms


def _collisions(source: str, existing: Terminology) -> tuple[str, ...]:
    """Glossary entries a new name shares a word with.

    This is the season-to-season trap: carry `Ryan Madison` into the next season
    and the model reads the new `Aaron Ryan` as the old character. Both names are
    really in the script, so neither entry can be dropped -- the new one has to be
    written down too, and that is what this warns about.
    """
    words = {word.casefold() for word in source.split() if len(word) >= MIN_COLLISION_TOKEN}
    if not words:
        return ()
    return tuple(
        term.source
        for term in existing.terms
        if words & {word.casefold() for word in term.source.split()}
        and term.source.casefold() != source.casefold()
    )


def draft_terminology(
    cues: list[SubtitleCue],
    *,
    existing: Terminology | None = None,
    min_count: int = DEFAULT_MIN_COUNT,
    max_terms: int = DEFAULT_MAX_TERMS,
) -> Draft:
    """Names worth a glossary entry, minus the ones a glossary already has."""
    found = scan_candidates(cues, min_count=min_count, max_terms=max_terms)
    if existing is None:
        return Draft(candidates=tuple(found), scanned_cues=len(cues))

    known = _known_forms(existing)
    fresh: list[Candidate] = []
    covered: list[str] = []
    for candidate in found:
        if candidate.source.casefold() in known:
            covered.append(candidate.source)
            continue
        fresh.append(
            Candidate(
                source=candidate.source,
                count=candidate.count,
                match=candidate.match,
                examples=candidate.examples,
                collides_with=_collisions(candidate.source, existing),
                variants=candidate.variants,
            )
        )
    return Draft(
        candidates=tuple(fresh),
        scanned_cues=len(cues),
        already_covered=tuple(covered),
    )


def draft_to_dict(
    draft: Draft, *, source_language: str | None, target_language: str | None
) -> dict[str, object]:
    """The draft as a terminology file with every target still to be written."""
    return {
        "schema": "video-txt.terminology",
        "version": 1,
        "source_language": source_language,
        "target_language": target_language,
        "terms": [candidate.to_dict() for candidate in draft.candidates],
    }
