"""Duration-aware retranslation.

Text-to-speech does not care how long a subtitle stays on screen, so a faithful
translation often takes longer to read aloud than the slot the original speaker
used. This module measures the real speaking speed of the synthesized clips,
turns each overrunning slot into a character budget, and asks the model for a
shorter line that still says the same thing.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .parallel import map_in_parallel
from .subtitles import chunk_by_chars, join_text_parts
from .translate import (
    JsonModeState,
    TranslationConfig,
    request_items,
)

WHITESPACE_PATTERN = re.compile(r"\s+")
LATIN_RUN_PATTERN = re.compile(r"[A-Za-z0-9]+")
FALLBACK_UNITS_PER_SECOND = 4.5
MIN_REGRESSION_SAMPLES = 8
# Rewriting costs an API call and a new voice clip, so only bother when the
# budget buys a real cut.
MIN_GAIN_RATIO = 0.9

# Time every line's current text; returns seconds keyed by cue position. The whole
# set goes in each round because a rewrite can change how lines merge into clips,
# and a clip that was not rewritten may still have been re-cut around one that was.
Measure = Callable[[dict[int, str]], dict[int, float]]


@dataclass(frozen=True)
class SpeechRate:
    """Clip length as ``overhead + seconds_per_unit * speech units``.

    The overhead absorbs the lead-in and tail silence the engine adds to every
    clip, which would otherwise make short lines look far slower than long ones.
    """

    overhead: float
    seconds_per_unit: float

    @property
    def units_per_second(self) -> float:
        return 1 / self.seconds_per_unit

    def units_for(self, seconds: float) -> float:
        return max(1.0, (seconds - self.overhead) / self.seconds_per_unit)

    def seconds_for(self, units: float) -> float:
        return self.overhead + units * self.seconds_per_unit


@dataclass
class FitOptions:
    translation: TranslationConfig
    tempo: float = 1.15
    rounds: int = 2
    min_ratio: float = 0.65


@dataclass
class FitItem:
    position: int
    text: str
    slot: float
    source_text: str | None = None


@dataclass
class ShortenRequest:
    position: int
    text: str
    max_chars: int
    source_text: str | None = None


@dataclass
class FitOutcome:
    texts: dict[int, str]
    shortened: dict[int, str]
    durations: dict[int, float]
    rate: SpeechRate
    over_before: int
    over_after: int
    tempo_before: float
    tempo_after: float
    rounds_used: int = 0


def text_length(text: str) -> int:
    return len(WHITESPACE_PATTERN.sub("", text))


def speech_units(text: str) -> float:
    """Roughly how many Chinese characters this line costs to say.

    Counting raw characters prices Latin words far too high: ``Claude Code``
    is eleven characters but two spoken beats, which drags the measured speed
    up and then hands those lines an impossibly tight budget. Latin runs are
    charged by syllable instead, and acronyms per letter because that is how
    they are read out.
    """
    units = float(text_length(LATIN_RUN_PATTERN.sub("", text)))
    for run in LATIN_RUN_PATTERN.findall(text):
        units += len(run) if run.isupper() else max(2.0, round(len(run) / 3))
    return units


def flatten_line(text: str) -> str:
    return join_text_parts([part.strip() for part in text.splitlines()])


def measure_speech_rate(texts: dict[int, str], durations: dict[int, float]) -> SpeechRate:
    samples = [
        (speech_units(texts[position]), seconds)
        for position, seconds in durations.items()
        if position in texts and seconds > 0 and speech_units(texts[position]) > 0
    ]
    if not samples:
        return SpeechRate(overhead=0.0, seconds_per_unit=1 / FALLBACK_UNITS_PER_SECOND)

    count = len(samples)
    sum_units = sum(units for units, _ in samples)
    sum_seconds = sum(seconds for _, seconds in samples)
    average = SpeechRate(overhead=0.0, seconds_per_unit=sum_seconds / sum_units)
    if count < MIN_REGRESSION_SAMPLES:
        return average

    sum_units_squared = sum(units * units for units, _ in samples)
    sum_product = sum(units * seconds for units, seconds in samples)
    denominator = count * sum_units_squared - sum_units * sum_units
    if denominator <= 0:
        return average

    slope = (count * sum_product - sum_units * sum_seconds) / denominator
    if slope <= 0:
        return average
    intercept = (sum_seconds - slope * sum_units) / count
    return SpeechRate(overhead=max(0.0, intercept), seconds_per_unit=slope)


class ShortenCache:
    """Remembers ``shorten(text, budget) -> text`` so a rerun costs no API calls."""

    def __init__(self, path: Path, namespace: str) -> None:
        self.path = path
        self.namespace = namespace
        self._entries: dict[str, str] = {}
        self._lock = threading.Lock()
        self._loaded = False

    def _key(self, text: str, max_chars: int) -> str:
        raw = f"{self.namespace}|{max_chars}|{text}".encode()
        return hashlib.sha1(raw).hexdigest()[:16]

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key, text = record.get("key"), record.get("text")
            if isinstance(key, str) and isinstance(text, str):
                self._entries[key] = text

    def get(self, text: str, max_chars: int) -> str | None:
        with self._lock:
            self._load()
            return self._entries.get(self._key(text, max_chars))

    def put(self, text: str, max_chars: int, shortened: str) -> None:
        key = self._key(text, max_chars)
        with self._lock:
            self._load()
            if self._entries.get(key) == shortened:
                return
            self._entries[key] = shortened
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "text": shortened}, ensure_ascii=False) + "\n")


def build_shorten_messages(
    requests: list[ShortenRequest], *, config: TranslationConfig
) -> list[dict[str, str]]:
    system_prompt = (
        "You tighten dubbing lines so the spoken audio fits a fixed time slot. "
        "Return JSON only. Do not add Markdown fences or commentary."
    )

    items: list[dict[str, object]] = []
    for request in requests:
        item: dict[str, object] = {
            "id": str(request.position),
            "text": request.text,
            "current_chars": text_length(request.text),
            "max_chars": request.max_chars,
        }
        if request.source_text:
            item["source_text"] = request.source_text
        items.append(item)

    payload: dict[str, object] = {
        "task": "Rewrite each line shorter so it can be spoken within its time slot",
        "language": config.target_language,
        "rules": [
            "Return one item per input id, with the ids unchanged.",
            "Aim for max_chars characters; whitespace does not count.",
            "Every line must stay a fluent, natural spoken sentence.",
            "Never drop grammar, particles or meaning just to hit max_chars: "
            "when the budget cannot be met that way, return the shortest "
            "natural sentence you can instead.",
            "Cut filler, hedging and repetition first; keep facts, numbers, "
            "names and preserved terms.",
            "Return one line with no line breaks.",
            "source_text, when present, is the original line: rewrite from its meaning.",
        ],
        "preserve_terms": config.preserve_terms,
        "extra_note": config.note or "",
        "output_schema": {"items": [{"id": "same as input", "text": "the shortened line only"}]},
        "items": items,
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def request_shorter_texts(
    requests: list[ShortenRequest],
    *,
    config: TranslationConfig,
    cache: ShortenCache | None = None,
    debug_dir: Path | None = None,
) -> dict[int, str]:
    results: dict[int, str] = {}
    pending: list[ShortenRequest] = []
    for request in requests:
        cached = cache.get(request.text, request.max_chars) if cache else None
        if cached is None:
            pending.append(request)
        else:
            results[request.position] = cached

    if results:
        print(f"  Reusing {len(results)} cached rewrite{'s' if len(results) > 1 else ''}.")
    if not pending:
        return results

    batches = chunk_by_chars(pending, config.batch_chars, lambda request: len(request.text))
    json_mode = JsonModeState()
    lock = threading.Lock()

    def run_batch(batch_number: int, batch: list[ShortenRequest]) -> None:
        by_id = {str(request.position): request for request in batch}
        answers = request_items(
            ids=list(by_id),
            build_request=lambda ids: build_shorten_messages(
                [by_id[item_id] for item_id in ids], config=config
            ),
            config=config,
            json_mode=json_mode,
            debug_dir=debug_dir,
            batch_number=batch_number,
        )
        accepted: dict[int, str] = {}
        for item_id, answer in answers.items():
            request = by_id[item_id]
            text = flatten_line(answer)
            if not text:
                continue
            accepted[request.position] = text
            if cache is not None:
                cache.put(request.text, request.max_chars, text)
        with lock:
            results.update(accepted)

    map_in_parallel(
        list(enumerate(batches, start=1)),
        lambda numbered: run_batch(*numbered),
        workers=config.concurrency,
    )
    return results


def overrunning(items: list[FitItem], durations: dict[int, float], tempo: float) -> list[FitItem]:
    return [
        item for item in items if durations.get(item.position, 0.0) > item.slot * max(1.0, tempo)
    ]


def worst_tempo(items: list[FitItem], durations: dict[int, float]) -> float:
    """The speed-up the tightest line needs to stay inside its slot."""
    needed = (durations.get(item.position, 0.0) / item.slot for item in items if item.slot > 0)
    return max(1.0, max(needed, default=1.0))


def build_shorten_requests(
    items: list[FitItem],
    *,
    texts: dict[int, str],
    rate: SpeechRate,
    tempo: float,
    min_ratio: float,
) -> list[ShortenRequest]:
    requests: list[ShortenRequest] = []
    for item in items:
        text = texts[item.position]
        current = speech_units(text)
        if current <= 0:
            continue
        # The line may still be sped up to `tempo`, so it only has to get short
        # enough for that; and asking for a fraction of a line back yields
        # telegraphic notes, so min_ratio caps what one round may cut.
        budget = max(rate.units_for(item.slot * tempo), current * min_ratio)
        if budget > current * MIN_GAIN_RATIO:
            continue
        requests.append(
            ShortenRequest(
                position=item.position,
                text=text,
                # The model counts characters, not spoken beats, so the budget
                # is handed over in the same units it can check itself against.
                max_chars=max(1, round(text_length(text) * budget / current)),
                source_text=item.source_text,
            )
        )
    return requests


def fit_cues_to_slots(
    items: list[FitItem],
    *,
    measure: Measure,
    options: FitOptions,
    cache: ShortenCache | None = None,
    debug_dir: Path | None = None,
) -> FitOutcome:
    texts = {item.position: item.text for item in items}
    durations = measure(texts)
    rate = measure_speech_rate(texts, durations)
    print(
        f"  Voice speed: {rate.units_per_second:.1f} characters per second "
        f"(plus {rate.overhead:.2f}s of padding per clip)"
    )

    over_before = len(overrunning(items, durations, options.tempo))
    tempo_before = worst_tempo(items, durations)
    shortened: dict[int, str] = {}
    rounds_used = 0

    for _ in range(max(0, options.rounds)):
        over = overrunning(items, durations, options.tempo)
        if not over:
            break
        requests = build_shorten_requests(
            over, texts=texts, rate=rate, tempo=options.tempo, min_ratio=options.min_ratio
        )
        if not requests:
            break

        rounds_used += 1
        print(f"  Round {rounds_used}: rewriting {len(requests)} line(s) that overrun their slot")
        answers = request_shorter_texts(
            requests, config=options.translation, cache=cache, debug_dir=debug_dir
        )
        accepted = {
            position: answer
            for position, answer in answers.items()
            if speech_units(answer) < speech_units(texts[position])
        }
        if not accepted:
            break

        texts.update(accepted)
        shortened.update(accepted)
        durations = measure(texts)

    return FitOutcome(
        texts=texts,
        shortened=shortened,
        durations=durations,
        rate=rate,
        over_before=over_before,
        over_after=len(overrunning(items, durations, options.tempo)),
        tempo_before=tempo_before,
        tempo_after=worst_tempo(items, durations),
        rounds_used=rounds_used,
    )
