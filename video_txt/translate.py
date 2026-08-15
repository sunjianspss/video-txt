from __future__ import annotations

import contextlib
import hashlib
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    DEFAULT_BATCH_CHARS,
    DEFAULT_CONCURRENCY,
    DEFAULT_TARGET_LANGUAGE,
)
from .parallel import map_in_parallel
from .subtitles import SubtitleCue, chunk_cues, parse_srt, serialize_srt, write_srt

RETRYABLE_STATUS = {408, 409, 425, 429}


class TranslationError(RuntimeError):
    pass


class ApiHttpError(TranslationError):
    def __init__(self, status: int, body: str, retry_after: float | None = None) -> None:
        super().__init__(f"API request failed with HTTP {status}: {body}")
        self.status = status
        self.body = body
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE_STATUS or 500 <= self.status < 600


class ApiNetworkError(TranslationError):
    retryable = True


class ResponseFormatError(TranslationError):
    retryable = True


@dataclass
class TranslationConfig:
    base_url: str
    api_key: str
    model: str
    target_language: str = DEFAULT_TARGET_LANGUAGE
    source_language: str | None = None
    preserve_terms: list[str] = field(default_factory=list)
    note: str | None = None
    # The lines are written to be spoken by a voice, not read off the screen.
    spoken: bool = False
    batch_chars: int = DEFAULT_BATCH_CHARS
    retries: int = 2
    concurrency: int = DEFAULT_CONCURRENCY
    context_cues: int = 3
    timeout: float = 180.0
    backoff_base: float = 1.5
    backoff_cap: float = 30.0


@dataclass
class ContextCue:
    source: str
    translation: str | None


class JsonModeState:
    def __init__(self) -> None:
        self._supported = True
        self._lock = threading.Lock()

    @property
    def supported(self) -> bool:
        with self._lock:
            return self._supported

    def disable(self) -> None:
        with self._lock:
            self._supported = False


class PartialStore:
    def __init__(self, path: Path, meta: dict[str, object]) -> None:
        self.path = path
        self.meta = meta
        self._lock = threading.Lock()

    def load(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}

        lines = self.path.read_text(encoding="utf-8").splitlines()
        header: object = None
        if lines:
            try:
                header = json.loads(lines[0])
            except json.JSONDecodeError:
                header = None

        if header != self.meta:
            # Delete the unusable file so append() starts a fresh one with the
            # current header; otherwise new records would be appended under the
            # old header and never load again.
            if lines:
                print(
                    f"Ignoring stale resume file (input or settings changed): {self.path}",
                    file=sys.stderr,
                )
            self.path.unlink(missing_ok=True)
            return {}

        translations: dict[str, str] = {}
        for line in lines[1:]:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cue_id, text = record.get("id"), record.get("text")
            if isinstance(cue_id, str) and isinstance(text, str):
                translations[cue_id] = text
        return translations

    def append(self, translations: dict[str, str]) -> None:
        with self._lock:
            new_file = not self.path.is_file()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                if new_file:
                    handle.write(json.dumps(self.meta, ensure_ascii=False) + "\n")
                for cue_id, text in translations.items():
                    handle.write(
                        json.dumps({"id": cue_id, "text": text}, ensure_ascii=False) + "\n"
                    )

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


def partial_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.partial.jsonl")


def default_debug_dir(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.debug")


def cleanup_debug_dir(debug_dir: Path) -> int:
    """Drop raw-response snapshots; after a fully successful run they only
    describe failures the retries already recovered from."""
    if not debug_dir.is_dir():
        return 0
    removed = 0
    for path in debug_dir.glob("*-raw-response.txt"):
        path.unlink(missing_ok=True)
        removed += 1
    # A directory that will not empty holds files we did not write; leave it alone.
    with contextlib.suppress(OSError):
        debug_dir.rmdir()
    return removed


def build_partial_meta(cues: list[SubtitleCue], config: TranslationConfig) -> dict[str, object]:
    fingerprint = hashlib.sha1(serialize_srt(cues).encode("utf-8")).hexdigest()
    return {
        "kind": "video-txt.translation.partial",
        "version": 2,
        "source_sha1": fingerprint,
        "cue_count": len(cues),
        "base_url": config.base_url,
        "model": config.model,
        "target_language": config.target_language,
        "source_language": config.source_language,
        "preserve_terms": config.preserve_terms,
        "note": config.note,
        "spoken": config.spoken,
        "batch_chars": config.batch_chars,
        "context_cues": config.context_cues,
    }


def build_messages(
    batch: list[tuple[str, SubtitleCue]],
    *,
    config: TranslationConfig,
    context_before: list[ContextCue],
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a precise subtitle translator. "
        "Return JSON only. Do not add Markdown fences or commentary."
    )

    rules = [
        "Keep the same number of subtitle items.",
        "Keep each item's id exactly the same as the input.",
        "Translate only the text field.",
        "Preserve internal line breaks when they help readability.",
        "Do not merge or split subtitle items.",
        "Keep terminology consistent across the batch.",
        "context_before is only for continuity: never translate or return those lines.",
    ]
    if config.spoken:
        # Two rules, not one: asking for spoken phrasing and for spoken spelling in
        # the same sentence reliably gets the phrasing and drops the spelling.
        rules.append(
            "The translation will be read aloud by a voice-over, not shown as text: "
            "use natural spoken phrasing a narrator would actually say, and prefer "
            "short clauses over bookish wording."
        )
        rules.append(
            "Spell out whatever a voice cannot pronounce as written. Percent signs, "
            "currency amounts, maths symbols, units of measure and section labels "
            "become the words the target language says them as, so '30%' and '3(b)' "
            "must not survive as digits and punctuation. Initialisms that are "
            "normally said letter by letter, and anything in preserve_terms, stay "
            "exactly as they are."
        )

    payload: dict[str, object] = {
        "task": "Translate subtitle text",
        "target_language": config.target_language,
        "source_language": config.source_language or "auto-detect",
        "rules": rules,
        "preserve_terms": config.preserve_terms,
        "extra_note": config.note or "",
        "output_schema": {
            "items": [{"id": "same as input", "text": "translated subtitle text only"}]
        },
        "items": [{"id": cue_id, "text": cue.text} for cue_id, cue in batch],
    }
    if context_before:
        payload["context_before"] = [
            {"source": item.source, "translation": item.translation}
            if item.translation
            else {"source": item.source}
            for item in context_before
        ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()


def extract_message_content(response_data: dict) -> str:
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ResponseFormatError(f"Unexpected API response shape: {response_data}") from exc

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "output_text"}
            and item.get("text")
        ]
        if text_parts:
            return "\n".join(text_parts)

    raise ResponseFormatError(f"Unable to read message content from API response: {response_data}")


def parse_translation_json(raw_content: str) -> dict[str, str]:
    cleaned = strip_code_fence(raw_content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ResponseFormatError(f"API did not return valid JSON: {exc}") from exc

    items = parsed if isinstance(parsed, list) else parsed.get("items")
    if not isinstance(items, list):
        raise ResponseFormatError("API did not return an 'items' list.")

    translations: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ResponseFormatError("API returned a non-object item in the translation list.")
        item_id = str(item.get("id", "")).strip()
        text = item.get("text")
        if not item_id:
            raise ResponseFormatError("API returned an item without an id.")
        if not isinstance(text, str):
            raise ResponseFormatError(f"API returned a non-string translation for id {item_id}.")
        translations[item_id] = text
    return translations


def safe_debug_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return label[:48].strip("-") or "unknown"


def write_raw_response_debug(
    *,
    debug_dir: Path,
    batch_number: int | None,
    attempt: int,
    expected_ids: list[str],
    raw_content: str,
    error: Exception,
) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    batch_label = f"batch-{batch_number:03d}" if batch_number is not None else "batch-unknown"
    cue_range = (
        safe_debug_label(expected_ids[0])
        if len(expected_ids) == 1
        else f"{safe_debug_label(expected_ids[0])}-{safe_debug_label(expected_ids[-1])}"
    )
    filename = (
        f"{timestamp}-{time.time_ns()}-{batch_label}-"
        f"attempt-{attempt:02d}-cues-{cue_range}-raw-response.txt"
    )
    path = debug_dir / filename
    path.write_text(
        "\n".join(
            [
                f"error: {type(error).__name__}: {error}",
                f"batch_number: {batch_number if batch_number is not None else 'unknown'}",
                f"attempt: {attempt}",
                f"expected_ids: {', '.join(expected_ids)}",
                "",
                "---- raw API message content ----",
                raw_content,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def post_chat_completion(
    *,
    config: TranslationConfig,
    messages: list[dict[str, str]],
    json_mode: JsonModeState,
) -> dict:
    payload: dict[str, object] = {
        "model": config.model,
        "temperature": 0,
        "messages": messages,
    }
    using_json_mode = json_mode.supported
    if using_json_mode:
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        config.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if using_json_mode and exc.code == 400 and "response_format" in body:
            json_mode.disable()
            print("API rejected response_format; falling back to plain JSON prompting.")
            return post_chat_completion(config=config, messages=messages, json_mode=json_mode)
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        try:
            retry_seconds = float(retry_after) if retry_after else None
        except ValueError:
            retry_seconds = None
        raise ApiHttpError(exc.code, body, retry_seconds) from exc
    except urllib.error.URLError as exc:
        raise ApiNetworkError(f"Network error while calling the translation API: {exc}") from exc
    except TimeoutError as exc:
        raise ApiNetworkError(f"Timed out while calling the translation API: {exc}") from exc


def backoff_delay(attempt: int, config: TranslationConfig, error: Exception) -> float:
    if isinstance(error, ApiHttpError) and error.retry_after:
        return min(error.retry_after, config.backoff_cap)
    delay = config.backoff_base * (2 ** (attempt - 1))
    return min(delay, config.backoff_cap) * (0.75 + random.random() * 0.5)


def is_retryable(error: Exception) -> bool:
    if isinstance(error, ApiHttpError):
        return error.retryable
    return isinstance(error, (ApiNetworkError, ResponseFormatError))


def request_items(
    *,
    ids: list[str],
    build_request: Callable[[list[str]], list[dict[str, str]]],
    config: TranslationConfig,
    json_mode: JsonModeState,
    debug_dir: Path | None,
    batch_number: int | None = None,
) -> dict[str, str]:
    """Ask the model for one text per id, retrying and splitting until the ids line up."""
    messages = build_request(ids)

    last_error: Exception | None = None
    for attempt in range(1, config.retries + 2):
        raw_content = ""
        try:
            response_data = post_chat_completion(
                config=config, messages=messages, json_mode=json_mode
            )
            raw_content = extract_message_content(response_data)
            answers = parse_translation_json(raw_content)
            missing = [item_id for item_id in ids if item_id not in answers]
            extra = [item_id for item_id in answers if item_id not in ids]
            if missing or extra:
                raise ResponseFormatError(
                    "Translated batch ids did not match the input. "
                    f"Missing: {missing}; extra: {extra}"
                )
            return {item_id: answers[item_id] for item_id in ids}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if isinstance(exc, ResponseFormatError) and debug_dir is not None and raw_content:
                debug_path = write_raw_response_debug(
                    debug_dir=debug_dir,
                    batch_number=batch_number,
                    attempt=attempt,
                    expected_ids=ids,
                    raw_content=raw_content,
                    error=exc,
                )
                print(f"Saved invalid API response to: {debug_path}", file=sys.stderr)
                last_error = ResponseFormatError(f"{exc} Raw response saved to: {debug_path}")
            if not is_retryable(exc):
                break
            if attempt <= config.retries:
                time.sleep(backoff_delay(attempt, config, exc))

    if last_error is None:
        raise TranslationError(f"The API returned neither a translation nor an error for {ids}.")
    if len(ids) > 1 and isinstance(last_error, ResponseFormatError):
        midpoint = len(ids) // 2
        left, right = ids[:midpoint], ids[midpoint:]
        print(
            "Batch validation failed; retrying with smaller chunks "
            f"({len(left)} + {len(right)} subtitle blocks)."
        )
        merged: dict[str, str] = {}
        for half in (left, right):
            merged.update(
                request_items(
                    ids=half,
                    build_request=build_request,
                    config=config,
                    json_mode=json_mode,
                    debug_dir=debug_dir,
                    batch_number=batch_number,
                )
            )
        return merged

    raise last_error


def translate_batch(
    *,
    batch: list[tuple[str, SubtitleCue]],
    config: TranslationConfig,
    context_before: list[ContextCue],
    json_mode: JsonModeState,
    debug_dir: Path | None,
    batch_number: int | None = None,
) -> dict[str, str]:
    cues_by_id = dict(batch)
    return request_items(
        ids=[cue_id for cue_id, _ in batch],
        build_request=lambda ids: build_messages(
            [(cue_id, cues_by_id[cue_id]) for cue_id in ids],
            config=config,
            context_before=context_before,
        ),
        config=config,
        json_mode=json_mode,
        debug_dir=debug_dir,
        batch_number=batch_number,
    )


def build_context(
    *,
    cues: list[SubtitleCue],
    first_id: str,
    translations: dict[str, str],
    limit: int,
) -> list[ContextCue]:
    if limit <= 0:
        return []

    position = int(first_id) - 1
    context: list[ContextCue] = []
    for offset in range(position - 1, -1, -1):
        cue = cues[offset]
        if cue.is_empty:
            continue
        context.append(ContextCue(source=cue.text, translation=translations.get(str(offset + 1))))
        if len(context) >= limit:
            break
    return list(reversed(context))


def translate_cues(
    cues: list[SubtitleCue],
    config: TranslationConfig,
    *,
    debug_dir: Path | None = None,
    partial_store: PartialStore | None = None,
) -> list[SubtitleCue]:
    pending = [
        (str(position), cue) for position, cue in enumerate(cues, start=1) if not cue.is_empty
    ]
    translations: dict[str, str] = {}

    if partial_store is not None:
        translations.update(partial_store.load())
        reused = sum(1 for cue_id, _ in pending if cue_id in translations)
        if reused:
            print(f"Resuming: {reused}/{len(pending)} subtitle blocks already translated.")

    todo = [(cue_id, cue) for cue_id, cue in pending if cue_id not in translations]
    if not todo:
        return _apply_translations(cues, translations)

    batches = chunk_cues([cue for _, cue in todo], config.batch_chars)
    grouped: list[list[tuple[str, SubtitleCue]]] = []
    cursor = 0
    for batch in batches:
        grouped.append(todo[cursor : cursor + len(batch)])
        cursor += len(batch)

    total = len(grouped)
    lock = threading.Lock()
    json_mode = JsonModeState()
    completed = 0

    def run_batch(batch_number: int, batch: list[tuple[str, SubtitleCue]]) -> dict[str, str]:
        nonlocal completed
        with lock:
            snapshot = dict(translations)
        context_before = build_context(
            cues=cues,
            first_id=batch[0][0],
            translations=snapshot,
            limit=config.context_cues,
        )
        result = translate_batch(
            batch=batch,
            config=config,
            context_before=context_before,
            json_mode=json_mode,
            debug_dir=debug_dir,
            batch_number=batch_number,
        )
        if partial_store is not None:
            partial_store.append(result)
        with lock:
            translations.update(result)
            completed += 1
            print(f"Translated batch {completed}/{total} ({len(batch)} subtitle blocks).")
        return result

    workers = max(1, min(config.concurrency, total))
    print(
        f"Translating {len(todo)} subtitle blocks in {total} batches "
        f"with {workers} concurrent request{'s' if workers > 1 else ''}..."
    )
    map_in_parallel(
        list(enumerate(grouped, start=1)),
        lambda numbered: run_batch(*numbered),
        workers=workers,
    )
    return _apply_translations(cues, translations)


def translate_subtitle_file(
    *,
    input_path: Path,
    output_path: Path,
    config: TranslationConfig,
    debug_dir: Path | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> Path:
    cues = parse_srt(input_path)
    translatable = [cue for cue in cues if not cue.is_empty]
    batches = chunk_cues(translatable, config.batch_chars)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Target language: {config.target_language}")
    print(f"Model: {config.model}")
    print(f"Subtitle blocks: {len(cues)} ({len(translatable)} with text)")
    print(f"Batches: {len(batches)} (about {config.batch_chars} chars each)")
    print(f"Concurrency: {max(1, config.concurrency)}")
    if config.source_language:
        print(f"Source language hint: {config.source_language}")
    if config.preserve_terms:
        print(f"Preserve terms: {', '.join(config.preserve_terms)}")
    print()

    if dry_run:
        return output_path

    store = (
        PartialStore(partial_path_for(output_path), build_partial_meta(cues, config))
        if resume
        else None
    )
    auto_debug_dir = default_debug_dir(output_path) if debug_dir is None else None
    translated = translate_cues(
        cues,
        config,
        debug_dir=debug_dir or auto_debug_dir,
        partial_store=store,
    )
    write_srt(output_path, translated)
    if store is not None:
        store.discard()
    if auto_debug_dir is not None:
        removed = cleanup_debug_dir(auto_debug_dir)
        if removed:
            print(f"Removed {removed} debug snapshot(s); every batch succeeded after retries.")
    print(f"Wrote translated subtitles to: {output_path}")
    return output_path


def _apply_translations(cues: list[SubtitleCue], translations: dict[str, str]) -> list[SubtitleCue]:
    result: list[SubtitleCue] = []
    for position, cue in enumerate(cues, start=1):
        if cue.is_empty:
            result.append(cue)
            continue
        text = translations.get(str(position))
        if text is None:
            raise TranslationError(f"Missing translation for subtitle block {cue.index}.")
        result.append(cue.with_text(text))
    return result
