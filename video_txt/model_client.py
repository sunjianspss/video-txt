"""Structured model requests shared by translation and duration fitting."""

from __future__ import annotations

import contextlib
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class RequestConfig(Protocol):
    base_url: str
    api_key: str
    model: str
    thinking: str
    reasoning_effort: str
    timeout: float
    retries: int
    backoff_base: float
    backoff_cap: float


class ModelError(RuntimeError):
    pass


RETRYABLE_STATUS = {408, 409, 425, 429}


class ApiHttpError(ModelError):
    def __init__(self, status: int, body: str, retry_after: float | None = None) -> None:
        super().__init__(f"API request failed with HTTP {status}: {body}")
        self.status = status
        self.body = body
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE_STATUS or 500 <= self.status < 600


class ApiNetworkError(ModelError):
    retryable = True


class ResponseFormatError(ModelError):
    retryable = True


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

    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = parsed.get("items")
    else:
        raise ResponseFormatError("API JSON must be an object or list.")
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
        if not text.strip():
            raise ResponseFormatError(f"API returned an empty translation for id {item_id}.")
        if item_id in translations:
            raise ResponseFormatError(f"API returned a duplicate id: {item_id}.")
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


def loaded_local_model(base_url: str, *, timeout: float = 10.0) -> str:
    """The chat model LM Studio will answer with.

    Local models are named after the file they were downloaded as, which nobody
    wants to retype, so ask the server instead. This uses LM Studio's own model
    listing rather than the OpenAI-compatible one: that one also lists models
    that are merely downloaded, and embedding models among them, so its first
    entry is often something that cannot translate a word.

    A loaded chat model wins. With none loaded — LM Studio unloads idle models
    when a JIT TTL is set — the one downloaded chat model is just as good: the
    first request loads it (a few seconds), and the 500s that LM Studio hands the
    requests racing that load are retried like any other 5xx. Several downloaded
    and none loaded is a guess we do not make.
    """
    host = base_url.rstrip("/").removesuffix("/v1")
    url = host + "/api/v0/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ModelError(f"Could not ask {url} which model is loaded: {exc}") from exc

    chat = [
        entry
        for entry in payload.get("data") or []
        if entry.get("id") and entry.get("type") in {"llm", "vlm"}
    ]
    loaded = [entry["id"] for entry in chat if entry.get("state") == "loaded"]
    if loaded:
        return loaded[0]
    downloaded = [entry["id"] for entry in chat]
    if len(downloaded) == 1:
        print(
            f"No chat model loaded; {downloaded[0]} is the only one downloaded, "
            "LM Studio will load it on the first request.",
            file=sys.stderr,
        )
        return downloaded[0]
    if downloaded:
        raise ModelError(
            f"{host} has no chat model loaded and several downloaded "
            f"({', '.join(downloaded)}); pass --model to pick one."
        )
    raise ModelError(f"{host} has no chat model loaded or downloaded.")


def post_chat_completion(
    *,
    config: RequestConfig,
    messages: list[dict[str, str]],
    json_mode: JsonModeState,
) -> dict:
    payload: dict[str, object] = {
        "model": config.model,
        "temperature": 0,
        "messages": messages,
    }
    if config.thinking and config.thinking != "auto":
        payload["thinking"] = {"type": config.thinking}
    if config.reasoning_effort and config.reasoning_effort != "auto":
        payload["reasoning_effort"] = config.reasoning_effort
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
            try:
                return json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ResponseFormatError(
                    f"API returned an invalid JSON response body: {exc}"
                ) from exc
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


def backoff_delay(attempt: int, config: RequestConfig, error: Exception) -> float:
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
    config: RequestConfig,
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
            if isinstance(exc, ResponseFormatError) and len(ids) > 1:
                # Not bad luck: temperature is 0, so asking the same question again
                # gets the same wrong answer back. Smaller questions are the way out,
                # and the split below is right there. A single id has nothing to split,
                # so that one keeps its retries.
                break
            if attempt <= config.retries:
                time.sleep(backoff_delay(attempt, config, exc))

    if last_error is None:
        raise ModelError(f"The API returned neither a translation nor an error for {ids}.")
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
