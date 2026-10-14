from __future__ import annotations

import hashlib
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from .model_client import JsonModeState, cleanup_debug_dir, default_debug_dir, request_items
from .model_client import ModelError as TranslationError
from .model_client import loaded_local_model as loaded_local_model
from .parallel import map_in_parallel
from .reuse import PreviousTranslation, plan_reuse
from .subtitles import SubtitleCue, chunk_cues, parse_srt, serialize_srt, write_srt
from .terminology import (
    audit_translation,
    enforce_terminology,
    translation_audit_path_for,
    write_translation_audit_report,
)
from .translation_config import TranslationConfig as TranslationConfig


@dataclass
class ContextCue:
    source: str
    translation: str | None


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
            if isinstance(cue_id, str) and isinstance(text, str) and text.strip():
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


def build_partial_meta(cues: list[SubtitleCue], config: TranslationConfig) -> dict[str, object]:
    fingerprint = hashlib.sha1(serialize_srt(cues).encode("utf-8")).hexdigest()
    return {
        "kind": "video-txt.translation.partial",
        "version": 4,
        "source_sha1": fingerprint,
        "cue_count": len(cues),
        "base_url": config.base_url,
        "model": config.model,
        "target_language": config.target_language,
        "source_language": config.source_language,
        "preserve_terms": config.preserve_terms,
        "terminology": config.terminology.to_dict() if config.terminology else None,
        "note": config.note,
        "spoken": config.spoken,
        "batch_chars": config.batch_chars,
        "context_cues": config.context_cues,
        "thinking": config.thinking,
        "reasoning_effort": config.reasoning_effort,
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
            "must not survive as digits and punctuation. Keep an initialism as "
            "letters only where a listener in the target language hears those "
            "letters in ordinary speech, the way Chinese keeps CPU and GPS; where "
            "the target language says the thing in words instead, use the words, "
            "so a finance talk's AR and AP are read as the accounts they name. "
            "Anything in preserve_terms stays exactly as it is."
        )
    if config.terminology and config.terminology.terms:
        rules.append(
            "Whenever a source term listed in terminology appears, use its target value exactly."
        )

    payload: dict[str, object] = {
        "task": "Translate subtitle text",
        "target_language": config.target_language,
        "source_language": config.source_language or "auto-detect",
        "rules": rules,
        "preserve_terms": config.preserve_terms,
        "terminology": [
            {"source": term.source, "target": term.target} for term in config.terminology.terms
        ]
        if config.terminology
        else [],
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
    reused: dict[str, str] | None = None,
) -> list[SubtitleCue]:
    pending = [
        (str(position), cue) for position, cue in enumerate(cues, start=1) if not cue.is_empty
    ]
    translations: dict[str, str] = {}

    # Lines an earlier translation already covers. They also give the batches
    # around them their context, the same way finished batches do.
    if reused:
        translations.update(reused)

    if partial_store is not None:
        stored = partial_store.load()
        translations.update(stored)
        resumed = sum(1 for cue_id, _ in pending if cue_id in stored)
        if resumed:
            print(f"Resuming: {resumed}/{len(pending)} subtitle blocks already translated.")

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
    overwrite_audit: bool = False,
    previous: PreviousTranslation | None = None,
) -> Path:
    cues = parse_srt(input_path)
    translatable = [cue for cue in cues if not cue.is_empty]
    plan = plan_reuse(cues, previous) if previous is not None else None
    reused = plan.by_position if plan is not None else {}
    outstanding = [
        cue
        for position, cue in enumerate(cues, start=1)
        if not cue.is_empty and str(position) not in reused
    ]
    batches = chunk_cues(outstanding, config.batch_chars)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Target language: {config.target_language}")
    print(f"Model: {config.model}")
    print(f"Subtitle blocks: {len(cues)} ({len(translatable)} with text)")
    if plan is not None:
        print(
            f"Reusing: {plan.reused} unchanged block(s) from {plan.previous.translation}, "
            f"{plan.fresh} left to translate"
        )
    print(f"Batches: {len(batches)} (about {config.batch_chars} chars each)")
    print(f"Concurrency: {max(1, config.concurrency)}")
    if config.reasoning_effort and config.reasoning_effort != "auto":
        print(f"Reasoning effort: {config.reasoning_effort}")
    if config.source_language:
        print(f"Source language hint: {config.source_language}")
    if config.preserve_terms:
        print(f"Preserve terms: {', '.join(config.preserve_terms)}")
    if config.terminology:
        print(f"Terminology: {len(config.terminology.terms)} approved mappings")
    print(f"Translation audit: {translation_audit_path_for(output_path)}")
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
        reused=reused,
    )
    enforcement = (
        enforce_terminology(cues, translated, config.terminology) if config.terminology else None
    )
    if enforcement is not None:
        translated = enforcement.cues
    write_srt(output_path, translated)
    if store is not None:
        store.discard()
    if auto_debug_dir is not None:
        removed = cleanup_debug_dir(auto_debug_dir)
        if removed:
            print(f"Removed {removed} debug snapshot(s); every batch succeeded after retries.")
    print(f"Wrote translated subtitles to: {output_path}")
    audit = audit_translation(cues, translated, config.terminology)
    report_path = write_translation_audit_report(
        source_path=input_path,
        translation_path=output_path,
        terminology=config.terminology,
        audit=audit,
        enforcement=enforcement,
        overwrite=overwrite_audit,
    )
    changed = len(enforcement.changes) if enforcement is not None else 0
    if changed:
        print(f"Terminology normalized: {changed} subtitle block(s)")
    print(
        f"Translation audit: {audit.summary['error']} errors, "
        f"{audit.summary['warning']} warnings -> {report_path}"
    )
    if audit.has_errors:
        raise TranslationError(
            f"Translation audit found {audit.summary['error']} error(s). Review: {report_path}"
        )
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
