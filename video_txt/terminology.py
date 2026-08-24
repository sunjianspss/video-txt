from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .subtitles import SubtitleCue


class TerminologyError(ValueError):
    pass


@dataclass(frozen=True)
class Term:
    source: str
    target: str
    match: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Terminology:
    source_language: str | None
    target_language: str | None
    terms: tuple[Term, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_language": self.source_language,
            "target_language": self.target_language,
            "terms": [
                {
                    "source": term.source,
                    "target": term.target,
                    "match": term.match,
                    "aliases": list(term.aliases),
                }
                for term in self.terms
            ],
        }


@dataclass(frozen=True)
class TerminologyChange:
    position: int
    cue_index: str
    terms: tuple[str, ...]
    before: str
    after: str


@dataclass(frozen=True)
class EnforcementResult:
    cues: list[SubtitleCue]
    changes: tuple[TerminologyChange, ...]


@dataclass(frozen=True)
class TranslationFinding:
    code: str
    severity: str
    message: str
    source_position: int | None = None
    translation_position: int | None = None
    term_source: str | None = None
    expected_target: str | None = None

    def to_dict(self) -> dict[str, object]:
        values: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "source_position": self.source_position,
            "translation_position": self.translation_position,
            "term_source": self.term_source,
            "expected_target": self.expected_target,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class TranslationAudit:
    source_cue_count: int
    translation_cue_count: int
    findings: tuple[TranslationFinding, ...]

    @property
    def is_clean(self) -> bool:
        return not self.findings

    @property
    def has_errors(self) -> bool:
        return any(finding.severity == "error" for finding in self.findings)

    @property
    def summary(self) -> dict[str, object]:
        by_code: dict[str, int] = {}
        for finding in self.findings:
            by_code[finding.code] = by_code.get(finding.code, 0) + 1
        return {
            "error": sum(finding.severity == "error" for finding in self.findings),
            "warning": sum(finding.severity == "warning" for finding in self.findings),
            "by_code": by_code,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "source_cue_count": self.source_cue_count,
            "translation_cue_count": self.translation_cue_count,
            "is_clean": self.is_clean,
            "has_errors": self.has_errors,
            "summary": self.summary,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _pattern(value: str, *, word: bool) -> re.Pattern[str]:
    escaped = re.escape(value)
    if word:
        boundary = r"A-Za-z0-9_" if re.search(r"[A-Za-z]", value) else r"\w"
        escaped = rf"(?<![{boundary}]){escaped}(?![{boundary}])"
    return re.compile(escaped, re.IGNORECASE)


def _source_matches(text: str, term: Term) -> bool:
    return _pattern(term.source, word=term.match == "word").search(text) is not None


def _ends_sentence(text: str) -> bool:
    return re.search(r"[.!?…][\"'”’)]*\s*$", text) is not None


def _reflow_offsets(source_cues: list[SubtitleCue], cue_offset: int) -> range:
    start = cue_offset
    stop = cue_offset + 1
    if cue_offset > 0 and not _ends_sentence(source_cues[cue_offset - 1].text):
        start -= 1
    if cue_offset + 1 < len(source_cues) and not _ends_sentence(source_cues[cue_offset].text):
        stop += 1
    return range(start, stop)


def enforce_terminology(
    source_cues: list[SubtitleCue],
    translated_cues: list[SubtitleCue],
    terminology: Terminology,
) -> EnforcementResult:
    """Normalize approved forms within the one-cue subtitle reflow window."""
    result = list(translated_cues)
    changes: list[TerminologyChange] = []
    for position, translated_cue in enumerate(translated_cues, start=1):
        cue_offset = position - 1
        if cue_offset >= len(source_cues):
            continue
        source_context = "\n".join(
            source_cues[offset].text
            for offset in _reflow_offsets(source_cues, cue_offset)
            if offset < len(source_cues)
        )
        text = translated_cue.text
        applied: list[str] = []
        for term in sorted(terminology.terms, key=lambda item: len(item.source), reverse=True):
            if not _source_matches(source_context, term):
                continue
            forms = sorted((term.source, *term.aliases), key=len, reverse=True)
            for form in forms:
                if _normalized(form) == _normalized(term.target):
                    continue
                pattern = _pattern(form, word=form == term.source and term.match == "word")
                text, count = pattern.subn(
                    lambda _match, target=term.target: target,
                    text,
                )
                if count and term.source not in applied:
                    applied.append(term.source)
        if text != translated_cue.text:
            result[position - 1] = translated_cue.with_text(text)
            changes.append(
                TerminologyChange(
                    position=position,
                    cue_index=translated_cue.index,
                    terms=tuple(applied),
                    before=translated_cue.text,
                    after=text,
                )
            )
    return EnforcementResult(cues=result, changes=tuple(changes))


def _contains(text: str, value: str) -> bool:
    return _normalized(value) in _normalized(text)


def _looks_wholly_untranslated(source: str, translation: str) -> bool:
    normalized_source = re.sub(r"\s+", " ", _normalized(source)).strip()
    normalized_translation = re.sub(r"\s+", " ", _normalized(translation)).strip()
    words = re.findall(r"[a-z]+(?:['’-][a-z]+)?", normalized_source)
    return normalized_source == normalized_translation and (
        len(words) >= 2 or len(normalized_source) >= 12
    )


def _source_residue(source: str, translation: str) -> str | None:
    source_words = re.findall(r"[a-z]+(?:['’-][a-z]+)?", _normalized(source))
    translated_words = re.findall(r"[a-z]+(?:['’-][a-z]+)?", _normalized(translation))
    for size in range(min(6, len(source_words), len(translated_words)), 2, -1):
        translated_windows = {
            tuple(translated_words[offset : offset + size])
            for offset in range(len(translated_words) - size + 1)
        }
        for offset in range(len(source_words) - size + 1):
            window = tuple(source_words[offset : offset + size])
            phrase = " ".join(window)
            if len(phrase) >= 12 and window in translated_windows:
                return phrase
    return None


def audit_translation(
    source_cues: list[SubtitleCue],
    translated_cues: list[SubtitleCue],
    terminology: Terminology | None = None,
) -> TranslationAudit:
    """Compare a source/translation pair without changing either subtitle."""
    findings: list[TranslationFinding] = []
    if len(source_cues) != len(translated_cues):
        findings.append(
            TranslationFinding(
                code="cue_count_mismatch",
                severity="error",
                message=(
                    f"Source has {len(source_cues)} cues but translation has "
                    f"{len(translated_cues)}."
                ),
            )
        )

    terms = terminology.terms if terminology is not None else ()
    for position, (source, translated) in enumerate(
        zip(source_cues, translated_cues, strict=False), start=1
    ):
        if source.index != translated.index:
            findings.append(
                TranslationFinding(
                    code="cue_index_mismatch",
                    severity="error",
                    message=(
                        f"Cue {position} index changed from {source.index!r} "
                        f"to {translated.index!r}."
                    ),
                    source_position=position,
                    translation_position=position,
                )
            )
        if source.time_range != translated.time_range:
            findings.append(
                TranslationFinding(
                    code="cue_timing_mismatch",
                    severity="error",
                    message=f"Cue {source.index} timing differs from its translation.",
                    source_position=position,
                    translation_position=position,
                )
            )
        if not source.is_empty and translated.is_empty:
            findings.append(
                TranslationFinding(
                    code="translation_empty",
                    severity="error",
                    message=f"Cue {source.index} has source text but no translation.",
                    source_position=position,
                    translation_position=position,
                )
            )
            continue

        for term in terms:
            if not _source_matches(source.text, term):
                continue
            cue_offset = position - 1
            translation_context = "\n".join(
                translated_cues[offset].text
                for offset in _reflow_offsets(source_cues, cue_offset)
                if offset < len(translated_cues)
            )
            context_without_target = _pattern(term.target, word=False).sub(
                "", translation_context
            )
            aliases_found = [
                alias for alias in term.aliases if _contains(context_without_target, alias)
            ]
            if aliases_found:
                findings.append(
                    TranslationFinding(
                        code="glossary_alias",
                        severity="error",
                        message=(
                            f"Cue {source.index} uses {', '.join(aliases_found)}; "
                            f"the approved form is {term.target}."
                        ),
                        source_position=position,
                        translation_position=position,
                        term_source=term.source,
                        expected_target=term.target,
                    )
                )
            if not _contains(translation_context, term.target):
                findings.append(
                    TranslationFinding(
                        code="glossary_target_missing",
                        severity="error",
                        message=(
                            f"Cue {source.index} contains {term.source!r} but its translation "
                            f"does not contain the approved form {term.target!r}."
                        ),
                        source_position=position,
                        translation_position=position,
                        term_source=term.source,
                        expected_target=term.target,
                    )
                )

        wholly_untranslated = _looks_wholly_untranslated(source.text, translated.text)
        if wholly_untranslated:
            findings.append(
                TranslationFinding(
                    code="source_text_unchanged",
                    severity="warning",
                    message=f"Cue {source.index} appears to be unchanged source text.",
                    source_position=position,
                    translation_position=position,
                )
            )
        else:
            residue = _source_residue(source.text, translated.text)
            if residue:
                findings.append(
                    TranslationFinding(
                        code="source_text_residue",
                        severity="warning",
                        message=(
                            f"Cue {source.index} still contains the source fragment "
                            f"{residue!r}."
                        ),
                        source_position=position,
                        translation_position=position,
                    )
                )
        source_size = len(re.sub(r"\s+", "", source.text))
        translated_size = len(re.sub(r"\s+", "", translated.text))
        if translated_size > max(80, source_size * 3 + 20):
            findings.append(
                TranslationFinding(
                    code="translation_unusually_long",
                    severity="warning",
                    message=(
                        f"Cue {source.index} grew from {source_size} to "
                        f"{translated_size} non-space characters."
                    ),
                    source_position=position,
                    translation_position=position,
                )
            )

    return TranslationAudit(
        source_cue_count=len(source_cues),
        translation_cue_count=len(translated_cues),
        findings=tuple(findings),
    )


def translation_audit_path_for(translation_path: Path) -> Path:
    return translation_path.with_name(f"{translation_path.stem}.translation-audit.json")


def write_translation_audit_report(
    *,
    source_path: Path,
    translation_path: Path,
    terminology: Terminology | None,
    audit: TranslationAudit,
    enforcement: EnforcementResult | None = None,
    report_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    destination = report_path or translation_audit_path_for(translation_path)
    if destination.exists() and not overwrite:
        raise TerminologyError(
            f"Translation audit report already exists: {destination}. "
            "Explicitly allow replacement to overwrite it."
        )
    changes = enforcement.changes if enforcement is not None else ()
    payload = {
        "schema": "video-txt.translation-audit",
        "version": 1,
        "source": str(source_path),
        "translation": str(translation_path),
        "terminology": terminology.to_dict() if terminology is not None else None,
        "enforcement": {
            "changed_cue_count": len(changes),
            "changes": [
                {
                    "position": change.position,
                    "cue_index": change.cue_index,
                    "terms": list(change.terms),
                    "before": change.before,
                    "after": change.after,
                }
                for change in changes
            ],
        },
        **audit.to_dict(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _optional_language(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise TerminologyError(f"{key} must be a non-empty string when provided.")
    return value.strip() if isinstance(value, str) else None


def load_terminology(path: Path) -> Terminology:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise TerminologyError(f"Could not read terminology file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TerminologyError(f"Terminology file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TerminologyError("Terminology file must contain one JSON object.")
    if payload.get("schema") != "video-txt.terminology":
        raise TerminologyError("Terminology schema must be 'video-txt.terminology'.")
    if payload.get("version") != 1:
        raise TerminologyError("Terminology version must be 1.")
    raw_terms = payload.get("terms")
    if not isinstance(raw_terms, list):
        raise TerminologyError("Terminology terms must be a JSON list.")

    terms: list[Term] = []
    seen_sources: set[str] = set()
    approved_forms: dict[str, str] = {}
    for position, item in enumerate(raw_terms, start=1):
        if not isinstance(item, dict):
            raise TerminologyError(f"Term #{position} must be a JSON object.")
        source, target = item.get("source"), item.get("target")
        if not isinstance(source, str) or not source.strip():
            raise TerminologyError(f"Term #{position} source must be non-empty.")
        if not isinstance(target, str) or not target.strip():
            raise TerminologyError(f"Term #{position} target must be non-empty.")
        source, target = source.strip(), target.strip()
        source_key = _normalized(source)
        if source_key in seen_sources:
            raise TerminologyError(f"Term #{position} has a duplicate source: {source}")
        seen_sources.add(source_key)

        match = item.get("match", "phrase")
        if not isinstance(match, str) or match not in {"phrase", "word"}:
            raise TerminologyError(
                f"Term #{position} match must be 'phrase' or 'word', got: {match!r}"
            )
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise TerminologyError(f"Term #{position} aliases must be non-empty strings.")
        for form in (source, *(alias.strip() for alias in aliases)):
            form_key = _normalized(form)
            earlier_target = approved_forms.get(form_key)
            if earlier_target is not None and earlier_target != _normalized(target):
                raise TerminologyError(
                    f"Term #{position} makes form {form!r} ambiguous across target values."
                )
            approved_forms[form_key] = _normalized(target)
        terms.append(
            Term(
                source=source,
                target=target,
                match=match,
                aliases=tuple(alias.strip() for alias in aliases),
            )
        )

    return Terminology(
        source_language=_optional_language(payload, "source_language"),
        target_language=_optional_language(payload, "target_language"),
        terms=tuple(terms),
    )
