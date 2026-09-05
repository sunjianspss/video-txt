from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .mux import MuxOptions, run_mux
from .quality import check_transcript
from .refine import RefineError, load_word_document, word_document_matches_subtitle
from .reuse import PreviousTranslation
from .revise import apply_revisions, load_revisions, revised_subtitle_path
from .subtitles import parse_srt, translated_subtitle_path, write_srt
from .terminology import (
    Terminology,
    audit_translation,
    translation_audit_path_for,
    write_translation_audit_report,
)
from .transcribe import TranscribeError, TranscribeOptions, run_transcribe
from .translate import TranslationConfig, TranslationError, translate_subtitle_file


@dataclass
class TranscribeStage:
    model: str | None = None
    backend: str = "auto"
    language: str | None = None
    device: str | None = None
    initial_prompt: str | None = None
    extra_args: list[str] = field(default_factory=list)
    refine_subtitles: bool = False
    audio_stream: int | None = None
    retranscribe: bool = False
    skip_transcript_check: bool = False


@dataclass
class TranslateStage:
    config: TranslationConfig
    config_loader: Callable[[], TranslationConfig] | None = None
    output_path: Path | None = None
    output_dir: Path | None = None
    debug_dir: Path | None = None
    resume: bool = True
    retranslate: bool = False
    reuse_if_exists: bool = False
    # An earlier source/translation pair whose unchanged lines are carried over.
    previous: PreviousTranslation | None = None

    def require_config(self) -> TranslationConfig:
        """Load API settings only when this stage actually has work to do."""
        if self.config_loader is not None:
            config = self.config_loader()
            self.config = config
            self.config_loader = None
        return self.config


def stage_prefix(label: str | None) -> str:
    return f"{label} " if label else ""


def ensure_source_subtitle(
    video: Path,
    *,
    subtitle: Path | None,
    output_dir: Path,
    stage: TranscribeStage,
    dry_run: bool = False,
    label: str | None = "[1/3]",
) -> Path:
    prefix = stage_prefix(label)
    source = resolve_source_subtitle(
        video,
        subtitle=subtitle,
        output_dir=output_dir,
        stage=stage,
        dry_run=dry_run,
        prefix=prefix,
    )
    if not dry_run and not stage.skip_transcript_check:
        report = check_transcript(source, language=stage.language, media_path=video)
        if report:
            raise TranscribeError(f"{report}\nStopping before the steps that cost time.")
    return source


def resolve_source_subtitle(
    video: Path,
    *,
    subtitle: Path | None,
    output_dir: Path,
    stage: TranscribeStage,
    dry_run: bool,
    prefix: str,
) -> Path:
    if subtitle is not None:
        if stage.refine_subtitles:
            raise TranscribeError(
                "--refine-subtitles cannot be used with --subtitle because an .srt "
                "does not contain Whisper word timestamps."
            )
        if not subtitle.is_file():
            raise TranscribeError(f"Subtitle file not found: {subtitle}")
        print(f"{prefix}Source subtitle: reuse {subtitle}")
        return subtitle

    expected = output_dir / f"{video.stem}.srt"
    word_document = output_dir / f"{video.stem}.words.json"
    if (
        stage.refine_subtitles
        and expected.is_file()
        and word_document.is_file()
        and valid_word_document_contents(word_document)
        and not word_document_matches_subtitle(word_document, expected)
        and not stage.retranscribe
    ):
        raise TranscribeError(
            f"Refined subtitle and word timestamps no longer match: {expected}. "
            "The subtitle may have been edited by hand or a previous rewrite was interrupted. "
            "Keeping it unchanged; pass --retranscribe to replace it."
        )
    refinement_ready = not stage.refine_subtitles or valid_word_document(
        word_document,
        expected,
    )
    if expected.is_file() and refinement_ready and not stage.retranscribe:
        print(f"{prefix}Transcribe: skip, reusing {expected}")
        return expected

    print(f"{prefix}Transcribe: running Whisper" + (" (dry run)" if dry_run else ""))
    return run_transcribe(
        TranscribeOptions(
            input_path=video,
            output_dir=output_dir,
            mode="transcribe",
            model=stage.model,
            output_format="srt",
            language=stage.language,
            backend=stage.backend,
            device=stage.device,
            initial_prompt=stage.initial_prompt,
            extra_args=stage.extra_args,
            refine_subtitles=stage.refine_subtitles,
            audio_stream=stage.audio_stream,
        ),
        dry_run=dry_run,
    )


def valid_word_document(path: Path, subtitle_path: Path) -> bool:
    return path.is_file() and word_document_matches_subtitle(path, subtitle_path)


def valid_word_document_contents(path: Path) -> bool:
    try:
        load_word_document(path)
    except RefineError:
        return False
    return True


def ensure_translated_subtitle(
    source_subtitle: Path,
    *,
    stage: TranslateStage,
    dry_run: bool = False,
    label: str | None = "[2/3]",
) -> Path:
    prefix = stage_prefix(label)
    output_path = stage.output_path
    if output_path is None:
        default_path = translated_subtitle_path(source_subtitle, stage.config.target_language)
        output_path = stage.output_dir / default_path.name if stage.output_dir else default_path
    source_changed = (
        source_subtitle.is_file()
        and output_path.is_file()
        and not stage.reuse_if_exists
        and source_subtitle.stat().st_mtime_ns > output_path.stat().st_mtime_ns
    )
    if output_path.is_file() and not source_changed and not stage.retranslate:
        print(f"{prefix}Translate: skip, reusing {output_path}")
        if source_subtitle.is_file():
            report = audit_translation(
                parse_srt(source_subtitle),
                parse_srt(output_path),
                stage.config.terminology,
            )
            report_path = translation_audit_path_for(output_path)
            if not dry_run:
                if report_path.exists():
                    print(f"{prefix}Translation audit report exists; keeping it unchanged.")
                else:
                    write_translation_audit_report(
                        source_path=source_subtitle,
                        translation_path=output_path,
                        terminology=stage.config.terminology,
                        audit=report,
                    )
            print(
                f"{prefix}Translation audit: {report.summary['error']} errors, "
                f"{report.summary['warning']} warnings"
                + (f" -> {report_path}" if not dry_run else " (dry run; report not written)")
            )
            if report.has_errors:
                raise TranslationError(
                    f"Reused translation failed audit with {report.summary['error']} error(s). "
                    "Pass --retranslate to regenerate it with the current terminology, "
                    f"or review: {report_path}"
                )
        return output_path

    report_path = translation_audit_path_for(output_path)
    if report_path.exists() and not stage.retranslate and not dry_run:
        raise TranslationError(
            f"Translation audit report already exists: {report_path}. "
            "Pass --retranslate to replace the translation and its generated audit report."
        )

    print(f"{prefix}Translate: calling the API" + (" (dry run)" if dry_run else ""))
    if dry_run and not source_subtitle.is_file():
        print(f"Would translate {source_subtitle} -> {output_path}")
        print()
        return output_path

    return translate_subtitle_file(
        input_path=source_subtitle,
        output_path=output_path,
        config=stage.require_config(),
        debug_dir=stage.debug_dir,
        resume=stage.resume,
        dry_run=dry_run,
        overwrite_audit=stage.retranslate,
        previous=stage.previous,
    )


def ensure_revised_subtitle(
    source_subtitle: Path,
    translated_subtitle: Path,
    *,
    revisions: Path | None,
    terminology: Terminology | None = None,
    dry_run: bool = False,
    label: str | None = None,
) -> Path:
    """Put the hand-corrected lines back on top of a freshly translated subtitle.

    The translation is regenerated on every run; the corrections are not. They
    are reapplied here rather than edited into the translation, so that the file
    a person maintains stays the one they wrote, and a retranslation costs them
    nothing.

    The revised subtitle is the one that gets muxed or spoken, so it is the one
    worth auditing. `translate` already wrote a report about its own output; this
    one describes what actually ships, with the hand-finalized lines marked as
    signed off rather than counted against it.
    """
    if revisions is None:
        return translated_subtitle

    prefix = stage_prefix(label)
    revision_set = load_revisions(revisions)
    if not (source_subtitle.is_file() and translated_subtitle.is_file()):
        # A dry run reaches here with nothing on disk yet: the anchors cannot be
        # checked, but the file itself has already been read and validated.
        print(
            f"{prefix}Revise: would apply {len(revision_set.revisions)} "
            f"hand-corrected line(s) from {revisions}"
        )
        return translated_subtitle

    source_cues = parse_srt(source_subtitle)
    result = apply_revisions(source_cues, parse_srt(translated_subtitle), revision_set)
    for item in result.drifted:
        print(
            f"{prefix}Anchor moved: cue {item.cue_index} is now {item.drift:.1f}s from "
            f"where {item.revision.source!r} used to be.",
            file=sys.stderr,
        )
    output = revised_subtitle_path(translated_subtitle)
    if dry_run:
        print(
            f"{prefix}Revise: would restore {result.changed_count} hand-corrected "
            f"line(s) -> {output}"
        )
        return translated_subtitle

    write_srt(output, result.cues)
    already = len(result.already_current)
    print(
        f"{prefix}Revise: restored {result.changed_count} hand-corrected line(s)"
        + (f", {already} already current" if already else "")
        + f" -> {output}"
    )

    audit = audit_translation(
        source_cues,
        result.cues,
        terminology,
        revised={item.position for item in result.applied},
    )
    report_path = write_translation_audit_report(
        source_path=source_subtitle,
        translation_path=output,
        terminology=terminology,
        audit=audit,
        overwrite=True,
    )
    accepted = audit.summary["accepted"]
    print(
        f"{prefix}Translation audit: {audit.summary['error']} errors, "
        f"{audit.summary['warning']} warnings"
        + (f", {accepted} accepted on revised lines" if accepted else "")
        + f" -> {report_path}"
    )
    if audit.has_errors:
        raise TranslationError(
            f"The revised translation failed audit with {audit.summary['error']} error(s). "
            f"Correct the line in {revisions}, fix the terminology, or review: {report_path}"
        )
    return output


def run_pipeline(
    *,
    video: Path,
    subtitle: Path | None,
    output_dir: Path,
    transcribe_stage: TranscribeStage,
    translate_stage: TranslateStage,
    mux_options_for: Callable[[Path], MuxOptions],
    revisions: Path | None = None,
    dry_run: bool = False,
) -> Path:
    source_subtitle = ensure_source_subtitle(
        video,
        subtitle=subtitle,
        output_dir=output_dir,
        stage=transcribe_stage,
        dry_run=dry_run,
    )
    translated_subtitle = ensure_translated_subtitle(
        source_subtitle, stage=translate_stage, dry_run=dry_run
    )
    translated_subtitle = ensure_revised_subtitle(
        source_subtitle,
        translated_subtitle,
        revisions=revisions,
        terminology=translate_stage.config.terminology,
        dry_run=dry_run,
    )
    options = mux_options_for(translated_subtitle)
    print(f"[3/3] Mux: {options.mux_mode} subtitles -> {options.video_output}")
    return run_mux(options, dry_run=dry_run)
