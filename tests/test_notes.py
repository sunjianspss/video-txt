from __future__ import annotations

import fcntl
import json
from dataclasses import replace
from pathlib import Path

import pytest

from video_txt import notes
from video_txt.cli import main

SOURCE = (
    "1\n00:00:01,000 --> 00:00:05,000\nThis is not\na replacement for judgment.\n\n"
    "2\n00:00:05,000 --> 00:00:12,000\nWe tested three examples.\n"
)
TRANSLATION = (
    "1\n00:00:01,000 --> 00:00:05,000\n这不能替代\n判断。\n\n"
    "2\n00:00:05,000 --> 00:00:12,000\n我们测试了三个例子。\n"
)


@pytest.fixture
def options(tmp_path, monkeypatch):
    monkeypatch.setattr(notes, "probe_duration", lambda _: 20.0)
    video = tmp_path / "interview.mp4"
    video.write_bytes(b"media")
    source = tmp_path / "interview.srt"
    source.write_text(SOURCE)
    translation = tmp_path / "interview.zh.srt"
    translation.write_text(TRANSLATION)
    vault = tmp_path / "vault"
    vault.mkdir()
    return notes.NotesOptions(
        video, source, translation, vault, "AI/raw/videos", "interview-1", "Interview: a test"
    )


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_export_preserves_text_timing_and_source_bytes_and_reuses_exact_revision(options):
    inputs = [p.read_bytes() for p in (options.source, options.translation)]
    first = notes.export_notes(options)
    original, translated = [Path(p).read_text() for p in first["documents"]]
    assert "This is not a replacement for judgment." in original
    assert "这不能替代判断。" in translated
    assert "00:00:01.000 — 00:00:12.000" in original
    assert "^p-" in translated
    before = snapshot(options.vault)
    assert notes.export_notes(options) == first
    assert snapshot(options.vault) == before
    assert [p.read_bytes() for p in (options.source, options.translation)] == inputs


def test_empty_cue_error_names_the_offending_cue_number(options):
    options.translation.write_text(
        "1\n00:00:01,000 --> 00:00:05,000\n这不能替代\n判断。\n\n"
        "2\n00:00:05,000 --> 00:00:12,000\n\n"
    )
    with pytest.raises(notes.NotesError, match=r"empty_cue（第 2 条）"):
        notes.export_notes(options)
    assert list(options.vault.iterdir()) == []


def test_dry_run_is_read_only_even_when_source_folder_does_not_exist(options):
    result = notes.export_notes(replace(options, dry_run=True))
    assert result["dry_run"]
    assert list(options.vault.iterdir()) == []


def test_known_misaligned_translation_fails_before_creating_notes(options):
    options.translation.write_text(TRANSLATION.replace("00:00:05,000", "00:00:06,000"))
    with pytest.raises(notes.NotesError, match="do not align"):
        notes.export_notes(options)
    assert list(options.vault.iterdir()) == []
    result = notes.export_notes(replace(options, alignment="independent"))
    assert "不表示逐句翻译对齐" in Path(result["documents"][0]).read_text()


def test_changed_subtitle_retains_old_quotes_and_reports_human_notes_stale(options):
    first = notes.export_notes(options)
    old_text = Path(first["documents"][1]).read_bytes()
    human = options.vault / "我的文章.md"
    human.write_text(
        f'---\ntype: source\nvideo_source_id: "interview-1"\n'
        f'video_revision: "{first["revision"]}"\n---\n我的判断保持原样。\n'
    )
    before = human.read_bytes()
    options.translation.write_text(TRANSLATION.replace("三个", "3 个"))
    second = notes.export_notes(options)
    assert first["revision"] != second["revision"]
    assert second["needs_review"] == ["我的文章.md"]
    assert human.read_bytes() == before
    assert Path(first["documents"][1]).read_bytes() == old_text


def test_human_edits_in_generated_copy_are_reported_not_overwritten(options):
    first = notes.export_notes(options)
    source_note = Path(first["documents"][0])
    source_note.write_text(source_note.read_text() + "\n我的批注\n")
    before = snapshot(options.vault)
    with pytest.raises(notes.NotesError, match="human modification"):
        notes.export_notes(options)
    assert snapshot(options.vault) == before
    options.translation.write_text(TRANSLATION.replace("三个", "3 个"))
    with pytest.raises(notes.NotesError, match="human modification"):
        notes.export_notes(options)
    assert snapshot(options.vault) == before


def test_failure_advancing_head_is_recoverable_without_duplicate_notes(options, monkeypatch):
    original_write = notes.write_json
    monkeypatch.setattr(notes, "write_json", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(notes.NotesError, match="disk"):
        notes.export_notes(options)
    assert not list(options.vault.rglob("latest.json"))
    assert len(list(options.vault.rglob("manifest.json"))) == 1
    monkeypatch.setattr(notes, "write_json", original_write)
    notes.export_notes(options)
    assert len(list(options.vault.rglob("manifest.json"))) == 1
    assert len(list(options.vault.rglob("latest.json"))) == 1


def test_mid_pair_write_failure_leaves_old_revision_intact(options, monkeypatch):
    first = notes.export_notes(options)
    before = snapshot(options.vault)
    options.translation.write_text(TRANSLATION.replace("三个", "3 个"))
    original = Path.write_bytes

    def fail_second(path, data):
        if path.name == "中文译文.md":
            raise OSError("second note failed")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_second)
    with pytest.raises(notes.NotesError, match="second note"):
        notes.export_notes(options)
    assert snapshot(options.vault) == before
    assert Path(first["documents"][0]).is_file()


def test_rollup_deduplicates_display_lines_but_preserves_spoken_repetitions():
    raw = (
        "1\n00:00:01,000 --> 00:00:02,000\n\nHi. Hi.\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nHi. Hi.\nWe tested it.\n\n"
        "3\n00:00:03,000 --> 00:00:04,000\nWe tested it.\n\n"
        "4\n00:00:04,000 --> 00:00:05,000\n \n\n"
        "5\n00:00:08,000 --> 00:00:09,000\nWe tested it.\n"
    )
    cues, report = notes.read_rollup(raw)
    assert [cue.text for cue in cues] == ["Hi. Hi.", "We tested it.", "We tested it."]
    assert report["input_cues"] == 5
    assert report["actions"][-1]["empty_display_cleared"]


@pytest.mark.parametrize("bad", ["../escape", "A|B", "", "x/y"])
def test_source_id_cannot_escape_destination(options, bad):
    with pytest.raises(notes.NotesError, match="source-id"):
        notes.export_notes(replace(options, source_id=bad))
    assert list(options.vault.iterdir()) == []


def test_concurrent_export_is_refused(options):
    destination = options.vault / options.folder / options.source_id
    destination.mkdir(parents=True)
    with (destination / ".export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(notes.NotesError, match="Another export"):
            notes.export_notes(options)


def test_moving_input_paths_does_not_create_a_second_source_revision(options):
    first = notes.export_notes(options)
    renamed = options.source.with_name("renamed.srt")
    options.source.rename(renamed)
    second = notes.export_notes(replace(options, source=renamed))
    assert first["revision"] == second["revision"]
    assert len(list(options.vault.rglob("manifest.json"))) == 1


def test_cli_has_no_model_requirement_and_leaves_a_valid_manifest(options, capsys):
    assert (
        main(
            [
                "export-notes",
                str(options.video),
                "--source",
                str(options.source),
                "--translation",
                str(options.translation),
                "--vault",
                str(options.vault),
                "--source-id",
                options.source_id,
                "--title",
                options.title,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    manifest = json.loads((Path(result["documents"][0]).parent / "manifest.json").read_text())
    assert manifest["audits"]["translation_pair"]["is_clean"]
    assert len(manifest["paragraphs"]["source"]) == 1


def test_zero_duration_and_dubbing_shortcuts_are_not_silently_accepted(options):
    options.source.write_text(SOURCE.replace("00:00:01,000", "00:00:05,000"))
    with pytest.raises(notes.NotesError, match="zero/reversed"):
        notes.export_notes(options)
    with pytest.raises(notes.NotesError, match="complete monolingual"):
        notes.export_notes(replace(options, translation=Path("clip.zh.fitted.srt")))
