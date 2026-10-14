from __future__ import annotations

import fcntl
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_txt import notes, notes_summary, notes_sync


@pytest.fixture
def sync(tmp_path, monkeypatch):
    vault, media, state = (tmp_path / name for name in ("vault", "media", "state"))
    vault.mkdir()
    media.mkdir()
    now = [1000.0]
    calls = []
    monkeypatch.setattr(notes, "probe_duration", lambda _: 20.0)
    monkeypatch.setattr(notes_sync.time, "time", lambda: now[0])

    def model(prompt, model, cwd):
        calls.append(prompt)
        blocks = list(dict.fromkeys(re.findall(r"\^(p-[a-f0-9]+)", prompt)))
        return {
            "question": "讨论什么？",
            "overview": "讨论具体按钮的颜色。",
            "claims": [{"text": "只针对这一个按钮。", "block": block} for block in blocks],
            "analysis": ["不能泛化。"],
            "questions": ["应回听核验。"],
            "related": [],
        }

    monkeypatch.setattr(notes_summary, "invoke_claude", model)

    def clip(name="interview", captions=True):
        video = media / f"{name}.mp4"
        video.write_bytes(b"media fixture")
        if captions:
            (media / f"{name}.en.srt").write_text(
                "1\n00:00:01,000 --> 00:00:07,000\nClaude changes this button.\n"
            )
            (media / f"{name}.zh.srt").write_text(
                "1\n00:00:01,000 --> 00:00:07,000\nClaude 改这个按钮。\n"
            )
        return video

    def run(action="run", **kwargs):
        return notes_sync.synchronize(vault, media, state, action=action, **kwargs)

    def jobs():
        return json.loads((state / "state.json").read_text())["jobs"]

    return SimpleNamespace(**locals())


def test_baseline_is_read_only_until_selected_and_rerun_is_idempotent(sync):
    sync.clip()
    assert sync.run()["counts"] == {"historical": 1}
    assert not list(sync.vault.iterdir())
    job_id = next(iter(sync.jobs()))
    sync.run("enqueue", job_id=job_id)
    sync.now[0] += 121
    assert sync.run()["processed"] == [job_id]
    job = sync.jobs()[job_id]
    assert job["status"] == "done"
    files = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sync.vault.rglob("*.md")}
    assert "#^p-" in (sync.vault / job["summary"]).read_text()
    assert sync.run()["processed"] == []
    assert files == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    assert len(sync.calls) == 1


def test_new_pair_waits_for_stability_then_runs_once(sync):
    sync.run("scan")
    sync.clip()
    assert sync.run()["counts"] == {"queued": 1}
    assert not sync.calls
    sync.now[0] += 121
    assert len(sync.run()["processed"]) == 1
    assert sync.run()["processed"] == []


def test_old_video_receiving_first_captions_gets_classified_and_auto_queued(sync):
    sync.clip(captions=False)
    assert sync.run()["counts"] == {"waiting": 1}
    sync.clip(captions=True)
    sync.run("scan")
    job = next(iter(sync.jobs().values()))
    assert job["enabled"] and job["status"] == "queued"
    assert job["category"] == "ClaudeAI"
    assert job["options"]["folder"] == "ClaudeAI/raw/videos"


def test_failed_summary_is_reported_no_automatic_retry_and_source_is_preserved(sync, monkeypatch):
    sync.run("scan")
    sync.clip()
    original = notes_summary.invoke_claude

    def fail(*args):
        raise notes.NotesError("fixture: Claude unavailable")

    monkeypatch.setattr(notes_summary, "invoke_claude", fail)
    report = sync.run(min_age=0)
    job_id = next(iter(sync.jobs()))
    assert report["failed"] == [job_id]
    job = sync.jobs()[job_id]
    assert job["status"] == "error"
    sources = {p: p.read_bytes() for p in sync.vault.rglob("raw/**/*.md")}
    assert len(sources) == 2
    monkeypatch.setattr(notes_summary, "invoke_claude", original)
    assert sync.run(min_age=0)["processed"] == []
    sync.run("retry", job_id=job_id)
    assert sync.run(min_age=0)["processed"] == [job_id]
    assert sources == {p: p.read_bytes() for p in sources}


def test_summary_cache_retries_partial_publish_without_second_model_call(sync, monkeypatch):
    sync.run("scan")
    sync.clip()
    original = notes_summary.append_reference
    monkeypatch.setattr(
        notes_summary,
        "append_reference",
        lambda *args: (_ for _ in ()).throw(OSError("fixture: disk")),
    )
    sync.run(min_age=0)
    job_id = next(iter(sync.jobs()))
    assert sync.jobs()[job_id]["status"] == "error"
    assert len(list(sync.vault.rglob("wiki/sources/*.md"))) == 1
    monkeypatch.setattr(notes_summary, "append_reference", original)
    sync.run("retry", job_id=job_id)
    assert sync.run(min_age=0)["processed"] == [job_id]
    assert len(sync.calls) == 1
    assert len(list(sync.vault.rglob("wiki/sources/*.md"))) == 1


def test_edited_summary_is_never_overwritten(sync):
    sync.run("scan")
    sync.clip()
    sync.run(min_age=0)
    job_id, job = next(iter(sync.jobs().items()))
    p = sync.vault / job["summary"]
    p.write_text(p.read_text() + "\n人工分析\n")
    before = p.read_bytes()
    sync.run("retry", job_id=job_id)
    assert sync.run(min_age=0)["counts"] == {"error": 1}
    assert "保留已有或人工改过" in sync.jobs()[job_id]["error"]
    assert p.read_bytes() == before
    assert len(sync.calls) == 1


def test_rebuilt_state_recognizes_quoted_generated_summary(sync):
    sync.run("scan")
    sync.clip()
    sync.run(min_age=0)
    job = next(iter(sync.jobs().values()))
    assert notes_sync.existing_summary(sync.vault, job["id"], job["revision"]) == job["summary"]
    state = {"jobs": {}}
    notes_sync.scan(sync.vault, sync.media, state, sync.now[0])
    assert state["jobs"][job["id"]]["status"] == "done"


def test_alignment_error_never_calls_model_or_writes_notes(sync):
    sync.run("scan")
    sync.clip()
    p = sync.media / "interview.zh.srt"
    p.write_text(p.read_text().replace("00:00:07,000", "00:00:09,000"))
    assert sync.run(min_age=0)["counts"] == {"error": 1}
    assert not sync.calls
    assert not list(sync.vault.rglob("*.md"))


def test_ignore_survives_input_changes_and_only_one_job_runs_per_pass(sync):
    sync.run("scan")
    sync.clip("one")
    sync.clip("two")
    sync.clip("three")
    sync.run("scan")
    job_id = next(iter(sync.jobs()))
    sync.run("ignore", job_id=job_id)
    job = sync.jobs()[job_id]
    Path(job["options"]["source"]).write_text("invalid new content")
    assert len(sync.run(min_age=0)["processed"]) == 1
    assert sync.jobs()[job_id]["status"] == "ignored"
    assert len(sync.calls) == 1


def test_interrupted_job_requires_retry_and_queue_lock_prevents_second_writer(sync):
    sync.run("scan")
    sync.clip()
    sync.run("scan")
    state_path = sync.state / "state.json"
    state = json.loads(state_path.read_text())
    job_id = next(iter(state["jobs"]))
    state["jobs"][job_id]["status"] = "summarizing"
    state_path.write_text(json.dumps(state))
    assert sync.run(min_age=0)["counts"] == {"error": 1}
    assert not sync.calls
    with (sync.state / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(notes.NotesError, match="正在处理"):
            sync.run("scan")


def test_nonexistent_evidence_is_rejected_before_publication():
    bad = {
        "question": "问题",
        "overview": "概述",
        "claims": [{"text": "观点", "block": "invented"}],
        "analysis": [],
        "questions": ["核实？"],
        "related": [],
    }
    with pytest.raises(notes.NotesError, match="不存在的来源段落"):
        notes_summary.validate_summary(bad, {"p-real"}, [])


def _claims(n):
    return [{"text": f"观点{i}", "block": f"p-{i}"} for i in range(n)]


def _reply(claims):
    return {
        "question": "问题",
        "overview": "概述",
        "claims": claims,
        "analysis": [],
        "questions": ["核实？"],
        "related": [],
    }


def test_overlong_claims_are_thinned_evenly_and_the_omission_is_disclosed():
    blocks = {f"p-{i}" for i in range(16)}
    data = notes_summary.validate_summary(_reply(_claims(16)), blocks, [], maximum=15)
    kept = [c["block"] for c in data["claims"]]
    assert len(kept) == 15 and kept[0] == "p-0" and kept[-1] == "p-15"
    assert kept == sorted(kept, key=lambda b: int(b[2:]))
    assert "模型返回 16 条观点" in data["questions"][-1]


def test_claims_far_over_the_cap_are_still_rejected_and_report_the_count():
    blocks = {f"p-{i}" for i in range(25)}
    with pytest.raises(notes.NotesError, match="3–10 条.*模型返回 25 条"):
        notes_summary.validate_summary(_reply(_claims(25)), blocks, [])


def test_within_cap_claims_are_untouched():
    data = notes_summary.validate_summary(_reply(_claims(10)), {f"p-{i}" for i in range(10)}, [])
    assert len(data["claims"]) == 10 and data["questions"] == ["核实？"]


def test_overlong_reply_costs_one_model_call_and_publishes(sync, monkeypatch):
    sync.run("scan")
    sync.clip()
    calls = []

    def model(prompt, model, cwd):
        calls.append(1)
        block = re.findall(r"\^(p-[a-f0-9]+)", prompt)[0]
        return _reply([{"text": "观点", "block": block}] * 12)

    monkeypatch.setattr(notes_summary, "invoke_claude", model)
    monkeypatch.setattr(notes_summary, "max_claims", lambda _: 10)
    report = sync.run(min_age=0)
    assert len(calls) == 1 and len(report["processed"]) == 1 and report["failed"] == []


def test_raw_reply_is_kept_even_when_it_is_not_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(notes_summary.shutil, "which", lambda _: __file__)
    monkeypatch.setattr(
        notes_summary.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="not json at all", stderr=""),
    )
    with pytest.raises(notes.NotesError, match="未返回有效摘要"):
        notes_summary.invoke_claude("prompt", "sonnet", tmp_path)
    assert (tmp_path / "summaries" / "last-raw.txt").read_text() == "not json at all"


def test_claim_cap_grows_for_long_sources_and_prompt_states_it():
    assert notes_summary.max_claims(5_000) == 10
    assert notes_summary.max_claims(notes_summary.LONG_SOURCE_CHARS) == 10
    assert notes_summary.max_claims(notes_summary.LONG_SOURCE_CHARS + 1) == 15
    blocks = {f"p-{i}" for i in range(15)}
    notes_summary.validate_summary(_reply(_claims(15)), blocks, [], maximum=15)


def test_long_source_prompt_asks_for_the_larger_range(sync, monkeypatch):
    monkeypatch.setattr(notes_summary, "LONG_SOURCE_CHARS", 10)
    sync.run("scan")
    sync.clip()
    sync.run(min_age=0)
    assert "3–15 条核心观点" in sync.calls[0]
