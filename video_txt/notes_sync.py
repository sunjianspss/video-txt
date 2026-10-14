"""One bounded queue pass; Obsidian owns scheduling, this module owns state and writes."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from .artifacts import write_json
from .notes import NotesError, NotesOptions, digest, export_notes, verify_revision
from .notes_summary import generate_summary, publish_summary

SCHEMA = "video-txt.notes-sync"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
CATEGORIES = ("ClaudeAI", "OpenAI", "AI", "技术与工具")
ACTIVE = {"exporting", "summarizing"}


def default_state_dir(vault: Path, source_dir: Path) -> Path:
    key = digest(f"{vault.resolve()}\n{source_dir.resolve()}".encode())[:12]
    return Path.home() / "Library/Application Support/sun-video-sync" / key


def signature(options: dict) -> str:
    records = []
    for name in ("video", "source", "translation"):
        p = Path(options.get(name, ""))
        if not p.is_file():
            records.append([name, "missing"])
            continue
        stat = p.stat()
        records.append(
            [
                name,
                str(p),
                stat.st_size,
                stat.st_mtime_ns,
                digest(p.read_bytes()) if name != "video" else "media",
            ]
        )
    return digest(json.dumps(records).encode())


def pair(video: Path) -> tuple[Path | None, Path | None, str]:
    base = video.with_suffix("")
    sources = [p for p in (Path(f"{base}.srt"), Path(f"{base}.en.srt")) if p.is_file()]
    revised = [Path(f"{base}{suffix}") for suffix in (".zh.revised.srt", ".zh-CN.revised.srt")]
    plain = [Path(f"{base}{suffix}") for suffix in (".zh.srt", ".zh-CN.srt")]
    translations = [p for p in revised if p.is_file()] or [p for p in plain if p.is_file()]
    if len(sources) > 1 or len(translations) > 1:
        return None, None, "同名字幕有多个候选，请在导出命令中明确选择"
    if not sources or not translations:
        return None, None, "等待同名原文 .srt（或 .en.srt）和中文字幕 .zh.srt"
    return sources[0], translations[0], ""


def category_for(title: str, source: Path | None) -> str:
    text = (
        source.read_text(encoding="utf-8-sig", errors="replace")[:8000] if source else ""
    ).lower()
    weighted = (title.lower() + " ") * 8 + text
    claude = sum(weighted.count(k) for k in ("claude", "anthropic", "cowork", "cherny"))
    openai = sum(weighted.count(k) for k in ("openai", "codex", "chatgpt", "gpt-"))
    return "ClaudeAI" if claude > openai else "OpenAI" if openai else "AI"


def existing_summary(vault: Path, source_id: str, revision: str) -> str | None:
    for category in CATEGORIES:
        for p in (vault / category / "wiki/sources").glob("*.md"):
            with p.open(encoding="utf-8") as handle:
                header = handle.read(3000).split("\n---", 1)[0]

            def matches(key: str, value: str, header: str = header) -> bool:
                match = re.search(rf"(?m)^{key}:\s*(.+)$", header)
                if not match:
                    return False
                raw = match[1].strip()
                return raw in (value, json.dumps(value), "'" + value + "'")

            if matches("video_source_id", source_id) and matches("video_revision", revision):
                return p.relative_to(vault).as_posix()
    return None


def adopt_exports(vault: Path, state: dict, now: float) -> None:
    """Preserve explicit mappings such as the differently named Boris subtitle pair."""
    for category in CATEGORIES:
        for head in (vault / category / "raw/videos").glob("*/latest.json"):
            receipt = json.loads(head.read_text())
            source_id = receipt.get("source_id")
            if source_id in state["jobs"]:
                continue
            revision = receipt.get("revision", "")
            if not re.fullmatch(r"[a-f0-9]{20}", revision):
                continue
            m = verify_revision(head.parent / revision, revision)
            meta = m["metadata"]
            options = {
                **meta,
                "video": m["video"],
                **m["inputs"],
                "vault": str(vault),
                "folder": head.parent.parent.relative_to(vault).as_posix(),
                "dry_run": False,
            }
            documents = [head.parent / revision / name for name in m["documents"]]
            documents.sort(key=lambda p: p.name != "原文.md")
            summary = existing_summary(vault, source_id, revision)
            state["jobs"][source_id] = {
                "id": source_id,
                "title": meta["title"],
                "category": category,
                "options": options,
                "explicit_mapping": True,
                "ready": True,
                "enabled": True,
                "signature": signature(options),
                "stable_since": now,
                "status": "done" if summary else "queued",
                "revision": revision,
                "documents": [p.relative_to(vault).as_posix() for p in documents],
                "summary": summary,
                "summary_kind": "existing",
                "updated": now,
            }


def scan(vault: Path, source_dir: Path, state: dict, now: float) -> None:
    first = not state.get("initialized")
    adopt_exports(vault, state, now)
    jobs = state["jobs"]
    by_video = {j["options"]["video"]: j for j in jobs.values()}
    aliases = {
        (j["options"].get("source"), j["options"].get("translation"))
        for j in jobs.values()
        if j.get("explicit_mapping")
    }
    for video in sorted(source_dir.iterdir()):
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if re.search(r"(?:[.-]zh(?:-burned|-subbed|-dubbed)?|\.dubbed)$", video.stem, re.I):
            continue
        video = video.resolve()
        job = by_video.get(str(video))
        if job and job.get("explicit_mapping"):
            ready = all(Path(job["options"][k]).is_file() for k in ("source", "translation"))
            issue = "" if ready else "已登记的字幕路径不存在，请恢复文件或重新登记"
        else:
            source, translated, issue = pair(video)
            if source and translated and (str(source), str(translated)) in aliases:
                continue
            ready = bool(source and translated)
            if job is None:
                identifier = "video-" + digest(str(video).encode())[:16]
                category = category_for(video.stem, source)
                job = {
                    "id": identifier,
                    "title": video.stem,
                    "category": category,
                    "enabled": not first,
                    "ready": False,
                    "updated": now,
                    "status": "historical" if first and ready else "waiting",
                    "options": asdict(
                        NotesOptions(
                            video,
                            source or Path("/missing-source"),
                            translated or Path("/missing-translation"),
                            vault,
                            f"{category}/raw/videos",
                            identifier,
                            video.stem,
                        )
                    ),
                }
                job["options"] = {
                    k: str(v) if isinstance(v, Path) else v for k, v in job["options"].items()
                }
                jobs[identifier] = job
            elif ready and not job.get("ready") and job["status"] != "ignored":
                # Old media receiving its first subtitle pair is new work.
                job["enabled"] = True
                if not job.get("revision") and not job.get("category_explicit"):
                    job["category"] = category_for(video.stem, source)
                    job["options"]["folder"] = f"{job['category']}/raw/videos"
            if ready:
                job["options"].update(source=str(source), translation=str(translated))
        current = signature(job["options"])
        if current != job.get("signature"):
            job.update(signature=current, stable_since=now, updated=now)
            if job["enabled"]:
                job["status"] = "queued" if ready else "waiting"
                job.pop("error", None)
        job["ready"], job["issue"] = ready, issue
        if ready and job["enabled"] and job["status"] == "missing":
            job["status"] = "queued"
        if not ready and job["enabled"]:
            job["status"] = "waiting"
    for job in jobs.values():
        if not Path(job["options"]["video"]).is_file():
            job.update(status="missing", issue="视频文件不在原路径，已暂停此项目")
    state.update(initialized=True, last_scan=now)


def snapshot_vault(vault: Path) -> None:
    script = vault / ".obsidian/plugins/sun-history/cli.js"
    if script.is_file():
        subprocess.run(
            ["node", str(script), "snapshot", "-m", "改前：视频资料自动入库"],
            cwd=vault,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )


def process_job(vault: Path, job: dict, state: dict, state_path: Path, model: str) -> None:
    snapshot_vault(vault)
    job.update(status="exporting", updated=time.time())
    write_json(state_path, state)
    values = dict(job["options"])
    for name in ("video", "source", "translation", "vault"):
        values[name] = Path(values[name])
    result = export_notes(NotesOptions(**values))
    job.update(
        revision=result["revision"],
        documents=[Path(p).relative_to(vault).as_posix() for p in result["documents"]],
        needs_review=result["needs_review"],
        status="summarizing",
        updated=time.time(),
    )
    write_json(state_path, state)
    payload = generate_summary(vault, job, model, state_path.parent)
    job["summary"] = publish_summary(vault, job, payload)
    job.update(status="done", summary_kind="automatic", updated=time.time(), error="")
    write_json(state_path, state)


def synchronize(
    vault: Path,
    source_dir: Path,
    state_dir: Path,
    *,
    action: str = "run",
    job_id: str | None = None,
    category: str | None = None,
    min_age: float = 120,
    model: str = "sonnet",
) -> dict:
    vault, source_dir, state_dir = vault.resolve(), source_dir.resolve(), state_dir.resolve()
    if not vault.is_dir() or not source_dir.is_dir():
        raise NotesError("知识库或视频目录不存在")
    if state_dir.is_relative_to(vault):
        raise NotesError("自动同步状态必须放在知识库外")
    if category is not None and category not in CATEGORIES:
        raise NotesError("未知分类")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    with (state_dir / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise NotesError("视频资料正在处理，稍后再试") from exc
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if (
                state.get("schema") != SCHEMA
                or state.get("version") != 1
                or state.get("vault") != str(vault)
                or state.get("source_dir") != str(source_dir)
            ):
                raise NotesError("状态与当前知识库/视频目录不匹配，保留原状态")
        else:
            state = {
                "schema": SCHEMA,
                "version": 1,
                "vault": str(vault),
                "source_dir": str(source_dir),
                "jobs": {},
                "initialized": False,
            }
        for job in state["jobs"].values():
            if job["status"] in ACTIVE:
                job.update(status="error", error="上次处理被中断，来源稿与摘要缓存保留；可重试")
        scan(vault, source_dir, state, time.time())
        if action in ("enqueue", "retry", "ignore"):
            if job_id not in state["jobs"]:
                raise NotesError("没有找到这个视频项目")
            job = state["jobs"][job_id]
            if category:
                if job.get("revision") and category != job["category"]:
                    raise NotesError("已入库来源保留原分类；移动需另行整理已有引用")
                job["category"] = category
                job["category_explicit"] = True
                job["options"]["folder"] = f"{category}/raw/videos"
            job.update(
                enabled=action != "ignore",
                status="ignored" if action == "ignore" else "queued" if job["ready"] else "waiting",
                error="",
            )
        write_json(state_path, state)
        processed, failed = [], []
        if action == "run":
            ready = [
                j
                for j in state["jobs"].values()
                if j["enabled"]
                and j["ready"]
                and j["status"] == "queued"
                and time.time() - j["stable_since"] >= min_age
            ]
            if ready:
                job = sorted(ready, key=lambda j: j["updated"])[0]
                try:
                    process_job(vault, job, state, state_path, model)
                    processed.append(job["id"])
                except Exception as exc:
                    job.update(status="error", error=str(exc)[-600:], updated=time.time())
                    write_json(state_path, state)
                    failed.append(job["id"])
        counts = {}
        for job in state["jobs"].values():
            counts[job["status"]] = counts.get(job["status"], 0) + 1
        return {
            "state_path": str(state_path),
            "processed": processed,
            "failed": failed,
            "counts": counts,
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument(
        "--action", choices=("scan", "run", "enqueue", "retry", "ignore"), default="run"
    )
    parser.add_argument("--job-id")
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument(
        "--min-age",
        type=float,
        default=120,
        help="Seconds since the last observed input change before processing.",
    )
    parser.add_argument("--model", default="sonnet")


def command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        result = synchronize(
            args.vault,
            args.source_dir,
            args.state_dir or default_state_dir(args.vault, args.source_dir),
            action=args.action,
            job_id=args.job_id,
            category=args.category,
            min_age=max(0, args.min_age),
            model=args.model,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise NotesError(str(exc)) from exc
