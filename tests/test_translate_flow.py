from __future__ import annotations

import json
import threading

import pytest

from video_txt import translate as translate_module
from video_txt.reuse import PreviousTranslation
from video_txt.subtitles import parse_srt, parse_srt_text, write_srt
from video_txt.terminology import Term, Terminology
from video_txt.translate import (
    ApiHttpError,
    PartialStore,
    TranslationConfig,
    build_partial_meta,
    partial_path_for,
    translate_subtitle_file,
)

SOURCE = "\n\n".join(
    f"{index}\n00:00:{index:02d},000 --> 00:00:{index + 1:02d},000\nline {index}"
    for index in range(1, 7)
)


@pytest.fixture
def srt(tmp_path):
    path = tmp_path / "clip.srt"
    path.write_text(SOURCE + "\n", encoding="utf-8")
    return path


def make_config(**overrides) -> TranslationConfig:
    defaults = dict(
        base_url="https://api.test",
        api_key="k",
        model="test-model",
        batch_chars=6,
        concurrency=3,
        retries=0,
    )
    return TranslationConfig(**{**defaults, **overrides})


class FakeApi:
    def __init__(self, *, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()
        self.requests: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, *, config, messages, json_mode) -> dict:
        payload = json.loads(messages[1]["content"])
        with self.lock:
            self.requests.append(payload)
        ids = [item["id"] for item in payload["items"]]
        if self.fail_ids & set(ids):
            raise ApiHttpError(500, "boom")
        items = [{"id": item["id"], "text": f"译文 {item['text']}"} for item in payload["items"]]
        return {"choices": [{"message": {"content": json.dumps({"items": items})}}]}


@pytest.fixture
def fake_api(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(translate_module, "post_chat_completion", api)
    return api


def test_translates_every_cue_across_concurrent_batches(srt, fake_api, tmp_path):
    output = tmp_path / "clip.zh.srt"
    translate_subtitle_file(input_path=srt, output_path=output, config=make_config())

    cues = parse_srt(output)
    assert [cue.text for cue in cues] == [f"译文 line {index}" for index in range(1, 7)]
    assert len(fake_api.requests) == 6
    assert not partial_path_for(output).exists()


def test_resume_skips_cues_already_in_the_partial_file(srt, fake_api, tmp_path):
    output = tmp_path / "clip.zh.srt"
    config = make_config()
    store = PartialStore(partial_path_for(output), build_partial_meta(parse_srt(srt), config))
    store.append({"1": "来自续传文件", "2": "同上"})

    translate_subtitle_file(input_path=srt, output_path=output, config=config)

    cues = parse_srt(output)
    assert cues[0].text == "来自续传文件"
    assert cues[1].text == "同上"
    assert cues[2].text == "译文 line 3"
    requested = {item["id"] for payload in fake_api.requests for item in payload["items"]}
    assert requested == {"3", "4", "5", "6"}


def test_only_the_changed_lines_are_translated_again(srt, fake_api, tmp_path):
    """A repair rewrites a few lines; the rest of the film is already translated."""
    previous_translation = tmp_path / "clip.zh.srt"
    write_srt(previous_translation, [
        cue for cue in parse_srt_text(
            "\n\n".join(
                f"{index}\n00:00:{index:02d},000 --> 00:00:{index + 1:02d},000\n旧译文 {index}"
                for index in range(1, 7)
            )
        )
    ])
    repaired = tmp_path / "clip.repaired.srt"
    repaired.write_text(
        SOURCE.replace("line 3", "what was really said").replace("line 4", "and this too") + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "clip.repaired.zh.srt"

    translate_subtitle_file(
        input_path=repaired,
        output_path=output,
        config=make_config(),
        previous=PreviousTranslation(source=srt, translation=previous_translation),
    )

    requested = {item["text"] for payload in fake_api.requests for item in payload["items"]}
    assert requested == {"what was really said", "and this too"}
    assert [cue.text for cue in parse_srt(output)] == [
        "旧译文 1",
        "旧译文 2",
        "译文 what was really said",
        "译文 and this too",
        "旧译文 5",
        "旧译文 6",
    ]


def test_carried_over_lines_are_not_reported_as_a_resumed_run(srt, fake_api, tmp_path, capsys):
    """--reuse is not a resume. Counting the carried-over lines as resumed work
    claims a partial file that was never written."""
    previous_translation = tmp_path / "clip.zh.srt"
    write_srt(previous_translation, list(parse_srt_text(
        "\n\n".join(
            f"{index}\n00:00:{index:02d},000 --> 00:00:{index + 1:02d},000\n旧译文 {index}"
            for index in range(1, 7)
        )
    )))
    repaired = tmp_path / "clip.repaired.srt"
    repaired.write_text(SOURCE.replace("line 4", "what was really said") + "\n", encoding="utf-8")

    translate_subtitle_file(
        input_path=repaired,
        output_path=tmp_path / "out.srt",
        config=make_config(),
        previous=PreviousTranslation(source=srt, translation=previous_translation),
    )

    output = capsys.readouterr().out
    assert "Reusing: 5 unchanged block(s)" in output
    assert "Resuming:" not in output


def test_carried_over_lines_are_the_context_the_new_ones_are_translated_in(
    srt, fake_api, tmp_path
):
    previous_translation = tmp_path / "clip.zh.srt"
    write_srt(previous_translation, list(parse_srt_text(
        "\n\n".join(
            f"{index}\n00:00:{index:02d},000 --> 00:00:{index + 1:02d},000\n旧译文 {index}"
            for index in range(1, 7)
        )
    )))
    repaired = tmp_path / "clip.repaired.srt"
    repaired.write_text(SOURCE.replace("line 4", "what was really said") + "\n", encoding="utf-8")

    translate_subtitle_file(
        input_path=repaired,
        output_path=tmp_path / "out.srt",
        config=make_config(context_cues=2),
        previous=PreviousTranslation(source=srt, translation=previous_translation),
    )

    context = [payload["context_before"] for payload in fake_api.requests]
    assert context and context[0][-1]["translation"] == "旧译文 3"


def test_a_batch_the_model_answers_short_is_split_instead_of_asked_again(
    srt, monkeypatch, tmp_path
):
    """Temperature is 0: the same question would only get the same short answer back."""

    class DropsLastId(FakeApi):
        def __call__(self, *, config, messages, json_mode) -> dict:
            response = super().__call__(config=config, messages=messages, json_mode=json_mode)
            items = json.loads(response["choices"][0]["message"]["content"])["items"]
            if len(items) > 1:
                items = items[:-1]
            content = json.dumps({"items": items})
            return {"choices": [{"message": {"content": content}}]}

    api = DropsLastId()
    monkeypatch.setattr(translate_module, "post_chat_completion", api)
    output = tmp_path / "clip.zh.srt"

    translate_subtitle_file(
        input_path=srt,
        output_path=output,
        config=make_config(batch_chars=1000, concurrency=1, retries=2),
    )

    assert [cue.text for cue in parse_srt(output)] == [f"译文 line {i}" for i in range(1, 7)]
    asked = [tuple(item["id"] for item in payload["items"]) for payload in api.requests]
    assert len(asked) == len(set(asked)), f"asked the same ids twice: {asked}"


def test_a_failed_batch_leaves_the_finished_ones_on_disk(srt, monkeypatch, tmp_path):
    api = FakeApi(fail_ids={"4"})
    monkeypatch.setattr(translate_module, "post_chat_completion", api)
    output = tmp_path / "clip.zh.srt"

    with pytest.raises(ApiHttpError):
        translate_subtitle_file(
            input_path=srt, output_path=output, config=make_config(concurrency=1)
        )

    assert not output.exists()
    resumable = PartialStore(
        partial_path_for(output), build_partial_meta(parse_srt(srt), make_config())
    ).load()
    assert resumable == {"1": "译文 line 1", "2": "译文 line 2", "3": "译文 line 3"}


def test_success_cleans_up_snapshots_from_batches_that_recovered(srt, monkeypatch, tmp_path):
    inner = FakeApi()
    served_garbage = threading.Event()

    def flaky(*, config, messages, json_mode):
        payload = json.loads(messages[1]["content"])
        ids = [item["id"] for item in payload["items"]]
        if "2" in ids and not served_garbage.is_set():
            served_garbage.set()
            return {"choices": [{"message": {"content": "not json"}}]}
        return inner(config=config, messages=messages, json_mode=json_mode)

    monkeypatch.setattr(translate_module, "post_chat_completion", flaky)
    output = tmp_path / "clip.zh.srt"

    translate_subtitle_file(
        input_path=srt,
        output_path=output,
        config=make_config(retries=1, concurrency=1, backoff_base=0.0),
    )

    assert parse_srt(output)[1].text == "译文 line 2"
    assert not (tmp_path / "clip.zh.debug").exists()


def test_success_also_clears_snapshots_left_by_an_earlier_run(srt, fake_api, tmp_path):
    debug_dir = tmp_path / "clip.zh.debug"
    debug_dir.mkdir()
    stale = debug_dir / "20260101-000000-0-batch-001-attempt-01-cues-1-raw-response.txt"
    stale.write_text("stale", encoding="utf-8")

    translate_subtitle_file(
        input_path=srt, output_path=tmp_path / "clip.zh.srt", config=make_config()
    )

    assert not debug_dir.exists()


def test_a_run_that_ultimately_fails_keeps_the_snapshots(srt, monkeypatch, tmp_path):
    def always_garbage(*, config, messages, json_mode):
        return {"choices": [{"message": {"content": "not json"}}]}

    monkeypatch.setattr(translate_module, "post_chat_completion", always_garbage)
    output = tmp_path / "clip.zh.srt"

    with pytest.raises(translate_module.TranslationError):
        translate_subtitle_file(
            input_path=srt,
            output_path=output,
            config=make_config(concurrency=1, backoff_base=0.0),
        )

    assert list((tmp_path / "clip.zh.debug").glob("*-raw-response.txt"))


def test_an_explicit_debug_dir_is_not_cleaned_up(srt, monkeypatch, tmp_path):
    inner = FakeApi()
    served_garbage = threading.Event()

    def flaky(*, config, messages, json_mode):
        payload = json.loads(messages[1]["content"])
        ids = [item["id"] for item in payload["items"]]
        if "2" in ids and not served_garbage.is_set():
            served_garbage.set()
            return {"choices": [{"message": {"content": "not json"}}]}
        return inner(config=config, messages=messages, json_mode=json_mode)

    monkeypatch.setattr(translate_module, "post_chat_completion", flaky)
    debug_dir = tmp_path / "kept-debug"

    translate_subtitle_file(
        input_path=srt,
        output_path=tmp_path / "clip.zh.srt",
        config=make_config(retries=1, concurrency=1, backoff_base=0.0),
        debug_dir=debug_dir,
    )

    assert list(debug_dir.glob("*-raw-response.txt"))


def test_later_batches_carry_earlier_lines_as_context(srt, fake_api, tmp_path):
    translate_subtitle_file(
        input_path=srt,
        output_path=tmp_path / "clip.zh.srt",
        config=make_config(concurrency=1, context_cues=2),
    )

    first, second, third = fake_api.requests[0], fake_api.requests[1], fake_api.requests[2]
    assert "context_before" not in first
    assert second["context_before"] == [{"source": "line 1", "translation": "译文 line 1"}]
    assert third["context_before"] == [
        {"source": "line 1", "translation": "译文 line 1"},
        {"source": "line 2", "translation": "译文 line 2"},
    ]


def test_empty_cues_are_kept_without_being_sent_to_the_api(tmp_path, fake_api):
    source = tmp_path / "gappy.srt"
    write_srt(
        source,
        parse_srt_text(
            "1\n00:00:01,000 --> 00:00:02,000\nspoken\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\n\n"
            "3\n00:00:03,000 --> 00:00:04,000\nalso spoken"
        ),
    )
    output = tmp_path / "gappy.zh.srt"
    translate_subtitle_file(input_path=source, output_path=output, config=make_config())

    cues = parse_srt(output)
    assert [cue.index for cue in cues] == ["1", "2", "3"]
    assert cues[1].is_empty
    requested = {item["id"] for payload in fake_api.requests for item in payload["items"]}
    assert requested == {"1", "3"}


def test_dry_run_writes_nothing(srt, fake_api, tmp_path):
    output = tmp_path / "clip.zh.srt"
    translate_subtitle_file(input_path=srt, output_path=output, config=make_config(), dry_run=True)
    assert not output.exists()
    assert fake_api.requests == []


def test_translation_normalizes_terms_and_writes_a_clean_audit_report(tmp_path, monkeypatch):
    source = tmp_path / "story.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nWoody is here.\n", encoding="utf-8"
    )
    output = tmp_path / "story.zh.srt"

    def translated_alias(**_kwargs):
        content = json.dumps({"items": [{"id": "1", "text": "伍迪来了。"}]})
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(translate_module, "post_chat_completion", translated_alias)
    terminology = Terminology(
        source_language="en",
        target_language="Simplified Chinese",
        terms=(Term("Woody", "胡迪", "word", ("伍迪",)),),
    )

    translate_subtitle_file(
        input_path=source,
        output_path=output,
        config=make_config(terminology=terminology),
    )

    assert parse_srt(output)[0].text == "胡迪来了。"
    report = json.loads(
        (tmp_path / "story.zh.translation-audit.json").read_text(encoding="utf-8")
    )
    assert report["schema"] == "video-txt.translation-audit"
    assert report["is_clean"] is True
    assert report["enforcement"]["changed_cue_count"] == 1


def test_unresolved_terminology_stops_after_preserving_the_translation_and_report(
    tmp_path, monkeypatch
):
    source = tmp_path / "story.srt"
    source.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nWoody is here.\n", encoding="utf-8"
    )
    output = tmp_path / "story.zh.srt"

    def missing_term(**_kwargs):
        content = json.dumps({"items": [{"id": "1", "text": "他来了。"}]})
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(translate_module, "post_chat_completion", missing_term)
    terminology = Terminology(
        source_language="en",
        target_language="Simplified Chinese",
        terms=(Term("Woody", "胡迪", "word"),),
    )

    with pytest.raises(translate_module.TranslationError, match="audit found 1 error"):
        translate_subtitle_file(
            input_path=source,
            output_path=output,
            config=make_config(terminology=terminology),
        )

    assert output.is_file()
    report = json.loads(
        (tmp_path / "story.zh.translation-audit.json").read_text(encoding="utf-8")
    )
    assert report["has_errors"] is True
    assert report["findings"][0]["code"] == "glossary_target_missing"
