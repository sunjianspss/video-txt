from __future__ import annotations

import json
import threading

import pytest

from video_txt import translate as translate_module
from video_txt.subtitles import parse_srt, parse_srt_text, write_srt
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
