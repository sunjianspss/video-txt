from __future__ import annotations

import json

import pytest

from video_txt.subtitles import SubtitleCue, parse_srt_text
from video_txt.translate import (
    ApiHttpError,
    ApiNetworkError,
    ContextCue,
    PartialStore,
    ResponseFormatError,
    TranslationConfig,
    TranslationError,
    _apply_translations,
    backoff_delay,
    build_context,
    build_messages,
    build_partial_meta,
    is_retryable,
    parse_translation_json,
    partial_path_for,
    strip_code_fence,
)


def make_config(**overrides) -> TranslationConfig:
    defaults = dict(base_url="https://example.test", api_key="k", model="test-model")
    return TranslationConfig(**{**defaults, **overrides})


def test_parse_translation_json_accepts_items_object():
    payload = json.dumps({"items": [{"id": "1", "text": "你好"}, {"id": "2", "text": "再见"}]})
    assert parse_translation_json(payload) == {"1": "你好", "2": "再见"}


def test_parse_translation_json_accepts_bare_list_and_code_fence():
    payload = '```json\n[{"id": 1, "text": "你好"}]\n```'
    assert parse_translation_json(payload) == {"1": "你好"}


def test_parse_translation_json_rejects_bad_payloads():
    with pytest.raises(ResponseFormatError, match="valid JSON"):
        parse_translation_json("not json at all")
    with pytest.raises(ResponseFormatError, match="items"):
        parse_translation_json(json.dumps({"result": "nope"}))
    with pytest.raises(ResponseFormatError, match="without an id"):
        parse_translation_json(json.dumps({"items": [{"text": "x"}]}))
    with pytest.raises(ResponseFormatError, match="non-string"):
        parse_translation_json(json.dumps({"items": [{"id": "1", "text": 5}]}))


def test_strip_code_fence_leaves_plain_text_alone():
    assert strip_code_fence('  {"a": 1}  ') == '{"a": 1}'


def test_build_messages_carries_ids_terms_and_context():
    batch = [("1", SubtitleCue("1", "00:00:01,000 --> 00:00:02,000", ["hello"]))]
    messages = build_messages(
        batch,
        config=make_config(preserve_terms=["Claude"], note="casual"),
        context_before=[ContextCue(source="prior line", translation="上一句")],
    )
    payload = json.loads(messages[1]["content"])
    assert payload["items"] == [{"id": "1", "text": "hello"}]
    assert payload["preserve_terms"] == ["Claude"]
    assert payload["extra_note"] == "casual"
    assert payload["context_before"] == [{"source": "prior line", "translation": "上一句"}]
    assert messages[0]["role"] == "system"


def test_build_messages_omits_context_when_empty():
    batch = [("1", SubtitleCue("1", "00:00:01,000 --> 00:00:02,000", ["hello"]))]
    payload = json.loads(
        build_messages(batch, config=make_config(), context_before=[])[1]["content"]
    )
    assert "context_before" not in payload


def test_build_context_walks_backwards_and_skips_empty_cues():
    cues = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\none\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\ntwo\n\n"
        "4\n00:00:07,000 --> 00:00:08,000\nthree"
    )
    context = build_context(cues=cues, first_id="4", translations={"3": "二"}, limit=2)
    assert [(item.source, item.translation) for item in context] == [("one", None), ("two", "二")]


def test_build_context_returns_nothing_for_the_first_batch():
    cues = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\none")
    assert build_context(cues=cues, first_id="1", translations={}, limit=3) == []
    assert build_context(cues=cues, first_id="1", translations={}, limit=0) == []


def test_partial_store_round_trip(tmp_path):
    cues = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\none")
    config = make_config()
    store = PartialStore(partial_path_for(tmp_path / "out.srt"), build_partial_meta(cues, config))
    assert store.load() == {}
    store.append({"1": "一"})
    store.append({"2": "二"})
    assert store.load() == {"1": "一", "2": "二"}
    store.discard()
    assert not store.path.exists()


def test_partial_store_ignores_a_stale_file(tmp_path):
    cues = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\none")
    path = partial_path_for(tmp_path / "out.srt")
    PartialStore(path, build_partial_meta(cues, make_config())).append({"1": "一"})

    other_cues = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\nchanged")
    assert PartialStore(path, build_partial_meta(other_cues, make_config())).load() == {}


def test_partial_store_replaces_a_stale_file_so_new_appends_survive(tmp_path):
    cues = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\none")
    path = partial_path_for(tmp_path / "out.srt")
    PartialStore(path, build_partial_meta(cues, make_config())).append({"1": "旧的"})

    changed = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\nchanged")
    store = PartialStore(path, build_partial_meta(changed, make_config()))
    assert store.load() == {}
    store.append({"1": "新的"})
    assert store.load() == {"1": "新的"}


def test_partial_store_discards_an_unreadable_file(tmp_path):
    cues = parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\none")
    path = partial_path_for(tmp_path / "out.srt")
    path.write_text("not a json header\n", encoding="utf-8")

    store = PartialStore(path, build_partial_meta(cues, make_config()))
    assert store.load() == {}
    store.append({"1": "一"})
    assert store.load() == {"1": "一"}


def test_partial_path_and_meta_shape(tmp_path):
    assert partial_path_for(tmp_path / "a.zh.srt").name == "a.zh.srt.partial.jsonl"
    meta = build_partial_meta(
        parse_srt_text("1\n00:00:01,000 --> 00:00:02,000\none"), make_config()
    )
    assert meta["cue_count"] == 1
    assert meta["model"] == "test-model"


def test_apply_translations_keeps_empty_cues_and_requires_the_rest():
    cues = parse_srt_text(
        "1\n00:00:01,000 --> 00:00:02,000\n\n2\n00:00:03,000 --> 00:00:04,000\nhi"
    )
    result = _apply_translations(cues, {"2": "你好"})
    assert result[0].is_empty
    assert result[1].text == "你好"
    with pytest.raises(TranslationError, match="Missing translation"):
        _apply_translations(cues, {})


def test_retryable_classification():
    assert is_retryable(ApiHttpError(429, "slow down"))
    assert is_retryable(ApiHttpError(503, "unavailable"))
    assert is_retryable(ApiNetworkError("boom"))
    assert is_retryable(ResponseFormatError("bad json"))
    assert not is_retryable(ApiHttpError(401, "bad key"))
    assert not is_retryable(ApiHttpError(400, "bad request"))
    assert not is_retryable(ValueError("unrelated"))


def test_backoff_delay_grows_and_respects_retry_after():
    config = make_config(backoff_base=1.0, backoff_cap=10.0)
    first = backoff_delay(1, config, ApiNetworkError("x"))
    third = backoff_delay(3, config, ApiNetworkError("x"))
    assert 0.7 <= first <= 1.3
    assert 2.9 <= third <= 5.1
    assert backoff_delay(1, config, ApiHttpError(429, "", retry_after=4.0)) == 4.0
    assert backoff_delay(1, config, ApiHttpError(429, "", retry_after=99.0)) == 10.0
