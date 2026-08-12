from __future__ import annotations

import json

import pytest

from video_txt import fit as fit_module
from video_txt.fit import (
    FitItem,
    FitOptions,
    ShortenCache,
    ShortenRequest,
    SpeechRate,
    build_shorten_messages,
    build_shorten_requests,
    fit_cues_to_slots,
    flatten_line,
    measure_speech_rate,
    request_shorter_texts,
    speech_units,
    text_length,
)
from video_txt.translate import TranslationConfig


def make_config(**overrides) -> TranslationConfig:
    defaults = dict(base_url="https://example.test", api_key="k", model="test-model")
    return TranslationConfig(**{**defaults, **overrides})


def four_chars_per_second(texts: dict[int, str]) -> dict[int, float]:
    return {position: len(text) / 4 for position, text in texts.items()}


def test_text_length_ignores_whitespace():
    assert text_length("你好 世界\n再见") == 6
    assert text_length("   ") == 0


def test_flatten_line_only_adds_a_space_between_western_words():
    assert flatten_line("你好\n世界") == "你好世界"
    assert flatten_line("hello\nworld") == "hello world"
    assert flatten_line("  \n \n ") == ""


def test_speech_units_charge_latin_words_by_syllable():
    assert speech_units("今天的内容就到这里") == 9
    assert speech_units("Claude Code 演示") == 6
    assert speech_units("Anthropic 更新") == 5
    # Acronyms are spelled out letter by letter, so they cost one beat each.
    assert speech_units("MCP 服务器") == 6


def test_units_for_removes_the_clip_padding_first():
    rate = SpeechRate(overhead=0.5, seconds_per_unit=0.25)
    assert rate.units_per_second == 4.0
    assert rate.units_for(2.5) == 8
    assert rate.seconds_for(8) == 2.5
    assert rate.units_for(0.4) == 1


def test_measure_speech_rate_recovers_the_padding_and_the_slope():
    texts = {position: "字" * position for position in range(4, 20)}
    durations = {position: 0.4 + 0.2 * position for position in texts}
    rate = measure_speech_rate(texts, durations)
    assert rate.overhead == pytest.approx(0.4)
    assert rate.seconds_per_unit == pytest.approx(0.2)


def test_measure_speech_rate_averages_when_there_are_too_few_clips():
    rate = measure_speech_rate({1: "四个字啊", 2: "八个字啊八个字啊"}, {1: 1.0, 2: 2.0})
    assert rate.overhead == 0.0
    assert rate.seconds_per_unit == pytest.approx(0.25)


def test_measure_speech_rate_falls_back_when_every_line_is_the_same_length():
    texts = {position: "四个字啊" for position in range(1, 11)}
    rate = measure_speech_rate(texts, dict.fromkeys(texts, 1.0))
    assert rate.overhead == 0.0
    assert rate.seconds_per_unit == pytest.approx(0.25)


def test_measure_speech_rate_without_samples_uses_a_default():
    assert measure_speech_rate({}, {}).units_per_second == pytest.approx(4.5)


def test_build_shorten_requests_skips_lines_that_already_fit():
    rate = SpeechRate(overhead=0.0, seconds_per_unit=0.25)
    items = [
        FitItem(position=1, text="字" * 9, slot=3.0),
        FitItem(position=2, text="字" * 20, slot=2.0),
    ]
    texts = {item.position: item.text for item in items}
    requests = build_shorten_requests(items, texts=texts, rate=rate, tempo=1.15, min_ratio=0.65)

    assert [request.position for request in requests] == [2]
    # The slot plus the tolerated speed-up pays for 9 beats, but one round may
    # not cut past 65% of the line.
    assert requests[0].max_chars == 13


def test_build_shorten_requests_hand_the_model_a_character_target():
    rate = SpeechRate(overhead=0.0, seconds_per_unit=0.25)
    item = FitItem(position=1, text="用 Claude Code 做一个很长很长的演示", slot=2.0)
    requests = build_shorten_requests(
        [item], texts={1: item.text}, rate=rate, tempo=1.0, min_ratio=0.65
    )

    # 21 characters but only 15 spoken beats, so the 9.75-beat budget scales
    # back up to 14 characters instead of the 8 a raw character count would ask for.
    assert requests[0].max_chars == 14


def test_build_shorten_messages_carries_the_budget_and_the_original_line():
    messages = build_shorten_messages(
        [ShortenRequest(position=7, text="很长的一句中文", max_chars=4, source_text="a long line")],
        config=make_config(preserve_terms=["MCP"], note="casual"),
    )
    payload = json.loads(messages[1]["content"])

    assert messages[0]["role"] == "system"
    assert payload["items"] == [
        {
            "id": "7",
            "text": "很长的一句中文",
            "current_chars": 7,
            "max_chars": 4,
            "source_text": "a long line",
        }
    ]
    assert payload["preserve_terms"] == ["MCP"]
    assert payload["extra_note"] == "casual"


def test_build_shorten_messages_omits_the_source_line_when_it_is_unknown():
    messages = build_shorten_messages(
        [ShortenRequest(position=1, text="中文", max_chars=1)], config=make_config()
    )
    assert "source_text" not in json.loads(messages[1]["content"])["items"][0]


def test_shorten_cache_survives_a_reload_and_keys_on_the_budget(tmp_path):
    path = tmp_path / "shortened.jsonl"
    cache = ShortenCache(path, namespace="model-a")
    assert cache.get("原句", 5) is None

    cache.put("原句", 5, "短句")
    assert cache.get("原句", 5) == "短句"
    assert ShortenCache(path, namespace="model-a").get("原句", 5) == "短句"
    assert ShortenCache(path, namespace="model-a").get("原句", 4) is None
    assert ShortenCache(path, namespace="model-b").get("原句", 5) is None


def test_request_shorter_texts_only_asks_for_the_uncached_lines(tmp_path, monkeypatch):
    cache = ShortenCache(tmp_path / "shortened.jsonl", namespace="n")
    cache.put("第一句原文", 3, "第一句")
    asked: list[list[str]] = []

    def fake_request_items(*, ids, build_request, **_kwargs):
        asked.append(list(ids))
        build_request(ids)
        return dict.fromkeys(ids, "短")

    monkeypatch.setattr(fit_module, "request_items", fake_request_items)
    result = request_shorter_texts(
        [
            ShortenRequest(position=1, text="第一句原文", max_chars=3),
            ShortenRequest(position=2, text="第二句原文", max_chars=3),
        ],
        config=make_config(),
        cache=cache,
    )

    assert result == {1: "第一句", 2: "短"}
    assert asked == [["2"]]
    assert cache.get("第二句原文", 3) == "短"


def test_fit_cues_to_slots_shortens_only_the_lines_that_overrun(monkeypatch):
    monkeypatch.setattr(
        fit_module,
        "request_shorter_texts",
        lambda requests, **_kwargs: {
            request.position: "字" * request.max_chars for request in requests
        },
    )
    items = [
        FitItem(position=1, text="字" * 12, slot=2.0),
        FitItem(position=2, text="字" * 4, slot=2.0),
    ]
    outcome = fit_cues_to_slots(
        items, measure=four_chars_per_second, options=FitOptions(translation=make_config())
    )

    assert outcome.over_before == 1
    assert outcome.over_after == 0
    assert outcome.shortened == {1: "字" * 9}
    assert outcome.texts == {1: "字" * 9, 2: "字" * 4}
    assert outcome.durations[1] == 2.25
    assert outcome.rounds_used == 1
    assert (outcome.tempo_before, outcome.tempo_after) == (1.5, 1.125)


def test_fit_cues_to_slots_stops_when_a_rewrite_gains_nothing(monkeypatch):
    monkeypatch.setattr(
        fit_module,
        "request_shorter_texts",
        lambda requests, **_kwargs: {request.position: request.text for request in requests},
    )
    outcome = fit_cues_to_slots(
        [FitItem(position=1, text="字" * 8, slot=1.0)],
        measure=four_chars_per_second,
        options=FitOptions(translation=make_config(), rounds=3),
    )

    assert outcome.shortened == {}
    assert outcome.rounds_used == 1
    assert outcome.over_after == 1


def test_fit_cues_to_slots_leaves_a_line_that_is_barely_over_to_the_speed_up(monkeypatch):
    monkeypatch.setattr(
        fit_module,
        "request_shorter_texts",
        lambda *_args, **_kwargs: pytest.fail("a 10% overrun is not worth an API call"),
    )
    # 2.5 s of speech where 2.3 s is allowed: atempo can absorb that.
    outcome = fit_cues_to_slots(
        [FitItem(position=1, text="字" * 10, slot=2.0)],
        measure=four_chars_per_second,
        options=FitOptions(translation=make_config()),
    )

    assert outcome.over_before == 1
    assert outcome.rounds_used == 0
    assert outcome.shortened == {}
    assert outcome.tempo_after == 1.25


def test_fit_cues_to_slots_leaves_a_fitting_timeline_alone(monkeypatch):
    monkeypatch.setattr(
        fit_module,
        "request_shorter_texts",
        lambda *_args, **_kwargs: pytest.fail("should not call the API"),
    )
    outcome = fit_cues_to_slots(
        [FitItem(position=1, text="字" * 4, slot=2.0)],
        measure=four_chars_per_second,
        options=FitOptions(translation=make_config()),
    )

    assert outcome.rounds_used == 0
    assert outcome.shortened == {}
    assert outcome.tempo_before == 1.0
