from __future__ import annotations

import json

import pytest

from video_txt import model_client
from video_txt.cli import main
from video_txt.screen import (
    SCHEMA,
    ScreenError,
    compile_plan,
    load_plan,
    sha256,
    translate_plan,
)
from video_txt.screen_scan import draft_captions, parse_ocr
from video_txt.translate import TranslationConfig


@pytest.fixture
def plan(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"source-video")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"source-frame")
    payload = {
        "schema": SCHEMA,
        "version": 1,
        "video": {"path": video.name, "sha256": sha256(video), "duration": 10},
        "captions": [
            {
                "id": "screen-1",
                "start": 0,
                "end": 3,
                "source": "Final subscriptions 3100",
                "translation": "最终订阅数3,100",
                "status": "ready",
                "evidence": [{"image": image.name, "sha256": sha256(image), "time": 2}],
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload))
    return path


def edit(plan, callback):
    payload = json.loads(plan.read_text())
    callback(payload)
    plan.write_text(json.dumps(payload))


def test_screen_and_speech_keep_order_and_aligned_translations(plan, tmp_path):
    source, target = tmp_path / "speech.srt", tmp_path / "speech.zh.srt"
    source.write_text("1\n00:00:04,000 --> 00:00:06,000\nIt is convenient.\n")
    target.write_text("1\n00:00:04,000 --> 00:00:06,000\n用起来很方便。\n")
    _, english, chinese, warnings = compile_plan(
        plan, speech_source=source, speech_translation=target
    )
    assert [c.text for c in chinese] == ["最终订阅数3,100", "用起来很方便。"]
    assert [c.time_range for c in english] == [c.time_range for c in chinese]
    assert not warnings


@pytest.mark.parametrize("failure", ["draft", "empty", "duration", "duplicate", "dense"])
def test_rejects_known_unsafe_render_plans(plan, failure):
    def change(p):
        item = p["captions"][0]
        if failure == "draft":
            item["status"] = "draft"
        elif failure == "empty":
            item["translation"] = ""
        elif failure == "duration":
            item["end"] = 11
        elif failure == "duplicate":
            p["captions"].append(dict(item))
        else:
            item["translation"] = "密集内容" * 25

    edit(plan, change)
    with pytest.raises(ScreenError):
        compile_plan(plan)


@pytest.mark.parametrize("filename", ["clip.mp4", "frame.jpg"])
def test_changed_source_or_evidence_is_not_silently_reused(plan, filename):
    (plan.parent / filename).write_bytes(b"changed")
    with pytest.raises(ScreenError, match="changed"):
        compile_plan(plan)


def test_colliding_speech_is_an_error_not_an_implicit_priority_rule(plan, tmp_path):
    speech = tmp_path / "speech.srt"
    speech.write_text("1\n00:00:02,000 --> 00:00:05,000\n重叠的口述\n")
    with pytest.raises(ScreenError, match="collision.*screen-1.*speech-1"):
        compile_plan(plan, speech_source=speech, speech_translation=speech)


def test_speech_pair_must_have_matching_times(plan, tmp_path):
    source, target = tmp_path / "speech.srt", tmp_path / "speech.zh.srt"
    source.write_text("1\n00:00:04,000 --> 00:00:05,000\nHello\n")
    target.write_text("1\n00:00:05,000 --> 00:00:06,000\n你好\n")
    with pytest.raises(ScreenError, match="timing mismatch"):
        compile_plan(plan, speech_source=source, speech_translation=target)


def test_omit_requires_reason_and_does_not_erase_evidence(plan):
    edit(plan, lambda p: p["captions"][0].update(status="omit", note=""))
    with pytest.raises(ScreenError, match="needs a note"):
        load_plan(plan)
    edit(plan, lambda p: p["captions"][0].update(note="Repeated logo; no new information"))
    payload, _ = load_plan(plan)
    assert payload["captions"][0]["evidence"]


def test_dynamic_values_keep_variants_and_latest_sample_stays_unreviewed():
    samples = [
        {"time": i, "image": f"{i}.jpg", "sha256": str(i), "text": text}
        for i, text in enumerate(["Subscriptions 3048", "Subscriptions 3100", "Subscriptions 3100"])
    ]
    captions = draft_captions(samples, 3)
    assert len(captions) == 1
    assert captions[0]["source"] == "Subscriptions 3100"
    assert captions[0]["variants"] == ["Subscriptions 3048", "Subscriptions 3100"]
    assert captions[0]["status"] == "draft"
    assert len(captions[0]["evidence"]) == 3


def test_empty_frame_breaks_a_caption_instead_of_covering_unrelated_scene():
    samples = [
        {"time": i, "image": f"{i}.jpg", "sha256": str(i), "text": text}
        for i, text in enumerate(["Hello", "", "Hello"])
    ]
    captions = draft_captions(samples, 3)
    assert [(c["start"], c["end"]) for c in captions] == [(0, 1), (2, 3)]


def test_ocr_keeps_positions_confidence_and_quoted_words():
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        '5\t1\t1\t1\t1\t20\t30\t80\t20\t94.5\t"hello"\n'
    )
    text, confidence, boxes = parse_ocr(tsv)
    assert text == '"hello"'
    assert confidence == 94.5 and boxes[0]["left"] == 20


def test_translate_reuses_real_translation_engine_but_never_promotes_review(
    plan, tmp_path, monkeypatch
):
    edit(plan, lambda p: p["captions"][0].update(translation="", status="draft"))
    before = plan.read_bytes()

    def api(*, config, messages, json_mode):
        payload = json.loads(messages[1]["content"])
        assert payload["items"][0]["text"] == "Final subscriptions 3100"
        result = {"items": [{"id": "1", "text": "最终订阅数为3100"}]}
        return {"choices": [{"message": {"content": json.dumps(result)}}]}

    monkeypatch.setattr(model_client, "post_chat_completion", api)
    output = tmp_path / "elsewhere" / "translated.json"
    config = TranslationConfig(base_url="http://fixture", api_key="", model="fixture")
    translate_plan(plan, output, config)
    translated, _ = load_plan(output)
    assert translated["captions"][0]["translation"] == "最终订阅数为3100"
    assert translated["captions"][0]["status"] == "draft"
    assert plan.read_bytes() == before
    with pytest.raises(ScreenError, match="status=ready"):
        compile_plan(output)
    with pytest.raises(ScreenError, match="already exists"):
        translate_plan(plan, output, config)


def test_translate_dry_run_has_no_model_or_output_side_effects(plan, tmp_path):
    output = tmp_path / "dry.json"
    assert main(["screen", "translate", str(plan), "-o", str(output), "--dry-run"]) == 0
    assert not output.exists()


def test_cli_reports_failed_review_as_nonzero(plan, capsys):
    edit(plan, lambda p: p["captions"][0].update(status="draft"))
    assert main(["screen", "validate", str(plan)]) == 1
    assert "status=ready" in capsys.readouterr().err
