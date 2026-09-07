from __future__ import annotations

import json

import pytest

from video_txt.cli import main
from video_txt.draft import draft_terminology, draft_to_dict, scan_candidates
from video_txt.subtitles import parse_srt_text
from video_txt.terminology import Term, Terminology, TerminologyError, load_terminology


def cues(*lines: str):
    blocks = [
        f"{index}\n00:00:{index:02d},000 --> 00:00:{index + 1:02d},000\n{line}"
        for index, line in enumerate(lines, start=1)
    ]
    return parse_srt_text("\n\n".join(blocks))


def sources(candidates) -> list[str]:
    return [candidate.source for candidate in candidates]


def test_a_name_is_a_word_the_subtitle_capitalizes_mid_sentence():
    """No stopword list decides this. `Well` and `But` are capitalized only where a
    sentence starts; a name keeps its capital in the middle of one."""
    found = scan_candidates(
        cues(
            "Well, Barry is here.",
            "But Barry left.",
            "I told Barry so.",
            "Well, that went well.",
            "But it did not.",
        ),
        min_count=2,
    )

    assert sources(found) == ["Barry"]


def test_a_word_the_script_also_writes_in_lower_case_is_not_a_name():
    found = scan_candidates(
        cues(
            "Meet Hope, my friend.",
            "There is Hope for us.",
            "I lost all hope today.",
            "She had no hope left.",
            "Without hope we are done.",
        ),
        min_count=2,
    )

    assert sources(found) == []


def test_the_words_english_always_capitalizes_are_never_proposed():
    found = scan_candidates(
        cues(
            "Yes, I am Woody.",
            "And I think Woody knows.",
            "Well I said OK to Woody.",
            "So I told him OK.",
        ),
        min_count=2,
    )

    assert sources(found) == ["Woody"]


def test_one_name_is_one_candidate_however_the_line_capitalized_it():
    """A transcript writes `bonnie` sometimes. Splitting that into two candidates
    hides how often the name really appears and proposes it twice."""
    found = scan_candidates(
        cues(
            "Say hi to Bonnie now.",
            "Where is Bonnie going?",
            "Give Bonnie's toy back.",
            "I saw Bonnie leave.",
            "That is Bonnie there.",
            "It was bonnie who asked.",
        ),
        min_count=2,
    )

    # Five capitalized sightings, one of them possessive, one line in lower case.
    assert sources(found) == ["Bonnie"]
    assert found[0].count == 5


def test_a_comma_between_two_names_does_not_make_one_name():
    """`Oh, Lilypan!` is a greeting and a name. Counted as the phrase `Oh Lilypan`
    it both invents a term and steals the count the real name needed."""
    found = scan_candidates(
        cues(
            "Oh, Lilypan!",
            "Hi there, I am Lilypan.",
            "Her name is Lilypan.",
            "Oh, look at that.",
            "Oh, no.",
        ),
        min_count=3,
    )

    assert sources(found) == ["Lilypan"]
    assert found[0].count == 3


def test_neighbouring_names_are_proposed_as_one_phrase():
    found = scan_candidates(
        cues(
            "Say hello to NoHo Hank.",
            "That was NoHo Hank again.",
            "I saw NoHo Hank leave.",
        ),
        min_count=3,
    )

    assert sources(found) == ["NoHo Hank"]
    assert found[0].match == "phrase"


def test_two_spellings_one_edit_apart_are_flagged_as_the_same_name():
    """Whisper writes Jessie and Jesse for one character. Both reach the glossary
    as separate entries unless somebody is told they might be the same person."""
    found = scan_candidates(
        cues(
            "Where is Jessie now?",
            "Ask Jessie about it.",
            "I saw Jesse there.",
            "Tell Jesse to wait.",
        ),
        min_count=2,
    )

    variants = {candidate.source: candidate.variants for candidate in found}
    assert variants == {"Jesse": ("Jessie",), "Jessie": ("Jesse",)}
    assert "Jessie" in found[0].to_dict()["aliases"] + found[1].to_dict()["aliases"]


def test_names_an_existing_glossary_already_covers_are_left_out():
    existing = Terminology(
        source_language="en",
        target_language="zh-CN",
        terms=(Term(source="Woody", target="胡迪", match="word"),),
    )

    draft = draft_terminology(
        cues(
            "Ask Woody first.",
            "Then Woody left.",
            "Tell Sally now.",
            "I saw Sally go.",
        ),
        existing=existing,
        min_count=2,
    )

    assert sources(draft.candidates) == ["Sally"]
    assert draft.already_covered == ("Woody",)


def test_a_new_name_sharing_a_word_with_an_old_entry_is_flagged():
    """The season-to-season trap, verbatim: carry `Ryan Madison` into the next
    season and the model reads the new `Aaron Ryan` as the old character. Both
    names are really in the script, so the new one has to be written down too."""
    existing = Terminology(
        source_language="en",
        target_language="zh-CN",
        terms=(Term(source="Ryan Madison", target="瑞恩·麦迪逊", match="phrase"),),
    )

    draft = draft_terminology(
        cues("Aaron Ryan called.", "That was Aaron Ryan.", "Aaron Ryan again."),
        existing=existing,
        min_count=2,
    )

    assert sources(draft.candidates) == ["Aaron Ryan"]
    assert draft.candidates[0].collides_with == ("Ryan Madison",)
    assert draft.colliding == draft.candidates
    assert "Ryan Madison" in draft.candidates[0].to_dict()["draft_warning"]


def test_an_unfilled_draft_refuses_to_load_as_a_glossary(tmp_path):
    """Fail closed. A draft that loaded would quietly translate names to nothing."""
    draft = draft_terminology(cues("Ask Woody first.", "Tell Woody again."), min_count=2)
    path = tmp_path / "draft.json"
    path.write_text(
        json.dumps(draft_to_dict(draft, source_language="en", target_language="zh-CN")),
        encoding="utf-8",
    )

    with pytest.raises(TerminologyError, match="target must be non-empty"):
        load_terminology(path)


def test_a_filled_in_draft_loads_unchanged(tmp_path):
    draft = draft_terminology(cues("Ask Woody first.", "Tell Woody again."), min_count=2)
    payload = draft_to_dict(draft, source_language="en", target_language="zh-CN")
    payload["terms"][0]["target"] = "胡迪"
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    loaded = load_terminology(path)

    # The evidence fields ride along and the loader ignores them.
    assert loaded.terms[0] == Term(source="Woody", target="胡迪", match="word")


def test_draft_terms_command_scans_a_whole_season_at_once(tmp_path, capsys):
    first = tmp_path / "e01.srt"
    first.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nAsk Woody first.\n", encoding="utf-8"
    )
    second = tmp_path / "e02.srt"
    second.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nThen Woody left.\n", encoding="utf-8"
    )
    output = tmp_path / "season.terms.json"

    assert (
        main(
            [
                "draft-terms",
                str(first),
                str(second),
                "-o",
                str(output),
                "--min-count",
                "2",
                "--source-language",
                "en",
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "video-txt.terminology"
    assert payload["source_language"] == "en"
    assert [term["source"] for term in payload["terms"]] == ["Woody"]
    assert payload["terms"][0]["target"] == ""
    assert "Scanned 2 cues in 2 subtitle file(s)" in capsys.readouterr().out


def test_draft_terms_command_refuses_to_overwrite_an_input(tmp_path):
    subtitle = tmp_path / "e01.srt"
    subtitle.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nAsk Woody first.\n", encoding="utf-8"
    )

    with pytest.raises(SystemExit):
        main(["draft-terms", str(subtitle), "-o", str(subtitle)])

    assert subtitle.read_text(encoding="utf-8").endswith("Ask Woody first.\n")
