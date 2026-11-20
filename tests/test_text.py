import pytest
from pydantic import ValidationError

from florea.styles import ARTISTIC_STYLES, EMOTIONS_FR
from florea.text import (FloreaConcept, build_image_prompt, clean_text, detect_emotion, detect_season, extract_json,
                         final_mood, first_sentence, looks_english, normalize_colors, trim_words)

OK = dict(title="Le murmure du violet", concept="Un champ de lavande où la mémoire revient doucement.",
          subject="rows of blooming purple lavender at sunset with small birds", palette="bleu, rouge",
          symbols=["lavande"], mood="calm")


def test_extract_json_tolerant():
    assert extract_json('Voici :\n```json\n{"a": [1, 2,],}\n``` fin') == {"a": [1, 2]}


@pytest.mark.parametrize("bad", ["pas de json", '{"a": ', None, ""])
def test_extract_json_invalid(bad):
    assert extract_json(bad) is None


def test_small_helpers():
    assert trim_words("un deux trois quatre", 2) == "un deux"
    assert first_sentence("Bonjour toi. La suite.") == "Bonjour toi"
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_emotion_and_season():
    assert detect_emotion("Les souvenirs de mon enfance chez ma grand-mère") == "Nostalgie"
    assert detect_emotion("xyz", None) is None
    assert detect_season("un matin de printemps") == "spring"
    assert detect_season("la mer") is None


def test_mood_is_always_english():
    assert all(final_mood(e, "douceur nostalgique") == EMOTIONS_FR[e] for e in EMOTIONS_FR)
    assert final_mood("Nostalgie", "gentle longing") == "nostalgia, gentle longing"


def test_colors_are_translated_and_filtered():
    assert normalize_colors(["bleu ciel", "rouge", "vert clair", "حمراء"]) == ["sky blue", "red", "light green"]


def test_looks_english():
    assert not looks_english("un champ de lavande")
    assert not looks_english("champ lavande soleil")
    assert looks_english("purple lavender at sunset")


def test_concept_accepts_valid_data_and_translates_palette():
    assert FloreaConcept(**OK).palette == ["blue", "red"]


@pytest.mark.parametrize("bad", [{"title": ""}, {"subject": "un champ de lavande au coucher du soleil"},
                                 {"palette": ["violet"]}, {"concept": "court"}])
def test_concept_rejects_invalid_fields(bad):
    with pytest.raises(ValidationError):
        FloreaConcept(**{**OK, **bad})


def test_image_prompt_is_short_for_every_style():
    for s in ARTISTIC_STYLES:
        p = build_image_prompt(s, "a lavender field at sunset with small birds above warm hills",
                               ["violet", "gold", "soft green", "cream"], "nostalgia, gentle longing")
        assert len(p.split()) <= 60
