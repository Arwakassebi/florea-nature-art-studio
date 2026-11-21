import pytest

from florea import lite
from florea.kb import ENTITY_BY_NAME
from florea.styles import ARTISTIC_STYLES


@pytest.fixture(scope="module")
def engine():
    return lite.LiteEngine(epochs=60)


def test_every_alias_points_to_a_real_entity():
    assert set(lite.FR_ALIASES.values()) <= set(ENTITY_BY_NAME)


def test_find_entities_french_and_english():
    assert lite.find_entities("Un champ de lavande au coucher du soleil") == ["lavender", "golden hour"]
    assert "olive tree" in lite.find_entities("des oliviers sur la colline")
    assert "cherry blossom" in lite.find_entities("cherry blossoms in spring")
    assert lite.find_entities("pinceau") == []                      # « pin » ne doit pas matcher « pinceau »


def test_analyze_french_text(engine):
    r = engine.analyze("Un champ de lavande au coucher du soleil, des oiseaux chantent, je pense à mon enfance")
    assert "lavender" in r["entities"] and r["emotion"] == "Nostalgie"
    assert r["style"] in ARTISTIC_STYLES and len(r["styles"]) == 5
    assert r["links"] and r["docs"]
    assert len(r["prompt"].split()) <= 60 and "no watermark" in r["prompt"]


def test_analyze_respects_user_choices(engine):
    r = engine.analyze("mer calme", emotion="Joie", style="Hokusai — Ukiyo-e")
    assert r["emotion"] == "Joie" and r["style"] == "Hokusai — Ukiyo-e" and "Hokusai" in r["prompt"]


def test_analyze_english_subject_is_kept(engine):
    r = engine.analyze("Misty mountains at dawn, a lonely pine tree")
    assert "Misty mountains at dawn" in r["prompt"]


def test_empty_input_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.analyze("   ")


def test_markdown_renders(engine):
    md = lite.result_markdown(engine.analyze("jasmin au bord de la mer"))
    assert "Prompt image généré" in md and "jasmine" in md


def test_noisy_vector_hits_are_filtered(engine):
    r = engine.analyze("Un champ de lavande au coucher du soleil, des oiseaux chantent, je pense à mon enfance")
    assert r["docs"][0]["document"].startswith("Lavender") and r["docs"][0]["via"] == "mot-clé reconnu"
    assert all(d["via"] == "mot-clé reconnu" or d["score"] >= engine.min_score for d in r["docs"])
    assert "foxglove" not in r["entities"]
