import pytest

import florea_server as srv
from florea.kb import KnowledgeBase, StyleRecommender, hash_embed


@pytest.fixture(autouse=True)
def light_backends(monkeypatch):
    monkeypatch.setattr(srv, "_kb", KnowledgeBase(embed_fn=hash_embed))
    monkeypatch.setattr(srv, "_reco", StyleRecommender(epochs=40))


def test_list_styles_filter():
    assert len(srv.list_styles()) > len(srv.list_styles("Joie")) > 0
    assert all("Joie" in s["emotions"] for s in srv.list_styles("Joie"))


def test_recommend_and_explain():
    r = srv.recommend_styles("lavender, rain sound", "Sérénité", "summer", "violet", 3)
    assert len(r) == 3 and {"style", "score"} <= set(r[0])
    assert srv.explain_links("lavender", "Sérénité")


def test_input_validation():
    with pytest.raises(ValueError):
        srv.recommend_styles(emotion="Colère")
    with pytest.raises(ValueError):
        srv.generate_image_prompt("Style inconnu", "x y z w")


def test_search_clamps_top_k():
    assert len(srv.search_knowledge("olive tree", top_k=999)) <= 10


@pytest.mark.skipif(srv._Server is None, reason="paquet mcp absent")
def test_server_registers_five_tools():
    assert srv.build_server() is not None and len(srv.TOOLS) == 5
