import numpy as np

from florea.kb import (ENTITIES, KnowledgeBase, StyleRecommender, TinyGCN, build_graph, doc_entity,
                       evaluate_gcn, hash_embed, to_graph_colors)
from florea.styles import ARTISTIC_STYLES, EMOTIONS_FR


def test_graph_shape_and_path():
    g = build_graph()
    assert len(g.nodes) > 200 and len(g.edges()) > 500
    assert g.path("lavender", "Nostalgie") == ["lavender évoque Nostalgie"]
    assert g.path("lavender", "nope") is None


def test_every_style_emotion_is_known():
    assert all(e in EMOTIONS_FR for v in ARTISTIC_STYLES.values() for e in v["emotions"])
    assert all(e in EMOTIONS_FR for ent in ENTITIES for e in ent["emotions"])


def test_doc_entity_and_colors():
    assert doc_entity("Lavender: purple flower…") == "lavender"
    assert doc_entity("Qui n'existe pas: truc") is None
    assert to_graph_colors(["pale pink", "deep green", "charcoal"]) == ["pink", "green", "grey"]


def test_gcn_gradients_match_finite_differences():
    g = build_graph()
    gcn = TinyGCN(g, d1=8, d2=4, seed=1)
    E = g.edges()
    A = gcn._norm_adj(E)
    pos = np.array([[gcn.ix[a], gcn.ix[b]] for a, b in E[:40]])
    neg = np.array([[0, 5], [3, 9], [7, 2], [11, 40]] * 10)
    gcn.W1 *= 5
    gcn.W2 *= 5
    _, g1, g2 = gcn.loss_and_grads(A, pos, neg, l2=1e-3)
    h = 1e-5
    for W, grad, idx in ((gcn.W1, g1, (3, 2)), (gcn.W2, g2, (1, 1))):
        old = W[idx]
        W[idx] = old + h
        lp = gcn.loss_and_grads(A, pos, neg, l2=1e-3)[0]
        W[idx] = old - h
        lm = gcn.loss_and_grads(A, pos, neg, l2=1e-3)[0]
        W[idx] = old
        assert abs((lp - lm) / (2 * h) - grad[idx]) < 1e-6


def test_recommender_returns_valid_styles():
    r = StyleRecommender(epochs=60)
    out = r.recommend(["lavender"], "Sérénité", "summer", ["violet"], k=3)
    assert len(out) == 3 and all(n in ARTISTIC_STYLES for n, _ in out)
    assert r.recommend([], None) == []
    links = r.explain(["lavender"], "Sérénité")
    assert links and all("path" in l for l in links)


def test_evaluate_gcn_smoke():
    res = evaluate_gcn(n_splits=1, epochs=30)
    assert set(res) >= {"gcn", "vote", "popularité"} and all(0 <= res[k] <= 1 for k in ("gcn", "vote", "popularité"))


def test_knowledge_base_search_without_model():
    kb = KnowledgeBase(embed_fn=hash_embed)
    hits = kb.search("lavender purple flower Mediterranean summer", top_k=3)
    assert len(hits) == 3 and hits[0]["document"].startswith("Lavender")
    assert all(h["document"] in kb.docs[:kb.n_nature] for h in hits)      # only_nature par défaut
