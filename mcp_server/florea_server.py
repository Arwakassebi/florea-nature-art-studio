"""Serveur MCP de FLOREA : les outils du notebook, réutilisables par Claude Desktop, Cursor, etc.

Outils exposés
  search_knowledge(query, top_k)                              recherche sémantique (nature, ambiances, sons, styles)
  generate_image_prompt(style_name, subject, palette, mood)   prompt prêt pour un modèle texte→image
  list_styles(emotion)                                        les 44 styles (filtrables par émotion)
  recommend_styles(entities, emotion, season, colors)         styles recommandés par le graphe + GCN
  explain_links(entities, emotion, season, colors)            chemins du graphe de connaissances (Graph-RAG explicable)

Lancer :   python mcp_server/florea_server.py          (transport stdio)
Claude Desktop (claude_desktop_config.json) :
  {"mcpServers": {"florea": {"command": "python", "args": ["/chemin/vers/florea-nature-art-studio/mcp_server/florea_server.py"]}}}
Dépendances : pip install mcp numpy pydantic     (+ sentence-transformers et faiss-cpu en option : meilleure recherche)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from florea.kb import KnowledgeBase, StyleRecommender  # noqa: E402
from florea.styles import ARTISTIC_STYLES, EMOTIONS_FR  # noqa: E402
from florea.text import as_list, build_image_prompt, normalize_colors  # noqa: E402

try:                                              # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:                               # mcp 2.x : FastMCP a été renommé MCPServer
    try:
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:
        _Server = None

_kb, _reco = None, None


def _get_kb():
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()                     # MiniLM si disponible, sinon embeddings hachés
    return _kb


def _get_reco():
    global _reco
    if _reco is None:
        _reco = StyleRecommender()
    return _reco


# ----------------------------------------------------------------------------- fonctions-outils (testables sans MCP)
def search_knowledge(query: str, top_k: int = 4) -> list[dict]:
    """Recherche sémantique dans la base de connaissances FLOREA (fleurs, arbres, paysages, météo, sons, culture
    méditerranéenne). Renvoie les documents les plus proches avec leur score."""
    return _get_kb().search(query, top_k=max(1, min(int(top_k), 10)))


def generate_image_prompt(style_name: str, subject: str, palette: str = "", mood: str = "") -> str:
    """Assemble un prompt texte→image : style artistique + sujet + palette (liste séparée par des virgules) + ambiance."""
    if style_name not in ARTISTIC_STYLES:
        raise ValueError(f"Style inconnu : {style_name!r}. Utilisez list_styles pour voir les noms valides.")
    return build_image_prompt(style_name, subject, normalize_colors(as_list(palette, 8)) or as_list(palette, 5), mood)


def list_styles(emotion: str = "") -> list[dict]:
    """Liste les styles artistiques ; filtre optionnel par émotion (Paix, Nostalgie, Joie, Mélancolie, Émerveillement,
    Sérénité, Mystère, Passion, Admiration, Espoir, Solitude, Tendresse)."""
    return [{"name": n, "description": v["description"], "emotions": v["emotions"]}
            for n, v in ARTISTIC_STYLES.items() if not emotion or emotion in v["emotions"]]


def recommend_styles(entities: str = "", emotion: str = "", season: str = "", colors: str = "", k: int = 3) -> list[dict]:
    """Recommande des styles pour des entités (ex. « lavender, rain sound »), une émotion, une saison et des couleurs,
    avec un GCN entraîné sur le graphe de connaissances."""
    if emotion and emotion not in EMOTIONS_FR:
        raise ValueError(f"Émotion inconnue : {emotion!r}. Valeurs : {', '.join(EMOTIONS_FR)}")
    ents = [e.strip().lower() for e in as_list(entities, 8)]
    ranked = _get_reco().recommend(ents, emotion or None, season or None, as_list(colors, 6), k=max(1, min(int(k), 8)))
    return [{"style": n, "score": round(s, 3)} for n, s in ranked]


def explain_links(entities: str = "", emotion: str = "", season: str = "", colors: str = "") -> list[dict]:
    """Chemins du graphe de connaissances entre vos entrées et des styles / symboles (Graph-RAG explicable)."""
    ents = [e.strip().lower() for e in as_list(entities, 8)]
    r = _get_reco()
    return r.explain(ents, emotion or None, season or None, as_list(colors, 6))


TOOLS = [search_knowledge, generate_image_prompt, list_styles, recommend_styles, explain_links]


def build_server():
    if _Server is None:
        raise RuntimeError("Le paquet « mcp » n'est pas installé : pip install mcp")
    server = _Server("florea")
    for fn in TOOLS:
        server.tool()(fn)
    return server


if __name__ == "__main__":
    build_server().run(transport="stdio")
