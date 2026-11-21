"""FLOREA Lite : version CPU, sans LLM ni génération d'image, pour un hébergement gratuit 24 h/24.

Ce que la démo montre en direct : détection d'émotion et de saison, recherche dans la base de connaissances,
Graph-RAG (chemins explicables), classement des styles par le GCN entraîné au démarrage, prompt image prêt à l'emploi.
Ce qu'elle ne fait pas : écrire le poème et peindre l'image (cela demande un GPU : voir le notebook Colab).

Dépendances : numpy, pydantic, gradio. Embeddings : sacs de mots hachés par défaut (1 Go de RAM suffit) ;
FLOREA_EMBEDDER=minilm active MiniLM si sentence-transformers est installé.

Lancer :  python -m florea.lite        (variables : FLOREA_HOST, PORT)
"""
import os
import re

from florea.kb import ENTITY_BY_NAME, KnowledgeBase, StyleRecommender, doc_entity, hash_embed
from florea.styles import ARTISTIC_STYLES, EMOTIONS_FR, STYLE_AUTO
from florea.text import (build_image_prompt, detect_emotion, detect_season, final_mood, first_sentence, looks_english,
                         trim_words)

COLAB_URL = os.environ.get("FLOREA_COLAB_URL", "https://colab.research.google.com/github/Arwakassebi/florea-nature-art-studio/blob/main/notebooks/FLOREA_V6.ipynb")

# Français -> nom d'entité du graphe (la base de connaissances est en anglais ; sans modèle multilingue, on traduit les mots-clés)
FR_ALIASES = {
    "lavande": "lavender", "rose": "rose", "tournesol": "sunflower", "coquelicot": "poppy", "pâquerette": "daisy",
    "marguerite": "daisy", "cerisier": "cherry blossom", "cerisiers": "cherry blossom", "tulipe": "tulip",
    "muguet": "lily of the valley", "myosotis": "forget-me-not", "jacinthe": "bluebell", "pissenlit": "dandelion",
    "lotus": "lotus", "nénuphar": "water lily", "iris": "iris", "pivoine": "peony", "glycine": "wisteria",
    "jasmin": "jasmine", "bougainvillier": "bougainvillea", "bougainvillée": "bougainvillea", "perce-neige": "snowdrop",
    "bleuet": "cornflower", "digitale": "foxglove", "orchidée": "orchid", "souci": "marigold", "lilas": "lilac",
    "olivier": "olive tree", "oliviers": "olive tree", "palmier": "date palm", "figuier de barbarie": "prickly pear cactus",
    "chêne": "oak", "saule": "weeping willow", "bouleau": "silver birch", "pin": "pine tree", "érable": "japanese maple",
    "ginkgo": "ginkgo", "fougère": "fern", "mousse": "moss", "bambou": "bamboo", "cyprès": "cypress", "figuier": "fig tree",
    "prairie": "spring meadow", "pré": "spring meadow", "sous-bois": "ancient woodland in spring",
    "forêt d'automne": "autumn woodland", "étang": "still pond", "mare": "still pond", "colline": "mediterranean hillside",
    "jardin d'hiver": "winter garden", "forêt sous la pluie": "rain in the forest", "rosée": "dawn dew",
    "herbes hautes": "wind in tall grass", "montagne": "misty mountains", "montagnes": "misty mountains",
    "mer": "seashore", "plage": "seashore", "bord de mer": "seashore", "dune": "desert dunes at sunset",
    "dunes": "desert dunes at sunset", "désert": "desert dunes at sunset", "jardin de nuit": "night garden",
    "heure dorée": "golden hour", "coucher de soleil": "golden hour", "coucher du soleil": "golden hour",
    "brouillard": "fog in a valley", "brume": "fog in a valley", "arc-en-ciel": "rainbow after rain",
    "pluie": "rain sound", "oiseaux": "birdsong", "oiseau": "birdsong", "chant d'oiseaux": "birdsong", "vent": "wind sound",
    "ruisseau": "stream sound", "rivière": "stream sound", "tonnerre": "thunder sound", "orage": "thunder sound",
    "cigales": "cicadas", "cigale": "cicadas", "grillons": "cicadas", "vagues": "waves sound", "feuilles": "rustling leaves",
    "silence": "silence in nature",
}
_ALIAS_RES = [(re.compile(r"(?<!\w)" + re.escape(a) + r"(s|x)?(?!\w)"), e) for a, e in sorted(FR_ALIASES.items(), key=lambda t: -len(t[0]))]
_NAME_RES = [(re.compile(r"(?<!\w)" + re.escape(n) + r"s?(?!\w)"), n) for n in sorted(ENTITY_BY_NAME, key=len, reverse=True)]

EXAMPLES = [
    "Un champ de lavande au coucher du soleil, des oiseaux chantent, je pense à mon enfance",
    "Un étang calme sous la brume, un nénuphar, le silence",
    "Pluie sur la forêt en automne, un peu triste, une odeur de terre mouillée",
    "Jasmin et bougainvillier sur un mur blanc au bord de la mer, un soir d'été",
    "Misty mountains at dawn, a lonely pine tree",
]


def find_entities(text):
    """Entités du graphe citées dans le texte (noms français via FR_ALIASES, ou noms anglais directs)."""
    t = (text or "").lower()
    hits = {}
    for rx, ent in _ALIAS_RES + _NAME_RES:
        m = rx.search(t)
        if m and (ent not in hits or m.start() < hits[ent]):
            hits[ent] = m.start()
    return sorted(hits, key=hits.get)                                # dans l'ordre d'apparition dans le texte


class LiteEngine:
    """Recherche + graphe + GCN, sans GPU. Entraînement du GCN au démarrage (quelques secondes)."""

    def __init__(self, epochs=300, embedder=None):
        lexical = (embedder or os.environ.get("FLOREA_EMBEDDER", "hash")) != "minilm"
        self.kb = KnowledgeBase(embed_fn=hash_embed if lexical else None)
        self.min_score = 0.2 if lexical else 0.3          # en dessous : résultat de la recherche vectorielle jugé trop bruité
        self.reco = StyleRecommender(epochs=epochs)
        self.entity_docs = {}                              # entité -> sa fiche (accès direct pour les mots-clés reconnus)
        for d in self.kb.docs[:self.kb.n_nature]:
            e = doc_entity(d)
            if e and e not in self.entity_docs:
                self.entity_docs[e] = d

    def analyze(self, text, emotion="Auto (détecter)", style=STYLE_AUTO, top_k=4):
        text = (text or "").strip()
        if not text:
            raise ValueError("Écrivez quelques mots : une fleur, un paysage, un son, un souvenir.")
        found = find_entities(text)
        docs = [{"document": self.entity_docs[e], "score": None, "via": "mot-clé reconnu"} for e in found if e in self.entity_docs][:top_k]
        seen = {d["document"] for d in docs}
        for h in self.kb.search(text + " " + " ".join(found), top_k=top_k):
            if len(docs) < top_k and h["score"] >= self.min_score and h["document"] not in seen:
                docs.append({**h, "via": f"similarité {h['score']}"})
        ents = list(found)
        for d in docs:
            e = doc_entity(d["document"])
            if e and e not in ents:
                ents.append(e)
        ents = ents[:6]
        emo = emotion if emotion in EMOTIONS_FR else detect_emotion(text, None)
        if not emo:                                                  # sinon : émotion la plus fréquente des entités trouvées
            counts = {}
            for e in ents:
                for em in ENTITY_BY_NAME[e]["emotions"]:
                    counts[em] = counts.get(em, 0) + 1
            emo = max(counts, key=counts.get) if counts else "Sérénité"
        season = detect_season(text)
        ranked = self.reco.recommend(ents, emo, season, [], k=5)
        chosen = style if style in ARTISTIC_STYLES else (ranked[0][0] if ranked else sorted(ARTISTIC_STYLES)[0])
        links = self.reco.explain(ents, emo, season, [])
        subject = trim_words(text, 25) if looks_english(text) else self._subject(ents, docs)
        palette = ENTITY_BY_NAME[ents[0]]["colors"][:4] if ents else []
        prompt = build_image_prompt(chosen, subject, palette, final_mood(emo))
        return {"emotion": emo, "season": season, "entities": ents, "styles": [(n, round(s, 3)) for n, s in ranked],
                "style": chosen, "links": links, "docs": docs, "prompt": prompt}

    @staticmethod
    def _subject(ents, docs):
        if docs:
            return first_sentence(docs[0]["document"].split(":", 1)[-1], 18)
        return "a quiet natural landscape in soft light"


def result_markdown(r):
    styles = "\n".join(f"{i}. **{n}** : score {s}" for i, (n, s) in enumerate(r["styles"], 1)) or "—"
    links = "\n".join(f"- **{l['label']}** ({l['type']}) : " + " → ".join(l["path"] or []) for l in r["links"]) or "- (aucun)"
    docs = "\n".join(f"- {d['document']}  \n  *({d['via']})*" for d in r["docs"]) or "- (aucune)"
    return (f"### Analyse\n**Émotion** : {r['emotion']} · **Saison** : {r['season'] or '—'} · "
            f"**Entités reconnues** : {', '.join(r['entities']) or '—'}\n\n"
            f"### Styles recommandés par le GCN\n{styles}\n\n**Style retenu** : {r['style']}\n\n"
            f"### Graphe de connaissances (Graph-RAG)\n{links}\n\n### Connaissances retrouvées (recherche vectorielle)\n{docs}\n\n"
            f"### Prompt image généré\n> {r['prompt']}")


def build_ui(engine):
    import gradio as gr
    from florea.styles import EMOTION_CHOICES, STYLE_CHOICES

    css = ".header{text-align:center;padding:1.2rem;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:16px;margin-bottom:1rem}" \
          ".header h1{color:#fff;margin:0}.header p{color:#f0f0f0;margin:.4rem 0 0}"
    theme = gr.themes.Soft(primary_hue="violet", secondary_hue="slate")
    gr_major = int(gr.__version__.split(".")[0])
    blocks_kw = {} if gr_major >= 6 else {"theme": theme, "css": css}
    launch_kw = {"theme": theme, "css": css} if gr_major >= 6 else {}

    def run(text, emotion, style):
        try:
            return result_markdown(engine.analyze(text, emotion, style))
        except ValueError as e:
            raise gr.Error(str(e))

    def styles_table(emotion):
        rows = [[n, v["description"], ", ".join(v["emotions"])] for n, v in ARTISTIC_STYLES.items()
                if emotion in ("Toutes", None) or emotion in v["emotions"]]
        return rows

    with gr.Blocks(title="FLOREA Lite — Nature Art Studio", **blocks_kw) as demo:
        gr.HTML('<div class="header"><h1>🌿 FLOREA Lite</h1><p>Graph-RAG + GCN en direct : décrivez une scène de nature, '
                'voyez comment FLOREA choisit un style</p></div>')
        gr.Markdown(f"Version **CPU** de la démo : elle n'écrit pas le poème et ne peint pas l'image (il faut un GPU). "
                    f"[Lancer la version complète sur Colab]({COLAB_URL}).")
        with gr.Tab("🔎 Analyser une scène"):
            with gr.Row():
                with gr.Column(scale=1):
                    text = gr.Textbox(label="✍️ Une fleur, un paysage, un son, un souvenir (français ou anglais)", lines=4, value=EXAMPLES[0])
                    emotion = gr.Dropdown(label="💭 Émotion", choices=EMOTION_CHOICES, value=EMOTION_CHOICES[0])
                    style = gr.Dropdown(label="🎨 Style", choices=STYLE_CHOICES, value=STYLE_CHOICES[0])
                    btn = gr.Button("🚀 Analyser", variant="primary")
                    gr.Examples(EXAMPLES, inputs=text, label="Exemples")
                with gr.Column(scale=2):
                    out = gr.Markdown()
            btn.click(run, [text, emotion, style], out)
        with gr.Tab("🎨 Les 44 styles"):
            f = gr.Dropdown(label="Filtrer par émotion", choices=["Toutes"] + list(EMOTIONS_FR), value="Toutes")
            table = gr.Dataframe(headers=["Style", "Description", "Émotions"], value=styles_table("Toutes"), wrap=True)
            f.change(styles_table, f, table)
        with gr.Tab("ℹ️ À propos"):
            gr.Markdown("**Comment ça marche** : le texte est analysé (émotion, saison, entités) → recherche vectorielle dans 70 fiches "
                        "nature → graphe fleur/ambiance/son → couleur → saison → émotion → style (≈ 250 nœuds) → un **GCN écrit en NumPy**, "
                        "entraîné au démarrage, classe les styles.\n\n"
                        "**Limites** : sans modèle multilingue, les mots-clés français sont traduits par une table d'alias ; "
                        "les embeddings sont lexicaux (pas sémantiques) ; la base de connaissances est écrite à la main.\n\n"
                        "Code, tests et résultats d'évaluation : voir le dépôt GitHub.")
    return demo, launch_kw


def main():
    engine = LiteEngine()
    demo, launch_kw = build_ui(engine)
    demo.queue(max_size=16).launch(server_name=os.environ.get("FLOREA_HOST", "127.0.0.1"),
                                   server_port=int(os.environ.get("PORT", "7860")), show_error=True, **launch_kw)


if __name__ == "__main__":
    main()
