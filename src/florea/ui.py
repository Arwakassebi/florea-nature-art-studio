"""Interface Gradio de FLOREA (onglets « Créer » et « Galerie »). À utiliser après florea.pipeline.init()."""
import json
import os
import zipfile

import gradio as gr

from florea import pipeline as P
from florea.pipeline import FloreaError, run_florea
from florea.styles import EMOTION_CHOICES, LANGS, STYLE_CHOICES

CSS = """
.gradio-container { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
.header { text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 4px 20px rgba(102,126,234,0.3); }
.header h1 { color: white; margin: 0; font-size: 2.4rem; font-weight: 700; letter-spacing: 2px; }
.header p { color: #f0f0f0; margin-top: 0.5rem; font-size: 1.05rem; }
"""
THEME = gr.themes.Soft(primary_hue="violet", secondary_hue="slate")
DISABLED_TXT = "(désactivé : cochez « Écrire le poème et l'histoire » pour les obtenir)"


def details_md(r):
    a, ci = r["audio"], r["concept_info"]
    kb = "\n".join(f"- {d['document']}" for d in r["retrieved"]) or "- (aucune)"
    ranked = " · ".join(f"{n.split(' — ')[0]} ({s})" for n, s in r["ranked_styles"]) or "—"
    links = "\n".join(f"- **{l['label']}** ({l['type']}) : " + " → ".join(l["path"] or []) for l in r["graph_links"]) or "- (aucun)"
    lines = [f"**Style** : {r['style_name']}  \n**Émotion** : {r['emotion']}  \n**Saison détectée** : {r['season'] or '—'}  \n**Graine** : {r['seed']}",
             f"**Classement des styles par le GCN** : {ranked}"]
    if r["image_description"]:
        lines.append(f"**Photo vue par FLOREA** : {r['image_description']} ; couleurs : {', '.join(r['palette_names'])}")
    if a.get("summary"):
        lines.append(f"**Sons détectés** : {a['summary']}  \n**Sons de nature retenus** : {', '.join(a['nature']) or '—'}")
    if a.get("transcription"):
        lines.append(f"**Voix transcrite** : {a['transcription']}")
    lines.append("**Agent / recherche** : " + (" ; ".join(r["trace"]) or "—"))
    lines.append(f"**Connaissances utilisées** :\n{kb}")
    lines.append(f"**Graphe de connaissances (Graph-RAG)** :\n{links}")
    lines.append(f"**Concept** : {ci['attempts']} tentative(s) ; " + ("champs complétés automatiquement : " + ", ".join(ci["fixed"]) if ci["fixed"] else "tous les champs validés par le schéma Pydantic"))
    lines.append(f"**Prompt envoyé au modèle d'image** :\n> {r['final_prompt']}")
    lines.append("**Temps (s)** : " + " · ".join(f"{k} {v}" for k, v in r["timings"].items()) + f"  \n**Modèles en mémoire** : {P.MM.status()}")
    return "\n\n".join(lines)


def florea_interface(image, audio_path, user_text, emotion, style_name, language, creativity, n_variants, seed, write_text, use_agent,
                     progress=gr.Progress()):
    try:
        r = run_florea(image, audio_path, user_text, emotion, style_name, language, creativity, n_variants, seed, write_text, use_agent,
                       progress=lambda f, desc: progress(f, desc=desc))
    except (ValueError, FloreaError) as e:                               # message clair, pas de stack trace
        raise gr.Error(str(e))
    c = r["concept"]
    poem = r["poem"] if write_text else DISABLED_TXT
    story = r["story"] if write_text else DISABLED_TXT
    return (r["images"], c["title"], c["concept"], poem, story, ", ".join(c["palette"]), ", ".join(c["symbols"]), details_md(r))


# ---------- Galerie ----------
def list_gallery():
    d = P.CFG.save_dir
    metas = []
    for f in sorted(os.listdir(d), reverse=True):
        if f.endswith(".json"):
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                metas.append(json.load(fh))
    metas = [m for m in metas if os.path.exists(os.path.join(d, m["id"] + ".png"))]
    return [(os.path.join(d, m["id"] + ".png"), f"{m['title']} · {m['style'].split(' — ')[0]}") for m in metas], [m["id"] for m in metas]


def gallery_pick(ids, evt: gr.SelectData):
    with open(os.path.join(P.CFG.save_dir, ids[evt.index] + ".json"), encoding="utf-8") as f:
        m = json.load(f)
    poem = m["poem"] or DISABLED_TXT
    return f"### {m['title']}\n*{m['style']} · {m['emotion']} · graine {m['seed']}*\n\n{poem}\n\n{m['story'] or ''}\n\n> {m['prompt']}"


def export_zip():
    d = P.CFG.save_dir
    path = os.path.join(os.path.dirname(os.path.abspath(d)), "florea_gallery.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(d):
            z.write(os.path.join(d, f), f)
    return path


def toggle_text(v):
    return gr.update(visible=bool(v)), gr.update(visible=bool(v))


def build_ui():
    """Renvoie (demo, launch_kwargs). Gradio ≥ 6 : thème et CSS se passent à launch()."""
    gr_major = int(gr.__version__.split(".")[0])
    blocks_kw = {} if gr_major >= 6 else {"theme": THEME, "css": CSS}
    launch_kw = {"theme": THEME, "css": CSS} if gr_major >= 6 else {}

    with gr.Blocks(title="FLOREA — Studio d'art inspiré de la nature", **blocks_kw) as demo:
        gr.HTML('<div class="header"><h1>🌿 FLOREA</h1><p>Une photo, un son ou quelques mots → une œuvre, un poème, une histoire</p></div>')
        with gr.Tab("🎨 Créer"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Entrées (au moins une, ou plusieurs ensemble)")
                    input_image = gr.Image(label="📷 Photo (optionnel)", type="pil")
                    input_audio = gr.Audio(label="🎵 Son (optionnel) : pluie, vent, oiseaux, eau, voix…", sources=["upload", "microphone"], type="filepath")
                    user_text = gr.Textbox(label="✍️ Souvenirs / texte (optionnel)", lines=3, placeholder="Racontez ce que cela vous évoque, ou décrivez une scène…")
                    gr.Markdown("### 🎛️ Réglages")
                    style_name = gr.Dropdown(label="🎨 Style artistique (44)", choices=STYLE_CHOICES, value=STYLE_CHOICES[0])
                    emotion = gr.Dropdown(label="💭 Émotion", choices=EMOTION_CHOICES, value=EMOTION_CHOICES[0])
                    language = gr.Dropdown(label="🌍 Langue du poème et de l'histoire", choices=list(LANGS), value="Français")
                    creativity = gr.Slider(label="✨ Transformation de la photo (si photo)", minimum=0.3, maximum=0.9, value=0.65, step=0.05)
                    n_variants = gr.Slider(label="🖼️ Nombre de variantes", minimum=1, maximum=4, value=1, step=1)
                    seed = gr.Number(label="🎲 Graine (-1 = aléatoire ; même graine = même résultat)", value=-1, precision=0)
                    write_text = gr.Checkbox(label="Écrire le poème et l'histoire", value=True)
                    use_agent = gr.Checkbox(label="🤖 Mode agent : le LLM décide d'appeler la recherche de connaissances", value=P.CFG.use_agent)
                    btn = gr.Button("🚀 Créer l'œuvre", variant="primary", size="lg")
                with gr.Column(scale=2):
                    out_gallery = gr.Gallery(label="🖼️ Œuvre(s) générée(s)", columns=2, height=520)
                    out_title = gr.Textbox(label="Titre")
                    out_concept = gr.Textbox(label="Concept", lines=2)
                    out_poem = gr.Textbox(label="Poème", lines=8)
                    out_story = gr.Textbox(label="Histoire", lines=4)
                    with gr.Row():
                        out_palette = gr.Textbox(label="Palette")
                        out_symbols = gr.Textbox(label="Symboles")
                    with gr.Accordion("🔍 Pourquoi cette œuvre ? (agent, graphe, GCN, prompt, temps)", open=False):
                        out_details = gr.Markdown()
            write_text.change(toggle_text, write_text, [out_poem, out_story])
            btn.click(florea_interface,
                      [input_image, input_audio, user_text, emotion, style_name, language, creativity, n_variants, seed, write_text, use_agent],
                      [out_gallery, out_title, out_concept, out_poem, out_story, out_palette, out_symbols, out_details])
        with gr.Tab("🗂️ Galerie"):
            refresh = gr.Button("🔄 Actualiser")
            ids = gr.State([])
            grid = gr.Gallery(label="Vos créations (cliquez pour lire le poème)", columns=4, height=480)
            g_text = gr.Markdown()
            zbtn = gr.Button("📦 Préparer un ZIP de toute la galerie")
            zfile = gr.File(label="Téléchargement")
            refresh.click(list_gallery, None, [grid, ids])
            demo.load(list_gallery, None, [grid, ids])
            grid.select(gallery_pick, [ids], [g_text])
            zbtn.click(export_zip, None, zfile)
    return demo, launch_kw


def launch(share=True, debug=False):
    demo, launch_kw = build_ui()
    demo.queue().launch(share=share, debug=debug, **launch_kw)
    return demo
