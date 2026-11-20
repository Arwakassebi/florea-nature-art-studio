"""Pipeline FLOREA (partie GPU) : modèles de perception, LLM, SD-Turbo et workflow LangGraph.

    perception → émotion → recherche (agent ReAct + garde-fou) → style (graphe + GCN)
               → concept → validate ⟲ → écriture → image → sauvegarde

Ce module importe PyTorch, Transformers et Diffusers : il n'est utilisé que sur machine GPU (Colab T4).
Tout le reste du paquet (styles, text, kb, mem, chain) est testé sans GPU.

Usage :
    from florea import pipeline as P
    P.init(llm_name="Qwen/Qwen2.5-1.5B-Instruct", save_dir="/content/florea_gallery")
    r = P.run_florea(user_text="Un champ de lavande au coucher du soleil", seed=42)
"""
import gc
import json
import math
import os
import random
import re
import tempfile
import time
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, TypedDict

import numpy as np
import torch
from PIL import Image, ImageOps
from pydantic import ValidationError

from florea.chain import build_chains, generate_concept, make_tools, run_research_agent
from florea.kb import ENTITY_BY_NAME, KnowledgeBase, StyleRecommender, doc_entity
from florea.mem import ModelManager
from florea.styles import ARTISTIC_STYLES, EMOTIONS_FR, LANGS, STYLE_AUTO
from florea.text import (FloreaConcept, clean_text, detect_emotion, detect_season, extract_palette, fallback_poem,
                         final_mood, first_sentence, trim_words)

# ----------------------------------------------------------------------------- état global (rempli par init())
CFG = SimpleNamespace(llm_name="Qwen/Qwen2.5-1.5B-Instruct", image_model="stabilityai/sd-turbo",
                      save_dir="/content/florea_gallery", use_agent=True, gpu_gb=0.0, low_memory=False)
MM: Optional[ModelManager] = None
KB = RECO = CHAINS = APP = None
_embedder = None
_progress, _stage = None, "démarrage"


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================ 1. Modèles
def _embed(texts):
    return np.asarray(_embedder.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True), dtype="float32")


# ---- BLIP : décrit la photo
BLIP_MODEL = "Salesforce/blip-image-captioning-base"


def _load_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL, torch_dtype=torch.float16).to("cuda").eval()
    return proc, model


def caption_image(image):
    proc, model = MM.get("blip")
    inputs = proc(images=image.convert("RGB"), return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, num_beams=4)
    return proc.decode(out[0], skip_special_tokens=True).strip()


def analyze_image(image):
    if image is None:
        return {"description": "", "palette": []}
    return {"description": caption_image(image), "palette": extract_palette(image)}


# ---- Audio : AST (sons) et Whisper (parole)
def _load_ast():
    from transformers import pipeline
    return pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", device=0)


def _load_whisper():
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0)


def classify_audio(audio, top_k=8):
    res = MM.get("ast")({"raw": audio, "sampling_rate": 16000}, top_k=top_k)
    return [{"label": r["label"], "score": float(r["score"])} for r in res]


def transcribe(wav_path):
    return MM.get("whisper")(wav_path, generate_kwargs={"task": "transcribe"})["text"].strip()


# Correspondance « son » → « ce qu'on voit sur l'image » (table lisible et modifiable)
SOUND_VISUAL = {
    "rain":    {"marks": "fine vertical strokes and droplets", "colors": ["slate grey", "silver blue", "deep green"]},
    "wind":    {"marks": "sweeping directional strokes, bending grass and branches", "colors": ["pale grey-green", "soft gold"]},
    "birds":   {"marks": "small bright flicks of color, light and airy", "colors": ["sunlit yellow", "fresh green"]},
    "water":   {"marks": "flowing horizontal ribbons and ripples, glassy reflections", "colors": ["aqua", "deep blue"]},
    "thunder": {"marks": "heavy dark clouds and sharp contrast", "colors": ["charcoal", "storm violet"]},
    "insects": {"marks": "dense stippled dots, warm summer haze", "colors": ["amber", "olive"]},
    "leaves":  {"marks": "overlapping layers of small leaves, soft rustling texture", "colors": ["leaf green", "ochre"]},
    "sea":     {"marks": "rolling waves and foam", "colors": ["teal", "sea foam white"]},
    "frogs":   {"marks": "dark damp ground with tiny glowing specks", "colors": ["deep green", "firefly yellow"]},
    "silence": {"marks": "wide empty space and a few delicate lines", "colors": ["pale cream", "soft grey"]},
}
KEYWORDS = {"rain": "rain", "raindrop": "rain", "drip": "rain", "thunder": "thunder", "wind": "wind", "rustle": "leaves",
            "rustling": "leaves", "leaves": "leaves", "thunderstorm": "thunder", "bird": "birds", "chirp": "birds",
            "tweet": "birds", "owl": "birds", "crow": "birds", "pigeon": "birds", "stream": "water", "water": "water",
            "trickle": "water", "gurgle": "water", "river": "water", "waves": "sea", "ocean": "sea", "surf": "sea",
            "insect": "insects", "cricket": "insects", "cicada": "insects", "bee": "insects", "buzz": "insects",
            "frog": "frogs", "croak": "frogs", "silence": "silence"}
# Émotion « naturelle » associée à chaque son (utilisée en mode Auto quand le texte ne dit rien)
SOUND_EMOTION = {"rain": "Mélancolie", "birds": "Joie", "wind": "Espoir", "water": "Sérénité", "thunder": "Admiration",
                 "insects": "Nostalgie", "leaves": "Tendresse", "sea": "Paix", "silence": "Solitude", "frogs": "Mystère"}
NOT_NATURE = ("instrument", "chime", "music", "guitar", "piano", "organ", "whistl")
SPEECH = ("speech", "conversation", "narration", "monologue")


def prepare_audio(audio_path):
    """Convertit n'importe quel format en WAV 16 kHz mono (30 s max)."""
    import librosa
    from pydub import AudioSegment
    wav_path = os.path.join(tempfile.gettempdir(), "florea_audio.wav")
    try:
        sound = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(16000)
        sound.export(wav_path, format="wav")
        audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    except Exception:
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        import soundfile as sf
        sf.write(wav_path, audio, 16000)
    return wav_path, audio[: 16000 * 30]


def analyze_audio(audio_path):
    empty = {"sounds": [], "nature": [], "transcription": "", "visual": {}, "summary": ""}
    if not audio_path:
        return empty
    wav_path, audio = prepare_audio(audio_path)
    if len(audio) < 1600:
        return empty
    sounds = classify_audio(audio)
    nature = []
    for s in sounds:
        lab = s["label"].lower()
        if any(x in lab for x in NOT_NATURE):
            continue
        for kw, key in KEYWORDS.items():
            if re.search(r"\b" + re.escape(kw) + r"(s|ing)?\b", lab) and key not in nature:
                nature.append(key)
    nature = nature[:3]
    speech_score = max([s["score"] for s in sounds if any(x in s["label"].lower() for x in SPEECH)] or [0])
    text = ""
    if speech_score >= 0.2:                      # Whisper seulement s'il y a de la parole (évite les textes inventés)
        try:
            text = transcribe(wav_path)
        except Exception as e:
            print("[Whisper]", e)
    visual = {"marks": [SOUND_VISUAL[k]["marks"] for k in nature], "colors": [c for k in nature for c in SOUND_VISUAL[k]["colors"]]}
    summary = ", ".join(f"{s['label']} ({s['score']:.2f})" for s in sounds[:4])
    return {"sounds": sounds, "nature": nature, "transcription": text, "visual": visual, "summary": summary}


# ---- LLM Qwen2.5
def _load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG.llm_name)
    model = AutoModelForCausalLM.from_pretrained(CFG.llm_name, torch_dtype=torch.float16, device_map="auto")
    return tok, model


def llm_generate(messages, max_new_tokens=400, temperature=0.7, tools=None):
    """messages = [{"role": ..., "content": ...}] ; tools = schémas d'outils (le gabarit Qwen2.5 sait les présenter au modèle)."""
    tok, model = MM.get("llm")
    extra = {"tools": tools} if tools else {}
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                             temperature=temperature if temperature > 0 else None, top_p=0.9 if temperature > 0 else None,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---- SD-Turbo : un seul jeu de poids pour texte→image ET photo→image
def _load_sd():
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
    t2i = AutoPipelineForText2Image.from_pretrained(CFG.image_model, torch_dtype=torch.float16, variant="fp16")
    if CFG.gpu_gb >= 14:
        t2i = t2i.to("cuda")                   # tout sur le GPU : rapide
    else:
        t2i.enable_model_cpu_offload()         # petit GPU : plus lent mais ça passe
    return t2i, AutoPipelineForImage2Image.from_pipe(t2i)      # img2img partage les poids : aucune mémoire en plus


def sd_generate(prompt, init_image=None, strength=0.65, steps=4, seed=42):
    """Sans photo : texte→image. Avec photo : la photo est transformée dans le style demandé."""
    t2i, i2i = MM.get("sd")
    g = torch.Generator(device="cuda").manual_seed(int(seed))
    if init_image is not None:
        n = max(steps, math.ceil(1.0 / max(strength, 0.05)))   # SD-Turbo exige steps × strength ≥ 1
        return i2i(prompt=prompt, image=init_image, strength=strength, num_inference_steps=n,
                   guidance_scale=0.0, generator=g).images[0]
    return t2i(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0, height=512, width=512, generator=g).images[0]


# ============================================================================ 2. Création (LCEL + Pydantic)
def inputs_summary(st, with_style=True):
    rows = []
    if st["image_description"]:
        rows.append(f"Photo: {st['image_description']}. Dominant colors: {', '.join(st['palette_names'])}.")
    a = st["audio"]
    if a.get("nature"):
        rows.append("Ambient sounds: " + ", ".join(a["nature"]) + ". How they look: " + "; ".join(a["visual"]["marks"]) + ".")
    if a.get("transcription"):
        rows.append(f"Spoken words (transcribed): {a['transcription']}")
    if st["user_text"].strip():
        rows.append(f"The person's words: {st['user_text'].strip()}")
    rows.append(f"Emotion: {EMOTIONS_FR[st['emotion']]}.")
    if with_style and st.get("style_name"):
        rows.append(f"Art style: {st['style_name']}. {ARTISTIC_STYLES[st['style_name']]['description']}")
    return "\n".join(rows)


def graph_text(links):
    return "\n".join(f"- {l['label']} ({l['type']}): " + " ; ".join(l["path"] or []) for l in links) or "(none)"


def fallback_concept(st):
    """Concept de secours, valide par construction : utilisé si le LLM ne produit rien d'exploitable."""
    ents = st.get("entities") or []
    label = (ents[0] if ents else (trim_words(st["user_text"], 2) or "nature")).title()
    ent = ENTITY_BY_NAME.get(ents[0]) if ents else None
    short = st["style_name"].split(" — ")[0]
    cands = [st["image_description"]] if st["image_description"] else []
    if st["retrieved"]:
        cands.append(first_sentence(st["retrieved"][0]["document"].split(":", 1)[-1], 20))
    cands.append("a quiet natural landscape in soft light")
    palette = (st["palette_names"][:4] or st["audio"].get("visual", {}).get("colors", [])[:3] or ["soft green", "cream"])
    base = dict(title=f"{label} — {st['emotion']}",
                concept=f"Une œuvre {short} inspirée par {label.lower()} et par un sentiment de {st['emotion'].lower()}.",
                palette=palette, symbols=(ent["symbols"][:3] if ent else []) or ["nature"], mood=EMOTIONS_FR[st["emotion"]])
    for subject in cands:                                  # premier sujet qui passe la validation (anglais, ≥ 4 mots)
        try:
            return FloreaConcept(subject=trim_words(subject, 28), **base)
        except ValidationError:
            continue
    return FloreaConcept(subject="a quiet natural landscape in soft light", **{**base, "palette": ["soft green", "cream"]})


def make_concept(st):
    variables = {"inputs": inputs_summary(st), "language": LANGS[st["language"]], "graph": graph_text(st["graph_links"]),
                 "knowledge": "\n".join(f"[{i + 1}] {d['document']}" for i, d in enumerate(st["retrieved"])) or "(none)"}
    concept, info = generate_concept(CHAINS, variables, fallback_concept(st))
    d = concept.model_dump()
    d["mood"] = final_mood(st["emotion"], d["mood"])        # l'ambiance envoyée à l'image est toujours en anglais
    return d, info


def write_texts(st, c):
    ctx = (f"Title: {c['title']}\nIdea: {c['concept']}\nSymbols: {', '.join(c['symbols'])}\nMood: {c['mood']}\n"
           f"Style: {st['style_name']}\nThe person's words: {st['user_text'].strip() or '(none)'}\nWhat the picture shows: {c['subject']}")
    variables = {"language": LANGS[st["language"]], "context": ctx}
    try:
        poem = clean_text(CHAINS["poem"].invoke(variables))
    except Exception as e:
        print("[poème]", e)
        poem = ""
    if len([l for l in poem.splitlines() if l.strip()]) < 4:
        poem = fallback_poem(c["title"], c["symbols"][0], st["language"])
    try:
        story = clean_text(CHAINS["story"].invoke(variables))
    except Exception as e:
        print("[histoire]", e)
        story = ""
    if len(story.split()) < 12:
        story = c["concept"]
    return poem, story


# ============================================================================ 3. Workflow LangGraph
class FloreaError(Exception):
    """Erreur lisible pour l'utilisateur (mémoire GPU saturée, étape qui échoue…)."""


class FloreaState(TypedDict, total=False):
    image: object
    audio_path: Optional[str]
    user_text: str
    emotion_in: str
    style_in: str
    language: str
    creativity: float
    n_variants: int
    seed: int
    write_text: bool
    use_agent: bool
    image_description: str
    palette_names: list
    audio: dict
    emotion: str
    retrieved: list
    entities: list
    season: Optional[str]
    ranked_styles: list
    graph_links: list
    style_name: str
    concept: dict
    concept_info: dict
    concept_retries: int
    validation_ok: bool
    poem: str
    story: str
    final_prompt: str
    images: list
    paths: list
    timings: dict
    trace: list


def tick(frac, desc):
    global _stage
    _stage = desc
    print(f"[{int(frac * 100):3d}%] {desc}")
    if _progress:
        _progress(frac, desc=desc)


def _t(state, name, t0):
    return {**state.get("timings", {}), name: round(time.time() - t0, 1)}


def perception_node(state):
    t0 = time.time()
    tick(0.05, "Analyse de la photo et du son")
    img = analyze_image(state.get("image"))
    aud = analyze_audio(state.get("audio_path"))
    if CFG.low_memory:                                   # petit GPU -> on libère les modèles de perception
        for name in ("blip", "ast", "whisper"):
            MM.unload(name)
    return {"image_description": img["description"], "palette_names": img["palette"], "audio": aud,
            "timings": _t(state, "perception", t0), "trace": []}


def emotion_node(state):
    emo = state.get("emotion_in", "Auto (détecter)")
    if emo not in EMOTIONS_FR:
        text = f"{state.get('user_text', '')} {state['audio'].get('transcription', '')}"
        emo = detect_emotion(text, None) or SOUND_EMOTION.get((state["audio"].get("nature") or [""])[0], "Sérénité")
    return {"emotion": emo}


def research_node(state):
    """Un agent ReAct DÉCIDE d'appeler search_knowledge ; sinon un garde-fou lance la recherche (le petit LLM peut ne pas appeler l'outil)."""
    t0 = time.time()
    tick(0.15, "Recherche de connaissances (agent)")
    ctx = {"style_name": None, "retrieved": [], "calls": []}
    tools = make_tools(KB, ctx)
    trace = []
    if state.get("use_agent", True):
        try:
            trace += run_research_agent(llm_generate, tools, inputs_summary(state, with_style=False))
        except Exception as e:
            trace.append(f"agent: erreur ({type(e).__name__}) -> garde-fou")
    if not ctx["retrieved"]:
        a = state["audio"]
        q = " ".join(filter(None, [state["image_description"], " ".join(f"{k} sound" for k in a.get("nature", [])),
                                   a.get("transcription", ""), state["user_text"], EMOTIONS_FR[state["emotion"]]]))
        tools["search_knowledge"].invoke({"query": q})
        trace.append("garde-fou : recherche lancée automatiquement")
    return {"retrieved": ctx["retrieved"][:6], "trace": state.get("trace", []) + trace + ctx["calls"],
            "timings": _t(state, "recherche", t0)}


def style_node(state):
    """Graph-RAG : entités trouvées → graphe (chemins explicables) + GCN (classement des styles). Mode Auto : tirage parmi le top 3."""
    ents = [e for e in (doc_entity(d["document"]) for d in state["retrieved"]) if e]
    text = f"{state['user_text']} {state['audio'].get('transcription', '')}"
    season = detect_season(text)
    ranked = RECO.recommend(ents, state["emotion"], season, state["palette_names"], k=4)
    rng = random.Random(state["seed"])
    if state.get("style_in") in ARTISTIC_STYLES:
        style = state["style_in"]
    elif ranked:
        style = rng.choice([n for n, _ in ranked[:3]])
    else:
        style = rng.choice(sorted(ARTISTIC_STYLES))
    return {"entities": ents, "season": season, "style_name": style, "ranked_styles": [(n, round(s, 2)) for n, s in ranked],
            "graph_links": RECO.explain(ents, state["emotion"], season, state["palette_names"])}


def concept_node(state):
    t0 = time.time()
    tick(0.3, "Conception artistique (chaîne LCEL + Pydantic)")
    concept, info = make_concept(state)
    return {"concept": concept, "concept_info": info, "timings": _t(state, "concept", t0)}


def validate_node(state):
    """Bloque si le concept est invalide : relance une fois, sinon route_validate lève une ValueError."""
    try:
        FloreaConcept.model_validate(state["concept"])
        return {"validation_ok": True}
    except ValidationError:
        return {"validation_ok": False, "concept_retries": state.get("concept_retries", 0) + 1}


def route_validate(state):
    if state["validation_ok"]:
        return "ecriture"
    if state.get("concept_retries", 0) <= 1:
        return "concept"
    raise ValueError("Concept invalide (title ou subject vide) après relance.")


def writing_node(state):
    t0 = time.time()
    if not state.get("write_text", True):
        return {"poem": "", "story": ""}
    tick(0.5, "Écriture du poème et de l'histoire")
    poem, story = write_texts(state, state["concept"])
    return {"poem": poem, "story": story, "timings": _t(state, "texte", t0)}


def image_node(state):
    t0 = time.time()
    c, photo = state["concept"], state.get("image")
    tools = make_tools(KB, {"style_name": state["style_name"]})          # même outil que celui du serveur MCP
    prompt = tools["generate_image_prompt"].invoke({"subject": c["subject"], "palette": ", ".join(c["palette"]), "mood": c["mood"]})
    if CFG.low_memory:
        MM.unload("llm")                                                 # on libère le LLM avant de peindre
    src = ImageOps.fit(photo.convert("RGB"), (512, 512), method=Image.LANCZOS) if photo is not None else None
    n = max(1, min(4, int(state.get("n_variants", 1))))
    imgs = []
    for k in range(n):
        tick(0.6 + 0.3 * k / n, f"Peinture {k + 1}/{n}")
        imgs.append(sd_generate(prompt, src, float(state.get("creativity", 0.65)), 4, state["seed"] + k))
    return {"final_prompt": prompt, "images": imgs, "timings": _t(state, "image", t0)}


def save_node(state):
    base = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    paths = []
    for k, img in enumerate(state["images"]):
        pid = f"{base}-{k + 1}"
        img.save(os.path.join(CFG.save_dir, pid + ".png"))
        meta = {"id": pid, "created_at": datetime.now().isoformat(timespec="seconds"), "title": state["concept"]["title"],
                "style": state["style_name"], "emotion": state["emotion"], "language": state["language"], "seed": state["seed"] + k,
                "prompt": state["final_prompt"], "concept": state["concept"], "poem": state["poem"], "story": state["story"],
                "inputs": {"text": state["user_text"], "photo": state["image_description"],
                           "sounds": state["audio"].get("summary", ""), "voice": state["audio"].get("transcription", "")},
                "knowledge": [d["document"] for d in state["retrieved"]], "ranked_styles": state["ranked_styles"],
                "graph_links": state["graph_links"], "trace": state["trace"], "timings": state["timings"],
                "models": {"llm": CFG.llm_name, "image": CFG.image_model}}
        with open(os.path.join(CFG.save_dir, pid + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        paths.append(os.path.join(CFG.save_dir, pid + ".png"))
    tick(1.0, "Terminé ✅")
    return {"paths": paths}


def build_workflow():
    from langgraph.graph import END, START, StateGraph
    wf = StateGraph(FloreaState)
    for name, fn in [("perception", perception_node), ("emotion", emotion_node), ("recherche", research_node),
                     ("style", style_node), ("concept", concept_node), ("validate", validate_node),
                     ("ecriture", writing_node), ("image", image_node), ("sauvegarde", save_node)]:
        wf.add_node(name, fn)
    wf.add_edge(START, "perception")
    for a, b in [("perception", "emotion"), ("emotion", "recherche"), ("recherche", "style"), ("style", "concept"),
                 ("concept", "validate"), ("ecriture", "image"), ("image", "sauvegarde")]:
        wf.add_edge(a, b)
    wf.add_conditional_edges("validate", route_validate, {"concept": "concept", "ecriture": "ecriture"})
    wf.add_edge("sauvegarde", END)
    return wf.compile()


# ============================================================================ 4. Initialisation et point d'entrée
def init(llm_name=None, image_model=None, save_dir=None, use_agent=None, evaluate=True):
    """Charge l'embedder, construit la base de connaissances / le graphe / le GCN, enregistre les modèles, compile le workflow."""
    global MM, KB, RECO, CHAINS, APP, _embedder
    if not torch.cuda.is_available():
        raise RuntimeError("Activez le GPU : Exécution > Modifier le type d'exécution > GPU (T4).")
    from sentence_transformers import SentenceTransformer
    CFG.llm_name = llm_name or CFG.llm_name
    CFG.image_model = image_model or CFG.image_model
    CFG.save_dir = save_dir or CFG.save_dir
    CFG.use_agent = CFG.use_agent if use_agent is None else bool(use_agent)
    CFG.gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    CFG.low_memory = CFG.gpu_gb < 12
    os.makedirs(CFG.save_dir, exist_ok=True)
    print(f"GPU : {torch.cuda.get_device_name(0)} ({CFG.gpu_gb:.0f} Go) | mode économe : {CFG.low_memory}")

    MM = ModelManager(budget_gb=max(CFG.gpu_gb - 3.0, 4.0), free_fn=clean_memory)     # réserve 3 Go pour les calculs
    llm_gb = 1.3 if "0.5B" in CFG.llm_name else 3.5 if "1.5B" in CFG.llm_name else 6.8 if "3B" in CFG.llm_name else 15.0
    MM.register("blip", _load_blip, 1.0)
    MM.register("ast", _load_ast, 0.5)
    MM.register("whisper", _load_whisper, 0.4)
    MM.register("llm", _load_llm, llm_gb)
    MM.register("sd", _load_sd, 3.0)

    _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")     # petit : gardé en mémoire
    KB = KnowledgeBase(embed_fn=_embed)
    RECO = StyleRecommender()
    CHAINS = build_chains(llm_generate)
    APP = build_workflow()
    print(f"✅ FAISS : {len(KB.docs)} documents | graphe : {len(RECO.graph.nodes)} nœuds, {len(RECO.graph.edges())} liens | GCN entraîné.")
    if evaluate:
        from florea.kb import evaluate_gcn
        print("Évaluation du GCN sur liens masqués :", evaluate_gcn(n_splits=5))


def run_florea(image=None, audio_path=None, user_text="", emotion="Auto (détecter)", style_name=STYLE_AUTO, language="Français",
               creativity=0.65, n_variants=1, seed=-1, write_text=True, use_agent=None, progress=None):
    """Photo, audio, texte : n'importe quelle combinaison. Renvoie l'état final ; lève ValueError (entrée) ou FloreaError (échec)."""
    global _progress, _stage
    if APP is None:
        raise RuntimeError("Appelez florea.pipeline.init() d'abord.")
    if image is None and not audio_path and not (user_text or "").strip():
        raise ValueError("Donnez au moins une entrée : une photo, un son ou quelques mots.")
    if seed in (None, "", -1):
        seed = random.randint(0, 2**31 - 100)
    _progress, _stage = progress, "démarrage"
    state = {"image": image.convert("RGB") if image is not None else None, "audio_path": audio_path, "user_text": user_text or "",
             "emotion_in": emotion, "style_in": style_name, "language": language, "creativity": float(creativity),
             "n_variants": int(n_variants), "seed": int(seed), "write_text": bool(write_text),
             "use_agent": CFG.use_agent if use_agent is None else bool(use_agent), "timings": {}, "trace": [], "concept_retries": 0}
    try:                                                                  # jamais de stack trace brute pour l'utilisateur
        return APP.invoke(state)
    except FloreaError:
        raise
    except Exception as e:
        msg = str(e).lower()
        if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
            MM.unload_all()
            clean_memory()
            raise FloreaError(f"⚠️ Mémoire GPU saturée à l'étape « {_stage} ». Les modèles ont été libérés : réessayez avec moins de "
                              f"variantes, ou utilisez Qwen2.5-0.5B-Instruct (paramètre llm_name de init()).") from e
        raise FloreaError(f"Erreur à l'étape « {_stage} » ({type(e).__name__}) : {e}") from e
    finally:
        _progress = None
