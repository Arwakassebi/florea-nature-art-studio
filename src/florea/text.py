"""Outils texte : émotions, couleurs, JSON tolérant, validation Pydantic du concept, prompt image.

Aucune dépendance GPU : testé par pytest, utilisé par le notebook et par le serveur MCP.
"""
import json
import re
import unicodedata

from pydantic import BaseModel, Field, field_validator

from florea.styles import ARTISTIC_STYLES, EMOTIONS_FR

# ----------------------------------------------------------------------------- émotions
EMOTION_LEXICON = {
    "Nostalgie": ["souvenir", "enfance", "grand-mère", "grand-père", "autrefois", "oublié", "jadis", "memor", "remember", "childhood", "forgotten", "nostalg"],
    "Mélancolie": ["triste", "mélanc", "pluie", "gris", "automne", "larme", "sad", "rain", "autumn", "melanch"],
    "Joie": ["joie", "heureu", "rire", "fête", "soleil", "joy", "happy", "laugh", "sunny"],
    "Sérénité": ["calme", "silence", "doux", "douce", "tranquill", "serein", "quiet", "calm", "gentle", "still"],
    "Paix": ["paix", "repos", "apais", "peace", "rest"],
    "Mystère": ["mystè", "secret", "brume", "brouillard", "ombre", "caché", "mystery", "mist", "fog", "hidden", "shadow"],
    "Émerveillement": ["merveill", "magi", "rêve", "féerique", "wonder", "magic", "dream"],
    "Passion": ["passion", "amour", "ardent", "feu", "love", "fire"],
    "Espoir": ["espoir", "aube", "renaiss", "printemps", "nouveau", "hope", "dawn", "spring", "renew"],
    "Solitude": ["seul", "solitude", "vide", "isolé", "alone", "empty"],
    "Tendresse": ["tendre", "fragile", "délicat", "petite", "tender", "delicate", "tiny"],
    "Admiration": ["immense", "majest", "vaste", "grandios", "sublime", "vast", "awe"],
}


def detect_emotion(text, default="Sérénité"):
    """Émotion dominante d'un texte français/anglais (None possible si default=None)."""
    t = (text or "").lower()
    best, score = None, 0
    for emo, words in EMOTION_LEXICON.items():
        s = sum(1 for w in words if re.search(r"\b" + re.escape(w), t))
        if s > score:
            best, score = emo, s
    return best or default


SEASON_WORDS = {"spring": ["printemps", "spring"], "summer": ["été", "ete", "summer", "estival"],
                "autumn": ["automne", "autumn", "fall"], "winter": ["hiver", "winter", "neige", "snow"]}


def detect_season(text):
    """'spring' | 'summer' | 'autumn' | 'winter' | None (mots français ou anglais)."""
    t = (text or "").lower()
    for season, words in SEASON_WORDS.items():
        if any(re.search(r"\b" + w + r"\b", t) for w in words):
            return season
    return None


# ----------------------------------------------------------------------------- couleurs
NAMED_COLORS = {
    "white": (245, 245, 240), "cream": (240, 230, 200), "pale pink": (245, 200, 210), "pink": (235, 130, 170),
    "magenta": (200, 40, 130), "red": (200, 30, 40), "crimson": (150, 20, 40), "orange": (240, 140, 40),
    "golden yellow": (240, 200, 50), "pale yellow": (245, 235, 150), "lime green": (160, 200, 60),
    "leaf green": (70, 150, 60), "deep green": (25, 80, 45), "sage green": (150, 170, 130), "teal": (40, 140, 140),
    "sky blue": (130, 190, 235), "blue": (50, 100, 200), "indigo": (50, 50, 130), "lavender": (180, 160, 220),
    "violet": (120, 60, 160), "brown": (120, 80, 50), "tan": (200, 170, 130), "grey": (130, 130, 135),
    "charcoal": (45, 45, 50), "black": (15, 15, 15), "silver": (190, 195, 200)}


def name_color(rgb):
    return min(NAMED_COLORS, key=lambda n: sum((a - b) ** 2 for a, b in zip(rgb, NAMED_COLORS[n])))


def extract_palette(image, k=5):
    """Couleurs dominantes d'une photo PIL, en noms anglais (quantification median-cut)."""
    from PIL import Image
    q = image.convert("RGB").resize((96, 96)).quantize(colors=k, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()[:k * 3]
    names = []
    for _, idx in sorted(q.getcolors(), reverse=True):
        n = name_color(tuple(pal[idx * 3: idx * 3 + 3]))
        if n not in names:
            names.append(n)
    return names


FR_COLORS = {"rouge": "red", "bleu": "blue", "bleue": "blue", "vert": "green", "verte": "green", "jaune": "yellow",
             "rose": "pink", "blanc": "white", "blanche": "white", "noir": "black", "noire": "black", "gris": "grey",
             "grise": "grey", "marron": "brown", "doré": "golden", "dorée": "golden", "argenté": "silver",
             "argentée": "silver", "violette": "violet", "lavande": "lavender", "ciel": "sky", "clair": "light",
             "claire": "light", "foncé": "dark", "foncée": "dark", "pâle": "pale", "doux": "soft", "douce": "soft",
             "beige": "beige", "turquoise": "turquoise", "ocre": "ochre", "bordeaux": "burgundy", "corail": "coral"}
FR_STOP = {"le", "la", "les", "un", "une", "des", "du", "de", "et", "au", "aux", "avec", "qui", "que", "est", "sous",
           "dans", "sur", "d", "l", "ce", "cette", "ses", "son", "sa"}

# Mots français courants SANS accent (donc invisibles pour le test des lettres accentuées) ; on évite les mots aussi anglais.
FR_WORDS = {"champ", "champs", "lavande", "soleil", "coucher", "ciel", "mer", "plage", "arbre", "arbres", "fleur", "fleurs",
            "jardin", "foret", "montagne", "montagnes", "pluie", "vent", "nuit", "matin", "soir", "lumiere", "couleur",
            "couleurs", "eau", "riviere", "oiseau", "oiseaux", "nuage", "nuages", "paysage", "souvenir", "souvenirs",
            "enfance", "douceur", "calme", "tendresse", "doux", "douce", "vert", "verte", "bleu", "bleue", "rouge",
            "jaune", "blanc", "blanche", "noir", "noire", "avec", "pour", "vers", "entre", "sans", "tres", "mon", "ma",
            "mes", "ton", "ta", "nostalgique", "nostalgie", "mystere", "reve", "ete", "hiver", "automne", "printemps"}


def looks_english(text):
    """Heuristique : pas de lettres accentuées / arabes, pas de mots-outils ni de mots courants français."""
    for ch in text:
        if ord(ch) > 127 and unicodedata.category(ch).startswith("L"):
            return False
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and not any(w in FR_STOP or w in FR_WORDS for w in words)


# Liste blanche : une précision d'ambiance du LLM n'est acceptée que si TOUS ses mots sont ici (donc anglais sûr).
ENGLISH_MOOD_WORDS = set("""gentle quiet warm soft dreamy dreamlike calm serene bright tender longing wistful peaceful hopeful
joyful playful melancholic nostalgic mysterious magical luminous delicate fragile tranquil still deep lonely solitary vast
majestic dramatic intense vibrant lively cheerful sweet sad gloomy misty hazy golden cozy fresh airy serenity nostalgia wonder
mystery hope joy peace awe passion tenderness solitude melancholy stillness silence reverie bittersweet sunny moody romantic
spiritual meditative reflective mellow ethereal enchanted graceful harmonious timeless haunting uplifting radiant soothing
restful contemplative and""".split())


def normalize_colors(items):
    """Traduit les couleurs françaises courantes ; supprime ce qui n'est pas de l'anglais."""
    out = []
    for c in items:
        words = [FR_COLORS.get(w, w) for w in str(c).lower().replace("-", " ").split()]
        if len(words) > 1 and words[-1] in ("light", "dark", "pale", "soft", "sky"):    # « vert clair » -> « light green »
            words = [words[-1]] + words[:-1]
        t = " ".join(words).strip()
        if t and looks_english(t) and t not in out:
            out.append(t)
    return out


# ----------------------------------------------------------------------------- JSON tolérant
def extract_json(text):
    """Premier objet JSON d'un texte de LLM (``` , bavardage, virgules en trop). None si impossible."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text).strip()
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start:i + 1]
                for cand in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                    try:
                        obj = json.loads(cand)
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        pass
                return None
    return None


def as_list(v, max_n=6):
    if isinstance(v, str):
        v = [p.strip() for p in re.split(r"[;,\n]", v)]
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()][:max_n]
    return []


def trim_words(s, n):
    return " ".join(str(s).split()[:n]).rstrip(",;:.")


def first_sentence(s, n=25):
    return trim_words(re.split(r"(?<=[.!?])\s+", s.strip(), maxsplit=1)[0], n)


def clean_text(t, max_lines=16):
    t = re.sub(r"```.*?```", "", t, flags=re.S).strip().strip('"')
    lines, blank = [], 0
    for line in t.splitlines():
        if not line.strip():
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        lines.append(line.rstrip())
    return "\n".join(lines[:max_lines]).strip()


# ----------------------------------------------------------------------------- schéma Pydantic du concept
class FloreaConcept(BaseModel):
    """Concept artistique. Le LLM doit produire exactement ces champs ; chaque champ est validé."""
    title: str = Field(description="Titre évocateur de 2 à 6 mots, dans la langue demandée")
    concept: str = Field(description="Deux phrases expliquant l'idée artistique, dans la langue demandée")
    subject: str = Field(description="EN ANGLAIS : description visuelle concrète de ce que montre l'image (max 25 mots)")
    palette: list[str] = Field(description="EN ANGLAIS : liste de 3 à 5 noms de couleurs")
    symbols: list[str] = Field(description="Liste de 2 ou 3 symboles naturels courts, dans la langue demandée")
    mood: str = Field(description="EN ANGLAIS : 2 ou 3 mots d'ambiance")

    @field_validator("title")
    @classmethod
    def _title(cls, v):
        v = " ".join(str(v).split())
        if not v or len(v.split()) > 10:
            raise ValueError("title vide ou trop long")
        return v

    @field_validator("concept")
    @classmethod
    def _concept(cls, v):
        v = " ".join(str(v).split())
        if len(v.split()) < 5:
            raise ValueError("concept trop court (5 mots minimum)")
        return v

    @field_validator("subject")
    @classmethod
    def _subject(cls, v):
        v = trim_words(v, 28)
        if len(v.split()) < 4:
            raise ValueError("subject trop court (4 mots minimum)")
        if not looks_english(v):
            raise ValueError("subject doit être en anglais")
        return v

    @field_validator("palette", mode="before")
    @classmethod
    def _palette(cls, v):
        colors = normalize_colors(as_list(v, 8))[:5]
        if len(colors) < 2:
            raise ValueError("palette : au moins 2 couleurs en anglais")
        return colors

    @field_validator("symbols", mode="before")
    @classmethod
    def _symbols(cls, v):
        s = as_list(v, 5)[:3]
        if not s:
            raise ValueError("symbols vide")
        return s

    @field_validator("mood")
    @classmethod
    def _mood(cls, v):
        v = trim_words(v, 4)
        if not v:
            raise ValueError("mood vide")
        return v


def final_mood(emotion_fr, llm_mood=""):
    """Ambiance TOUJOURS en anglais : émotion canonique (EMOTIONS_FR) + précision du LLM seulement si elle est en anglais."""
    base = EMOTIONS_FR[emotion_fr]
    extra = trim_words(llm_mood or "", 3)
    ok = bool(extra) and all(w in ENGLISH_MOOD_WORDS for w in re.findall(r"[a-z]+", extra.lower())) and not re.search(r"[^\x00-\x7f]", extra)
    return f"{base}, {extra}" if ok and extra.lower() != base else base


# ----------------------------------------------------------------------------- prompt image
def build_image_prompt(style_name, subject, palette, mood, max_words=60):
    """Prompt assemblé par le code : l'ordre = la priorité (le CLIP de Stable Diffusion lit ~77 jetons)."""
    parts = [ARTISTIC_STYLES[style_name]["booster"], subject,
             ("color palette of " + ", ".join(palette[:4])) if palette else "",
             (mood + " mood") if mood else "", "no text, no watermark"]
    return trim_words(", ".join(p for p in parts if p), max_words)


def fallback_poem(title, symbol, language):
    if language == "Français":
        return (f"{title}\nUne lumière douce se pose sur {symbol}.\nLes couleurs respirent, lentes et claires.\n"
                "Quelque chose d'oublié revient tout bas.\nLe temps s'arrête au bord d'un souffle.\n"
                "Je regarde, et le monde me regarde.\nRien n'est pressé.\nTout est là.")
    return (f"{title}\nA soft light settles on {symbol}.\nThe colors breathe, slow and clear.\n"
            "Something forgotten returns in a whisper.\nTime stops at the edge of a breath.\n"
            "I look, and the world looks back.\nNothing is in a hurry.\nEverything is here.")
