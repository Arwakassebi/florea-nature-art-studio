"""Base de connaissances : documents, entités structurées, recherche FAISS, graphe (Graph-RAG) et GCN en NumPy.

Aucune dépendance GPU. FAISS est utilisé s'il est installé, sinon NumPy (même résultat).
"""
import hashlib
import re

import numpy as np

from florea.styles import ARTISTIC_STYLES, style_family

# ============================================================================ 1. Documents (texte, pour la recherche vectorielle)
NATURE_DOCS = [
    # ---- Fleurs ----
    "Lavender: purple flower spikes on silver-green leaves, Mediterranean summer, humming bees, calm and devotion.",
    "Rose: layered petals around a golden center, thorny stems; love, beauty and the fragility of time.",
    "Sunflower: a huge golden head of spiraling florets that follows the sun; warmth, optimism and loyalty.",
    "Poppy: four crumpled scarlet petals in fields, very short-lived; remembrance, sleep and consolation.",
    "Daisy: white rays around a yellow disc in short grass, closes at dusk; innocence and new beginnings.",
    "Cherry blossom: clouds of pale pink five-petalled flowers lasting about a week; transience and renewal.",
    "Tulip: a single cup-shaped flower on a smooth stem in spring; perfect love and joyful colors.",
    "Lily of the valley: tiny white bells on an arching stem in shady woods, sweet scent; return of happiness.",
    "Forget-me-not: small sky-blue flowers with a yellow eye near streams; remembrance and faithful love.",
    "Bluebell: drooping blue-violet bells forming carpets in spring woodland; constancy and nostalgia.",
    "Dandelion: yellow flower that becomes a round seed clock blown by wind; wishes, resilience and childhood.",
    "Lotus: pink flower rising from mud above water-repellent leaves; purity, calm and rebirth.",
    "Water lily: round floating leaves and star-shaped flowers on still ponds that close at dusk; tranquility.",
    "Iris: three upright and three drooping petals, blue or purple, beside water; hope and messages.",
    "Peony: heavy ruffled blooms in pink, white or red in late spring; prosperity and tenderness.",
    "Wisteria: hanging chains of fragrant violet flowers on twisting climbers; devotion and longing.",
    "Jasmine: small star-shaped white flowers with a scent strongest at dusk, strung into bouquets in Tunisia; welcome and tenderness.",
    "Bougainvillea: vivid magenta papery bracts spilling over white sunny walls in Mediterranean towns; summer warmth.",
    "Snowdrop: nodding white flowers pushing through frost in late winter; hope and consolation.",
    "Cornflower: intense blue ruffled flowers in wheat fields; delicacy, hope and summer nostalgia.",
    "Foxglove: tall spires of speckled purple bells in woodland clearings; enchantment, poisonous and folkloric.",
    "Orchid: elegant asymmetric flowers with a lip petal, often on tree branches in humid forests; rare beauty and luxury.",
    "Marigold: bright orange daisy-like flowers that bloom for months; warmth and remembrance.",
    "Lilac: dense fragrant purple clusters in late spring; first love and memories of youth.",
    # ---- Arbres & plantes ----
    "Olive tree: silver-green leaves, gnarled ancient trunk, groves on dry Mediterranean hills; peace and endurance.",
    "Date palm: tall trunk with a crown of feathery fronds in oases; hospitality, sweetness and desert life.",
    "Prickly pear cactus: flat paddles with spines and bright fruits along Mediterranean roads; resilience.",
    "Oak: a broad long-lived tree with lobed leaves and acorns; strength and endurance.",
    "Weeping willow: trailing branches over water, silver-green leaves flickering; mourning, grace and melancholy.",
    "Silver birch: white peeling bark and light dappled canopy, golden leaves in autumn; renewal and quiet.",
    "Pine tree: evergreen needles, orange bark and resin scent on mountain slopes; longevity and solitude.",
    "Japanese maple: delicate hand-shaped leaves turning crimson in autumn; passing time and beauty.",
    "Ginkgo: fan-shaped leaves turning bright yellow and falling in days; a living fossil, longevity and resilience.",
    "Fern: feathery fronds unrolling from fiddleheads in damp shade, no flowers; ancient mystery and calm.",
    "Moss: soft green velvet carpets on stones and bark in damp shade; patience and quiet time.",
    "Bamboo: hollow jointed stems that bend in the wind and rustle; flexibility and serenity.",
    "Cypress: tall dark flame-shaped evergreen on hillsides and in cemeteries; eternity and quiet melancholy.",
    "Fig tree: broad lobed leaves casting deep shade and sweet fruit in summer; abundance and Mediterranean afternoons.",
    # ---- Ambiances & paysages ----
    "Spring meadow: short grass full of wildflowers, bees, dew on blades, soft green-gold morning light.",
    "Ancient woodland in spring: light reaches the forest floor before the canopy closes, carpets of bluebells and anemones, dappled light, birdsong.",
    "Autumn woodland: leaves turn gold, orange and red, then fall; low golden light, melancholy and transience.",
    "Still pond: calm water mirrors sky and trees, reeds, dragonflies, floating leaves; stillness and reflection.",
    "Mossy forest floor: damp shade, ferns, fungi, fallen logs slowly decomposing; green, brown and silver, mystery.",
    "Mediterranean hillside: hot dry light, sharp shadows, silver olive leaves, aromatic herbs, cicadas.",
    "Winter garden: bare branches, frost, seed heads, snowdrops starting early; silver, brown and pale green, quiet hope.",
    "Rain in the forest: drips and thin curtains through the canopy, dark glistening bark, the smell of wet earth called petrichor.",
    "Dawn dew: droplets on petals, grass tips and spider webs catching the first light like tiny lenses.",
    "Wind in tall grass: seed heads bend in waves that travel across a field; flowing lines and soft rushing sound.",
    "Misty mountains: layers of ridges fading into fog, pine silhouettes, silence and vastness.",
    "Seashore: waves, foam, wet sand reflecting the sky, salt wind, gulls; rhythm and openness.",
    "Desert dunes at sunset: smooth wind-carved ridges, long shadows, warm orange and violet light; silence and vastness.",
    "Night garden: moonlight on leaves, deep blue shadows, fireflies, scented jasmine; mystery and calm.",
    "Golden hour: low warm sun, long shadows, glowing edges on leaves and petals; nostalgia and warmth.",
    "Fog in a valley: soft grey veil, muted colors, trees appearing and disappearing; mystery and solitude.",
    "Rainbow after rain: wet leaves sparkling, clearing sky, fresh air; hope and wonder.",
    # ---- Sons -> ambiance ----
    "Rain sound: soft patter, vertical lines, cool grey-blue and green; calm, nostalgia and melancholy.",
    "Birdsong: bright scattered notes at dawn, small flicks of light; freedom, morning freshness and joy.",
    "Wind sound: sweeping whoosh, bending branches and grass, airy movement; freedom and longing.",
    "Stream sound: flowing water over stones, ripples and horizontal ribbons, cool glassy light; serenity.",
    "Thunder sound: deep rumble, dark heavy clouds, sharp contrast and electric light; awe and drama.",
    "Cicadas and crickets: dense rhythmic chirping on hot summer evenings, amber light; warmth and nostalgia.",
    "Waves sound: rolling rhythm of the sea, foam and spray; calm and vastness.",
    "Rustling leaves: layered whispering of foliage in a breeze; tenderness and mystery.",
    "Silence in nature: wide empty space, soft light, a few delicate details; peace and solitude.",
    # ---- Culture méditerranéenne / Tunisie ----
    "Sidi Bou Said: white houses with blue doors and windows above the sea, bougainvillea and jasmine; Mediterranean light.",
    "Carthage and Roman mosaics: tessellated floors showing plants, birds and sea creatures in ochre, blue and white.",
    "Zellige and arabesque: geometric tile patterns and curving plant motifs, turquoise, ochre and white, symmetry and meditation.",
    "Sahel olive groves: endless rows of ancient olive trees on red-brown earth under a pale sky; patience and peace.",
    "Oasis: date palms, shade, running water channels and green gardens in a dry landscape; refuge and abundance.",
    "Medina garden courtyard: central fountain, orange and lemon trees, jasmine, cool shade and patterned tiles; calm.",
]

STYLE_DOCS = [f"{name}: {v['description']}" for name, v in ARTISTIC_STYLES.items()]
ALL_DOCS = NATURE_DOCS + STYLE_DOCS

# ============================================================================ 2. Entités structurées (pour le graphe)
# type | nom | couleurs | saisons | émotions | symboles      (séparateur ; dans les listes, - = vide)
ENTITY_TABLE = """
plant|lavender|violet;silver|summer|Sérénité;Nostalgie;Tendresse|calm;devotion
plant|rose|red;pink;white;yellow|summer|Passion;Tendresse;Mélancolie|love;beauty
plant|sunflower|yellow;brown|summer|Joie;Espoir|adoration;loyalty
plant|poppy|red|spring;summer|Mélancolie;Nostalgie|remembrance;consolation
plant|daisy|white;yellow|spring;summer|Joie;Tendresse;Sérénité|innocence;new beginnings
plant|cherry blossom|pink;white|spring|Mélancolie;Émerveillement|transience;renewal
plant|tulip|red;yellow;pink;violet|spring|Joie;Espoir|perfect love
plant|lily of the valley|white;green|spring|Tendresse;Espoir;Nostalgie|return of happiness;purity
plant|forget-me-not|blue;yellow|spring|Nostalgie;Tendresse|remembrance;true love
plant|bluebell|blue;violet|spring|Nostalgie;Sérénité;Émerveillement|constancy;humility
plant|dandelion|yellow;white|spring;summer;autumn|Espoir;Joie|wishes;resilience
plant|lotus|pink;white|summer|Sérénité;Admiration;Émerveillement|purity;rebirth
plant|water lily|white;pink;yellow|summer|Sérénité;Solitude;Émerveillement|tranquility;purity
plant|iris|violet;blue;yellow|spring;summer|Émerveillement;Espoir;Sérénité|messages;hope
plant|peony|pink;white;red|spring|Joie;Tendresse;Admiration|prosperity;honour
plant|wisteria|violet;blue|spring|Nostalgie;Émerveillement|devotion;longing
plant|jasmine|white|summer|Tendresse;Nostalgie;Sérénité|welcome;sweetness
plant|bougainvillea|magenta;pink;orange|summer;spring|Joie;Émerveillement|summer warmth;vibrancy
plant|snowdrop|white;green|winter|Espoir;Tendresse|hope;consolation
plant|cornflower|blue|summer|Espoir;Nostalgie;Joie|delicacy;hope
plant|foxglove|violet;pink;white|summer|Mystère;Émerveillement|enchantment;caution
plant|orchid|white;pink;violet|summer;winter|Admiration;Passion|rare beauty;luxury
plant|marigold|orange;yellow|summer;autumn|Joie;Nostalgie|warmth;remembrance
plant|lilac|violet;white|spring|Nostalgie;Tendresse|first love;youth
plant|olive tree|silver;green;brown|summer;autumn|Sérénité;Paix;Nostalgie|peace;endurance
plant|date palm|green;brown;gold|summer|Admiration;Paix|hospitality;sweetness
plant|prickly pear cactus|green;orange;red|summer|Solitude;Passion|resilience
plant|oak|green;brown;gold|spring;autumn|Admiration;Solitude|strength;endurance
plant|weeping willow|green;silver|spring;summer|Mélancolie;Sérénité|mourning;grace
plant|silver birch|white;green;gold|spring;autumn|Espoir;Sérénité;Solitude|renewal;purity
plant|pine tree|green;brown|winter;summer|Solitude;Admiration|longevity;endurance
plant|japanese maple|red;orange;green|autumn;spring|Mélancolie;Émerveillement;Nostalgie|passing time;balance
plant|ginkgo|yellow;green;gold|autumn|Émerveillement;Admiration|longevity;resilience
plant|fern|green|spring;summer|Mystère;Sérénité|ancient time;shelter
plant|moss|green|winter;spring;autumn|Sérénité;Mystère;Tendresse|patience;quiet time
plant|bamboo|green|spring;summer;autumn;winter|Sérénité;Paix|flexibility;resilience
plant|cypress|green|summer;winter|Mélancolie;Solitude;Paix|eternity
plant|fig tree|green;brown;violet|summer|Joie;Paix|abundance
scene|spring meadow|green;yellow;white|spring|Joie;Sérénité;Espoir|dew
scene|ancient woodland in spring|green;blue;white|spring|Émerveillement;Sérénité;Nostalgie|dappled light
scene|autumn woodland|gold;orange;red;brown|autumn|Mélancolie;Nostalgie|falling leaves
scene|still pond|green;blue;silver|summer;spring|Sérénité;Solitude;Mystère|reflection
scene|mossy forest floor|green;brown;silver|autumn;winter;spring|Mystère;Sérénité;Tendresse|renewal
scene|mediterranean hillside|silver;gold;green|summer|Sérénité;Nostalgie|sunlight
scene|winter garden|silver;brown;green|winter|Solitude;Espoir;Sérénité|frost
scene|rain in the forest|green;grey;brown|spring;autumn;summer|Sérénité;Mélancolie;Mystère|petrichor
scene|dawn dew|green;silver|spring;summer;autumn|Tendresse;Émerveillement;Sérénité|first light
scene|wind in tall grass|gold;green|summer;autumn|Sérénité;Nostalgie;Admiration|movement
scene|misty mountains|grey;blue;green|autumn;winter|Mystère;Solitude;Admiration|silence
scene|seashore|blue;white;tan|summer|Paix;Joie;Admiration|rhythm
scene|desert dunes at sunset|orange;violet;gold|summer;winter|Solitude;Admiration;Paix|silence
scene|night garden|blue;silver;violet|summer|Mystère;Paix;Émerveillement|moonlight
scene|golden hour|gold;orange|summer;autumn|Nostalgie;Joie|warm light
scene|fog in a valley|grey;green|autumn;winter|Mystère;Solitude|veil
scene|rainbow after rain|blue;red;yellow;green|spring;summer|Espoir;Émerveillement|clearing sky
sound|rain sound|grey;blue;green|-|Mélancolie;Nostalgie;Sérénité|patter
sound|birdsong|yellow;green|spring|Joie;Espoir|freedom
sound|wind sound|grey;green|-|Espoir;Nostalgie|freedom
sound|stream sound|blue;silver|-|Sérénité;Paix|flow
sound|thunder sound|grey;violet|summer|Admiration;Mystère|drama
sound|cicadas|orange;gold|summer|Nostalgie;Joie|heat
sound|waves sound|blue;white|-|Paix;Admiration|rhythm
sound|rustling leaves|green;gold|-|Tendresse;Mystère|whisper
sound|silence in nature|white;grey|-|Paix;Solitude|stillness
"""


def _split(s):
    return [] if s.strip() in ("", "-") else [x.strip() for x in s.split(";")]


ENTITIES = []
for _line in ENTITY_TABLE.strip().splitlines():
    _k, _n, _c, _s, _e, _y = _line.split("|")
    ENTITIES.append({"kind": _k, "name": _n, "colors": _split(_c), "seasons": _split(_s), "emotions": _split(_e), "symbols": _split(_y)})
ENTITY_BY_NAME = {e["name"]: e for e in ENTITIES}


def doc_entity(doc):
    """'Lavender: purple flower…' -> 'lavender' (nom d'entité du graphe) ; None pour les documents sans entité."""
    name = doc.split(":", 1)[0].strip().lower()
    if name in ENTITY_BY_NAME:
        return name
    for n in ENTITY_BY_NAME:
        if name.startswith(n) or n in name:
            return n
    return None


# ============================================================================ 3. Recherche vectorielle (FAISS si disponible)
def hash_embed(texts, dim=384):
    """Embeddings de secours sans modèle (sac de mots hachés) : lexicaux, pas sémantiques."""
    out = []
    for t in texts:
        v = np.zeros(dim, dtype="float32")
        for w in re.findall(r"[a-zà-ÿ0-9]+", t.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % dim] += 1.0 if (h >> 64) % 2 == 0 else -1.0
        out.append(v / (np.linalg.norm(v) or 1.0))
    return np.array(out, dtype="float32")


def default_embedder():
    """MiniLM (sentence-transformers) si disponible, sinon embeddings hachés."""
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return lambda texts: np.asarray(m.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True), dtype="float32")
    except Exception:
        return hash_embed


class KnowledgeBase:
    """Recherche sémantique dans les documents (index FAISS « produit scalaire » sur vecteurs normalisés)."""

    def __init__(self, embed_fn=None, docs=None, n_nature=None):
        self.embed = embed_fn or default_embedder()
        self.docs = list(docs or ALL_DOCS)
        self.n_nature = len(NATURE_DOCS) if n_nature is None else n_nature
        vecs = np.asarray(self.embed(self.docs), dtype="float32")
        try:
            import faiss
            self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)
            self.vecs = None
        except ImportError:
            self.index, self.vecs = None, vecs

    def search(self, query, top_k=4, only_nature=True):
        q = np.asarray(self.embed([query]), dtype="float32")
        n = min(top_k * 3, len(self.docs))
        if self.index is not None:
            scores, ids = self.index.search(q, n)
            scores, ids = scores[0], ids[0]
        else:
            s = (self.vecs @ q[0])
            ids = np.argsort(-s)[:n]
            scores = s[ids]
        out = []
        for sc, i in zip(scores, ids):
            if i < 0 or (only_nature and i >= self.n_nature):
                continue
            out.append({"document": self.docs[i], "score": round(float(sc), 3)})
            if len(out) == top_k:
                break
        return out


# ============================================================================ 4. Graphe de connaissances (Graph-RAG)
REL = {"color": "a la couleur", "season": "fleurit/se vit en", "emotion": "évoque", "symbol": "symbolise",
       "style_emotion": "exprime", "family": "appartient à"}
COLOR_KEYWORDS = [("pink", "pink"), ("red", "red"), ("crimson", "red"), ("yellow", "yellow"), ("green", "green"),
                  ("blue", "blue"), ("indigo", "blue"), ("teal", "blue"), ("violet", "violet"), ("lavender", "violet"),
                  ("orange", "orange"), ("brown", "brown"), ("tan", "tan"), ("silver", "silver"), ("grey", "grey"),
                  ("charcoal", "grey"), ("white", "white"), ("cream", "white"), ("gold", "gold"), ("magenta", "magenta")]


def to_graph_colors(names):
    """Noms de couleurs libres (palette de photo) -> noms de nœuds couleur du graphe."""
    out = []
    for n in names:
        for kw, node in COLOR_KEYWORDS:
            if kw in n.lower() and node not in out:
                out.append(node)
                break
    return out


class KGraph:
    """Graphe non orienté minimal : nœuds typés, relation par arête, parcours en largeur."""

    def __init__(self):
        self.nodes, self.adj, self.rel = {}, {}, {}

    def add_edge(self, a, b, rel):
        for n in (a, b):
            self.nodes.setdefault(n[1], {"type": n[0], "label": n[1]})
            self.adj.setdefault(n[1], set())
        ida, idb = a[1], b[1]
        self.adj[ida].add(idb)
        self.adj[idb].add(ida)
        self.rel[(ida, idb)] = (rel, ida)
        self.rel[(idb, ida)] = (rel, ida)

    def edges(self):
        seen, out = set(), []
        for (a, b) in self.rel:
            if (b, a) not in seen:
                seen.add((a, b))
                out.append((a, b))
        return out

    def path(self, src, dst, max_hops=2):
        """Plus court chemin (BFS) en phrases lisibles : 'lavender évoque Nostalgie'."""
        prev, frontier, seen = {src: None}, [src], {src}
        for _ in range(max_hops):
            nxt = []
            for u in frontier:
                for v in self.adj.get(u, ()):
                    if v not in seen:
                        seen.add(v)
                        prev[v] = u
                        nxt.append(v)
            frontier = nxt
        if dst not in prev:
            return None
        chain = [dst]
        while prev[chain[-1]] is not None:
            chain.append(prev[chain[-1]])
        chain.reverse()
        steps = []
        for a, b in zip(chain, chain[1:]):
            rel, origin = self.rel[(a, b)]
            s, d = (a, b) if origin == a else (b, a)
            steps.append(f"{s} {REL[rel]} {d}")
        return steps

    def links(self, seeds, target_types=("style", "symbol"), per_type=3, cutoff=2):
        """Voisins (≤ cutoff sauts) des nœuds de départ, classés par proximité × poids : Graph-RAG explicable."""
        score, best = {}, {}
        for node, w in seeds:
            if node not in self.nodes:
                continue
            dist, frontier = {node: 0}, [node]
            for d in range(1, cutoff + 1):
                nxt = []
                for u in frontier:
                    for v in self.adj[u]:
                        if v not in dist:
                            dist[v] = d
                            nxt.append(v)
                frontier = nxt
            for v, d in dist.items():
                if d and self.nodes[v]["type"] in target_types:
                    score[v] = score.get(v, 0.0) + w / d
                    if v not in best or w / d > best[v][0]:
                        best[v] = (w / d, node)
        out = []
        for t in target_types:
            for v in sorted((k for k in score if self.nodes[k]["type"] == t), key=lambda k: -score[k])[:per_type]:
                out.append({"label": v, "type": t, "score": round(score[v], 3), "path": self.path(best[v][1], v, cutoff)})
        return out


def build_graph():
    g = KGraph()
    for e in ENTITIES:
        me = (e["kind"], e["name"])
        for c in e["colors"]:
            g.add_edge(me, ("color", c), "color")
        for s in e["seasons"]:
            g.add_edge(me, ("season", s), "season")
        for em in e["emotions"]:
            g.add_edge(me, ("emotion", em), "emotion")
        for sy in e["symbols"]:
            g.add_edge(me, ("symbol", sy), "symbol")
    for name, v in ARTISTIC_STYLES.items():
        for em in v["emotions"]:
            g.add_edge(("style", name), ("emotion", em), "style_emotion")
        g.add_edge(("style", name), ("family", style_family(name)), "family")
    return g


# ============================================================================ 5. GCN (NumPy) : pertinence entre nœuds
class TinyGCN:
    """GCN à 2 couches, entièrement en NumPy (pas de GPU, pas de PyTorch), entraîné par prédiction de liens.

        Â = D^-1/2 (A + I) D^-1/2        H1 = ReLU(Â W1)        Z = Â H1 W2        score(i, j) = z_i · z_j

    Les nœuds n'ont pas d'attributs (X = identité) : Z est un plongement appris qui rapproche les nœuds liés
    par le graphe (fleur ↔ émotion ↔ style). Gradients écrits à la main (vérifiés par différences finies dans les tests).
    """

    def __init__(self, graph, d1=32, d2=16, seed=0):
        self.g, self.ids = graph, sorted(graph.nodes)
        self.ix = {n: i for i, n in enumerate(self.ids)}
        self.n, self.d1, self.d2 = len(self.ids), d1, d2
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (self.n, d1))
        self.W2 = rng.normal(0, 0.1, (d1, d2))
        self.rng = rng
        self.Z = None

    def _norm_adj(self, edges):
        A = np.eye(self.n)
        for a, b in edges:
            A[self.ix[a], self.ix[b]] = A[self.ix[b], self.ix[a]] = 1.0
        d = A.sum(1) ** -0.5
        return A * d[:, None] * d[None, :]

    def forward(self, Ah, W1=None, W2=None):
        W1 = self.W1 if W1 is None else W1
        W2 = self.W2 if W2 is None else W2
        H1p = Ah @ W1
        H1 = np.maximum(H1p, 0)
        AH = Ah @ H1
        return H1p, H1, AH, AH @ W2

    def loss_and_grads(self, Ah, pos, neg, W1=None, W2=None, l2=1e-5):
        W1 = self.W1 if W1 is None else W1
        W2 = self.W2 if W2 is None else W2
        H1p, H1, AH, Z = self.forward(Ah, W1, W2)
        pairs = np.vstack([pos, neg])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        s = np.sum(Z[pairs[:, 0]] * Z[pairs[:, 1]], axis=1)
        p = 1 / (1 + np.exp(-s))
        eps = 1e-9
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + l2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        ds = (p - y) / len(y)
        dZ = np.zeros_like(Z)
        np.add.at(dZ, pairs[:, 0], ds[:, None] * Z[pairs[:, 1]])
        np.add.at(dZ, pairs[:, 1], ds[:, None] * Z[pairs[:, 0]])
        dW2 = AH.T @ dZ + 2 * l2 * W2
        dH1 = Ah.T @ (dZ @ W2.T)
        dW1 = Ah.T @ (dH1 * (H1p > 0)) + 2 * l2 * W1
        return loss, dW1, dW2

    def fit(self, edges=None, epochs=400, lr=0.05, verbose=False):
        edges = edges if edges is not None else self.g.edges()
        Ah = self._norm_adj(edges)
        pos = np.array([[self.ix[a], self.ix[b]] for a, b in edges])
        eset = {(i, j) for i, j in pos} | {(j, i) for i, j in pos}
        m1, v1, m2, v2 = (np.zeros_like(self.W1), np.zeros_like(self.W1), np.zeros_like(self.W2), np.zeros_like(self.W2))
        for t in range(1, epochs + 1):
            neg = []
            while len(neg) < len(pos):
                i, j = self.rng.integers(0, self.n, 2)
                if i != j and (i, j) not in eset:
                    neg.append((i, j))
            loss, g1, g2 = self.loss_and_grads(Ah, pos, np.array(neg))
            for W, g, m, v in ((self.W1, g1, m1, v1), (self.W2, g2, m2, v2)):          # Adam
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                W -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            if verbose and t % 100 == 0:
                print(f"  époque {t}: perte {loss:.3f}")
        self.Z = self.forward(Ah)[3]
        return self

    def score(self, a, b):
        return float(self.Z[self.ix[a]] @ self.Z[self.ix[b]])

    def rank_styles(self, seeds, k=3):
        """seeds : [(id_de_nœud, poids)] -> les k styles les plus pertinents (cosinus avec le vecteur-requête)."""
        seeds = [(n, w) for n, w in seeds if n in self.ix]
        if not seeds:
            return []
        q = sum(w * self.Z[self.ix[n]] for n, w in seeds) / sum(w for _, w in seeds)
        q = q / (np.linalg.norm(q) + 1e-9)
        res = []
        for n in self.ids:
            if self.g.nodes[n]["type"] == "style":
                z = self.Z[self.ix[n]]
                res.append((n, float(q @ z / (np.linalg.norm(z) + 1e-9))))
        return sorted(res, key=lambda t: -t[1])[:k]


def _auc(pos, neg):
    """AUC = probabilité qu'une paire positive soit mieux classée qu'une négative (égalités = 0,5)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def evaluate_gcn(n_splits=5, holdout=0.2, epochs=400, seed=0):
    """Prédiction de liens entité↔émotion masqués : AUC du GCN contre deux références simples.

    - vote : nombre d'entités « voisines » (même couleur/saison/symbole) qui ont déjà cette émotion
    - popularité : nombre d'entités ayant cette émotion
    """
    rng = np.random.default_rng(seed)
    full = build_graph()
    emo_edges = [(e["name"], em) for e in ENTITIES for em in e["emotions"]]
    emos = sorted({em for _, em in emo_edges})
    res = {"gcn": [], "vote": [], "popularité": []}
    for s in range(n_splits):
        idx = rng.permutation(len(emo_edges))
        test = []
        left = {e["name"]: len(e["emotions"]) for e in ENTITIES}
        for i in idx:                                    # on garde au moins 1 émotion par entité
            a, b = emo_edges[i]
            if left[a] > 1 and len(test) < int(holdout * len(emo_edges)):
                test.append((a, b))
                left[a] -= 1
        test_set = set(test)
        train = [(a, b) for (a, b) in full.edges() if not ((a, b) in test_set or (b, a) in test_set)]
        g = KGraph()
        g.nodes, g.adj = full.nodes, {k: set() for k in full.nodes}
        g.rel = {}
        for a, b in train:
            rel = full.rel[(a, b)]
            g.adj[a].add(b)
            g.adj[b].add(a)
            g.rel[(a, b)] = rel
            g.rel[(b, a)] = rel
        gcn = TinyGCN(full, seed=s).fit(train, epochs=epochs)
        has = {(a, b) for a, b in emo_edges if (a, b) not in test_set}
        negs = []
        while len(negs) < len(test):
            a = rng.choice([e["name"] for e in ENTITIES])
            b = rng.choice(emos)
            if (a, b) not in has and (a, b) not in test_set:
                negs.append((a, b))
        by_emo = {em: {a for a, b in has if b == em} for em in emos}

        def vote(a, b):
            nb = {x for x in g.adj[a] if g.nodes[x]["type"] in ("color", "season", "symbol")}
            return sum(1 for q in by_emo[b] if q != a and nb & g.adj[q])

        res["gcn"].append(_auc([gcn.score(a, b) for a, b in test], [gcn.score(a, b) for a, b in negs]))
        res["vote"].append(_auc([vote(a, b) for a, b in test], [vote(a, b) for a, b in negs]))
        res["popularité"].append(_auc([len(by_emo[b]) for a, b in test], [len(by_emo[b]) for a, b in negs]))
    return {k: round(float(np.mean(v)), 3) for k, v in res.items()} | {"n_test_liens": len(test), "n_splits": n_splits}


class StyleRecommender:
    """Graphe + GCN entraîné : propose des styles pour des fleurs / ambiances / émotions / couleurs."""

    def __init__(self, epochs=400, seed=0):
        self.graph = build_graph()
        self.gcn = TinyGCN(self.graph, seed=seed).fit(epochs=epochs)

    def seeds(self, entities=(), emotion=None, season=None, colors=()):
        s = [(n, 1.0) for n in entities if n in self.graph.nodes]
        if emotion in self.graph.nodes:
            s.append((emotion, 1.5))
        if season in self.graph.nodes:
            s.append((season, 0.4))
        s += [(c, 0.3) for c in to_graph_colors(colors) if c in self.graph.nodes]
        return s

    def recommend(self, entities=(), emotion=None, season=None, colors=(), k=3):
        return self.gcn.rank_styles(self.seeds(entities, emotion, season, colors), k)

    def explain(self, entities=(), emotion=None, season=None, colors=(), per_type=3):
        return self.graph.links(self.seeds(entities, emotion, season, colors), per_type=per_type)
