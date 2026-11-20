"""Styles artistiques et constantes (émotions, langues).

Aucune dépendance lourde : utilisable par le notebook, les tests et le serveur MCP.
"""
ARTISTIC_STYLES = {}


def S(name, booster, description, emotions):
    ARTISTIC_STYLES[name] = {"booster": booster, "description": description, "emotions": emotions}


# --- Impressionnisme & post-impressionnisme ---
S("Claude Monet — Impressionnisme", "in the style of Claude Monet, impressionist oil painting, soft broken brushstrokes, dappled light, water reflections, pastel colors",
  "Impressionist light, atmosphere and reflections, soft visible brushstrokes.", ["Paix", "Sérénité", "Nostalgie"])
S("Pierre-Auguste Renoir — Impressionnisme", "in the style of Pierre-Auguste Renoir, impressionist painting, warm luminous colors, feathery brushstrokes, sunlit lush garden",
  "Warm, luminous, joyful impressionism with feathery brushwork and sunlit gardens.", ["Joie", "Tendresse"])
S("Georges Seurat — Pointillisme", "in the style of Georges Seurat, pointillism, tiny dots of pure color, optical color mixing, luminous calm",
  "Thousands of small dots of pure color that blend in the eye, a calm shimmering light.", ["Sérénité", "Paix"])
S("Paul Cézanne — Post-impressionnisme", "in the style of Paul Cézanne, post-impressionism, structured planes of color, constructive brushstrokes, simplified geometric forms",
  "Nature built from planes and geometric volumes with constructive brushstrokes.", ["Paix", "Admiration"])
S("Vincent van Gogh — Post-impressionnisme", "in the style of Vincent van Gogh, post-impressionism, swirling brushstrokes, vibrant yellow and blue, thick impasto",
  "Expressive swirling brushwork, vivid colors, thick impasto and a vibrating sky.", ["Passion", "Émerveillement", "Mélancolie"])
S("Paul Gauguin — Synthétisme", "in the style of Paul Gauguin, synthetism, flat areas of saturated color, bold dark outlines, lush tropical plants",
  "Flat saturated colors, strong outlines and lush exotic vegetation.", ["Mystère", "Passion"])
# --- Fauvisme, expressionnisme, romantisme ---
S("Henri Matisse — Fauvisme", "in the style of Henri Matisse, fauvism, bold flat non-naturalistic colors, simplified cut-out shapes, joyful decorative composition",
  "Bold unnatural colors, simplified shapes and joyful decoration.", ["Joie", "Espoir"])
S("J.M.W. Turner — Romantisme lumineux", "in the style of J.M.W. Turner, romantic landscape, luminous mist, golden atmospheric light, dissolving forms",
  "Golden mist and light dissolving the landscape into atmosphere.", ["Admiration", "Mélancolie", "Émerveillement"])
S("Caspar David Friedrich — Romantisme", "in the style of Caspar David Friedrich, romantic landscape, sublime misty trees and mountains, solitary mood, cold dawn light",
  "Sublime misty landscapes, solitary contemplation, cold dawn light.", ["Solitude", "Admiration", "Mystère"])
S("Edvard Munch — Expressionnisme", "in the style of Edvard Munch, expressionism, wavy swirling lines, intense emotional colors, psychological landscape",
  "Wavy lines and intense colors that make the landscape carry the emotion.", ["Mélancolie", "Solitude", "Passion"])
# --- Symbolisme, Art nouveau, Art déco ---
S("Gustav Klimt — Symbolisme", "in the style of Gustav Klimt, symbolism, gold leaf patterns, intricate ornaments, floral motifs, decorative elegance",
  "Gold leaf, intricate patterns and decorative organic ornaments, flowers.", ["Passion", "Tendresse", "Mystère"])
S("Odilon Redon — Symbolisme (pastel)", "in the style of Odilon Redon, symbolist pastel painting, dreamlike floating flowers, luminous glowing colors, mysterious mood",
  "Dreamlike flowers floating in luminous, glowing pastel color.", ["Mystère", "Émerveillement", "Nostalgie"])
S("Alphonse Mucha — Art nouveau", "in the style of Alphonse Mucha, art nouveau, sinuous floral lines, decorative border, soft muted colors, flowing vines",
  "Sinuous plant lines, decorative frames and soft muted colors.", ["Tendresse", "Sérénité", "Nostalgie"])
S("Art déco — Fleurs stylisées", "art deco illustration, symmetrical geometric stylized flowers, gold and black, elegant clean lines",
  "Symmetry, geometric stylization, gold accents and elegance.", ["Passion", "Admiration"])
# --- Cubisme, surréalisme, avant-gardes ---
S("Pablo Picasso — Cubisme", "in the style of Pablo Picasso, cubism, fragmented geometric forms, multiple perspectives, bold black outlines",
  "Fragmented geometric forms seen from several viewpoints at once.", ["Mystère", "Passion"])
S("Salvador Dalí — Surréalisme", "in the style of Salvador Dalí, surrealism, dreamlike landscape, melting forms, impossible geometry, long shadows",
  "Dreamlike landscapes, melting forms and impossible juxtapositions.", ["Mystère", "Émerveillement"])
S("René Magritte — Surréalisme", "in the style of René Magritte, surrealism, smooth clean realism, impossible juxtaposition, floating objects, calm blue sky",
  "Calm precise realism that shows impossible, poetic juxtapositions.", ["Mystère", "Sérénité"])
S("Joan Miró — Surréalisme", "in the style of Joan Miró, surrealism, playful biomorphic shapes, bold primary colors, black lines, stars",
  "Playful organic shapes, primary colors and black calligraphic lines.", ["Joie", "Émerveillement"])
S("Marc Chagall — Poésie onirique", "in the style of Marc Chagall, dreamlike floating figures and flowers, rich blues and reds, folk-tale poetry",
  "Floating dreamlike scenes full of flowers in deep blues and reds.", ["Tendresse", "Nostalgie", "Joie"])
S("Wassily Kandinsky — Abstraction", "in the style of Wassily Kandinsky, abstract art, geometric shapes and lines, vibrant colors, musical rhythm",
  "Abstract shapes and lines with colors that feel like music.", ["Joie", "Émerveillement", "Passion"])
S("Paul Klee — Abstraction poétique", "in the style of Paul Klee, whimsical abstract painting, delicate lines, soft colored squares, childlike poetry",
  "Delicate lines and soft color blocks, a childlike and poetic abstraction.", ["Tendresse", "Joie"])
S("Piet Mondrian — De Stijl", "in the style of Piet Mondrian, de stijl, grid of black lines, flat primary color blocks, minimal composition",
  "A grid of straight black lines and flat blocks of primary color.", ["Paix", "Sérénité"])
S("Mark Rothko — Champs de couleur", "in the style of Mark Rothko, color field painting, large soft glowing rectangles of color, meditative",
  "Large soft glowing fields of color for pure feeling.", ["Mélancolie", "Admiration", "Solitude"])
S("Jackson Pollock — Expressionnisme abstrait", "in the style of Jackson Pollock, abstract expressionism, dripped and splattered paint, energetic tangled lines",
  "Energetic dripped paint, the movement of the body recorded in color.", ["Passion", "Joie"])
S("Andy Warhol — Pop art", "in the style of Andy Warhol, pop art, silkscreen print, flat bright contrasting colors, repeated image",
  "Silkscreen flat colors, bright contrasts, repetition.", ["Joie"])
# --- Réalisme magique, naïf, modernisme ---
S("Georgia O'Keeffe — Fleurs agrandies", "in the style of Georgia O'Keeffe, modernism, magnified close-up flower, smooth flowing curves, soft color gradients",
  "A huge close-up of a bloom, smooth curves, soft gradients.", ["Tendresse", "Sérénité", "Admiration"])
S("Frida Kahlo — Réalisme magique", "in the style of Frida Kahlo, magical realism, lush tropical leaves, symbolic flowers, folk art colors, intense",
  "Lush leaves and symbolic flowers, folk colors and intense feeling.", ["Passion", "Mélancolie"])
S("Henri Rousseau — Art naïf", "in the style of Henri Rousseau, naive art, dense lush leaves, flat stylized plants, moonlit stillness",
  "Dense stylized jungle foliage and dreamlike stillness.", ["Mystère", "Émerveillement"])
# --- Orient, Maghreb, Méditerranée ---
S("Hokusai — Ukiyo-e", "in the style of Hokusai, ukiyo-e japanese woodblock print, flowing lines, stylized waves, flat colors, balanced composition",
  "Japanese woodblock print with flowing lines and stylized natural forms.", ["Admiration", "Sérénité"])
S("Hiroshige — Ukiyo-e (pluie et brume)", "in the style of Utagawa Hiroshige, ukiyo-e woodblock print, rain streaks, misty gradients, deep indigo blue, serene landscape",
  "Woodblock landscapes with rain, mist and indigo gradients.", ["Mélancolie", "Sérénité", "Nostalgie"])
S("Sumi-e — Lavis d'encre japonais", "sumi-e Japanese ink wash painting, minimalist brush strokes, black ink gradients, empty space, rice paper",
  "A few confident ink strokes and generous empty space.", ["Sérénité", "Solitude", "Paix"])
S("Peinture chinoise — Shan shui", "traditional Chinese shan shui landscape painting, misty mountains, delicate ink and light color wash, silk scroll",
  "Misty mountains and water in ink and pale washes on silk.", ["Paix", "Admiration", "Solitude"])
S("Miniature persane", "Persian miniature painting, intricate floral patterns, jewel colors, gold details, flat perspective, manuscript illumination",
  "Jewel-like colors, gold, dense flowers and flat perspective.", ["Tendresse", "Passion", "Émerveillement"])
S("Arabesque et zellige (Maghreb)", "Islamic geometric art, Maghrebi zellige tilework, arabesque floral patterns, symmetrical, turquoise ochre and white",
  "Symmetrical arabesques and zellige tile geometry with plant motifs.", ["Sérénité", "Admiration"])
S("Mosaïque romaine (style Carthage)", "Roman mosaic, small tesserae tiles, bold outlines, earthy ochre and blue, plants and birds, Carthage mosaic style",
  "Roman mosaic of small colored tiles showing plants and animals.", ["Nostalgie", "Admiration"])
# --- Techniques & traditions ---
S("Aquarelle botanique (Redouté)", "in the style of Pierre-Joseph Redouté, botanical illustration, delicate watercolor, fine detail, white paper, scientific plate",
  "Precise delicate botanical watercolor plate on white paper.", ["Tendresse", "Sérénité"])
S("Aquarelle humide — flou poétique", "wet-on-wet watercolor painting, soft bleeding colors, luminous transparent washes, visible paper texture",
  "Colors bleeding softly on wet paper, transparent and luminous.", ["Nostalgie", "Tendresse", "Mélancolie"])
S("Cyanotype — bleu de Prusse", "cyanotype print, deep Prussian blue with white plant silhouettes, sun-print texture",
  "Deep blue sun-print with white silhouettes of plants.", ["Nostalgie", "Mystère"])
S("Nature morte hollandaise (vanité)", "Dutch Golden Age flower still life, dark background, dramatic chiaroscuro, dewdrops and insects, opulent bouquet",
  "An opulent bouquet in dark light, a reminder that beauty is fleeting.", ["Mélancolie", "Admiration"])
S("Préraphaélites — détail et bijoux", "Pre-Raphaelite painting, hyper-detailed plants, jewel-like colors, romantic mood",
  "Hyper-detailed plants in jewel colors, romantic and wistful.", ["Nostalgie", "Passion", "Tendresse"])
S("Vitrail — verre coloré", "stained glass window, bold black lead lines, jewel-colored glass, glowing backlight",
  "Jewel-colored glass in black lead lines, glowing with light.", ["Admiration", "Espoir", "Émerveillement"])
S("Gravure sur bois / linogravure", "linocut block print, bold carved lines, two or three flat colors, handmade texture",
  "Bold carved lines and flat colors with a handmade texture.", ["Solitude", "Paix"])
S("Peinture à l'huile réaliste", "classical realistic oil painting, rich chiaroscuro, fine glazing, museum quality, natural light",
  "Realistic oil painting with warm chiaroscuro and glazes.", ["Admiration", "Nostalgie"])
S("Pixel art 16 bits", "16-bit pixel art, limited color palette, crisp pixels, nature scene",
  "Retro pixel art with a limited palette.", ["Joie", "Émerveillement"])

# --- Émotions (menu en français, prompts image en anglais) ---
EMOTIONS_FR = {"Paix": "peace", "Nostalgie": "nostalgia", "Joie": "joy", "Mélancolie": "melancholy",
               "Émerveillement": "wonder", "Sérénité": "serenity", "Mystère": "mystery", "Passion": "passion",
               "Admiration": "awe", "Espoir": "hope", "Solitude": "solitude", "Tendresse": "tenderness"}
EMOTION_CHOICES = ["Auto (détecter)"] + list(EMOTIONS_FR)
LANGS = {"Français": "French", "English": "English", "العربية": "Arabic"}
STYLE_AUTO = "✨ Auto (FLOREA choisit)"
STYLE_CHOICES = [STYLE_AUTO] + sorted(ARTISTIC_STYLES)


def style_family(name):
    """'Claude Monet — Impressionnisme' -> 'Impressionnisme' (utilisé comme nœud du graphe)."""
    return name.split(" — ", 1)[1] if " — " in name else name
