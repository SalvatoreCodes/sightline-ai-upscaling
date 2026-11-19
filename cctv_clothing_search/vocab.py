"""Shared open-vocabulary prompt banks, color/type metadata, and query synonyms for Sightline."""

from __future__ import annotations

from typing import Any

# Stage 1 human-vs-fixture zero-shot verification prompts
HUMAN_POS_PROMPTS = [
    "a CCTV photo of a standing or walking human person",
    "a surveillance view of a person, worker, or shopper wearing clothes",
    "a human body with torso, arms, or legs visible outdoors or indoors",
]
HUMAN_NEG_PROMPTS = [
    "an empty construction fence, metal barrier, scaffolding, or building wall with no person",
    "machinery, traffic cone, tarp, excavator, or vehicle with no human",
    "an indoor store shelf, mannequin, shopping cart, or floor with no person",
]

# Universal open-vocabulary 17 clothing color families
VARIABLE_COLORS: list[dict[str, Any]] = [
    {"name": "black", "short": "Black", "hex": "#1E293B", "prompts": ["black clothing garment", "deep black fabric"], "synonyms": ["black"]},
    {"name": "dark grey / charcoal", "short": "Dark Grey", "hex": "#475569", "prompts": ["dark grey charcoal clothing", "charcoal grey fabric"], "synonyms": ["dark grey", "dark gray", "charcoal", "grey", "gray"]},
    {"name": "light grey / silver", "short": "Light Grey", "hex": "#94A3B8", "prompts": ["light grey clothing", "pale silver grey fabric"], "synonyms": ["light grey", "light gray", "silver", "grey", "gray"]},
    {"name": "white", "short": "White", "hex": "#F1F5F9", "prompts": ["white clothing garment", "clean white fabric"], "synonyms": ["white", "cream", "off white"]},
    {"name": "red", "short": "Red", "hex": "#EF4444", "prompts": ["red clothing garment", "bright red or crimson fabric"], "synonyms": ["red", "crimson", "scarlet", "maroon"]},
    {"name": "orange", "short": "Orange", "hex": "#F97316", "prompts": ["orange clothing garment", "bright high-visibility orange fabric"], "synonyms": ["orange", "hi-vis orange", "safety orange"]},
    {"name": "yellow", "short": "Yellow", "hex": "#EAB308", "prompts": ["yellow clothing garment", "bright high-visibility neon yellow fabric"], "synonyms": ["yellow", "neon yellow", "hi-vis yellow", "gold"]},
    {"name": "green", "short": "Green", "hex": "#22C55E", "prompts": ["green clothing garment", "green fabric"], "synonyms": ["green", "olive", "lime", "emerald"]},
    {"name": "teal / cyan", "short": "Teal", "hex": "#06B6D4", "prompts": ["teal cyan clothing garment", "turquoise fabric"], "synonyms": ["teal", "cyan", "turquoise", "aqua"]},
    {"name": "light blue", "short": "Light Blue", "hex": "#38BDF8", "prompts": ["light blue clothing garment", "sky blue fabric"], "synonyms": ["light blue", "sky blue", "baby blue", "blue"]},
    {"name": "blue", "short": "Blue", "hex": "#2563EB", "prompts": ["blue denim or medium blue clothing", "blue fabric"], "synonyms": ["blue"]},
    {"name": "royal blue", "short": "Royal Blue", "hex": "#1D4ED8", "prompts": ["royal blue cobalt clothing", "bright cobalt blue jacket"], "synonyms": ["royal blue", "royal", "cobalt", "blue"]},
    {"name": "navy blue", "short": "Navy Blue", "hex": "#1E3A8A", "prompts": ["navy blue dark blue clothing", "deep navy fabric"], "synonyms": ["navy blue", "navy", "indigo", "blue"]},
    {"name": "purple / violet", "short": "Purple", "hex": "#8B5CF6", "prompts": ["purple violet clothing", "lavender violet fabric"], "synonyms": ["purple", "violet", "lavender", "plum"]},
    {"name": "pink / magenta", "short": "Pink", "hex": "#EC4899", "prompts": ["pink magenta clothing garment", "hot pink fuchsia shirt"], "synonyms": ["pink", "magenta", "fuchsia", "rose"]},
    {"name": "brown", "short": "Brown", "hex": "#78350F", "prompts": ["brown clothing garment", "chocolate dark brown fabric"], "synonyms": ["brown", "chocolate", "chestnut"]},
    {"name": "beige / khaki", "short": "Beige / Khaki", "hex": "#D6B88D", "prompts": ["beige khaki tan clothing", "tan khaki trench fabric"], "synonyms": ["beige", "khaki", "tan", "sand", "camel"]},
]

# Universal open-vocabulary 21 clothing types
VARIABLE_TYPES: list[dict[str, Any]] = [
    {"name": "vest / safety vest", "short": "Safety Vest", "role": "upper", "prompts": ["high-visibility safety vest", "reflective construction vest waistcoat", "sleeveless utility safety vest"], "synonyms": ["vest", "safety vest", "hi-vis", "hivis", "high visibility", "reflective vest", "construction vest", "waistcoat"]},
    {"name": "t-shirt / polo shirt", "short": "T-Shirt / Polo", "role": "upper", "prompts": ["short-sleeve t-shirt", "polo shirt tee", "casual t-shirt top"], "synonyms": ["t-shirt", "tshirt", "t shirt", "tee", "polo", "polo shirt", "shirt", "top"]},
    {"name": "dress shirt / blouse", "short": "Shirt / Blouse", "role": "upper", "prompts": ["long-sleeve button-up shirt", "casual blouse or work shirt", "collared cotton shirt worn by a person"], "synonyms": ["shirt", "blouse", "dress shirt", "button shirt", "work shirt", "top"]},
    {"name": "flannel / plaid shirt", "short": "Plaid Shirt", "role": "upper", "prompts": ["plaid flannel button shirt", "checkered long sleeve shirt"], "synonyms": ["flannel", "plaid", "plaid shirt", "checkered shirt", "shirt"]},
    {"name": "tank top / sleeveless top", "short": "Tank Top", "role": "upper", "prompts": ["sleeveless tank top", "singlet undershirt"], "synonyms": ["tank top", "sleeveless", "singlet", "top"]},
    {"name": "sweater / knitwear", "short": "Sweater", "role": "upper", "prompts": ["crewneck sweater", "knitted pullover sweatshirt jumper"], "synonyms": ["sweater", "jumper", "pullover", "knitwear", "sweatshirt"]},
    {"name": "hoodie / sweatshirt", "short": "Hoodie", "role": "upper", "prompts": ["hoodie sweatshirt with hood", "pullover hooded sweatshirt"], "synonyms": ["hoodie", "hooded", "sweatshirt", "top"]},
    {"name": "zip-up jacket / bomber jacket", "short": "Jacket", "role": "upper", "prompts": ["zip-up casual jacket or work jacket", "bomber jacket outerwear"], "synonyms": ["jacket", "zip jacket", "bomber", "work jacket", "coat"]},
    {"name": "open jacket / coat", "short": "Open Jacket", "role": "upper", "prompts": ["open front casual jacket worn over shirt", "unzipped colorful coat or cardigan"], "synonyms": ["open jacket", "cardigan", "jacket", "coat"]},
    {"name": "puffer jacket", "short": "Puffer Jacket", "role": "upper", "prompts": ["quilted puffer jacket", "padded down winter jacket"], "synonyms": ["puffer", "puffer jacket", "quilted", "jacket", "coat"]},
    {"name": "buttoned coat", "short": "Buttoned Coat", "role": "upper", "prompts": ["buttoned car coat jacket", "single breasted mid-length coat"], "synonyms": ["buttoned coat", "coat", "jacket"]},
    {"name": "long overcoat", "short": "Long Overcoat", "role": "upper", "prompts": ["long winter overcoat", "knee-length wool overcoat coat"], "synonyms": ["overcoat", "long coat", "coat"]},
    {"name": "windbreaker / rain jacket", "short": "Windbreaker", "role": "upper", "prompts": ["nylon windbreaker shell jacket", "waterproof high-vis rain work jacket"], "synonyms": ["windbreaker", "rain jacket", "raincoat", "shell jacket", "jacket", "coat"]},
    {"name": "suit jacket / blazer", "short": "Blazer", "role": "upper", "prompts": ["tailored suit jacket or blazer", "formal sport coat blazer"], "synonyms": ["blazer", "suit jacket", "suit", "jacket"]},
    {"name": "overalls / workwear", "short": "Work Overalls", "role": "upper", "prompts": ["industrial work coveralls or overalls", "boiler suit uniform"], "synonyms": ["overalls", "workwear", "coveralls", "jumpsuit"]},
    {"name": "trousers / pants", "short": "Trousers / Pants", "role": "lower", "prompts": ["casual plain fabric trousers or work pants", "dark or khaki slacks trousers"], "synonyms": ["trousers", "pants", "slacks", "work pants"]},
    {"name": "blue denim jeans", "short": "Denim Jeans", "role": "lower", "prompts": ["blue denim jeans", "indigo denim pants legs"], "synonyms": ["jeans", "denim jeans", "denim", "pants"]},
    {"name": "shorts / cargo shorts", "short": "Shorts", "role": "lower", "prompts": ["person wearing knee-length shorts", "summer casual shorts or cargo shorts"], "synonyms": ["shorts", "cargo shorts", "short pants"]},
    {"name": "joggers / track pants", "short": "Joggers", "role": "lower", "prompts": ["person wearing sweatpants joggers", "athletic track pants with cuffs"], "synonyms": ["joggers", "track pants", "sweatpants", "pants"]},
    {"name": "cargo pants", "short": "Cargo Pants", "role": "lower", "prompts": ["utility cargo pants with pockets", "heavy work cargo trousers"], "synonyms": ["cargo pants", "cargo", "pants", "trousers"]},
    {"name": "dress / skirt", "short": "Dress / Skirt", "role": "lower", "prompts": ["person wearing a skirt or dress hem", "flowing skirt or dress"], "synonyms": ["dress", "skirt"]},
]

COLOR_NAMES = [c["name"] for c in VARIABLE_COLORS]
SHORT_COLOR_LABELS = {c["name"]: c["short"] for c in VARIABLE_COLORS}
COLOR_HEX_MAP = {c["name"]: c["hex"] for c in VARIABLE_COLORS}
COLOR_PROMPTS_MAP = {c["name"]: c["prompts"] for c in VARIABLE_COLORS}

TYPE_NAMES = [t["name"] for t in VARIABLE_TYPES]
TYPE_ROLES = [t["role"] for t in VARIABLE_TYPES]
SHORT_TYPE_LABELS = {t["name"]: t["short"] for t in VARIABLE_TYPES}
TYPE_PROMPTS_MAP = {t["name"]: t["prompts"] for t in VARIABLE_TYPES}

QUERY_COLOR_SYNONYMS: dict[str, set[str]] = {}
for _c in VARIABLE_COLORS:
    QUERY_COLOR_SYNONYMS.setdefault(_c["name"].lower(), set()).add(_c["name"])
    QUERY_COLOR_SYNONYMS.setdefault(_c["short"].lower(), set()).add(_c["name"])
    for _syn in _c["synonyms"]:
        QUERY_COLOR_SYNONYMS.setdefault(_syn.lower(), set()).add(_c["name"])

QUERY_TYPE_SYNONYMS: dict[str, set[str]] = {}
for _t in VARIABLE_TYPES:
    QUERY_TYPE_SYNONYMS.setdefault(_t["name"].lower(), set()).add(_t["name"])
    QUERY_TYPE_SYNONYMS.setdefault(_t["short"].lower(), set()).add(_t["name"])
    for _syn in _t["synonyms"]:
        QUERY_TYPE_SYNONYMS.setdefault(_syn.lower(), set()).add(_t["name"])
