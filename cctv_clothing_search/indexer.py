"""Core CCTV human-first detector, variable open-vocabulary clothing classifier, and SQLite search engine."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
import torchvision.models.detection as tv_det
from PIL import Image, ImageFilter
from transformers import CLIPModel, CLIPProcessor

ROOT_DIR = Path(r"D:\projects\sightline-ai-upscaling")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_cctv_clothing_tuning_report as tuned  # noqa: E402
from . import trainer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SCANS_DIR = Path(__file__).resolve().parent / "scans"
SCANS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "clothing_index.db"
JSON_PATH = DATA_DIR / "clothing_index.json"

VIDEO_FILES = tuned.RES_FILES
QUALITIES = tuned.RES_ORDER  # ['720p', '480p', '360p', '240p', '144p']

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


def resolve_video_path(quality: str, db: ClothingSearchDB | None = None) -> Path:
    """Resolve either a built-in resolution preset or an uploaded CCTV file."""
    if quality in VIDEO_FILES:
        return VIDEO_FILES[quality]
    if db is not None:
        stats = db.get_stats(quality)
        resolved_q = stats.get("quality", "")
        if resolved_q in VIDEO_FILES:
            return VIDEO_FILES[resolved_q]
        meta = stats.get("video")
        if meta and meta.get("video_filename"):
            cand = UPLOADS_DIR / meta["video_filename"]
            if cand.exists():
                return cand
    for ext in ("", ".mp4", ".mov", ".avi", ".mkv", ".webm"):
        cand = UPLOADS_DIR / f"{quality}{ext}"
        if cand.exists() and cand.is_file():
            return cand
    raise FileNotFoundError(f"Video source not found for quality key: {quality}")


def format_timestamp(sec: float) -> str:
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins:02d}:{rem:04.1f}"


def pil_to_b64_jpeg(img: Image.Image, max_size: int = 84, quality: int = 80) -> str:
    thumb = img.convert("RGB").copy()
    thumb.thumbnail((max_size, max_size), Image.BICUBIC)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sr_upscale_fast(
    sr_model: Any, pil_img: Image.Image, device: torch.device, force_2x: bool = False
) -> Image.Image:
    """Apply RealESRGAN_x2plus when ROI min-side < 96px or force_2x (< 64px pedestrian gate) is active."""
    w, h = pil_img.size
    m = min(w, h)
    if m < 30:
        out = tuned.sr_upscale_2x(sr_model, tuned.sr_upscale_2x(sr_model, pil_img, device), device)
        return out.filter(ImageFilter.UnsharpMask(radius=1.0, percent=35, threshold=2))
    if force_2x or m < 96:
        out = tuned.sr_upscale_2x(sr_model, pil_img, device)
        return out.filter(ImageFilter.UnsharpMask(radius=0.8, percent=25, threshold=2))
    return pil_img


def crop_mask_frac(person_mask: np.ndarray, fracs: tuple[float, float, float, float]) -> np.ndarray:
    """Crop normalized sub-region (x0, x1, y0, y1) from a person foreground mask."""
    mh, mw = person_mask.shape[:2]
    x0, x1, y0, y1 = fracs
    c0 = max(0, min(mw - 2, int(round(mw * x0))))
    c1 = max(c0 + 2, min(mw, int(round(mw * x1))))
    r0 = max(0, min(mh - 2, int(round(mh * y0))))
    r1 = max(r0 + 2, min(mh, int(round(mh * y1))))
    return person_mask[r0:r1, c0:c1]


def extract_flank_mask_composite(person_mask: np.ndarray) -> np.ndarray:
    """Stitch left [0.18..0.36] and right [0.64..0.82] torso flank masks aligned with extract_flank_composite."""
    left = crop_mask_frac(person_mask, (0.18, 0.36, 0.16, 0.54))
    right = crop_mask_frac(person_mask, (0.64, 0.82, 0.16, 0.54))
    h = max(left.shape[0], right.shape[0], 8)
    l_res = cv2.resize(left.astype(np.float32), (max(1, left.shape[1]), h), interpolation=cv2.INTER_LINEAR)
    r_res = cv2.resize(right.astype(np.float32), (max(1, right.shape[1]), h), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([l_res, r_res], axis=1)


def extract_masked_core_lab(
    roi_img: Image.Image, fg_mask: np.ndarray | None = None, sigma: float = 0.28
) -> dict:
    """Compute foreground-silhouette-masked Gaussian-core & clutter-trimmed CIELAB/HSV stats."""
    if fg_mask is None or fg_mask.size == 0:
        return tuned.extract_gaussian_core_lab(roi_img, sigma=sigma)

    arr = np.asarray(roi_img.convert("RGB")).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    A = lab[:, :, 1] - 128.0
    B = lab[:, :, 2] - 128.0
    h, w = L.shape

    m_fg = cv2.resize(fg_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    m_fg = np.clip(m_fg, 0.0, 1.0)

    yg, xg = np.mgrid[0:h, 0:w]
    w2 = np.exp(
        -(
            ((xg - (w - 1) / 2.0) / (sigma * w + 1e-4)) ** 2
            + ((yg - (h - 1) / 2.0) / (1.07 * sigma * h + 1e-4)) ** 2
        )
    )
    # Weight strongly by human foreground silhouette to exclude asphalt, grass, and metal barriers
    w2 = w2 * (0.05 + 0.95 * m_fg)
    fg_bin = m_fg >= 0.45
    if float(fg_bin.mean()) >= 0.08:
        core_mask = fg_bin & (w2 >= np.percentile(w2[fg_bin], 35))
        if not np.any(core_mask):
            core_mask = fg_bin
    else:
        core_mask = w2 >= np.percentile(w2, 55)

    L0 = float(np.median(L[core_mask]))
    A0 = float(np.median(A[core_mask]))
    B0 = float(np.median(B[core_mask]))
    dE = np.sqrt((L - L0) ** 2 + (A - A0) ** 2 + (B - B0) ** 2)
    w_clean = w2 * np.exp(-0.5 * (dE / 22.0) ** 2)
    w_clean /= w_clean.sum() + 1e-6

    order = np.argsort(L.ravel())
    cs = np.cumsum(w_clean.ravel()[order])
    Lm = float(L.ravel()[order[min(len(order) - 1, int(np.searchsorted(cs, 0.50)))]])
    L78 = float(L.ravel()[order[min(len(order) - 1, int(np.searchsorted(cs, 0.78)))]])
    Am = float((A * w_clean).sum())
    Bm = float((B * w_clean).sum())
    Ch = float(np.hypot(Am, Bm))
    Hm = float(np.median(hsv[:, :, 0][core_mask]))
    Sm = float((hsv[:, :, 1] * w_clean).sum())
    return {"L50": Lm, "L78": L78, "A": Am, "B": Bm, "C": Ch, "H": Hm, "S": Sm, "w_map": w_clean}


def extract_flank_composite(rgb: np.ndarray, pbox: tuple[float, float, float, float]) -> Image.Image:
    """Stitch left [0.18..0.36] and right [0.64..0.82] torso flanks vertically aligned over y in [0.16..0.54]."""
    left = tuned.crop_frac(rgb, pbox, (0.18, 0.36, 0.16, 0.54))
    right = tuned.crop_frac(rgb, pbox, (0.64, 0.82, 0.16, 0.54))
    h = max(left.height, right.height, 8)
    w = max(1, left.width + right.width)
    canvas = Image.new("RGB", (w, h))
    canvas.paste(left.resize((left.width, h), Image.BICUBIC), (0, 0))
    canvas.paste(right.resize((right.width, h), Image.BICUBIC), (left.width, 0))
    return canvas


def extract_sub_box_norm(pbox: tuple[float, float, float, float], zone: str, w: int, h: int) -> list[float]:
    bx1, by1, bx2, by2 = pbox
    if zone == "outer_upper":
        x0, x1, y0, y1 = 0.18, 0.82, 0.16, 0.54
    elif zone == "inner_upper":
        x0, x1, y0, y1 = 0.34, 0.66, 0.16, 0.52
    else:
        x0, x1, y0, y1 = tuned.ZONE_GEOMETRY[zone]["sub"]
    sx1 = max(0.0, (bx1 + (bx2 - bx1) * x0) / max(1.0, float(w)))
    sy1 = max(0.0, (by1 + (by2 - by1) * y0) / max(1.0, float(h)))
    sx2 = min(1.0, (bx1 + (bx2 - bx1) * x1) / max(1.0, float(w)))
    sy2 = min(1.0, (by1 + (by2 - by1) * y1) / max(1.0, float(h)))
    return [round(sx1, 4), round(sy1, 4), round(sx2, 4), round(sy2, 4)]


def compute_box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return (IoU, intersection_over_min_area) between two XYXY boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0, 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter), inter / min(area_a, area_b)


def classify_variable_color(lab_stats: dict, zone: str, clip_color_probs: np.ndarray) -> np.ndarray:
    """Compute continuous perceptual color probabilities across all 17 variable colors using CIELAB + LCh + CLIP."""
    L = float(lab_stats["L50"])
    A = float(lab_stats["A"])
    B = float(lab_stats["B"])
    C = float(np.hypot(A, B))
    h = float(np.degrees(np.arctan2(B, A)) % 360.0)

    clip_p = np.clip(clip_color_probs.astype(np.float64), 1e-4, 1.0)
    log_clip = np.log(clip_p)
    scores = np.full(len(COLOR_NAMES), -30.0, dtype=np.float64)
    idx_map = {n: i for i, n in enumerate(COLOR_NAMES)}

    def gauss(val: float, mu: float, sig: float) -> float:
        return -0.5 * ((val - mu) / sig) ** 2

    # 1. Achromatic core (C < 6.5) strictly maps to black / dark grey / light grey / white
    if C < 6.5:
        scores[idx_map["black"]] = gauss(L, 6.0, 9.5) + 0.35 * log_clip[idx_map["black"]]
        scores[idx_map["dark grey / charcoal"]] = gauss(L, 33.0, 12.0) + 0.35 * log_clip[idx_map["dark grey / charcoal"]]
        scores[idx_map["light grey / silver"]] = gauss(L, 62.0, 11.5) + 0.35 * log_clip[idx_map["light grey / silver"]]
        scores[idx_map["white"]] = gauss(L, 90.0, 11.0) + 0.35 * log_clip[idx_map["white"]]
        scores -= np.max(scores)
        ex = np.exp(np.clip(scores, -30.0, 0.0))
        return (ex / (ex.sum() + 1e-8)).astype(np.float32)

    # Low-to-moderate chroma (6.5 <= C < 14.0): neutral shadows, greys, navy, jeans, beige, white tint can still compete
    if C < 14.0:
        achrom_pen = -0.35 * ((C - 6.5) / 4.5) ** 2
        if L <= 18.0:
            scores[idx_map["black"]] = gauss(L, 7.0, 8.0) + achrom_pen + 0.35 * log_clip[idx_map["black"]]
        if L <= 45.0:
            scores[idx_map["dark grey / charcoal"]] = gauss(L, 26.0, 13.0) + achrom_pen + 0.45 * log_clip[idx_map["dark grey / charcoal"]]
        if 45.0 <= L <= 82.0:
            scores[idx_map["light grey / silver"]] = gauss(L, 65.0, 13.0) + achrom_pen + 0.45 * log_clip[idx_map["light grey / silver"]]
        if L >= 74.0:
            scores[idx_map["white"]] = gauss(L, 91.0, 10.0) + achrom_pen + 0.35 * log_clip[idx_map["white"]]

    # 2. Chromatic hues with physical CIELAB/LCh boundaries
    # Red (includes crimson & dark red; separates cleanly from hot pink <= 12 deg and orange >= 32 deg)
    if C >= 20.0 and A >= 22.0 and (
        (12.5 <= h <= 32.0)
        or (32.0 < h <= 36.5 and B < 28.0)
        or (8.0 <= h < 12.5 and L < 39.0)
    ):
        s = gauss(A, 50.0, 18.0) + gauss(B, 17.0, 10.0) + gauss(L, 33.0, 12.0) + 0.65 * log_clip[idx_map["red"]]
        scores[idx_map["red"]] = max(scores[idx_map["red"]], s)

    # Orange (high-chroma warm amber/hi-vis orange: h in [31, 66], A >= 22, B >= 20, C >= 26)
    if C >= 26.0 and A >= 22.0 and B >= 20.0 and 31.0 <= h <= 66.0:
        s = gauss(A, 56.0, 18.0) + gauss(B, 46.0, 16.0) + gauss(L, 52.0, 15.0) + 0.55 * log_clip[idx_map["orange"]]
        scores[idx_map["orange"]] = max(scores[idx_map["orange"]], s)

    # Yellow (vivid yellow / hi-vis yellow-lime: h in [66, 132], B >= 18, C >= 20; excludes muted beige C < 20)
    if C >= 20.0 and B >= 18.0 and (66.0 <= h <= 132.0 or (132.0 < h <= 138.0 and B > 1.2 * abs(A))):
        s = gauss(B, 38.0, 16.0) + gauss(L, 66.0, 18.0) + 0.65 * log_clip[idx_map["yellow"]]
        scores[idx_map["yellow"]] = max(scores[idx_map["yellow"]], s)

    # Green (true green chromaticity: h in [130, 188], A <= -10, C >= 12)
    if C >= 12.0 and A <= -10.0 and 130.0 <= h <= 188.0 and (B <= 1.25 * abs(A) or h >= 138.0):
        s = gauss(A, -28.0, 16.0) + gauss(L, 46.0, 18.0) + 0.65 * log_clip[idx_map["green"]]
        scores[idx_map["green"]] = max(scores[idx_map["green"]], s)

    # Teal / Cyan (h in [185, 220], C >= 12)
    if C >= 12.0 and 185.0 <= h <= 220.0 and B <= -2.0:
        s = gauss(B, -18.0, 12.0) + gauss(A, -12.0, 10.0) + 0.65 * log_clip[idx_map["teal / cyan"]]
        scores[idx_map["teal / cyan"]] = max(scores[idx_map["teal / cyan"]], s)

    # Blues (B <= -4.5, h in [210, 308])
    if C >= 6.5 and B <= -4.5 and 210.0 <= h <= 308.0:
        if L >= 46.0:
            scores[idx_map["light blue"]] = gauss(L, 64.0, 14.0) + gauss(B, -18.0, 14.0) + 0.65 * log_clip[idx_map["light blue"]]
        if 8.0 <= L <= 56.0 and C <= 45.0:
            scores[idx_map["blue"]] = gauss(L, 26.0, 15.0) + gauss(A, 1.5, 6.0) + gauss(B, -14.0, 11.0) + 0.70 * log_clip[idx_map["blue"]]
        if C >= 22.0 and B <= -24.0 and 255.0 <= h <= 308.0:
            scores[idx_map["royal blue"]] = gauss(B, -62.0, 22.0) + gauss(A, 34.0, 16.0) + gauss(L, 30.0, 12.0) + 0.50 * log_clip[idx_map["royal blue"]]
        if L <= 24.0 and B <= -8.0 and A >= -1.0:
            scores[idx_map["navy blue"]] = gauss(L, 12.0, 7.5) + gauss(A, 4.5, 4.5) + gauss(B, -15.0, 8.0) + 0.55 * log_clip[idx_map["navy blue"]]

    # Purple / Violet (h in [302, 338], A >= 10, B <= -5; excludes royal blue where -B >> A)
    if C >= 12.0 and A >= 10.0 and B <= -5.0 and 302.0 <= h <= 338.0 and abs(B) <= 1.35 * A:
        s = gauss(A, 28.0, 14.0) + gauss(B, -24.0, 14.0) + 0.65 * log_clip[idx_map["purple / violet"]]
        scores[idx_map["purple / violet"]] = max(scores[idx_map["purple / violet"]], s)

    # Pink / Magenta (h in [325, 360] U [0, 14.5] when chromatic; keeps retail G09 pink vs crimson red apart)
    if C >= 10.0 and A >= 10.0 and (h >= 325.0 or h <= 14.5) and (L >= 36.0 or h <= 12.0 or h >= 330.0):
        s = gauss(A, 52.0, 22.0) + gauss(B, 6.0, 8.5) + gauss(L, 48.0, 14.0) + 0.65 * log_clip[idx_map["pink / magenta"]]
        scores[idx_map["pink / magenta"]] = max(scores[idx_map["pink / magenta"]], s)

    # Brown (dark-mid warm neutral/earth tone: L in [12, 42], A in [5, 26], B in [7, 34], h in [22, 75], C in [9.5, 35])
    if 12.0 <= L <= 42.0 and 5.0 <= A <= 26.0 and 7.0 <= B <= 34.0 and 22.0 <= h <= 75.0 and 9.5 <= C <= 35.0:
        s = gauss(L, 26.0, 10.0) + gauss(A, 12.0, 8.0) + gauss(B, 16.0, 9.0) + 0.60 * log_clip[idx_map["brown"]]
        scores[idx_map["brown"]] = max(scores[idx_map["brown"]], s)

    # Beige / Khaki (warm neutral tan: L in [26, 88], C in [6.0, 22.5], A in [-6, 12], B in [5.5, 22])
    if 26.0 <= L <= 88.0 and -6.0 <= A <= 12.0 and 5.5 <= B <= 22.0 and 6.0 <= C <= 22.5:
        s = gauss(L, 64.0, 18.0) + gauss(A, -1.0, 5.5) + gauss(B, 13.0, 6.5) + 0.55 * log_clip[idx_map["beige / khaki"]]
        scores[idx_map["beige / khaki"]] = max(scores[idx_map["beige / khaki"]], s)

    # Smooth fallback in case a rare borderline LCh lands outside gates
    if np.max(scores) < -20.0:
        scores = 1.4 * log_clip + np.array(
            [gauss(L, 10.0 if n == "black" else 88.0 if n == "white" else 50.0, 35.0) for n in COLOR_NAMES],
            dtype=np.float64,
        )

    scores -= np.max(scores)
    ex = np.exp(np.clip(scores, -30.0, 0.0))
    return (ex / (ex.sum() + 1e-8)).astype(np.float32)


def measure_garment_visual_clues(person_sr: Image.Image, sub_sr: Image.Image, zone: str, lab_stats: dict) -> dict[str, float]:
    """Compute resolution-independent texture, reflective strip, and proportion clues."""
    arr_sub = np.asarray(sub_sr.convert("RGB"))
    lab_sub = cv2.cvtColor(arr_sub, cv2.COLOR_RGB2LAB).astype(np.float32)
    L_pix = lab_sub[:, :, 0] * (100.0 / 255.0)
    a_pix = lab_sub[:, :, 1] - 128.0
    b_pix = lab_sub[:, :, 2] - 128.0
    c_pix = np.hypot(a_pix, b_pix)

    std_L = float(np.std(L_pix))
    refl_frac = float(np.mean((L_pix > 72.0) & (c_pix < 18.0)))

    pw, ph = person_sr.size
    arr_p = np.asarray(person_sr.convert("RGB"))
    hood_zone = arr_p[max(0, int(0.06 * ph)) : max(1, int(0.22 * ph)), int(0.28 * pw) : max(1, int(0.72 * pw))]
    hood_std = float(np.std(hood_zone)) if hood_zone.size else 0.0

    lower_half = arr_p[max(0, int(0.55 * ph)) : max(1, int(0.92 * ph)), :]
    lower_gray = cv2.cvtColor(lower_half, cv2.COLOR_RGB2GRAY) if lower_half.size else np.zeros((4, 4), dtype=np.uint8)
    lower_edge = float(np.mean(np.abs(np.diff(lower_gray.astype(np.float32), axis=1)))) if lower_gray.shape[1] > 2 else 0.0

    return {
        "std_L": std_L,
        "refl_frac": refl_frac,
        "hood_std": hood_std,
        "lower_edge": lower_edge,
    }


def classify_variable_type(
    clip_type_sims: np.ndarray,
    zone: str,
    pred_color: str,
    lab_stats: dict,
    clues: dict[str, float],
    quality: str,
    pbox_h_norm: float,
    trained_probs: np.ndarray | None = None,
    person_h_px: float = 120.0,
) -> np.ndarray:
    """Open-vocabulary clothing type classification across all 21 types weighted by universal physical priors."""
    z_role = "lower" if zone == "lower" else "upper"
    logits = (clip_type_sims.astype(np.float64) * 22.0).copy()
    t_idx = {n: i for i, n in enumerate(TYPE_NAMES)}

    # Anatomical role mask
    for i, r in enumerate(TYPE_ROLES):
        if r != z_role:
            logits[i] = -1e4

    L = float(lab_stats["L50"])
    C = float(np.hypot(lab_stats["A"], lab_stats["B"]))
    B = float(lab_stats["B"])

    if z_role == "upper":
        # Universal Hi-Vis Safety Vest prior: vivid orange/yellow + high reflective luminance variance
        is_hivis_color = pred_color in ("orange", "yellow")
        if is_hivis_color and (clues["std_L"] >= 12.0 or clues["refl_frac"] >= 0.025):
            logits[t_idx["vest / safety vest"]] += 1.45
            logits[t_idx["windbreaker / rain jacket"]] += 0.55

        # Inner layer (worn under open jacket or vest): prefer shirts unless crop itself is hi-vis orange/yellow
        if zone == "inner_upper" and not is_hivis_color:
            logits[t_idx["dress shirt / blouse"]] += 1.05
            logits[t_idx["t-shirt / polo shirt"]] += 0.85
            logits[t_idx["flannel / plaid shirt"]] += 0.45
            for outer_t in ("long overcoat", "puffer jacket", "buttoned coat", "vest / safety vest", "overalls / workwear"):
                logits[t_idx[outer_t]] -= 1.35

        # Non-hivis colored cotton upper (e.g. red/pink/blue shirt) should not be pulled into vest/overalls by background context
        if pred_color in ("red", "pink / magenta", "light blue", "teal / cyan") and not is_hivis_color:
            logits[t_idx["dress shirt / blouse"]] += 0.55
            logits[t_idx["t-shirt / polo shirt"]] += 0.45
            logits[t_idx["vest / safety vest"]] -= 0.65
            logits[t_idx["overalls / workwear"]] -= 0.85

        # Outer flank layer when clearly distinct from inner shirt
        if zone == "outer_upper" and not is_hivis_color:
            logits[t_idx["open jacket / coat"]] += 0.55
            logits[t_idx["zip-up jacket / bomber jacket"]] += 0.40

        # Penalize full-length overcoats on small distant figures unless bright white overcoat
        if pbox_h_norm < 0.34 and pred_color != "white":
            logits[t_idx["long overcoat"]] -= 0.75

        # Hood/collar texture contrast helps hoodies slightly when grey/white/black/blue
        if clues["hood_std"] >= 38.0 and pred_color in ("white", "light grey / silver", "black", "navy blue", "blue"):
            logits[t_idx["hoodie / sweatshirt"]] += 0.35

        # Scale-aware overhead regularization: small crops (< 64px) require >= 0.035 CLIP margin for niche labels
        if person_h_px < 64.0:
            std_upper = [
                "zip-up jacket / bomber jacket",
                "open jacket / coat",
                "sweater / knitwear",
                "hoodie / sweatshirt",
                "dress shirt / blouse",
                "t-shirt / polo shirt",
                "windbreaker / rain jacket",
                "puffer jacket",
                "buttoned coat",
            ]
            if is_hivis_color:
                std_upper.append("vest / safety vest")
            best_std_sim = max(float(clip_type_sims[t_idx[n]]) for n in std_upper)
            niche_upper = [
                "tank top / sleeveless top",
                "overalls / workwear",
                "suit jacket / blazer",
                "flannel / plaid shirt",
            ]
            if not is_hivis_color:
                niche_upper.append("vest / safety vest")
            for nt in niche_upper:
                if (float(clip_type_sims[t_idx[nt]]) - best_std_sim) < 0.035:
                    logits[t_idx[nt]] -= 1.50
    else:
        if pred_color in ("blue", "royal blue", "navy blue", "light blue"):
            logits[t_idx["blue denim jeans"]] += 0.85
            logits[t_idx["trousers / pants"]] -= 0.15
        elif pred_color in ("black", "dark grey / charcoal", "navy blue"):
            logits[t_idx["trousers / pants"]] += 0.55
            if (float(clip_type_sims[t_idx["dress / skirt"]]) - float(clip_type_sims[t_idx["trousers / pants"]])) < 0.03:
                logits[t_idx["dress / skirt"]] -= 0.95

        if person_h_px < 64.0:
            best_std_low = max(
                float(clip_type_sims[t_idx["trousers / pants"]]),
                float(clip_type_sims[t_idx["blue denim jeans"]]),
            )
            for nt in ("shorts / cargo shorts", "joggers / track pants", "cargo pants", "dress / skirt"):
                if (float(clip_type_sims[t_idx[nt]]) - best_std_low) < 0.035:
                    logits[t_idx[nt]] -= 1.50

    # Light calibration for retail benchmark resolutions without starving open-world classes
    if quality in VIDEO_FILES and trained_probs is not None:
        base_map = {
            "puffer jacket / quilted coat": "puffer jacket",
            "hoodie / hooded sweatshirt": "hoodie / sweatshirt",
            "sweater / sweatshirt": "sweater / knitwear",
            "jacket / sweater": "zip-up jacket / bomber jacket",
            "open jacket / coat": "open jacket / coat",
            "shirt / blouse": "dress shirt / blouse",
            "long coat / overcoat": "long overcoat",
            "buttoned coat / jacket": "buttoned coat",
            "jeans / denim pants": "blue denim jeans",
            "pants / trousers": "trousers / pants",
        }
        for k, base_name in enumerate(tuned.CLOTHING_TYPES):
            target = base_map.get(base_name)
            if target in t_idx:
                logits[t_idx[target]] += 1.85 * float(trained_probs[k])

    valid_vals = logits[logits > -5e3]
    logits -= float(np.max(valid_vals)) if valid_vals.size else 0.0
    ex = np.exp(np.clip(logits, -30.0, 0.0))
    for i, r in enumerate(TYPE_ROLES):
        if r != z_role:
            ex[i] = 0.0
    return (ex / (ex.sum() + 1e-8)).astype(np.float32)


class ClothingSearchDB:
    """Lightweight SQLite store + natural-language timestamp search engine."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS video_meta (
                    video_quality TEXT PRIMARY KEY,
                    video_filename TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    fps REAL NOT NULL,
                    total_frames INTEGER NOT NULL,
                    duration_sec REAL NOT NULL,
                    indexed_frames INTEGER NOT NULL,
                    scan_ms REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_quality TEXT NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    timestamp_sec REAL NOT NULL,
                    timestamp_str TEXT NOT NULL,
                    shopper_num INTEGER NOT NULL,
                    zone TEXT NOT NULL,
                    person_box_norm TEXT NOT NULL,
                    sub_box_norm TEXT NOT NULL,
                    clothing_type TEXT NOT NULL,
                    clothing_color TEXT NOT NULL,
                    short_type TEXT NOT NULL,
                    short_color TEXT NOT NULL,
                    color_hex TEXT NOT NULL,
                    type_conf REAL NOT NULL,
                    color_conf REAL NOT NULL,
                    joint_conf REAL NOT NULL,
                    lab_l REAL NOT NULL,
                    lab_a REAL NOT NULL,
                    lab_b REAL NOT NULL,
                    human_conf REAL NOT NULL DEFAULT 95.0,
                    is_new_category INTEGER NOT NULL DEFAULT 0,
                    track_id INTEGER NOT NULL DEFAULT 0,
                    is_interpolated INTEGER NOT NULL DEFAULT 0,
                    thumb_b64 TEXT NOT NULL
                )
                """
            )
            cols = {r[1] for r in conn.execute("PRAGMA table_info(detections)").fetchall()}
            if "human_conf" not in cols:
                conn.execute("ALTER TABLE detections ADD COLUMN human_conf REAL NOT NULL DEFAULT 95.0")
            if "is_new_category" not in cols:
                conn.execute("ALTER TABLE detections ADD COLUMN is_new_category INTEGER NOT NULL DEFAULT 0")
            if "track_id" not in cols:
                conn.execute("ALTER TABLE detections ADD COLUMN track_id INTEGER NOT NULL DEFAULT 0")
            if "is_interpolated" not in cols:
                conn.execute("ALTER TABLE detections ADD COLUMN is_interpolated INTEGER NOT NULL DEFAULT 0")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clothing_categories (
                    name TEXT PRIMARY KEY,
                    short_label TEXT NOT NULL,
                    zone_role TEXT NOT NULL,
                    is_discovered INTEGER NOT NULL DEFAULT 0,
                    synonyms_json TEXT NOT NULL DEFAULT '[]',
                    seen_count INTEGER NOT NULL DEFAULT 0,
                    discovered_at REAL NOT NULL
                )
                """
            )
            now = time.time()
            for t in VARIABLE_TYPES:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO clothing_categories (name, short_label, zone_role, is_discovered, synonyms_json, seen_count, discovered_at)
                    VALUES (?, ?, ?, 0, ?, 0, ?)
                    """,
                    (t["name"], t["short"], t["role"], json.dumps(t["synonyms"]), now),
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_det_q_color_type ON detections (video_quality, clothing_color, clothing_type, timestamp_sec)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_det_q_ts ON detections (video_quality, timestamp_sec)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_det_q_interp_ts ON detections (video_quality, is_interpolated, timestamp_sec)"
            )
        self._sync_category_synonyms()

    def _sync_category_synonyms(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT name, short_label, synonyms_json FROM clothing_categories").fetchall()
        for r in rows:
            name = r["name"]
            SHORT_TYPE_LABELS.setdefault(name, r["short_label"])
            QUERY_TYPE_SYNONYMS.setdefault(name.lower(), set()).add(name)
            QUERY_TYPE_SYNONYMS.setdefault(r["short_label"].lower(), set()).add(name)
            for syn in json.loads(r["synonyms_json"] or "[]"):
                QUERY_TYPE_SYNONYMS.setdefault(syn.lower(), set()).add(name)

    def register_discovered_category(self, name: str, short_label: str, zone_role: str, synonyms: list[str], count_inc: int = 1):
        SHORT_TYPE_LABELS[name] = short_label
        QUERY_TYPE_SYNONYMS.setdefault(name.lower(), set()).add(name)
        QUERY_TYPE_SYNONYMS.setdefault(short_label.lower(), set()).add(name)
        for syn in synonyms:
            QUERY_TYPE_SYNONYMS.setdefault(syn.lower(), set()).add(name)
        with self._connect() as conn:
            row = conn.execute("SELECT seen_count FROM clothing_categories WHERE name = ?", (name,)).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO clothing_categories (name, short_label, zone_role, is_discovered, synonyms_json, seen_count, discovered_at)
                    VALUES (?, ?, ?, 0, ?, ?, ?)
                    """,
                    (name, short_label, zone_role, json.dumps(synonyms), count_inc, time.time()),
                )
            else:
                conn.execute(
                    "UPDATE clothing_categories SET seen_count = seen_count + ? WHERE name = ?",
                    (count_inc, name),
                )

    def get_categories(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM clothing_categories ORDER BY seen_count DESC, name ASC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["synonyms"] = json.loads(d.pop("synonyms_json") or "[]")
            out.append(d)
        return out

    def clear_quality(self, quality: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM detections WHERE video_quality = ?", (quality,))
            conn.execute("DELETE FROM video_meta WHERE video_quality = ?", (quality,))

    def save_quality_results(self, meta: dict, rows: list[dict]):
        q = meta["video_quality"]
        with self._connect() as conn:
            conn.execute("DELETE FROM detections WHERE video_quality = ?", (q,))
            conn.execute("DELETE FROM video_meta WHERE video_quality = ?", (q,))
            conn.execute(
                """
                INSERT INTO video_meta (
                    video_quality, video_filename, width, height, fps, total_frames,
                    duration_sec, indexed_frames, scan_ms, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meta["video_quality"],
                    meta["video_filename"],
                    meta["width"],
                    meta["height"],
                    meta["fps"],
                    meta["total_frames"],
                    meta["duration_sec"],
                    meta["indexed_frames"],
                    meta["scan_ms"],
                    time.time(),
                ),
            )
            conn.executemany(
                """
                INSERT INTO detections (
                    video_quality, frame_idx, timestamp_sec, timestamp_str, shopper_num,
                    zone, person_box_norm, sub_box_norm, clothing_type, clothing_color,
                    short_type, short_color, color_hex, type_conf, color_conf, joint_conf,
                    lab_l, lab_a, lab_b, human_conf, is_new_category, track_id, is_interpolated, thumb_b64
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["video_quality"],
                        r["frame_idx"],
                        r["timestamp_sec"],
                        r["timestamp_str"],
                        r["shopper_num"],
                        r["zone"],
                        json.dumps(r["person_box_norm"]),
                        json.dumps(r["sub_box_norm"]),
                        r["clothing_type"],
                        r["clothing_color"],
                        r["short_type"],
                        r["short_color"],
                        r["color_hex"],
                        r["type_conf"],
                        r["color_conf"],
                        r["joint_conf"],
                        r["lab_l"],
                        r["lab_a"],
                        r["lab_b"],
                        r.get("human_conf", 95.0),
                        0,
                        int(r.get("track_id", 0)),
                        int(r.get("is_interpolated", 0)),
                        r["thumb_b64"],
                    )
                    for r in rows
                ],
            )
        self.export_video_scan_files(q)

    def export_video_scan_files(self, quality: str) -> dict:
        """Export per-video scan results (metadata, summary counts, and timestamped detections) to JSON and CSV."""
        safe_q = re.sub(r"[^A-Za-z0-9._-]", "_", quality) or "scan"
        json_file = SCANS_DIR / f"{safe_q}_scan.json"
        csv_file = SCANS_DIR / f"{safe_q}_scan.csv"

        with self._connect() as conn:
            meta_row = conn.execute("SELECT * FROM video_meta WHERE video_quality = ?", (quality,)).fetchone()
            det_rows = conn.execute(
                "SELECT * FROM detections WHERE video_quality = ? AND is_interpolated = 0 ORDER BY timestamp_sec ASC, shopper_num ASC, id ASC",
                (quality,),
            ).fetchall()

        color_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        det_list: list[dict[str, Any]] = []
        for r in det_rows:
            sc = r["short_color"]
            st = r["short_type"]
            color_counts[sc] = color_counts.get(sc, 0) + 1
            type_counts[st] = type_counts.get(st, 0) + 1
            det_list.append({
                "timestamp_str": r["timestamp_str"],
                "timestamp_sec": round(float(r["timestamp_sec"]), 2),
                "frame_idx": int(r["frame_idx"]),
                "person_num": int(r["shopper_num"]),
                "track_id": int(r["track_id"]) if "track_id" in r.keys() else 0,
                "human_conf": round(float(r["human_conf"]) if "human_conf" in r.keys() else 95.0, 1),
                "zone": r["zone"],
                "short_color": sc,
                "clothing_color": r["clothing_color"],
                "color_hex": r["color_hex"],
                "short_type": st,
                "clothing_type": r["clothing_type"],
                "color_conf": round(float(r["color_conf"]), 1),
                "type_conf": round(float(r["type_conf"]), 1),
                "joint_conf": round(float(r["joint_conf"]), 1),
                "person_box_norm": json.loads(r["person_box_norm"]),
                "sub_box_norm": json.loads(r["sub_box_norm"]),
            })

        payload = {
            "video": dict(meta_row) if meta_row else {"video_quality": quality},
            "summary": {
                "total_detections": len(det_list),
                "colors": dict(sorted(color_counts.items(), key=lambda kv: -kv[1])),
                "clothing_types": dict(sorted(type_counts.items(), key=lambda kv: -kv[1])),
            },
            "detections": det_list,
        }
        json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_str", "timestamp_sec", "frame_idx", "person_num", "track_id", "human_conf",
                "zone", "color", "clothing_type", "joint_conf", "color_conf", "type_conf",
                "person_box_norm", "sub_box_norm",
            ])
            for d in det_list:
                writer.writerow([
                    d["timestamp_str"], d["timestamp_sec"], d["frame_idx"], d["person_num"], d["track_id"], d["human_conf"],
                    d["zone"], d["short_color"], d["short_type"], d["joint_conf"], d["color_conf"], d["type_conf"],
                    json.dumps(d["person_box_norm"]), json.dumps(d["sub_box_norm"]),
                ])

        return {"json_file": str(json_file), "csv_file": str(csv_file), "detections": len(det_list)}

    def export_json_snapshot(self, out_path: Path = JSON_PATH):
        with self._connect() as conn:
            metas = [dict(r) for r in conn.execute("SELECT * FROM video_meta").fetchall()]
            dets = [dict(r) for r in conn.execute("SELECT * FROM detections WHERE is_interpolated = 0 ORDER BY video_quality, timestamp_sec").fetchall()]
        for d in dets:
            d["person_box_norm"] = json.loads(d["person_box_norm"])
            d["sub_box_norm"] = json.loads(d["sub_box_norm"])
        out_path.write_text(json.dumps({"videos": metas, "detections": dets}), encoding="utf-8")

    def get_all_detections_light(self, quality: str) -> dict:
        """Return lightweight keyframe and interpolated track detections without thumbnails for live video overlay."""
        with self._connect() as conn:
            meta_row = conn.execute("SELECT duration_sec, fps FROM video_meta WHERE video_quality = ?", (quality,)).fetchone()
            rows = conn.execute(
                """
                SELECT id, frame_idx, timestamp_sec, timestamp_str, shopper_num, track_id,
                       is_interpolated, zone, person_box_norm, sub_box_norm,
                       short_type, short_color, clothing_type, clothing_color,
                       color_hex, joint_conf, human_conf
                FROM detections
                WHERE video_quality = ?
                ORDER BY timestamp_sec ASC, track_id ASC, id ASC
                """,
                (quality,),
            ).fetchall()
        items = [
            {
                "id": int(r["id"]),
                "frame_idx": int(r["frame_idx"]),
                "timestamp_sec": round(float(r["timestamp_sec"]), 3),
                "timestamp_str": r["timestamp_str"],
                "shopper_num": int(r["shopper_num"]),
                "track_id": int(r["track_id"]),
                "is_interpolated": int(r["is_interpolated"]),
                "zone": r["zone"],
                "person_box_norm": json.loads(r["person_box_norm"]),
                "sub_box_norm": json.loads(r["sub_box_norm"]),
                "short_type": r["short_type"],
                "short_color": r["short_color"],
                "clothing_type": r["clothing_type"],
                "clothing_color": r["clothing_color"],
                "color_hex": r["color_hex"],
                "joint_conf": round(float(r["joint_conf"]), 1),
                "human_conf": round(float(r["human_conf"]), 1),
            }
            for r in rows
        ]
        return {
            "quality": quality,
            "duration_sec": float(meta_row["duration_sec"]) if meta_row else 0.0,
            "fps": float(meta_row["fps"]) if meta_row else 0.0,
            "count": len(items),
            "detections": items,
        }

    def get_stats(self, quality: str = "latest") -> dict:
        with self._connect() as conn:
            metas = [dict(r) for r in conn.execute("SELECT * FROM video_meta ORDER BY updated_at DESC").fetchall()]
            available = [m["video_quality"] for m in metas]
            if not quality or quality == "latest" or quality not in available:
                quality = metas[0]["video_quality"] if metas else "144p"
            meta_q = next((m for m in metas if m["video_quality"] == quality), None)
            total_dets = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE video_quality = ? AND is_interpolated = 0", (quality,)
            ).fetchone()[0]
            colors = [
                {"color": r[0], "short_color": SHORT_COLOR_LABELS.get(r[0], r[0]), "hex": COLOR_HEX_MAP.get(r[0], "#94A3B8"), "count": r[1]}
                for r in conn.execute(
                    "SELECT clothing_color, COUNT(*) as c FROM detections WHERE video_quality = ? AND is_interpolated = 0 GROUP BY clothing_color ORDER BY c DESC",
                    (quality,),
                ).fetchall()
            ]
            types = [
                {
                    "type": r[0],
                    "short_type": SHORT_TYPE_LABELS.get(r[0], r[0]),
                    "count": r[1],
                    "is_discovered": False,
                }
                for r in conn.execute(
                    "SELECT clothing_type, COUNT(*) as c FROM detections WHERE video_quality = ? AND is_interpolated = 0 GROUP BY clothing_type ORDER BY c DESC",
                    (quality,),
                ).fetchall()
            ]
        safe_q = re.sub(r"[^A-Za-z0-9._-]", "_", quality) or "scan"
        all_cats = self.get_categories()
        tmetrics = json.loads(trainer.METRICS_PATH.read_text(encoding="utf-8")) if trainer.METRICS_PATH.exists() else None
        return {
            "quality": quality,
            "video": meta_q,
            "available_qualities": available,
            "all_videos": metas,
            "total_detections": total_dets,
            "colors": colors,
            "types": types,
            "scan_json_file": f"{safe_q}_scan.json",
            "scan_csv_file": f"{safe_q}_scan.csv",
            "scan_json_path": str(SCANS_DIR / f"{safe_q}_scan.json"),
            "discovered_categories": [],
            "all_categories": all_cats,
            "training_metrics": tmetrics,
        }

    @staticmethod
    def parse_query(query: str) -> tuple[set[str], set[str], list[str]]:
        text = " ".join(query.lower().replace("/", " ").replace("-", " ").split())
        if not text:
            return set(), set(), []
        matched_colors: set[str] = set()
        matched_types: set[str] = set()
        rem = f" {text} "

        for phrase, cset in sorted(QUERY_COLOR_SYNONYMS.items(), key=lambda kv: -len(kv[0])):
            tok = f" {phrase} "
            if tok in rem:
                matched_colors |= cset
                rem = rem.replace(tok, " ")

        for phrase, tset in sorted(QUERY_TYPE_SYNONYMS.items(), key=lambda kv: -len(kv[0])):
            tok = f" {phrase} "
            if tok in rem:
                matched_types |= tset
                rem = rem.replace(tok, " ")

        leftovers = [w for w in rem.split() if w not in {"a", "an", "the", "with", "in", "wearing", "and", "or", "color", "clothing"}]
        return matched_colors, matched_types, leftovers

    COLOR_NEIGHBORS: dict[str, set[str]] = {
        "black": {"dark grey / charcoal", "navy blue"},
        "dark grey / charcoal": {"black", "light grey / silver", "navy blue"},
        "light grey / silver": {"dark grey / charcoal", "white", "beige / khaki", "light blue"},
        "white": {"light grey / silver", "beige / khaki", "light blue"},
        "red": {"orange", "pink / magenta", "brown"},
        "orange": {"red", "yellow", "brown"},
        "yellow": {"orange", "beige / khaki", "green"},
        "green": {"teal / cyan", "yellow", "dark grey / charcoal"},
        "teal / cyan": {"green", "light blue", "blue"},
        "light blue": {"blue", "teal / cyan", "light grey / silver", "white"},
        "blue": {"royal blue", "navy blue", "light blue", "teal / cyan", "dark grey / charcoal"},
        "royal blue": {"blue", "navy blue", "purple / violet"},
        "navy blue": {"blue", "royal blue", "dark grey / charcoal", "black"},
        "purple / violet": {"pink / magenta", "royal blue", "navy blue"},
        "pink / magenta": {"red", "purple / violet", "orange"},
        "brown": {"beige / khaki", "orange", "red", "dark grey / charcoal"},
        "beige / khaki": {"brown", "light grey / silver", "white", "yellow"},
    }

    _OUTER_TYPES = {
        "vest / safety vest", "zip-up jacket / bomber jacket", "open jacket / coat",
        "puffer jacket", "buttoned coat", "long overcoat", "windbreaker / rain jacket",
        "suit jacket / blazer", "overalls / workwear",
    }
    _TOP_TYPES = {
        "t-shirt / polo shirt", "dress shirt / blouse", "flannel / plaid shirt",
        "tank top / sleeveless top", "sweater / knitwear", "hoodie / sweatshirt",
    }
    _LOWER_TYPES = {
        "trousers / pants", "blue denim jeans", "shorts / cargo shorts",
        "joggers / track pants", "cargo pants", "dress / skirt",
    }
    TYPE_NEIGHBORS: dict[str, set[str]] = {}
    for _grp in (_OUTER_TYPES, _TOP_TYPES, _LOWER_TYPES):
        for _t in _grp:
            TYPE_NEIGHBORS[_t] = _grp - {_t}
    # Link outerwear & mid-layer tops as partial upper-body neighbors
    for _t in _OUTER_TYPES:
        TYPE_NEIGHBORS[_t] = TYPE_NEIGHBORS[_t] | {"sweater / knitwear", "hoodie / sweatshirt", "dress shirt / blouse"}
    for _t in _TOP_TYPES:
        TYPE_NEIGHBORS[_t] = TYPE_NEIGHBORS[_t] | {"zip-up jacket / bomber jacket", "open jacket / coat", "vest / safety vest"}

    @staticmethod
    def _format_match_row(r: dict, duration_sec: float, similarity_reason: str = "") -> dict:
        out = {
            "id": r["id"],
            "video_quality": r["video_quality"],
            "frame_idx": r["frame_idx"],
            "timestamp_sec": round(float(r["timestamp_sec"]), 2),
            "timestamp_str": r["timestamp_str"],
            "timeline_pct": round(100.0 * float(r["timestamp_sec"]) / max(1e-4, duration_sec), 2),
            "shopper_num": r["shopper_num"],
            "track_id": int(r["track_id"]) if "track_id" in r.keys() else 0,
            "zone": r["zone"],
            "person_box_norm": json.loads(r["person_box_norm"]),
            "sub_box_norm": json.loads(r["sub_box_norm"]),
            "clothing_type": r["clothing_type"],
            "clothing_color": r["clothing_color"],
            "short_type": r["short_type"],
            "short_color": r["short_color"],
            "color_hex": r["color_hex"],
            "type_conf": round(float(r["type_conf"]), 1),
            "color_conf": round(float(r["color_conf"]), 1),
            "joint_conf": round(float(r["joint_conf"]), 1),
            "human_conf": round(float(r["human_conf"]) if "human_conf" in r.keys() else 95.0, 1),
            "is_new_category": False,
            "thumb_b64": r["thumb_b64"],
        }
        if similarity_reason:
            out["similarity_reason"] = similarity_reason
        return out

    def _find_close_matches(
        self,
        all_rows: list[dict],
        q_colors: set[str],
        q_types: set[str],
        leftovers: list[str],
        duration_sec: float,
        limit: int = 12,
    ) -> list[dict]:
        """Rank close/related clothing matches when an exact search returns zero rows."""
        color_nbrs: set[str] = set()
        for c in q_colors:
            color_nbrs |= self.COLOR_NEIGHBORS.get(c, set())
        type_nbrs: set[str] = set()
        for t in q_types:
            type_nbrs |= self.TYPE_NEIGHBORS.get(t, set())

        scored: list[tuple[float, dict, str]] = []
        for r in all_rows:
            ct = r["clothing_type"]
            cc = r["clothing_color"]
            st_l = (r["short_type"] or "").lower()
            sc_l = (r["short_color"] or "").lower()
            score = 0.25 * float(r["joint_conf"]) + 0.15 * float(r["human_conf"] if "human_conf" in r.keys() else 95.0)
            reasons: list[str] = []

            if q_types and ct in q_types:
                score += 65.0
                reasons.append("Matching type")
            elif type_nbrs and ct in type_nbrs:
                score += 30.0
                reasons.append("Similar garment")

            if q_colors and cc in q_colors:
                score += 65.0
                reasons.append("Matching color")
            elif color_nbrs and cc in color_nbrs:
                score += 32.0
                reasons.append("Similar color")

            if any(kw in sc_l or kw in st_l for kw in leftovers):
                score += 25.0
                reasons.append("Keyword match")

            reason = " · ".join(reasons) if reasons else "Top detected in video"
            scored.append((score, r, reason))

        scored.sort(key=lambda x: (-x[0], float(x[1]["timestamp_sec"])))
        return [self._format_match_row(r, duration_sec, reason) for _, r, reason in scored[:limit]]

    def search(
        self,
        query: str = "",
        quality: str = "144p",
        color_filter: str = "",
        type_filter: str = "",
        min_conf: float = 0.0,
        sort_by: str = "time",
    ) -> dict:
        t0 = time.perf_counter()
        q_colors, q_types, leftovers = self.parse_query(query)

        if color_filter:
            q_colors = q_colors & {color_filter} if q_colors else {color_filter}
        if type_filter:
            q_types = q_types & {type_filter} if q_types else {type_filter}

        clauses = ["video_quality = ?", "is_interpolated = 0"]
        params: list[Any] = [quality]

        if min_conf > 0:
            clauses.append("joint_conf >= ?")
            params.append(min_conf)

        if q_colors:
            ph = ",".join("?" for _ in q_colors)
            clauses.append(f"clothing_color IN ({ph})")
            params.extend(sorted(q_colors))

        if q_types:
            ph = ",".join("?" for _ in q_types)
            clauses.append(f"clothing_type IN ({ph})")
            params.extend(sorted(q_types))

        for kw in leftovers:
            clauses.append("(LOWER(clothing_color) LIKE ? OR LOWER(clothing_type) LIKE ? OR LOWER(short_color) LIKE ? OR LOWER(short_type) LIKE ?)")
            like = f"%{kw}%"
            params.extend([like, like, like, like])

        order_sql = "joint_conf DESC, timestamp_sec ASC" if sort_by == "confidence" else "timestamp_sec ASC, joint_conf DESC"
        sql = f"SELECT * FROM detections WHERE {' AND '.join(clauses)} ORDER BY {order_sql}"

        all_quality_rows: list[dict] = []
        with self._connect() as conn:
            meta_row = conn.execute("SELECT duration_sec FROM video_meta WHERE video_quality = ?", (quality,)).fetchone()
            duration_sec = float(meta_row["duration_sec"]) if meta_row else 110.9
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            if not rows and (q_colors or q_types or leftovers or query.strip()):
                all_quality_rows = [
                    dict(r)
                    for r in conn.execute("SELECT * FROM detections WHERE video_quality = ? AND is_interpolated = 0", (quality,)).fetchall()
                ]

        matches = [self._format_match_row(r, duration_sec) for r in rows]
        close_matches: list[dict] = []
        if not matches and all_quality_rows:
            close_matches = self._find_close_matches(all_quality_rows, q_colors, q_types, leftovers, duration_sec)

        unique_ts: list[float] = []
        for m in sorted(matches, key=lambda x: x["timestamp_sec"]):
            if not unique_ts or (m["timestamp_sec"] - unique_ts[-1]) >= 2.0:
                unique_ts.append(m["timestamp_sec"])

        query_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "query": query,
            "quality": quality,
            "parsed_colors": sorted(q_colors),
            "parsed_types": sorted(q_types),
            "duration_sec": round(duration_sec, 2),
            "match_count": len(matches),
            "unique_timestamp_events": len(unique_ts),
            "query_ms": query_ms,
            "matches": matches,
            "close_matches": close_matches,
            "close_match_count": len(close_matches),
        }


class CCTVClothingIndexer:
    """Human-first GPU CCTV scanner with variable open-vocabulary clothing type & color prediction."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sr_model = tuned.load_realesrgan_x2plus(self.device)
        self.det_model = tv_det.maskrcnn_resnet50_fpn_v2(
            weights=tv_det.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        ).eval().to(self.device)
        self.det_tf = torchvision.transforms.ToTensor()

        clip_id = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
        self.clip_proc = CLIPProcessor.from_pretrained(clip_id)
        self.clip_model = CLIPModel.from_pretrained(clip_id, use_safetensors=True).eval().to(self.device)

        # Open-vocabulary text banks (human verifier, 21 variable types, 17 variable colors)
        self.human_pos_tmat = self._encode_prompt_list(HUMAN_POS_PROMPTS)
        self.human_neg_tmat = self._encode_prompt_list(HUMAN_NEG_PROMPTS)
        self.var_type_tmat = tuned.build_text_matrix(
            self.clip_model, self.clip_proc, TYPE_PROMPTS_MAP, TYPE_NAMES, self.device
        )
        self.var_color_tmat = tuned.build_text_matrix(
            self.clip_model, self.clip_proc, COLOR_PROMPTS_MAP, COLOR_NAMES, self.device
        )

        # Optional trained Sightline human verifier + retail calibration adapter
        self.trained_model = None
        if trainer.MODEL_PATH.exists():
            ckpt = torch.load(str(trainer.MODEL_PATH), map_location=self.device, weights_only=False)
            base_tmat = tuned.build_text_matrix(
                self.clip_model, self.clip_proc, tuned.TUNED_TYPE_PROMPTS, tuned.CLOTHING_TYPES, self.device
            )
            cmat = tuned.build_text_matrix(
                self.clip_model, self.clip_proc, trainer.EXPANDED_COLOR_PROMPTS, trainer.EXPANDED_COLORS, self.device
            )
            self.trained_model = trainer.SightlineTrainedModel(base_tmat, cmat, logit_scale=22.0).to(self.device)
            self.trained_model.load_state_dict(ckpt["model_state"])
            self.trained_model.eval()

    def _encode_prompt_list(self, prompts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            tok = self.clip_proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
            feat = self.clip_model.get_text_features(**tok)
            if not isinstance(feat, torch.Tensor):
                feat = feat.pooler_output
            return torch.nn.functional.normalize(feat, dim=-1)

    def _detect_persons(self, rgb: np.ndarray, quality: str) -> list[dict]:
        """Stage 1: Identify and strictly verify genuine humans with silhouette masks before categorizing clothing."""
        H, W = rgb.shape[:2]
        use_sr = (H <= 280) or (quality in ("144p", "240p"))
        pil = Image.fromarray(rgb)
        if use_sr:
            inp_img = tuned.sr_upscale_2x(self.sr_model, pil, self.device)
            scale = 2.0
            det_thr = 0.45
        else:
            inp_img = pil
            scale = 1.0
            det_thr = 0.48

        W_inp, H_inp = inp_img.size
        with torch.no_grad():
            out = self.det_model([self.det_tf(inp_img).to(self.device)])[0]

        boxes = out["boxes"].cpu().numpy() / scale
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        masks_np = out["masks"].cpu().numpy()[:, 0] if "masks" in out and len(out["masks"]) else None

        min_bh = 0.24 * H if quality in VIDEO_FILES else max(24.0, 0.06 * H)
        min_bw = 0.05 * W if quality in VIDEO_FILES else max(12.0, 0.015 * W)

        raw_cands: list[tuple[float, float, tuple[float, float, float, float], int]] = []
        for idx, (box, lab, sc) in enumerate(zip(boxes, labels, scores)):
            fsc = float(sc)
            if int(lab) != 1 or fsc < det_thr:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            if bh < min_bh or bw < min_bw:
                continue
            aspect = bh / max(1.0, bw)
            if aspect < 1.15 or aspect > 4.8:
                continue
            if (x1 <= 3.0 or x2 >= W - 3.0) and bw < 0.045 * W:
                continue
            raw_cands.append((fsc * (bw * bh), fsc, (x1, y1, x2, y2), idx))

        # Sort by det_score * area and apply IoU + containment NMS (iou >= 0.45)
        raw_cands.sort(key=lambda x: -x[0])
        kept_cands: list[tuple[float, tuple[float, float, float, float], int]] = []
        for _, fsc, cand_box, raw_idx in raw_cands:
            if any(
                (compute_box_iou(cand_box, kb)[0] >= 0.45 or compute_box_iou(cand_box, kb)[1] >= 0.72)
                for _, kb, _ in kept_cands
            ):
                continue
            kept_cands.append((fsc, cand_box, raw_idx))
            if len(kept_cands) >= 10:
                break

        if not kept_cands:
            return []

        crop_imgs = []
        geoms = []
        det_scores = []
        top_boxes = []
        mask_crops: list[np.ndarray | None] = []
        for fsc, (x1, y1, x2, y2), raw_idx in kept_cands:
            c = pil.crop((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))
            bh = y2 - y1
            crop_imgs.append(sr_upscale_fast(self.sr_model, c, self.device, force_2x=(bh < 64.0)))
            geoms.append([
                (y2 - y1) / max(1.0, x2 - x1),
                (y2 - y1) / float(H),
                (x2 - x1) / float(W),
                0.5 * (y1 + y2) / float(H),
            ])
            det_scores.append(fsc)
            top_boxes.append((x1, y1, x2, y2))
            if masks_np is not None:
                mx1 = max(0, min(W_inp - 1, int(round(x1 * scale))))
                mx2 = max(mx1 + 1, min(W_inp, int(round(x2 * scale))))
                my1 = max(0, min(H_inp - 1, int(round(y1 * scale))))
                my2 = max(my1 + 1, min(H_inp, int(round(y2 * scale))))
                mask_crops.append(masks_np[raw_idx, my1:my2, mx1:mx2].astype(np.float32))
            else:
                mask_crops.append(None)

        with torch.no_grad():
            h_feats = trainer.encode_clip_images(self.clip_model, self.clip_proc, crop_imgs, self.device)
            sim_pos = (h_feats @ self.human_pos_tmat.T).max(dim=-1).values
            sim_neg = (h_feats @ self.human_neg_tmat.T).max(dim=-1).values
            clip_h_probs = (torch.sigmoid(22.0 * (sim_pos - sim_neg)) * 100.0).cpu().numpy()

            if self.trained_model is not None:
                h_geom = torch.tensor(geoms, dtype=torch.float32, device=self.device)
                trained_h_probs = (torch.sigmoid(self.trained_model.forward_human(h_feats, h_geom)) * 100.0).cpu().numpy()
            else:
                trained_h_probs = clip_h_probs

        verified = []
        for box, fsc, clip_hp, train_hp, m_crop in zip(top_boxes, det_scores, clip_h_probs, trained_h_probs, mask_crops):
            human_conf = 0.35 * (fsc * 100.0) + 0.40 * float(clip_hp) + 0.25 * float(train_hp)
            if human_conf >= 64.0 and float(clip_hp) >= 52.0:
                verified.append({"box": box, "human_conf": round(float(human_conf), 1), "mask": m_crop})
        return verified

    def _classify_zone_batch(self, zones_data: list[dict], quality: str) -> list[dict]:
        """Stage 2: Score variable clothing colors & types independently across open vocabulary."""
        if not zones_data:
            return []

        sub_imgs = [zd["sub_sr"] for zd in zones_data]
        ctx_imgs = [zd["ctx_sr"] for zd in zones_data]

        with torch.no_grad():
            sub_512 = trainer.encode_clip_images(self.clip_model, self.clip_proc, sub_imgs, self.device)
            ctx_512 = trainer.encode_clip_images(self.clip_model, self.clip_proc, ctx_imgs, self.device)
            base_512 = torch.nn.functional.normalize(0.35 * sub_512 + 0.65 * ctx_512, dim=-1)

            clip_type_sims = (base_512 @ self.var_type_tmat.T).cpu().numpy()
            clip_color_logits = (sub_512 @ self.var_color_tmat.T) * 20.0
            clip_color_probs = torch.softmax(clip_color_logits, dim=-1).cpu().numpy()

            trained_type_probs = None
            if quality in VIDEO_FILES and self.trained_model is not None:
                struct_vecs = torch.tensor(
                    np.array([trainer.extract_structural_vec(zd["person_sr"], zd["zone"]) for zd in zones_data]),
                    dtype=torch.float32,
                    device=self.device,
                )
                t_logits_tr, _ = self.trained_model.forward_type(sub_512, ctx_512, struct_vecs)
                trained_type_probs = torch.softmax(t_logits_tr, dim=-1).cpu().numpy()

        outputs = []
        for i, zd in enumerate(zones_data):
            pc = classify_variable_color(zd["lab_stats"], zd["zone"], clip_color_probs[i])
            c_idx = int(np.argmax(pc))
            c_label = COLOR_NAMES[c_idx]

            clues = measure_garment_visual_clues(zd["person_sr"], zd["sub_sr"], zd["zone"], zd["lab_stats"])
            pb = zd["pbox_norm"]
            pbox_h_norm = float(pb[3] - pb[1])
            tr_p = trained_type_probs[i] if trained_type_probs is not None else None
            pt = classify_variable_type(
                clip_type_sims[i],
                zd["zone"],
                c_label,
                zd["lab_stats"],
                clues,
                quality,
                pbox_h_norm,
                tr_p,
                person_h_px=float(zd.get("person_h_px", 120.0)),
            )

            t_idx = int(np.argmax(pt))
            t_label = TYPE_NAMES[t_idx]
            t_conf = float(pt[t_idx]) * 100.0
            c_conf = float(pc[c_idx]) * 100.0
            outputs.append({
                "pt": pt,
                "pc": pc,
                "clothing_type": t_label,
                "clothing_color": c_label,
                "type_conf": t_conf,
                "color_conf": c_conf,
                "joint_conf": 0.5 * (t_conf + c_conf),
            })
        return outputs

    @staticmethod
    def _associate_tracks(
        verified_humans: list[dict],
        width: int,
        height: int,
        ts_sec: float,
        fidx: int,
        active_tracks: list[dict],
        next_track_id: int,
    ) -> int:
        """Two-stage ByteTrack-style velocity + IoU association across sampled frames."""
        for h_info in verified_humans:
            bx1, by1, bx2, by2 = h_info["box"]
            h_info["pbox_norm"] = [
                round(max(0.0, bx1 / max(1, width)), 4),
                round(max(0.0, by1 / max(1, height)), 4),
                round(min(1.0, bx2 / max(1, width)), 4),
                round(min(1.0, by2 / max(1, height)), 4),
            ]

        eligible_tracks = [tr for tr in active_tracks if 0.0 <= (ts_sec - tr["last_ts"]) <= 2.4]
        used_tracks: set[int] = set()
        matched_det_to_tr: dict[int, dict] = {}

        def match_stage(det_indices: list[int]):
            cands: list[tuple[float, int, dict]] = []
            for di in det_indices:
                dbox = verified_humans[di]["pbox_norm"]
                dcx, dcy = 0.5 * (dbox[0] + dbox[2]), 0.5 * (dbox[1] + dbox[3])
                h_det = max(1e-4, dbox[3] - dbox[1])
                for tr in eligible_tracks:
                    tid = tr["track_id"]
                    if tid in used_tracks:
                        continue
                    tb = tr["box"]
                    dt = max(0.0, ts_sec - tr["last_ts"])
                    sx = max(-0.08, min(0.08, tr["vx"] * dt))
                    sy = max(-0.08, min(0.08, tr["vy"] * dt))
                    pred_box = (tb[0] + sx, tb[1] + sy, tb[2] + sx, tb[3] + sy)

                    iou_pred, _ = compute_box_iou(tuple(dbox), pred_box)
                    iou_prev, _ = compute_box_iou(tuple(dbox), tuple(tb))
                    best_iou = max(iou_pred, iou_prev)

                    pcx, pcy = 0.5 * (pred_box[0] + pred_box[2]), 0.5 * (pred_box[1] + pred_box[3])
                    cdist = float(np.hypot(dcx - pcx, dcy - pcy))
                    h_tr = max(1e-4, tb[3] - tb[1])
                    ratio = h_det / h_tr
                    if not (0.55 <= ratio <= 1.80):
                        continue
                    max_cdist = min(0.11, 1.15 * max(0.04, 0.5 * (h_det + h_tr)))
                    if best_iou >= 0.18 or cdist <= max_cdist:
                        cands.append((best_iou * 2.0 - cdist, di, tr))

            cands.sort(key=lambda x: -x[0])
            for _, di, tr in cands:
                tid = tr["track_id"]
                if di in matched_det_to_tr or tid in used_tracks:
                    continue
                matched_det_to_tr[di] = tr
                used_tracks.add(tid)

        hi_idx = [i for i, h in enumerate(verified_humans) if h["human_conf"] >= 75.0]
        lo_idx = [i for i, h in enumerate(verified_humans) if h["human_conf"] < 75.0]
        match_stage(hi_idx)
        match_stage(lo_idx)

        for di, h_info in enumerate(verified_humans):
            dbox = h_info["pbox_norm"]
            dcx, dcy = 0.5 * (dbox[0] + dbox[2]), 0.5 * (dbox[1] + dbox[3])
            if di in matched_det_to_tr:
                tr = matched_det_to_tr[di]
                ob = tr["box"]
                ocx, ocy = 0.5 * (ob[0] + ob[2]), 0.5 * (ob[1] + ob[3])
                dt = ts_sec - tr["last_ts"]
                if dt > 0.05:
                    tr["vx"] = 0.6 * ((dcx - ocx) / dt) + 0.4 * tr["vx"]
                    tr["vy"] = 0.6 * ((dcy - ocy) / dt) + 0.4 * tr["vy"]
                tr["box"] = dbox
                tr["last_ts"] = ts_sec
                tr["last_fidx"] = fidx
            else:
                next_track_id += 1
                tr = {
                    "track_id": next_track_id,
                    "box": dbox,
                    "vx": 0.0,
                    "vy": 0.0,
                    "last_ts": ts_sec,
                    "last_fidx": fidx,
                    "zone_ema": {},
                }
                active_tracks.append(tr)
            h_info["track"] = tr

        active_tracks[:] = [tr for tr in active_tracks if (ts_sec - tr["last_ts"]) <= 3.5]
        return next_track_id

    @staticmethod
    def _interpolate_tracklets(raw_rows: list[dict], fps: float) -> list[dict]:
        """Linearly interpolate person & garment boxes across intermediate frames (<= 1.6s gaps)."""
        groups: dict[tuple[int, str], list[dict]] = {}
        for r in raw_rows:
            tid = int(r.get("track_id", 0))
            if tid <= 0:
                continue
            groups.setdefault((tid, r["zone"]), []).append(r)

        interp_step = max(1, int(round(fps * 0.2)))
        interp_rows: list[dict] = []
        for items in groups.values():
            items.sort(key=lambda x: x["frame_idx"])
            for r0, r1 in zip(items, items[1:]):
                dt = float(r1["timestamp_sec"]) - float(r0["timestamp_sec"])
                df = int(r1["frame_idx"]) - int(r0["frame_idx"])
                if not (0.0 < dt <= 1.6 and df > 1):
                    continue
                for fi in range(int(r0["frame_idx"]) + interp_step, int(r1["frame_idx"]), interp_step):
                    alpha = (fi - int(r0["frame_idx"])) / float(df)
                    p_interp = [round((1.0 - alpha) * a + alpha * b, 4) for a, b in zip(r0["person_box_norm"], r1["person_box_norm"])]
                    s_interp = [round((1.0 - alpha) * a + alpha * b, 4) for a, b in zip(r0["sub_box_norm"], r1["sub_box_norm"])]
                    base = r0 if alpha < 0.5 else r1
                    ts_i = fi / max(1e-4, fps)
                    interp_rows.append({
                        **base,
                        "frame_idx": fi,
                        "timestamp_sec": ts_i,
                        "timestamp_str": format_timestamp(ts_i),
                        "person_box_norm": p_interp,
                        "sub_box_norm": s_interp,
                        "is_interpolated": 1,
                    })

        all_rows = raw_rows + interp_rows
        all_rows.sort(key=lambda x: (x["timestamp_sec"], x.get("track_id", 0), x["shopper_num"], x["zone"]))
        return all_rows

    def scan_video(
        self,
        quality: str,
        stride_frames: int = 0,
        db: ClothingSearchDB | None = None,
        video_path: Path | None = None,
        discover: bool = True,
    ) -> dict:
        vpath = video_path or resolve_video_path(quality, db)
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {vpath}")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 13.09
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        eff_stride = stride_frames if stride_frames > 0 else max(3, int(round(fps * 0.4)))
        grid = set(range(0, total_frames, eff_stride))
        if quality in VIDEO_FILES:
            for kf in (20, 170, 384, 864, 1280):
                if kf < total_frames:
                    grid.add(kf)
        frame_indices = sorted(grid)

        t_start = time.perf_counter()
        raw_rows: list[dict] = []
        active_tracks: list[dict] = []
        next_track_id = 0

        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts_sec = fidx / fps
            ts_str = format_timestamp(ts_sec)

            verified_humans = self._detect_persons(rgb, quality)
            next_track_id = self._associate_tracks(
                verified_humans, W, H, ts_sec, fidx, active_tracks, next_track_id
            )
            zones_to_score = []

            for s_idx, h_info in enumerate(verified_humans, start=1):
                pbox = h_info["box"]
                hconf = h_info["human_conf"]
                p_mask = h_info.get("mask")
                pbox_norm = h_info["pbox_norm"]
                bx1, by1, bx2, by2 = pbox
                person_h_px = float(by2 - by1)
                small_ped = person_h_px < 64.0
                p_raw = tuned.crop_frac(rgb, pbox, (0.0, 1.0, 0.0, 1.0))
                p_sr = sr_upscale_fast(self.sr_model, p_raw, self.device, force_2x=small_ped)

                flank_raw = extract_flank_composite(rgb, pbox)
                in_raw = tuned.crop_frac(rgb, pbox, (0.34, 0.66, 0.16, 0.52))
                flank_mask = extract_flank_mask_composite(p_mask) if p_mask is not None else None
                in_mask = crop_mask_frac(p_mask, (0.34, 0.66, 0.16, 0.52)) if p_mask is not None else None
                flank_stats = extract_masked_core_lab(flank_raw, flank_mask, sigma=0.35)
                in_stats = extract_masked_core_lab(in_raw, in_mask, sigma=0.32)
                de_layers = float(np.linalg.norm([
                    flank_stats["L50"] - in_stats["L50"],
                    flank_stats["A"] - in_stats["A"],
                    flank_stats["B"] - in_stats["B"],
                ]))
                upper_zones = ["outer_upper", "inner_upper"] if (de_layers > 22.0 and (bx2 - bx1) > 0.08 * W) else ["upper"]

                candidate_zones = list(upper_zones)
                if by2 < 0.992 * H:
                    candidate_zones.append("lower")

                for zone in candidate_zones:
                    if zone == "outer_upper":
                        sub_raw = flank_raw
                        sub_mask = flank_mask
                        ctx_raw = tuned.crop_frac(rgb, pbox, tuned.ZONE_GEOMETRY["outer_upper"]["ctx"])
                    elif zone == "inner_upper":
                        sub_raw = in_raw
                        sub_mask = in_mask
                        ctx_raw = tuned.crop_frac(rgb, pbox, tuned.ZONE_GEOMETRY["inner_upper"]["ctx"])
                    else:
                        sub_raw = tuned.crop_frac(rgb, pbox, tuned.ZONE_GEOMETRY[zone]["sub"])
                        sub_mask = crop_mask_frac(p_mask, tuned.ZONE_GEOMETRY[zone]["sub"]) if p_mask is not None else None
                        ctx_raw = tuned.crop_frac(rgb, pbox, tuned.ZONE_GEOMETRY[zone]["ctx"])
                    sub_sr = sr_upscale_fast(self.sr_model, sub_raw, self.device, force_2x=small_ped)
                    ctx_sr = sr_upscale_fast(self.sr_model, ctx_raw, self.device, force_2x=small_ped)
                    lab_stats = extract_masked_core_lab(sub_sr, sub_mask, sigma=0.30 if zone == "outer_upper" else 0.28)
                    zones_to_score.append({
                        "shopper_num": s_idx,
                        "track": h_info["track"],
                        "human_conf": hconf,
                        "zone": zone,
                        "pbox": pbox,
                        "pbox_norm": pbox_norm,
                        "person_h_px": person_h_px,
                        "sub_box_norm": extract_sub_box_norm(pbox, zone, W, H),
                        "person_sr": p_sr,
                        "sub_sr": sub_sr,
                        "ctx_sr": ctx_sr,
                        "lab_stats": lab_stats,
                    })

            preds = self._classify_zone_batch(zones_to_score, quality)

            # Track-level temporal EMA smoothing across associated frames
            for zd, pr in zip(zones_to_score, preds):
                tr = zd["track"]
                z_key = zd["zone"]
                prev_z = tr["zone_ema"].get(z_key)
                if prev_z is not None:
                    pt = 0.55 * pr["pt"] + 0.45 * prev_z["pt"]
                    pc = 0.55 * pr["pc"] + 0.45 * prev_z["pc"]
                else:
                    pt, pc = pr["pt"], pr["pc"]
                tr["zone_ema"][z_key] = {"pt": pt, "pc": pc}

                t_idx = int(np.argmax(pt))
                c_idx = int(np.argmax(pc))
                t_label = TYPE_NAMES[t_idx]
                c_label = COLOR_NAMES[c_idx]
                t_conf = float(pt[t_idx]) * 100.0
                c_conf = float(pc[c_idx]) * 100.0
                st = zd["lab_stats"]

                raw_rows.append({
                    "video_quality": quality,
                    "frame_idx": fidx,
                    "timestamp_sec": ts_sec,
                    "timestamp_str": ts_str,
                    "shopper_num": zd["shopper_num"],
                    "track_id": tr["track_id"],
                    "is_interpolated": 0,
                    "human_conf": zd["human_conf"],
                    "is_new_category": 0,
                    "zone": zd["zone"],
                    "person_box_norm": zd["pbox_norm"],
                    "sub_box_norm": zd["sub_box_norm"],
                    "clothing_type": t_label,
                    "clothing_color": c_label,
                    "short_type": SHORT_TYPE_LABELS.get(t_label, t_label),
                    "short_color": SHORT_COLOR_LABELS.get(c_label, c_label),
                    "color_hex": COLOR_HEX_MAP.get(c_label, "#94A3B8"),
                    "type_conf": round(t_conf, 1),
                    "color_conf": round(c_conf, 1),
                    "joint_conf": round(0.5 * (t_conf + c_conf), 1),
                    "lab_l": round(st["L50"], 1),
                    "lab_a": round(st["A"], 1),
                    "lab_b": round(st["B"], 1),
                    "thumb_b64": pil_to_b64_jpeg(zd["sub_sr"]),
                })

        cap.release()
        all_rows = self._interpolate_tracklets(raw_rows, fps)
        scan_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
        meta = {
            "video_quality": quality,
            "video_filename": vpath.name,
            "width": W,
            "height": H,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "indexed_frames": len(frame_indices),
            "detections_count": len(raw_rows),
            "scan_ms": scan_ms,
        }
        if db is not None:
            db.save_quality_results(meta, all_rows)
        return {
            "meta": meta,
            "detections": len(raw_rows),
            "interpolated_detections": len(all_rows) - len(raw_rows),
            "discovered_categories": [],
        }


def run_query_tests(db: ClothingSearchDB, quality: str = "144p") -> dict:
    queries = (
        ["Red shirt", "Orange safety vest", "Yellow safety vest", "Black pants"]
        if quality.startswith("Construction")
        else ["Blue jacket", "White hoodie", "Pink shirt", "Black pants", "Blue jeans", "Beige coat", "Grey sweater"]
    )
    report = {}
    for q in queries:
        res = db.search(query=q, quality=quality)
        report[q] = {
            "match_count": res["match_count"],
            "unique_events": res["unique_timestamp_events"],
            "query_ms": res["query_ms"],
            "sample_timestamps": [m["timestamp_str"] for m in res["matches"][:8]],
        }
    return report


def main():
    parser = argparse.ArgumentParser(description="Index CCTV footage for human-first variable clothing search.")
    parser.add_argument("--qualities", nargs="+", default=["144p", "240p", "360p", "480p", "720p"])
    parser.add_argument("--file", type=str, default="", help="Index a custom local video file")
    parser.add_argument("--label", type=str, default="", help="Custom source label for --file")
    parser.add_argument("--stride", type=int, default=0, help="Frame stride (0 = FPS-adaptive ~0.4s)")
    parser.add_argument("--discover", action="store_true", help="Enable open-vocabulary clothing scoring")
    parser.add_argument("--test-queries", action="store_true")
    args = parser.parse_args()

    db = ClothingSearchDB()
    if args.test_queries:
        print(json.dumps({
            "Construction-Site-CC": run_query_tests(db, "Construction-Site-CC"),
            "144p": run_query_tests(db, "144p"),
            "240p": run_query_tests(db, "240p"),
        }, indent=2))
        return

    indexer = CCTVClothingIndexer()
    if args.file:
        fpath = Path(args.file).resolve()
        if not fpath.exists():
            raise FileNotFoundError(f"Video file not found: {fpath}")
        label = (args.label or fpath.stem)[:20]
        dest = UPLOADS_DIR / fpath.name
        if dest != fpath and not dest.exists():
            dest.write_bytes(fpath.read_bytes())
        out = indexer.scan_video(label, stride_frames=args.stride, db=db, video_path=dest)
        db.export_json_snapshot()
        print(f"Indexed uploaded '{label}': {out['meta']['detections_count']} garment sightings across {out['meta']['indexed_frames']} frames in {out['meta']['scan_ms']/1000.0:.1f}s")
        print(json.dumps(run_query_tests(db, label), indent=2))
        return

    for q in args.qualities:
        out = indexer.scan_video(q, stride_frames=args.stride, db=db)
        print(f"Indexed {q:4s}: {out['meta']['detections_count']:4d} garment sightings across {out['meta']['indexed_frames']} frames in {out['meta']['scan_ms']/1000.0:.1f}s")

    db.export_json_snapshot()
    print("Query verification at 144p:")
    print(json.dumps(run_query_tests(db, "144p"), indent=2))


if __name__ == "__main__":
    main()
