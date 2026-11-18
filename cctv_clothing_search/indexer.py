"""Core CCTV clothing indexer, SQLite timestamp store, and natural-language search engine."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
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

ROOT_DIR = Path(r"D:\projects\affine-ai-upscaling")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_cctv_clothing_tuning_report as tuned  # noqa: E402
from . import trainer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "clothing_index.db"
JSON_PATH = DATA_DIR / "clothing_index.json"

VIDEO_FILES = tuned.RES_FILES
QUALITIES = tuned.RES_ORDER  # ['720p', '480p', '360p', '240p', '144p']


def resolve_video_path(quality: str, db: ClothingSearchDB | None = None) -> Path:
    """Resolve either a built-in resolution preset or an uploaded CCTV file."""
    if quality in VIDEO_FILES:
        return VIDEO_FILES[quality]
    if db is not None:
        meta = db.get_stats(quality).get("video")
        if meta and meta.get("video_filename"):
            cand = UPLOADS_DIR / meta["video_filename"]
            if cand.exists():
                return cand
    for ext in ("", ".mp4", ".mov", ".avi", ".mkv", ".webm"):
        cand = UPLOADS_DIR / f"{quality}{ext}"
        if cand.exists() and cand.is_file():
            return cand
    raise FileNotFoundError(f"Video source not found for quality key: {quality}")

SHORT_TYPE_LABELS = {
    "puffer jacket / quilted coat": "Puffer Jacket",
    "hoodie / hooded sweatshirt": "Hoodie",
    "sweater / sweatshirt": "Sweater",
    "jacket / sweater": "Jacket",
    "open jacket / coat": "Open Jacket",
    "shirt / blouse": "Shirt / Blouse",
    "long coat / overcoat": "Long Overcoat",
    "buttoned coat / jacket": "Buttoned Coat",
    "jeans / denim pants": "Denim Jeans",
    "pants / trousers": "Trousers / Pants",
}

SHORT_COLOR_LABELS = {
    "navy blue": "Navy Blue",
    "royal blue": "Royal Blue",
    "blue": "Blue",
    "white": "White",
    "light grey": "Light Grey",
    "dark grey / charcoal": "Dark Grey",
    "black": "Black",
    "pink / magenta": "Pink",
    "beige / khaki": "Beige / Khaki",
    "red": "Red",
    "green": "Green",
    "yellow / orange": "Yellow / Orange",
    "brown": "Brown",
    "purple": "Purple",
}

COLOR_HEX_MAP = {
    "navy blue": "#1E3A8A",
    "royal blue": "#2563EB",
    "blue": "#3B82F6",
    "white": "#F1F5F9",
    "light grey": "#94A3B8",
    "dark grey / charcoal": "#475569",
    "black": "#1E293B",
    "pink / magenta": "#EC4899",
    "beige / khaki": "#D6B88D",
    "red": "#DC2626",
    "green": "#16A34A",
    "yellow / orange": "#F59E0B",
    "brown": "#92400E",
    "purple": "#9333EA",
}

QUERY_COLOR_SYNONYMS = {
    "navy blue": {"navy blue"},
    "royal blue": {"royal blue"},
    "light grey": {"light grey"},
    "dark grey": {"dark grey / charcoal"},
    "dark gray": {"dark grey / charcoal"},
    "light gray": {"light grey"},
    "blue": {"blue", "royal blue", "navy blue"},
    "royal": {"royal blue"},
    "navy": {"navy blue"},
    "indigo": {"navy blue", "blue"},
    "white": {"white"},
    "cream": {"white", "beige / khaki"},
    "grey": {"light grey", "dark grey / charcoal"},
    "gray": {"light grey", "dark grey / charcoal"},
    "charcoal": {"dark grey / charcoal"},
    "black": {"black"},
    "dark": {"black", "dark grey / charcoal", "navy blue"},
    "pink": {"pink / magenta"},
    "magenta": {"pink / magenta"},
    "red": {"red", "pink / magenta"},
    "green": {"green"},
    "olive": {"green"},
    "yellow": {"yellow / orange"},
    "orange": {"yellow / orange"},
    "beige": {"beige / khaki"},
    "khaki": {"beige / khaki"},
    "tan": {"beige / khaki"},
    "brown": {"brown", "beige / khaki"},
    "purple": {"purple"},
}

QUERY_TYPE_SYNONYMS = {
    "puffer jacket": {"puffer jacket / quilted coat"},
    "long coat": {"long coat / overcoat"},
    "buttoned coat": {"buttoned coat / jacket"},
    "open jacket": {"open jacket / coat"},
    "denim jeans": {"jeans / denim pants"},
    "puffer": {"puffer jacket / quilted coat"},
    "quilted": {"puffer jacket / quilted coat"},
    "hoodie": {"hoodie / hooded sweatshirt"},
    "hooded": {"hoodie / hooded sweatshirt"},
    "sweater": {"sweater / sweatshirt", "jacket / sweater"},
    "sweatshirt": {"sweater / sweatshirt", "hoodie / hooded sweatshirt"},
    "jumper": {"sweater / sweatshirt"},
    "jacket": {
        "puffer jacket / quilted coat",
        "jacket / sweater",
        "open jacket / coat",
        "buttoned coat / jacket",
    },
    "coat": {
        "puffer jacket / quilted coat",
        "open jacket / coat",
        "long coat / overcoat",
        "buttoned coat / jacket",
    },
    "overcoat": {"long coat / overcoat"},
    "shirt": {"shirt / blouse"},
    "blouse": {"shirt / blouse"},
    "top": {"shirt / blouse", "sweater / sweatshirt", "hoodie / hooded sweatshirt"},
    "jeans": {"jeans / denim pants"},
    "denim": {"jeans / denim pants"},
    "pants": {"pants / trousers", "jeans / denim pants"},
    "trousers": {"pants / trousers"},
    "slacks": {"pants / trousers"},
}


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


def sr_upscale_fast(sr_model: Any, pil_img: Image.Image, device: torch.device) -> Image.Image:
    """Apply RealESRGAN_x2plus only while ROI min-side < 96px to keep app fast and light."""
    w, h = pil_img.size
    m = min(w, h)
    if m < 30:
        out = tuned.sr_upscale_2x(sr_model, tuned.sr_upscale_2x(sr_model, pil_img, device), device)
        return out.filter(ImageFilter.UnsharpMask(radius=1.0, percent=35, threshold=2))
    if m < 96:
        out = tuned.sr_upscale_2x(sr_model, pil_img, device)
        return out.filter(ImageFilter.UnsharpMask(radius=0.8, percent=25, threshold=2))
    return pil_img


def extract_sub_box_norm(pbox: tuple[float, float, float, float], zone: str, w: int, h: int) -> list[float]:
    bx1, by1, bx2, by2 = pbox
    x0, x1, y0, y1 = tuned.ZONE_GEOMETRY[zone]["sub"]
    sx1 = max(0.0, (bx1 + (bx2 - bx1) * x0) / max(1.0, float(w)))
    sy1 = max(0.0, (by1 + (by2 - by1) * y0) / max(1.0, float(h)))
    sx2 = min(1.0, (bx1 + (bx2 - bx1) * x1) / max(1.0, float(w)))
    sy2 = min(1.0, (by1 + (by2 - by1) * y1) / max(1.0, float(h)))
    return [round(sx1, 4), round(sy1, 4), round(sx2, 4), round(sy2, 4)]


class ClothingSearchDB:
    """Lightweight SQLite store + query engine for timestamped garment detections."""

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
                    thumb_b64 TEXT NOT NULL
                )
                """
            )
            cols = {r[1] for r in conn.execute("PRAGMA table_info(detections)").fetchall()}
            if "human_conf" not in cols:
                conn.execute("ALTER TABLE detections ADD COLUMN human_conf REAL NOT NULL DEFAULT 95.0")
            if "is_new_category" not in cols:
                conn.execute("ALTER TABLE detections ADD COLUMN is_new_category INTEGER NOT NULL DEFAULT 0")
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
            for idx, cname in enumerate(tuned.CLOTHING_TYPES):
                zrole = "lower" if idx in tuned.LOWER_TYPE_IDXS else "upper"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO clothing_categories (name, short_label, zone_role, is_discovered, synonyms_json, seen_count, discovered_at)
                    VALUES (?, ?, ?, 0, '[]', 0, ?)
                    """,
                    (cname, SHORT_TYPE_LABELS.get(cname, cname), zrole, now),
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_det_q_color_type ON detections (video_quality, clothing_color, clothing_type, timestamp_sec)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_det_q_ts ON detections (video_quality, timestamp_sec)"
            )
        self._sync_category_synonyms()

    def _sync_category_synonyms(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT name, short_label, synonyms_json FROM clothing_categories").fetchall()
        for r in rows:
            name = r["name"]
            SHORT_TYPE_LABELS[name] = r["short_label"]
            for syn in json.loads(r["synonyms_json"] or "[]"):
                QUERY_TYPE_SYNONYMS.setdefault(syn.lower(), set()).add(name)
            QUERY_TYPE_SYNONYMS.setdefault(r["short_label"].lower(), set()).add(name)

    def register_discovered_category(self, name: str, short_label: str, zone_role: str, synonyms: list[str], count_inc: int = 1):
        SHORT_TYPE_LABELS[name] = short_label
        for syn in synonyms:
            QUERY_TYPE_SYNONYMS.setdefault(syn.lower(), set()).add(name)
        QUERY_TYPE_SYNONYMS.setdefault(short_label.lower(), set()).add(name)
        with self._connect() as conn:
            row = conn.execute("SELECT seen_count FROM clothing_categories WHERE name = ?", (name,)).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO clothing_categories (name, short_label, zone_role, is_discovered, synonyms_json, seen_count, discovered_at)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                    """,
                    (name, short_label, zone_role, json.dumps(synonyms), count_inc, time.time()),
                )
            else:
                conn.execute(
                    "UPDATE clothing_categories SET is_discovered = 1, seen_count = seen_count + ? WHERE name = ?",
                    (count_inc, name),
                )

    def get_categories(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM clothing_categories ORDER BY is_discovered ASC, seen_count DESC, name ASC").fetchall()
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
                    lab_l, lab_a, lab_b, human_conf, is_new_category, thumb_b64
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        int(r.get("is_new_category", 0)),
                        r["thumb_b64"],
                    )
                    for r in rows
                ],
            )

    def export_json_snapshot(self, out_path: Path = JSON_PATH):
        with self._connect() as conn:
            metas = [dict(r) for r in conn.execute("SELECT * FROM video_meta").fetchall()]
            dets = [dict(r) for r in conn.execute("SELECT * FROM detections ORDER BY video_quality, timestamp_sec").fetchall()]
        for d in dets:
            d["person_box_norm"] = json.loads(d["person_box_norm"])
            d["sub_box_norm"] = json.loads(d["sub_box_norm"])
        out_path.write_text(json.dumps({"videos": metas, "detections": dets}), encoding="utf-8")

    def get_stats(self, quality: str = "144p") -> dict:
        with self._connect() as conn:
            metas = [dict(r) for r in conn.execute("SELECT * FROM video_meta ORDER BY width ASC").fetchall()]
            meta_q = next((m for m in metas if m["video_quality"] == quality), None)
            total_dets = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE video_quality = ?", (quality,)
            ).fetchone()[0]
            colors = [
                {"color": r[0], "short_color": SHORT_COLOR_LABELS.get(r[0], r[0]), "hex": COLOR_HEX_MAP.get(r[0], "#94A3B8"), "count": r[1]}
                for r in conn.execute(
                    "SELECT clothing_color, COUNT(*) as c FROM detections WHERE video_quality = ? GROUP BY clothing_color ORDER BY c DESC",
                    (quality,),
                ).fetchall()
            ]
            types = [
                {
                    "type": r[0],
                    "short_type": SHORT_TYPE_LABELS.get(r[0], r[0]),
                    "count": r[1],
                    "is_discovered": r[0] not in tuned.CLOTHING_TYPES,
                }
                for r in conn.execute(
                    "SELECT clothing_type, COUNT(*) as c FROM detections WHERE video_quality = ? GROUP BY clothing_type ORDER BY c DESC",
                    (quality,),
                ).fetchall()
            ]
        all_cats = self.get_categories()
        discovered_cats = [c for c in all_cats if c["is_discovered"]]
        tmetrics = json.loads(trainer.METRICS_PATH.read_text(encoding="utf-8")) if trainer.METRICS_PATH.exists() else None
        return {
            "quality": quality,
            "video": meta_q,
            "available_qualities": [m["video_quality"] for m in metas],
            "all_videos": metas,
            "total_detections": total_dets,
            "colors": colors,
            "types": types,
            "discovered_categories": discovered_cats,
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

        clauses = ["video_quality = ?"]
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

        with self._connect() as conn:
            meta_row = conn.execute("SELECT duration_sec FROM video_meta WHERE video_quality = ?", (quality,)).fetchone()
            duration_sec = float(meta_row["duration_sec"]) if meta_row else 110.9
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

        matches = []
        for r in rows:
            matches.append({
                "id": r["id"],
                "video_quality": r["video_quality"],
                "frame_idx": r["frame_idx"],
                "timestamp_sec": round(float(r["timestamp_sec"]), 2),
                "timestamp_str": r["timestamp_str"],
                "timeline_pct": round(100.0 * float(r["timestamp_sec"]) / max(1e-4, duration_sec), 2),
                "shopper_num": r["shopper_num"],
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
                "is_new_category": bool(r["is_new_category"]) if "is_new_category" in r.keys() else False,
                "thumb_b64": r["thumb_b64"],
            })

        # Compute distinct timestamp windows (>= 2.0s apart) for concise summary
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
        }


class CCTVClothingIndexer:
    """Automatic GPU video scanner using trained human verification + clothing & discovery heads."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sr_model = tuned.load_realesrgan_x2plus(self.device)
        self.det_model = tv_det.fasterrcnn_resnet50_fpn_v2(
            weights=tv_det.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        ).eval().to(self.device)
        self.det_tf = torchvision.transforms.ToTensor()

        clip_id = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
        self.clip_proc = CLIPProcessor.from_pretrained(clip_id)
        self.clip_model = CLIPModel.from_pretrained(clip_id, use_safetensors=True).eval().to(self.device)
        self.base_type_tmat = tuned.build_text_matrix(
            self.clip_model, self.clip_proc, tuned.TUNED_TYPE_PROMPTS, tuned.CLOTHING_TYPES, self.device
        )
        self.cmat = tuned.build_text_matrix(
            self.clip_model, self.clip_proc, trainer.EXPANDED_COLOR_PROMPTS, trainer.EXPANDED_COLORS, self.device
        )

        # Load trained Sightline multi-task PyTorch checkpoint
        if not trainer.MODEL_PATH.exists():
            trainer.train_model(epochs=65)
        ckpt = torch.load(str(trainer.MODEL_PATH), map_location=self.device, weights_only=False)
        self.base_types: list[str] = list(ckpt["base_types"])
        self.colors_list: list[str] = list(ckpt["expanded_colors"])
        self.open_world_candidates: list[dict] = list(ckpt["open_world_candidates"])
        self.open_world_embs: torch.Tensor = torch.nn.functional.normalize(
            ckpt["open_world_embs"].to(self.device), dim=-1
        )
        self.trained_model = trainer.SightlineTrainedModel(
            self.base_type_tmat, self.cmat, logit_scale=22.0
        ).to(self.device)
        self.trained_model.load_state_dict(ckpt["model_state"])
        self.trained_model.eval()

        # Runtime registry tracking active classes (10 base + dynamically discovered classes)
        self.active_types: list[str] = list(self.base_types)
        self.active_zone_roles: list[str] = [
            "lower" if idx in tuned.LOWER_TYPE_IDXS else "upper"
            for idx in range(len(self.base_types))
        ]
        self.extra_protos: list[torch.Tensor] = []
        self.extra_text_embs: list[torch.Tensor] = []
        self._load_existing_discovered_from_db()

    def _load_existing_discovered_from_db(self):
        db = ClothingSearchDB()
        for cat in db.get_categories():
            if cat["is_discovered"] and cat["name"] not in self.active_types:
                for cand_idx, cand in enumerate(self.open_world_candidates):
                    if cand["name"] == cat["name"]:
                        self._append_candidate(cand_idx, cand)
                        break

    def _append_candidate(self, cand_idx: int, cand: dict):
        if cand["name"] in self.active_types:
            return
        emb = self.open_world_embs[cand_idx].detach()
        self.active_types.append(cand["name"])
        self.active_zone_roles.append(cand["zone_role"])
        self.extra_protos.append(emb)
        self.extra_text_embs.append(emb)

    def _detect_persons(self, rgb: np.ndarray, quality: str) -> list[dict]:
        H, W = rgb.shape[:2]
        use_sr = (H <= 280) or (quality in ("144p", "240p"))
        pil = Image.fromarray(rgb)
        if use_sr:
            inp_img = tuned.sr_upscale_2x(self.sr_model, pil, self.device)
            scale = 2.0
            conf_thr = 0.42
        else:
            inp_img = pil
            scale = 1.0
            conf_thr = 0.50

        with torch.no_grad():
            out = self.det_model([self.det_tf(inp_img).to(self.device)])[0]

        boxes = out["boxes"].cpu().numpy() / scale
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()

        raw_cands = []
        min_bh = 0.24 * H if quality in VIDEO_FILES else max(24.0, 0.06 * H)
        min_bw = 0.05 * W if quality in VIDEO_FILES else max(10.0, 0.015 * W)
        for box, lab, sc in zip(boxes, labels, scores):
            if int(lab) != 1 or float(sc) < conf_thr:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            bw, bh = x2 - x1, y2 - y1
            if bh < min_bh or bw < min_bw:
                continue
            raw_cands.append((float(sc) * (bw * bh), (x1, y1, x2, y2)))

        raw_cands.sort(key=lambda x: -x[0])
        top_boxes = [b for _, b in raw_cands[:8]]
        if not top_boxes:
            return []

        # Verify each candidate bounding box with the trained Sightline human_head
        crop_imgs = []
        geoms = []
        for x1, y1, x2, y2 in top_boxes:
            c = pil.crop((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))
            crop_imgs.append(sr_upscale_fast(self.sr_model, c, self.device))
            geoms.append([
                (y2 - y1) / max(1.0, x2 - x1),
                (y2 - y1) / float(H),
                (x2 - x1) / float(W),
                0.5 * (y1 + y2) / float(H),
            ])

        with torch.no_grad():
            h_feats = trainer.encode_clip_images(self.clip_model, self.clip_proc, crop_imgs, self.device)
            h_geom = torch.tensor(geoms, dtype=torch.float32, device=self.device)
            h_probs = torch.sigmoid(self.trained_model.forward_human(h_feats, h_geom)).cpu().numpy() * 100.0

        verified = []
        for box, hprob in zip(top_boxes, h_probs):
            if float(hprob) >= 55.0:
                verified.append({"box": box, "human_conf": round(float(hprob), 1)})
        return verified

    def _classify_zone_batch(
        self,
        zones_data: list[dict],
        quality: str,
        discover: bool = True,
        db: ClothingSearchDB | None = None,
    ) -> list[dict]:
        """Score verified human clothing zones using trained adapters + open-world category discovery."""
        if not zones_data:
            return []

        sub_imgs = [zd["sub_sr"] for zd in zones_data]
        ctx_imgs = [zd["ctx_sr"] for zd in zones_data]
        struct_vecs = torch.tensor(
            np.array([trainer.extract_structural_vec(zd["person_sr"], zd["zone"]) for zd in zones_data]),
            dtype=torch.float32,
            device=self.device,
        )
        col_vecs = torch.tensor(
            np.array([trainer.extract_color_feature_vec(zd["lab_stats"], zd["zone"]) for zd in zones_data]),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            sub_512 = trainer.encode_clip_images(self.clip_model, self.clip_proc, sub_imgs, self.device)
            ctx_512 = trainer.encode_clip_images(self.clip_model, self.clip_proc, ctx_imgs, self.device)
            base_512 = torch.nn.functional.normalize(0.35 * sub_512 + 0.65 * ctx_512, dim=-1)

            extra_t = torch.stack(self.extra_protos, dim=0) if self.extra_protos else None
            t_logits, _ = self.trained_model.forward_type(sub_512, ctx_512, struct_vecs, extra_protos=extra_t)
            c_logits, _ = self.trained_model.forward_color(sub_512, col_vecs)

            # Open-world zero-shot similarity in shared 512-d CLIP space across base + discovery bank
            known_clip_tmat = self.base_type_tmat
            if self.extra_text_embs:
                known_clip_tmat = torch.cat([known_clip_tmat, torch.stack(self.extra_text_embs, dim=0)], dim=0)
            known_clip_sims = base_512 @ known_clip_tmat.T  # [B, N_active]
            novel_clip_sims = base_512 @ self.open_world_embs.T  # [B, N_cand]

        t_logits_np = t_logits.cpu().numpy()
        c_logits_np = c_logits.cpu().numpy()
        known_sims_np = known_clip_sims.cpu().numpy()
        novel_sims_np = novel_clip_sims.cpu().numpy()

        outputs = []
        for i, zd in enumerate(zones_data):
            zone = zd["zone"]
            z_role = "lower" if zone == "lower" else "upper"
            is_new_cat = False

            valid_known_idxs = [k for k, zr in enumerate(self.active_zone_roles) if zr == z_role]
            # Check Open-World Discovery Bank for unseen clothing types in this zone
            if discover:
                max_known_sim = float(np.max(known_sims_np[i, valid_known_idxs]))

                cand_scores = [
                    (c_i, cand, float(novel_sims_np[i, c_i]))
                    for c_i, cand in enumerate(self.open_world_candidates)
                    if cand["zone_role"] == z_role and cand["name"] not in self.active_types
                ]
                if cand_scores:
                    best_ci, best_cand, best_nsim = max(cand_scores, key=lambda x: x[2])
                    margin = best_nsim - max_known_sim
                    # Compute calibrated discovery confidence over [valid_known, best_novel]
                    comp_sims = np.concatenate([known_sims_np[i, valid_known_idxs], [best_nsim]]) * 22.0
                    comp_probs = np.exp(comp_sims - np.max(comp_sims))
                    comp_probs /= comp_probs.sum()
                    novel_prob = float(comp_probs[-1])
                    if margin >= 0.025 and novel_prob >= 0.52:
                        self._append_candidate(best_ci, best_cand)
                        known_sims_np = np.concatenate([known_sims_np, novel_sims_np[:, best_ci : best_ci + 1]], axis=1)
                        if db is not None:
                            db.register_discovered_category(
                                best_cand["name"],
                                best_cand["short"],
                                best_cand["zone_role"],
                                best_cand["synonyms"],
                                count_inc=0,
                            )
                        print(f"[DISCOVERY] Added new clothing category: '{best_cand['name']}' (margin={margin:+.3f}, conf={novel_prob*100:.1f}%)")
                        # Recompute trained logits including newly appended prototype
                        with torch.no_grad():
                            extra_t = torch.stack(self.extra_protos, dim=0)
                            tl_row, _ = self.trained_model.forward_type(
                                sub_512[i : i + 1],
                                ctx_512[i : i + 1],
                                struct_vecs[i : i + 1],
                                extra_protos=extra_t,
                            )
                        cur_logits = tl_row.squeeze(0).cpu().numpy()
                        is_new_cat = True
                    else:
                        cur_logits = t_logits_np[i]
                else:
                    cur_logits = t_logits_np[i]
            else:
                cur_logits = t_logits_np[i]

            # Ensure logit vector matches current active_types length (if earlier row in same frame expanded list)
            if len(cur_logits) < len(self.active_types):
                with torch.no_grad():
                    extra_t = torch.stack(self.extra_protos, dim=0)
                    tl_row, _ = self.trained_model.forward_type(
                        sub_512[i : i + 1], ctx_512[i : i + 1], struct_vecs[i : i + 1], extra_protos=extra_t
                    )
                cur_logits = tl_row.squeeze(0).cpu().numpy()

            # Mask logits by anatomical zone role
            mask = np.full(len(self.active_types), -1e4, dtype=np.float32)
            for k, zr in enumerate(self.active_zone_roles):
                if zr == z_role:
                    mask[k] = 0.0
            masked_t = cur_logits + mask
            n_base = len(self.base_types)
            valid_base_idxs = [k for k in valid_known_idxs if k < n_base]
            valid_disc_idxs = [k for k, zr in enumerate(self.active_zone_roles) if zr == z_role and k >= n_base]
            if valid_disc_idxs and valid_base_idxs:
                best_disc_k = max(valid_disc_idxs, key=lambda k: float(known_sims_np[i, k]))
                max_base_sim = float(np.max(known_sims_np[i, valid_base_idxs]))
                if is_new_cat or (float(known_sims_np[i, best_disc_k]) - max_base_sim >= 0.022):
                    target_k = (len(self.active_types) - 1) if is_new_cat else best_disc_k
                    masked_t[target_k] = float(np.max(masked_t[valid_base_idxs])) + 0.85
                    is_new_cat = True
            pt = tuned.softmax_temp(masked_t, 1.0)

            # Clothing color fusion: trained color_adapter probabilities + calibrated CIELAB Gaussian likelihood
            pc_neural = tuned.softmax_temp(c_logits_np[i], 1.0)
            pc_lab9 = tuned.lab_color_probs(zd["lab_stats"], zone)
            pc_lab = np.zeros(len(self.colors_list), dtype=np.float32)
            pc_lab[: len(pc_lab9)] = pc_lab9
            # Keep expanded colors available only when neural color score clearly peaks outside the 9 base colors
            if np.argmax(pc_neural) >= len(pc_lab9) and float(np.max(pc_neural)) >= 0.55:
                pc = 0.75 * pc_neural + 0.25 * pc_lab
            else:
                pc = 0.45 * pc_neural + 0.55 * pc_lab
            pc /= (pc.sum() + 1e-8)

            t_idx = int(np.argmax(pt))
            c_idx = int(np.argmax(pc))
            t_label = self.active_types[t_idx]
            c_label = self.colors_list[c_idx]
            t_conf = float(pt[t_idx]) * 100.0
            c_conf = float(pc[c_idx]) * 100.0
            if t_label not in self.base_types:
                is_new_cat = True

            outputs.append({
                "pt": pt,
                "pc": pc,
                "clothing_type": t_label,
                "clothing_color": c_label,
                "type_conf": t_conf,
                "color_conf": c_conf,
                "joint_conf": 0.5 * (t_conf + c_conf),
                "is_new_category": is_new_cat,
            })
        return outputs

    def scan_video(
        self,
        quality: str,
        stride_frames: int = 24,
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

        grid = set(range(0, total_frames, stride_frames))
        if quality in VIDEO_FILES:
            for kf in (20, 170, 384, 864, 1280):
                if kf < total_frames:
                    grid.add(kf)
        frame_indices = sorted(grid)

        t_start = time.perf_counter()
        raw_rows: list[dict] = []
        track_ema: list[dict] = []

        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts_sec = fidx / fps
            ts_str = format_timestamp(ts_sec)

            verified_humans = self._detect_persons(rgb, quality)
            zones_to_score = []

            for s_idx, h_info in enumerate(verified_humans, start=1):
                pbox = h_info["box"]
                hconf = h_info["human_conf"]
                bx1, by1, bx2, by2 = pbox
                p_raw = tuned.crop_frac(rgb, pbox, (0.0, 1.0, 0.0, 1.0))
                p_sr = sr_upscale_fast(self.sr_model, p_raw, self.device)

                out_raw = tuned.crop_frac(rgb, pbox, tuned.ZONE_GEOMETRY["outer_upper"]["sub"])
                in_raw = tuned.crop_frac(rgb, pbox, tuned.ZONE_GEOMETRY["inner_upper"]["sub"])
                out_stats = tuned.extract_gaussian_core_lab(out_raw, sigma=0.28)
                in_stats = tuned.extract_gaussian_core_lab(in_raw, sigma=0.28)
                de_layers = float(np.linalg.norm([
                    out_stats["L50"] - in_stats["L50"],
                    out_stats["A"] - in_stats["A"],
                    out_stats["B"] - in_stats["B"],
                ]))
                upper_zones = ["outer_upper", "inner_upper"] if (de_layers > 28.0 and (bx2 - bx1) > 0.11 * W) else ["upper"]

                candidate_zones = list(upper_zones)
                if by2 < 0.97 * H:
                    candidate_zones.append("lower")

                pbox_norm = [
                    round(max(0.0, bx1 / W), 4),
                    round(max(0.0, by1 / H), 4),
                    round(min(1.0, bx2 / W), 4),
                    round(min(1.0, by2 / H), 4),
                ]

                for zone in candidate_zones:
                    sub_raw = tuned.crop_frac(rgb, pbox, tuned.ZONE_GEOMETRY[zone]["sub"])
                    ctx_raw = tuned.crop_frac(rgb, pbox, tuned.ZONE_GEOMETRY[zone]["ctx"])
                    sub_sr = sr_upscale_fast(self.sr_model, sub_raw, self.device)
                    ctx_sr = sr_upscale_fast(self.sr_model, ctx_raw, self.device)
                    lab_stats = tuned.extract_gaussian_core_lab(sub_sr, sigma=0.28)
                    zones_to_score.append({
                        "shopper_num": s_idx,
                        "human_conf": hconf,
                        "zone": zone,
                        "pbox": pbox,
                        "pbox_norm": pbox_norm,
                        "sub_box_norm": extract_sub_box_norm(pbox, zone, W, H),
                        "person_sr": p_sr,
                        "sub_sr": sub_sr,
                        "ctx_sr": ctx_sr,
                        "lab_stats": lab_stats,
                    })

            preds = self._classify_zone_batch(zones_to_score, quality, discover=discover, db=db)

            # Temporal EMA smoothing with dynamic zero-padding if active_types grew mid-video
            for zd, pr in zip(zones_to_score, preds):
                pb = zd["pbox_norm"]
                cx, cy = 0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3])
                cur_pt = pr["pt"]
                if len(cur_pt) < len(self.active_types):
                    cur_pt = np.pad(cur_pt, (0, len(self.active_types) - len(cur_pt)))

                best_tr = None
                best_dist = 0.08
                for tr in track_ema:
                    if tr["zone"] == zd["zone"] and (ts_sec - tr["last_ts"]) <= 4.5:
                        dist = float(np.hypot(cx - tr["cx"], cy - tr["cy"]))
                        if dist < best_dist:
                            best_dist = dist
                            best_tr = tr
                if best_tr is not None:
                    prev_pt = best_tr["pt"]
                    if len(prev_pt) < len(cur_pt):
                        prev_pt = np.pad(prev_pt, (0, len(cur_pt) - len(prev_pt)))
                    pt = 0.55 * cur_pt + 0.45 * prev_pt
                    pc = 0.55 * pr["pc"] + 0.45 * best_tr["pc"]
                    best_tr["pt"] = pt
                    best_tr["pc"] = pc
                    best_tr["cx"] = cx
                    best_tr["cy"] = cy
                    best_tr["last_ts"] = ts_sec
                else:
                    pt, pc = cur_pt, pr["pc"]
                    track_ema.append({"zone": zd["zone"], "cx": cx, "cy": cy, "last_ts": ts_sec, "pt": pt, "pc": pc})

                t_idx = int(np.argmax(pt))
                c_idx = int(np.argmax(pc))
                t_label = self.active_types[t_idx]
                c_label = self.colors_list[c_idx]
                t_conf = float(pt[t_idx]) * 100.0
                c_conf = float(pc[c_idx]) * 100.0
                st = zd["lab_stats"]
                is_new = int(t_label not in self.base_types)
                if is_new and db is not None:
                    cand = next((c for c in self.open_world_candidates if c["name"] == t_label), None)
                    if cand:
                        db.register_discovered_category(t_label, cand["short"], cand["zone_role"], cand["synonyms"], count_inc=1)

                raw_rows.append({
                    "video_quality": quality,
                    "frame_idx": fidx,
                    "timestamp_sec": ts_sec,
                    "timestamp_str": ts_str,
                    "shopper_num": zd["shopper_num"],
                    "human_conf": zd["human_conf"],
                    "is_new_category": is_new,
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
            db.save_quality_results(meta, raw_rows)
        return {
            "meta": meta,
            "detections": len(raw_rows),
            "discovered_categories": [c["name"] for c in db.get_categories() if c["is_discovered"]] if db else [],
        }


def run_query_tests(db: ClothingSearchDB, quality: str = "144p") -> dict:
    queries = ["Blue jacket", "White hoodie", "Pink shirt", "Black pants", "Blue jeans", "Beige coat", "Grey sweater"]
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
    parser = argparse.ArgumentParser(description="Index retail CCTV footage for fast clothing search.")
    parser.add_argument("--qualities", nargs="+", default=["144p", "240p", "360p", "480p", "720p"])
    parser.add_argument("--file", type=str, default="", help="Index a custom local video file")
    parser.add_argument("--label", type=str, default="", help="Custom source label for --file")
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--discover", action="store_true", help="Enable open-world clothing category discovery")
    parser.add_argument("--test-queries", action="store_true")
    args = parser.parse_args()

    db = ClothingSearchDB()
    if args.test_queries:
        print(json.dumps({"144p": run_query_tests(db, "144p"), "240p": run_query_tests(db, "240p")}, indent=2))
        return

    indexer = CCTVClothingIndexer()
    use_discover = True
    if args.file:
        fpath = Path(args.file).resolve()
        if not fpath.exists():
            raise FileNotFoundError(f"Video file not found: {fpath}")
        label = (args.label or fpath.stem)[:20]
        dest = UPLOADS_DIR / fpath.name
        if dest != fpath and not dest.exists():
            dest.write_bytes(fpath.read_bytes())
        out = indexer.scan_video(label, stride_frames=args.stride, db=db, video_path=dest, discover=use_discover)
        db.export_json_snapshot()
        print(f"Indexed uploaded '{label}': {out['meta']['detections_count']} garment sightings across {out['meta']['indexed_frames']} frames in {out['meta']['scan_ms']/1000.0:.1f}s")
        print("Discovered categories:", out["discovered_categories"])
        return

    summary = []
    for q in args.qualities:
        out = indexer.scan_video(q, stride_frames=args.stride, db=db, discover=use_discover)
        summary.append(out["meta"])
        print(f"Indexed {q:4s}: {out['meta']['detections_count']:4d} garment sightings across {out['meta']['indexed_frames']} frames in {out['meta']['scan_ms']/1000.0:.1f}s")

    db.export_json_snapshot()
    print("Discovered categories:", [c["name"] for c in db.get_categories() if c["is_discovered"]])
    print("Query verification at 144p:")
    print(json.dumps(run_query_tests(db, "144p"), indent=2))


if __name__ == "__main__":
    main()
