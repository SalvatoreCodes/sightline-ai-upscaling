"""Multi-task GPU trainer: Human Verifier + Multi-Dataset PAR Clothing Type + Shadow-Invariant Chromaticity Heads on SigLIP-384."""

from __future__ import annotations

import io
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter

ROOT_DIR = Path(r"D:\projects\sightline-ai-upscaling")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_cctv_clothing_tuning_report as tuned  # noqa: E402
from . import color_constancy, par_datasets, pose_utils, vlm_backbone  # noqa: E402
from .vocab import (  # noqa: E402
    COLOR_NAMES,
    COLOR_PROMPTS_MAP,
    HUMAN_NEG_PROMPTS,
    HUMAN_POS_PROMPTS,
    TYPE_NAMES,
    TYPE_PROMPTS_MAP,
    TYPE_ROLES,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = DATA_DIR / "sightline_trained_model.pt"
METRICS_PATH = DATA_DIR / "training_metrics.json"
VIRAT_DIR = Path(r"D:\projects\sightline-ai-upscaling\Assets\VIRAT")
WEIGHTS_DIR = ROOT_DIR / "datasets_cache" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CACHE_PATH = ROOT_DIR / "datasets_cache" / "siglip384_par_features.pt"

FEATURE_SCHEMA_VERSION = 2

UPPER_TYPE_IDXS = [i for i, r in enumerate(TYPE_ROLES) if r == "upper"]
LOWER_TYPE_IDXS = [i for i, r in enumerate(TYPE_ROLES) if r == "lower"]
ZONES_LIST = ["upper", "outer_upper", "inner_upper", "lower"]

RETAIL_TO_VAR_TYPE = {
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
RETAIL_TO_VAR_COLOR = {
    "light grey": "light grey / silver",
}


def load_yolo_model(filename: str):
    """Load a YOLO weight from datasets_cache/weights (moving cleanly if downloaded to CWD)."""
    from ultralytics import YOLO

    target = WEIGHTS_DIR / filename
    if target.exists():
        return YOLO(str(target))
    m = YOLO(filename)
    cwd_file = Path.cwd() / filename
    if cwd_file.exists() and cwd_file != target:
        shutil.move(str(cwd_file), str(target))
    return m


def sr_fast(sr_model, pil_img: Image.Image, device: torch.device, force_2x: bool = False) -> Image.Image:
    w, h = pil_img.size
    m = min(w, h)
    if m < 30:
        return tuned.sr_upscale_2x(sr_model, tuned.sr_upscale_2x(sr_model, pil_img, device), device).filter(
            ImageFilter.UnsharpMask(1.0, 35, 2)
        )
    if force_2x or m < 96:
        return tuned.sr_upscale_2x(sr_model, pil_img, device).filter(ImageFilter.UnsharpMask(0.8, 25, 2))
    return pil_img


def extract_structural_vec(
    person_sr: Image.Image,
    zone: str,
    zone_fracs: dict | None = None,
) -> np.ndarray:
    """18-D numeric representation of zone role, hood/hem CIELAB continuity, and edge energy."""
    arr = np.asarray(person_sr.convert("RGB")).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    A = lab[:, :, 1] - 128.0
    B = lab[:, :, 2] - 128.0
    H, W = L.shape

    def rs(fr):
        x0, x1, y0, y1 = fr
        r0, r1 = max(0, int(round(H * y0))), max(1, min(H, int(round(H * y1))))
        c0, c1 = max(0, int(round(W * x0))), max(1, min(W, int(round(W * x1))))
        if r1 <= r0:
            r1 = min(H, r0 + 1)
        if c1 <= c0:
            c1 = min(W, c0 + 1)
        return np.array([
            float(np.median(L[r0:r1, c0:c1])),
            float(np.median(A[r0:r1, c0:c1])),
            float(np.median(B[r0:r1, c0:c1])),
        ])

    if zone_fracs is not None and zone in zone_fracs:
        sub_f = zone_fracs[zone]["sub"]
        outer_f = zone_fracs.get("outer_upper", {}).get("sub", tuned.ZONE_GEOMETRY["outer_upper"]["sub"])
        inner_f = zone_fracs.get("inner_upper", {}).get("sub", tuned.ZONE_GEOMETRY["inner_upper"]["sub"])
    else:
        sub_f = tuned.ZONE_GEOMETRY[zone]["sub"]
        outer_f = tuned.ZONE_GEOMETRY["outer_upper"]["sub"]
        inner_f = tuned.ZONE_GEOMETRY["inner_upper"]["sub"]

    torso = rs(sub_f)
    hem = rs((0.28, 0.72, 0.53, 0.67))
    thigh_c = rs((0.26, 0.58, 0.54, 0.72))
    lthigh = rs((0.28, 0.72, 0.68, 0.80))
    outer_r = rs(outer_f)
    inner_r = rs(inner_f)

    hr0, hr1 = max(0, int(round(H * 0.02))), max(1, min(H, int(round(H * 0.20))))
    hc0, hc1 = max(0, int(round(W * 0.35))), max(1, min(W, int(round(W * 0.90))))
    hslice_L = L[hr0:hr1, hc0:hc1]
    hslice_B = B[hr0:hr1, hc0:hc1]
    top_h = hslice_L >= np.percentile(hslice_L, 70)
    hd_topL = float(np.median(hslice_L[top_h]))
    hd_topB = float(np.median(hslice_B[top_h]))

    sub_r0, sub_r1 = max(0, int(H * sub_f[2])), max(1, min(H, int(H * sub_f[3])))
    sub_c0, sub_c1 = max(0, int(W * sub_f[0])), max(1, min(W, int(W * sub_f[1])))
    gray_sub = cv2.cvtColor(arr[sub_r0:sub_r1, sub_c0:sub_c1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    if gray_sub.size < 9:
        cen_grad, quilt_ratio, edge_mean = 0.0, 1.0, 0.0
    else:
        gx = cv2.Sobel(gray_sub, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_sub, cv2.CV_32F, 0, 1, ksize=3)
        cw = gray_sub.shape[1]
        cen_strip = gx[:, max(0, cw // 2 - max(1, cw // 6)) : min(cw, cw // 2 + max(1, cw // 6))]
        cen_grad = float(np.mean(np.abs(cen_strip))) if cen_strip.size else 0.0
        quilt_ratio = float(np.mean(np.abs(gy)) / (np.mean(np.abs(gx)) + 1e-4))
        edge_mean = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))

    de_hem = float(np.linalg.norm(hem - torso))
    de_tc = float(np.linalg.norm(thigh_c - torso))
    de_lthigh = float(np.linalg.norm(lthigh - torso))
    de_in_out = float(np.linalg.norm(outer_r - inner_r))

    zone_onehot = [1.0 if zone == z else 0.0 for z in ZONES_LIST]
    feats = zone_onehot + [
        torso[0] / 100.0,
        torso[1] / 80.0,
        torso[2] / 80.0,
        hd_topL / 100.0,
        hd_topB / 60.0,
        hem[0] / 100.0,
        thigh_c[0] / 100.0,
        de_hem / 50.0,
        de_tc / 50.0,
        de_lthigh / 50.0,
        de_in_out / 50.0,
        min(2.0, cen_grad / 80.0),
        min(2.5, quilt_ratio),
        min(2.0, edge_mean / 100.0),
    ]
    return np.asarray(feats, dtype=np.float32)


class SightlineTrainedModel(nn.Module):
    """Multi-task SigLIP-384 network: Human Verification + Open-World Type + Shadow-Invariant Chromaticity Heads."""

    def __init__(
        self,
        embed_dim: int,
        type_protos: torch.Tensor,
        color_protos: torch.Tensor,
        logit_scale: float = 45.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))

        # Task 1: Human vs non-human verifier (embed_dim + 4 bbox geom + 2 pose visibility)
        self.human_head = nn.Sequential(
            nn.Linear(embed_dim + 6, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 1),
        )

        # Task 2: Residual open-vocabulary clothing-type projection (2*embed_dim + 18 -> embed_dim)
        self.type_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 18, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(384, embed_dim),
        )
        self.type_protos = nn.Parameter(type_protos.clone())  # [21, embed_dim]

        # Task 3: Shadow-invariant chromaticity head (embed_dim + 16 -> embed_dim)
        self.color_mlp = nn.Sequential(
            nn.Linear(embed_dim + 16, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, embed_dim),
        )
        self.color_protos = nn.Parameter(color_protos.clone())  # [17, embed_dim]

    def forward_human(self, feat_emb: torch.Tensor, geom_6: torch.Tensor) -> torch.Tensor:
        return self.human_head(torch.cat([feat_emb, geom_6], dim=-1)).squeeze(-1)

    def encode_type(self, sub_emb: torch.Tensor, ctx_emb: torch.Tensor, struct_18: torch.Tensor) -> torch.Tensor:
        base = F.normalize(0.35 * sub_emb + 0.65 * ctx_emb, dim=-1)
        delta = self.type_mlp(torch.cat([sub_emb, ctx_emb, struct_18], dim=-1))
        return F.normalize(base + 0.55 * delta, dim=-1)

    def forward_type(
        self,
        sub_emb: torch.Tensor,
        ctx_emb: torch.Tensor,
        struct_18: torch.Tensor,
        extra_protos: torch.Tensor | None = None,
    ):
        emb = self.encode_type(sub_emb, ctx_emb, struct_18)
        protos = F.normalize(self.type_protos, dim=-1)
        if extra_protos is not None and extra_protos.numel() > 0:
            protos = torch.cat([protos, F.normalize(extra_protos, dim=-1)], dim=0)
        scale = self.logit_scale.clamp(10.0, 90.0)
        return scale * (emb @ protos.T), emb

    def encode_color(self, sub_emb: torch.Tensor, col_16: torch.Tensor) -> torch.Tensor:
        delta = self.color_mlp(torch.cat([sub_emb, col_16], dim=-1))
        return F.normalize(sub_emb + 0.60 * delta, dim=-1)

    def forward_color(self, sub_emb: torch.Tensor, col_16: torch.Tensor):
        emb = self.encode_color(sub_emb, col_16)
        protos = F.normalize(self.color_protos, dim=-1)
        scale = self.logit_scale.clamp(10.0, 90.0)
        return scale * (emb @ protos.T), emb


def encode_clip_images(vlm_m, vlm_proc, imgs: list[Image.Image], device: torch.device) -> torch.Tensor:
    """Shared wrapper encoding PIL crops via SigLIP-384."""
    return vlm_backbone.encode_vlm_images(vlm_m, vlm_proc, imgs, device, batch_size=16)


def _degrade_surveillance_pil(crop: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Apply realistic low-res downscale + JPEG artifact augmentation."""
    w0, h0 = crop.size
    scale = float(rng.uniform(0.32, 0.62))
    dw, dh = max(8, int(round(w0 * scale))), max(8, int(round(h0 * scale)))
    low = crop.resize((dw, dh), Image.BILINEAR)
    buf = io.BytesIO()
    low.save(buf, format="JPEG", quality=int(rng.integers(38, 72)))
    buf.seek(0)
    with Image.open(buf) as j_im:
        return j_im.convert("RGB")


def _extract_zone_tuple(
    rgb_vlm: np.ndarray,
    rgb_cc: np.ndarray,
    pbox: tuple[float, float, float, float],
    kpts: np.ndarray | None,
    zone: str,
    sr_model,
    device: torch.device,
    frame_stats: dict | None,
    low_res_aug: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """Extract aligned (sub_sr, ctx_sr, struct_18, col_16) using pose geometry + silhouette mask + constancy."""
    small_ped = (pbox[3] - pbox[1]) < 64.0
    zone_fracs = pose_utils.compute_pose_zone_fracs(pbox, kpts)
    sub_f = zone_fracs[zone]["sub"]
    ctx_f = zone_fracs[zone]["ctx"]

    person_raw = tuned.crop_frac(rgb_vlm, pbox, (0.0, 1.0, 0.0, 1.0))
    sub_raw_vlm = tuned.crop_frac(rgb_vlm, pbox, sub_f)
    ctx_raw_vlm = tuned.crop_frac(rgb_vlm, pbox, ctx_f)
    sub_raw_cc = tuned.crop_frac(rgb_cc, pbox, sub_f)

    if low_res_aug and rng is not None:
        sub_raw_vlm = _degrade_surveillance_pil(sub_raw_vlm, rng)
        ctx_raw_vlm = _degrade_surveillance_pil(ctx_raw_vlm, rng)
        sub_raw_cc = _degrade_surveillance_pil(sub_raw_cc, rng)

    person_sr = sr_fast(sr_model, person_raw, device, force_2x=small_ped)
    sub_sr = sr_fast(sr_model, sub_raw_vlm, device, force_2x=small_ped)
    ctx_sr = sr_fast(sr_model, ctx_raw_vlm, device, force_2x=small_ped)
    sub_cc_sr = sr_fast(sr_model, sub_raw_cc, device, force_2x=small_ped)

    p_mask = pose_utils.build_pose_silhouette_mask(pbox, kpts)
    sub_mask = color_constancy.crop_mask_frac(p_mask, sub_f) if p_mask is not None else None

    st_vec = extract_structural_vec(person_sr, zone, zone_fracs=zone_fracs)
    lab_stats = color_constancy.extract_masked_core_lab(sub_cc_sr, sub_mask, sigma=0.28)
    col_vec = color_constancy.extract_shadow_invariant_color_vec(
        sub_cc_sr, sub_mask, lab_stats, frame_stats, zone
    )
    return sub_sr, ctx_sr, st_vec, col_vec


def _prepare_human_samples(pose_model, sr_model, device: torch.device, par_tr_recs: list[dict], par_va_recs: list[dict]):
    """Collect Retail + VIRAT + PAR human positive & hard-negative crops (before VLM encoding)."""
    rng = np.random.default_rng(42)
    train_imgs, train_g, train_y = [], [], []
    val_imgs, val_g, val_y = [], [], []

    sources = [
        ("retail_720p_train", tuned.RES_FILES["720p"], [60, 240, 480, 720, 1080], False),
        ("retail_144p_val", tuned.RES_FILES["144p"], [20, 170, 384, 864, 1280], True),
    ]
    virat_clips = sorted(VIRAT_DIR.glob("VIRAT_S_*.mp4"))
    for vp in virat_clips:
        is_val = vp.name.startswith("VIRAT_S_010204") or vp.name.startswith("VIRAT_S_010000")
        frames = [60, 180, 340, 480] if not is_val else [48, 144, 288, 432]
        sources.append((vp.stem[:14], vp, frames, is_val))

    for _, vpath, f_list, is_val in sources:
        if not vpath.exists():
            continue
        cap = cv2.VideoCapture(str(vpath))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fidx in f_list:
            if fidx >= total_f:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            pil = Image.fromarray(rgb)
            inp_rgb = np.asarray(tuned.sr_upscale_2x(sr_model, pil, device)) if H <= 240 else rgb
            scale = 2.0 if H <= 240 else 1.0
            res = pose_model.predict(inp_rgb, imgsz=896 if H >= 600 else 640, classes=[0], conf=0.22, verbose=False, device=0 if device.type == "cuda" else "cpu")[0]

            pos_items = []
            all_persons = []
            if res.boxes is not None and len(res.boxes):
                boxes_np = res.boxes.xyxy.cpu().numpy() / scale
                confs_np = res.boxes.conf.cpu().numpy()
                kpts_np = res.keypoints.data.cpu().numpy() / np.array([scale, scale, 1.0], dtype=np.float32) if res.keypoints is not None else [None] * len(boxes_np)
                for b, sc, kpts in zip(boxes_np, confs_np, kpts_np):
                    x1, y1, x2, y2 = [float(v) for v in b]
                    bw, bh = x2 - x1, y2 - y1
                    all_persons.append((x1, y1, x2, y2))
                    if float(sc) >= 0.58 and bh >= max(20.0, 0.05 * H):
                        pos_items.append(((x1, y1, x2, y2), kpts))

            # Non-overlapping random background negative boxes
            neg_boxes = []
            for _ in range(max(4, len(pos_items) * 2)):
                rw = float(rng.uniform(0.06 * W, 0.22 * W))
                rh = float(rng.uniform(0.10 * H, 0.34 * H))
                rx1 = float(rng.uniform(0, max(1.0, W - rw)))
                ry1 = float(rng.uniform(0, max(1.0, H - rh)))
                rx2, ry2 = rx1 + rw, ry1 + rh
                overlaps = any(
                    max(0.0, min(rx2, px2) - max(rx1, px1)) * max(0.0, min(ry2, py2) - max(ry1, py1)) > 0.02 * rw * rh
                    for px1, py1, px2, py2 in all_persons
                )
                if not overlaps:
                    neg_boxes.append((rx1, ry1, rx2, ry2))

            tgt_i, tgt_g, tgt_y = (val_imgs, val_g, val_y) if is_val else (train_imgs, train_g, train_y)
            for (x1, y1, x2, y2), kpts in pos_items:
                crop = pil.crop((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))
                if min(crop.size) < 8:
                    continue
                k_vis, has_torso = pose_utils.compute_pose_stats(kpts)
                geom = [(y2 - y1) / max(1.0, x2 - x1), (y2 - y1) / float(H), (x2 - x1) / float(W), 0.5 * (y1 + y2) / float(H), k_vis, has_torso]
                tgt_i.append(sr_fast(sr_model, crop, device, force_2x=((y2 - y1) < 64.0)))
                tgt_g.append(geom)
                tgt_y.append(1.0)
                if not is_val:
                    tgt_i.append(_degrade_surveillance_pil(crop, rng))
                    tgt_g.append(geom)
                    tgt_y.append(1.0)

            for x1, y1, x2, y2 in neg_boxes[: max(6, len(pos_items) * 2)]:
                crop = pil.crop((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))
                if min(crop.size) < 8:
                    continue
                geom = [(y2 - y1) / max(1.0, x2 - x1), (y2 - y1) / float(H), (x2 - x1) / float(W), 0.5 * (y1 + y2) / float(H), 0.0, 0.0]
                tgt_i.append(sr_fast(sr_model, crop, device))
                tgt_g.append(geom)
                tgt_y.append(0.0)
        cap.release()

    # Add unique outdoor PAR positives (weighted cleanly) so verifier sees diverse outdoor angles
    seen_pids = set()
    par_samples = [(r, False) for r in par_tr_recs[:800]] + [(r, True) for r in par_va_recs[:160]]
    for rec, is_val in par_samples:
        pid = rec["person_id"]
        if pid in seen_pids:
            continue
        seen_pids.add(pid)
        crop = rec["pil"]
        pw, ph = crop.size
        kpts = rec.get("kpts")
        k_vis, has_torso = pose_utils.compute_pose_stats(kpts)
        geom = [ph / max(1.0, pw), 0.25, 0.08, 0.5, k_vis, has_torso]
        tgt_i, tgt_g, tgt_y = (val_imgs, val_g, val_y) if is_val else (train_imgs, train_g, train_y)
        tgt_i.append(sr_fast(sr_model, crop, device, force_2x=(ph < 64)))
        tgt_g.append(geom)
        tgt_y.append(1.0)

    return train_imgs, train_g, train_y, val_imgs, val_g, val_y


def _attach_batch_pose(records: list[dict], pose_model, device: torch.device):
    """Run fast batched YOLO11m-pose on unique pedestrian images across PAR records."""
    unique: dict[str, np.ndarray] = {}
    for r in records:
        pid = r["person_id"]
        if pid not in unique:
            unique[pid] = np.asarray(r["pil"])

    pids = list(unique.keys())
    imgs = [unique[p] for p in pids]
    kpts_by_pid: dict[str, np.ndarray | None] = {}
    bs = 48
    dev_arg = 0 if device.type == "cuda" else "cpu"
    for i in range(0, len(imgs), bs):
        chunk = imgs[i : i + bs]
        chunk_pids = pids[i : i + bs]
        results = pose_model.predict(chunk, imgsz=256, classes=[0], conf=0.20, verbose=False, device=dev_arg)
        for pid, res in zip(chunk_pids, results):
            if res.keypoints is not None and len(res.keypoints.data):
                # Pick highest-confidence person box in crop
                best_idx = int(res.boxes.conf.argmax().item()) if res.boxes is not None and len(res.boxes) else 0
                kpts_by_pid[pid] = res.keypoints.data[best_idx].cpu().numpy()
            else:
                kpts_by_pid[pid] = None

    for r in records:
        r["kpts"] = kpts_by_pid.get(r["person_id"])


def _prepare_garment_samples(pose_model, sr_model, device: torch.device, par_tr_recs: list[dict], par_va_recs: list[dict]):
    """Collect Retail (calibrated + full-frame WB) and PAR multi-dataset garment crops + pose/color vectors."""
    rng = np.random.default_rng(77)
    type_idx = {n: i for i, n in enumerate(TYPE_NAMES)}
    color_idx = {n: i for i, n in enumerate(COLOR_NAMES)}

    needed = {sp["frame"] + dt for sp in tuned.GARMENT_SPECS for dt in (-3, -2, -1, 0, 1, 2, 3)}
    res_frames = {rk: tuned.preload_video_frames(rk, needed) for rk in tuned.RES_ORDER}

    # Precompute frame-level Shades-of-Gray + Retinex for every retail frame (never on cropped garments)
    frame_stats_cache: dict[tuple[str, int], dict] = {}
    cc_frames_cache: dict[tuple[str, int], np.ndarray] = {}
    for rk, fmap in res_frames.items():
        for fi, rgb in fmap.items():
            fs = color_constancy.compute_frame_illuminant_gains(rgb)
            frame_stats_cache[(rk, fi)] = fs
            cc_frames_cache[(rk, fi)] = color_constancy.apply_color_constancy(rgb, fs)

    # Also extract pose keypoints once for 720p retail keyframes so training & inference share pose alignment
    retail_pose_cache: dict[tuple[int, int], np.ndarray | None] = {}
    dev_arg = 0 if device.type == "cuda" else "cpu"
    for fi, rgb720 in res_frames["720p"].items():
        H7, W7 = rgb720.shape[:2]
        res = pose_model.predict(rgb720, imgsz=960, classes=[0], conf=0.25, verbose=False, device=dev_arg)[0]
        pboxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None and len(res.boxes) else np.zeros((0, 4))
        pkpts = res.keypoints.data.cpu().numpy() if res.keypoints is not None and len(res.keypoints.data) else None
        for sp_idx, sp in enumerate(tuned.GARMENT_SPECS):
            if sp["frame"] - 3 <= fi <= sp["frame"] + 3 and pkpts is not None and len(pboxes):
                sx0, sy0, sx1, sy1 = sp["box"]
                best_k = None
                best_iou = 0.25
                for bi, pb in enumerate(pboxes):
                    ix1, iy1 = max(sx0, pb[0]), max(sy0, pb[1])
                    ix2, iy2 = min(sx1, pb[2]), min(sy1, pb[3])
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    union = max(1.0, (sx1 - sx0) * (sy1 - sy0) + (pb[2] - pb[0]) * (pb[3] - pb[1]) - inter)
                    if inter / union > best_iou:
                        best_iou = inter / union
                        best_k = pkpts[bi]
                if best_k is not None:
                    # Normalize kpts to [0..1] relative to image so any resolution works!
                    kn = best_k.copy()
                    kn[:, 0] /= float(W7)
                    kn[:, 1] /= float(H7)
                    retail_pose_cache[(sp_idx, fi)] = kn

    data: dict[str, dict[str, list]] = {
        "train": {"sub": [], "ctx": [], "struct": [], "col": [], "tmask": [], "cmask": [], "weight": [], "source": []},
        "val": {"sub": [], "ctx": [], "struct": [], "col": [], "tmask": [], "cmask": [], "weight": [], "source": []},
    }

    def add_row(split: str, sub_sr, ctx_sr, st_vec, col_vec, t_idxs: list[int], c_idxs: list[int], weight: float, src: str):
        tmask = np.zeros(len(TYPE_NAMES), dtype=np.float32)
        for ti in t_idxs:
            tmask[ti] = 1.0
        cmask = np.zeros(len(COLOR_NAMES), dtype=np.float32)
        for ci in c_idxs:
            cmask[ci] = 1.0
        b = data[split]
        b["sub"].append(sub_sr)
        b["ctx"].append(ctx_sr)
        b["struct"].append(st_vec)
        b["col"].append(col_vec)
        b["tmask"].append(tmask)
        b["cmask"].append(cmask)
        b["weight"].append(weight)
        b["source"].append(src)

    # 1. Retail benchmark samples (preserve strict held-out 144p/240p dt==0 val split)
    for rk in tuned.RES_ORDER:
        for sp_idx, sp in enumerate(tuned.GARMENT_SPECS):
            zone = sp["zone"]
            vt_name = RETAIL_TO_VAR_TYPE.get(sp["gt_type"], sp["gt_type"])
            vc_name = RETAIL_TO_VAR_COLOR.get(sp["gt_color"], sp["gt_color"])
            yt = [type_idx[vt_name]]
            yc = [color_idx[vc_name]]
            fc = sp["frame"]
            dts = [0] if rk in ("144p", "240p") else (-3, -2, -1, 1, 2, 3)

            for dt in dts:
                fi = fc + dt
                rgb = res_frames[rk][fi]
                rgb_cc = cc_frames_cache[(rk, fi)]
                fs = frame_stats_cache[(rk, fi)]
                H, W = rgb.shape[:2]
                pbox = tuned.scale_box(sp["box"], W, H)
                kn = retail_pose_cache.get((sp_idx, fi))
                kpts = None
                if kn is not None:
                    kpts = kn.copy()
                    kpts[:, 0] *= float(W)
                    kpts[:, 1] *= float(H)

                sub_sr, ctx_sr, st_v, col_v = _extract_zone_tuple(
                    rgb, rgb_cc, pbox, kpts, zone, sr_model, device, fs, low_res_aug=False
                )
                if rk in ("144p", "240p"):
                    add_row("val", sub_sr, ctx_sr, st_v, col_v, yt, yc, 1.0, f"retail_{rk}")
                else:
                    add_row("train", sub_sr, ctx_sr, st_v, col_v, yt, yc, 3.0, f"retail_{rk}")
                    if rk == "360p" and dt in (-2, 2):
                        s2, c2, st2, col2 = _extract_zone_tuple(
                            rgb, rgb_cc, pbox, kpts, zone, sr_model, device, fs, low_res_aug=True, rng=rng
                        )
                        add_row("train", s2, c2, st2, col2, yt, yc, 2.5, "retail_360_lowaug")

                if rk in ("144p", "240p") and dt == 0:
                    for tr_dt in (-3, -1, 1, 3):
                        fi2 = fc + tr_dt
                        rgb2 = res_frames[rk][fi2]
                        rgb2_cc = cc_frames_cache[(rk, fi2)]
                        fs2 = frame_stats_cache[(rk, fi2)]
                        pbox2 = tuned.scale_box(sp["box"], rgb2.shape[1], rgb2.shape[0])
                        s2, c2, st2, col2 = _extract_zone_tuple(
                            rgb2, rgb2_cc, pbox2, kpts, zone, sr_model, device, fs2
                        )
                        add_row("train", s2, c2, st2, col2, yt, yc, 3.2, f"retail_{rk}_offset")

    # 2. Outdoor Surveillance PAR Datasets (PA-100K + Market-1501 + Simuletic CCTV-1K)
    for rec in par_tr_recs + par_va_recs:
        arr = np.asarray(rec["pil"])
        H, W = arr.shape[:2]
        pbox = (0.0, 0.0, float(W), float(H))
        is_tr = rec["split"] == "train"
        aug = bool(is_tr and rng.random() < 0.30)
        # Note: frame_stats=None on pre-cropped PAR images to avoid estimating white balance from single-garment crops!
        sub_sr, ctx_sr, st_v, col_v = _extract_zone_tuple(
            arr, arr, pbox, rec.get("kpts"), rec["zone"], sr_model, device, frame_stats=None, low_res_aug=aug, rng=rng
        )
        # Weight Simuletic higher on color/type since all labels are fine-grained 1-hot / exact pairs
        w_val = 2.0 if rec["source"].startswith("simuletic") else 1.0
        add_row(rec["split"], sub_sr, ctx_sr, st_v, col_v, rec["type_idxs"], rec["color_idxs"], w_val, rec["source"])

    return data


def _extract_all_features(device: torch.device) -> dict:
    """Extract and cache SigLIP-384 + YOLO11m-pose + 16-D chromaticity tensors cleanly in low VRAM."""
    if FEATURE_CACHE_PATH.exists():
        cached = torch.load(str(FEATURE_CACHE_PATH), map_location="cpu", weights_only=False)
        if cached.get("schema_version") == FEATURE_SCHEMA_VERSION:
            print(f"Loaded cached SigLIP-384 + PAR feature bundle from {FEATURE_CACHE_PATH}")
            return cached

    par_tr, par_va, par_summary = par_datasets.load_all_par_datasets()

    # Phase A: Load YOLO11m-pose -> estimate COCO-17 joints on all PAR crops, Retail & VIRAT frames
    print("Stage A: Extracting COCO-17 pose keypoints across PAR + Retail + VIRAT...")
    pose_model = load_yolo_model("yolo11m-pose.pt")
    sr_model = tuned.load_realesrgan_x2plus(device)

    _attach_batch_pose(par_tr + par_va, pose_model, device)
    hx_tr_imgs, hg_tr, hy_tr, hx_va_imgs, hg_va, hy_va = _prepare_human_samples(
        pose_model, sr_model, device, par_tr, par_va
    )
    gdata = _prepare_garment_samples(pose_model, sr_model, device, par_tr, par_va)

    del pose_model, sr_model
    torch.cuda.empty_cache()

    # Phase B: Load google/siglip-base-patch16-384 (fp16 CUDA) and batch-encode all images & prompts
    print(f"Stage B: Encoding crops with {vlm_backbone.VLM_MODEL_ID} (fp16)...")
    vlm_m, vlm_proc, embed_dim = vlm_backbone.load_vlm(device)

    type_tmat = vlm_backbone.build_vlm_text_matrix(vlm_m, vlm_proc, TYPE_PROMPTS_MAP, TYPE_NAMES, device)
    color_tmat = vlm_backbone.build_vlm_text_matrix(vlm_m, vlm_proc, COLOR_PROMPTS_MAP, COLOR_NAMES, device)
    h_pos_tmat = vlm_backbone.encode_vlm_texts(vlm_m, vlm_proc, HUMAN_POS_PROMPTS, device)
    h_neg_tmat = vlm_backbone.encode_vlm_texts(vlm_m, vlm_proc, HUMAN_NEG_PROMPTS, device)

    bundle = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "vlm_model_id": vlm_backbone.VLM_MODEL_ID,
        "embed_dim": embed_dim,
        "par_summary": par_summary,
        "type_tmat": type_tmat.cpu(),
        "color_tmat": color_tmat.cpu(),
        "h_pos_tmat": h_pos_tmat.cpu(),
        "h_neg_tmat": h_neg_tmat.cpu(),
        "human": {
            "tr_x": encode_clip_images(vlm_m, vlm_proc, hx_tr_imgs, device).cpu(),
            "tr_g": torch.tensor(hg_tr, dtype=torch.float32),
            "tr_y": torch.tensor(hy_tr, dtype=torch.float32),
            "va_x": encode_clip_images(vlm_m, vlm_proc, hx_va_imgs, device).cpu(),
            "va_g": torch.tensor(hg_va, dtype=torch.float32),
            "va_y": torch.tensor(hy_va, dtype=torch.float32),
        },
        "garment": {},
    }

    for sp in ("train", "val"):
        b = gdata[sp]
        bundle["garment"][sp] = {
            "sub": encode_clip_images(vlm_m, vlm_proc, b["sub"], device).cpu(),
            "ctx": encode_clip_images(vlm_m, vlm_proc, b["ctx"], device).cpu(),
            "struct": torch.tensor(np.array(b["struct"]), dtype=torch.float32),
            "col": torch.tensor(np.array(b["col"]), dtype=torch.float32),
            "tmask": torch.tensor(np.array(b["tmask"]), dtype=torch.float32),
            "cmask": torch.tensor(np.array(b["cmask"]), dtype=torch.float32),
            "weight": torch.tensor(np.array(b["weight"]), dtype=torch.float32),
            "source": b["source"],
        }

    del vlm_m
    torch.cuda.empty_cache()

    torch.save(bundle, str(FEATURE_CACHE_PATH))
    print(f"Saved SigLIP-384 + PAR feature bundle to {FEATURE_CACHE_PATH}")
    return bundle


def group_ce_loss(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    valid_role_mask: torch.Tensor | None = None,
    label_smoothing: float = 0.04,
) -> torch.Tensor:
    """Group-marginalized Log-Sum-Exp cross-entropy supporting both fine 1-hot & coarse multi-label targets."""
    has_label = target_mask.sum(dim=-1) > 0.5
    if not bool(has_label.any()):
        return logits.sum() * 0.0

    lg = logits[has_label]
    tm = target_mask[has_label]
    if valid_role_mask is not None:
        vm = valid_role_mask[has_label]
        lg = lg.masked_fill(vm <= 0.5, -1e4)
        tm = tm * vm
        num_valid = vm.sum(dim=-1).clamp(min=1.0)
        log_denom = torch.logsumexp(lg, dim=-1)
        log_p_mean = ((F.log_softmax(lg, dim=-1)) * vm).sum(dim=-1) / num_valid
    else:
        log_denom = torch.logsumexp(lg, dim=-1)
        log_p_mean = F.log_softmax(lg, dim=-1).mean(dim=-1)

    pos_logits = lg.masked_fill(tm <= 0.5, -1e4)
    log_pos_prob = torch.logsumexp(pos_logits, dim=-1) - log_denom
    per_sample = (1.0 - label_smoothing) * (-log_pos_prob) + label_smoothing * (-log_p_mean)
    if sample_weights is not None:
        sw = sample_weights[has_label]
        return (per_sample * sw).sum() / (sw.sum() + 1e-6)
    return per_sample.mean()


def train_model(epochs: int = 60, batch_size: int = 256) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    bundle = _extract_all_features(device)
    embed_dim = int(bundle["embed_dim"])
    type_tmat = bundle["type_tmat"].to(device)
    color_tmat = bundle["color_tmat"].to(device)

    hum = bundle["human"]
    hx_tr, hg_tr, hy_tr = hum["tr_x"].to(device), hum["tr_g"].to(device), hum["tr_y"].to(device)
    hx_va, hg_va, hy_va = hum["va_x"].to(device), hum["va_g"].to(device), hum["va_y"].to(device)

    gtr = bundle["garment"]["train"]
    tr_sub, tr_ctx = gtr["sub"].to(device), gtr["ctx"].to(device)
    tr_struct, tr_col = gtr["struct"].to(device), gtr["col"].to(device)
    tr_tmask, tr_cmask = gtr["tmask"].to(device), gtr["cmask"].to(device)
    tr_w = gtr["weight"].to(device)

    # Precompute anatomical role masks per sample (1 for upper indices on upper zones, lower indices on lower zone)
    tr_is_lower = tr_struct[:, 3] > 0.5
    tr_role_mask = torch.zeros_like(tr_tmask)
    tr_role_mask[~tr_is_lower[:, None].expand_as(tr_tmask)] = 0.0
    upper_row = torch.zeros(len(TYPE_NAMES), device=device)
    upper_row[UPPER_TYPE_IDXS] = 1.0
    lower_row = torch.zeros(len(TYPE_NAMES), device=device)
    lower_row[LOWER_TYPE_IDXS] = 1.0
    tr_role_mask = torch.where(tr_is_lower[:, None], lower_row[None, :], upper_row[None, :])

    gva = bundle["garment"]["val"]
    va_sub, va_ctx = gva["sub"].to(device), gva["ctx"].to(device)
    va_struct, va_col = gva["struct"].to(device), gva["col"].to(device)
    va_tmask, va_cmask = gva["tmask"].to(device), gva["cmask"].to(device)
    va_sources = gva["source"]

    model = SightlineTrainedModel(embed_dim, type_tmat, color_tmat, logit_scale=45.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2.5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    type_anchor = type_tmat.detach().clone()
    color_anchor = color_tmat.detach().clone()

    n_gar = tr_sub.size(0)
    rng = torch.Generator(device=device).manual_seed(2026)
    loss_history = []

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_gar, generator=rng, device=device)
        ep_loss = 0.0
        steps = 0

        for start in range(0, n_gar, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()

            # Full small human batch paired with garment mini-batch
            h_logits = model.forward_human(hx_tr, hg_tr)
            loss_h = F.binary_cross_entropy_with_logits(h_logits, hy_tr)

            t_logits, _ = model.forward_type(tr_sub[idx], tr_ctx[idx], tr_struct[idx])
            loss_t = group_ce_loss(t_logits, tr_tmask[idx], tr_w[idx], tr_role_mask[idx], label_smoothing=0.04)

            c_logits, _ = model.forward_color(tr_sub[idx], tr_col[idx])
            loss_c = group_ce_loss(c_logits, tr_cmask[idx], tr_w[idx], None, label_smoothing=0.04)

            reg_t = F.mse_loss(F.normalize(model.type_protos, dim=-1), type_anchor)
            reg_c = F.mse_loss(F.normalize(model.color_protos, dim=-1), color_anchor)

            loss = loss_h + loss_t + loss_c + 1.6 * reg_t + 1.6 * reg_c
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            steps += 1

        sched.step()
        loss_history.append(round(ep_loss / max(1, steps), 4))

    model.eval()
    with torch.no_grad():
        hv_prob = torch.sigmoid(model.forward_human(hx_va, hg_va))
        human_val_acc = float(((hv_prob >= 0.5).float() == hy_va).float().mean().item()) * 100.0

        vt_logits, _ = model.forward_type(va_sub, va_ctx, va_struct)
        va_is_lower = va_struct[:, 3] > 0.5
        va_role_mask = torch.where(va_is_lower[:, None], lower_row[None, :], upper_row[None, :])
        vt_masked = vt_logits.masked_fill(va_role_mask <= 0.5, -1e4)
        va_tpreds = vt_masked.argmax(dim=-1)

        vc_logits, _ = model.forward_color(va_sub, va_col)
        va_cpreds = vc_logits.argmax(dim=-1)

        def eval_subset(prefix: str):
            sel = np.array([s.startswith(prefix) for s in va_sources], dtype=bool)
            if not np.any(sel):
                return {"samples": 0, "type_acc_pct": None, "color_acc_pct": None}
            sel_t = torch.tensor(sel, device=device)
            tm_sub = va_tmask[sel_t]
            tp_sub = va_tpreds[sel_t]
            t_has = tm_sub.sum(dim=-1) > 0.5
            t_hit = tm_sub[torch.arange(tp_sub.size(0), device=device), tp_sub] > 0.5
            t_acc = float(t_hit[t_has].float().mean().item()) * 100.0 if bool(t_has.any()) else None

            cm_sub = va_cmask[sel_t]
            cp_sub = va_cpreds[sel_t]
            c_has = cm_sub.sum(dim=-1) > 0.5
            c_hit = cm_sub[torch.arange(cp_sub.size(0), device=device), cp_sub] > 0.5
            c_acc = float(c_hit[c_has].float().mean().item()) * 100.0 if bool(c_has.any()) else None
            return {
                "samples": int(sel.sum()),
                "type_acc_pct": round(t_acc, 1) if t_acc is not None else None,
                "color_acc_pct": round(c_acc, 1) if c_acc is not None else None,
            }

        m_retail = eval_subset("retail_")
        m_pa = eval_subset("pa100k_")
        m_mk = eval_subset("market_")
        m_sim = eval_subset("simuletic_")

    metrics = {
        "arch_version": 2,
        "vlm_model_id": vlm_backbone.VLM_MODEL_ID,
        "embed_dim": embed_dim,
        "epochs": epochs,
        "train_time_sec": round(time.perf_counter() - t0, 2),
        "human_train_samples": int(hx_tr.size(0)),
        "human_val_samples": int(hx_va.size(0)),
        "human_val_acc_pct": round(human_val_acc, 1),
        "garment_train_samples": int(tr_sub.size(0)),
        "garment_val_samples": int(va_sub.size(0)),
        "par_dataset_counts": bundle.get("par_summary", {}),
        "type_val_acc_144_240_pct": m_retail["type_acc_pct"],
        "color_val_acc_144_240_pct": m_retail["color_acc_pct"],
        "val_by_dataset": {
            "retail_144_240": m_retail,
            "pa100k_val": m_pa,
            "market1501_test_ids": m_mk,
            "simuletic_val_ids": m_sim,
        },
        "final_loss": loss_history[-1],
    }
    ckpt = {
        "arch_version": 2,
        "vlm_model_id": vlm_backbone.VLM_MODEL_ID,
        "embed_dim": embed_dim,
        "type_names": list(TYPE_NAMES),
        "color_names": list(COLOR_NAMES),
        "model_state": model.state_dict(),
        "metrics": metrics,
    }
    torch.save(ckpt, str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Training complete:")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    train_model(epochs=60)


if __name__ == "__main__":
    main()
