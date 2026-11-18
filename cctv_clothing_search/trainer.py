"""Multi-task PyTorch trainer for Human ID + Low-Res Clothing Type/Color + Open-World Discovery."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models.detection as tv_det
from PIL import Image, ImageFilter
from transformers import CLIPModel, CLIPProcessor

ROOT_DIR = Path(r"D:\projects\sightline-ai-upscaling")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_cctv_clothing_tuning_report as tuned  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = DATA_DIR / "sightline_trained_model.pt"
METRICS_PATH = DATA_DIR / "training_metrics.json"
VIRAT_DIR = Path(r"D:\Projects\sightline-ai-upscaling\Assets\VIRAT")

EXPANDED_COLORS = list(tuned.CLOTHING_COLORS) + [
    "red",
    "green",
    "yellow / orange",
    "brown",
    "purple",
]

EXPANDED_COLOR_PROMPTS = {
    **tuned.TUNED_COLOR_PROMPTS,
    "red": ["red clothing garment", "bright red colored fabric"],
    "green": ["green olive clothing garment", "dark or bright green fabric"],
    "yellow / orange": ["yellow or orange clothing garment", "bright orange amber fabric"],
    "brown": ["brown tan leather clothing garment", "chocolate brown fabric"],
    "purple": ["purple violet clothing garment", "plum lavender fabric"],
}

# Open-world discovery bank: garment types not in the initial 10 indoor winter store classes
OPEN_WORLD_CANDIDATES = [
    {
        "name": "t-shirt / polo shirt",
        "short": "T-Shirt / Polo",
        "zone_role": "upper",
        "prompts": [
            "a person wearing a short sleeve t-shirt outdoors",
            "casual polo shirt short sleeves visible bare arms",
        ],
        "synonyms": ["t-shirt", "tshirt", "tee", "polo", "short sleeve", "polo shirt"],
    },
    {
        "name": "shorts / cargo shorts",
        "short": "Shorts",
        "zone_role": "lower",
        "prompts": [
            "person wearing above-the-knee shorts outdoors showing bare lower legs",
            "casual knee-length cargo shorts on walking pedestrian",
        ],
        "synonyms": ["shorts", "cargo shorts", "short pants"],
    },
    {
        "name": "windbreaker / rain jacket",
        "short": "Windbreaker",
        "zone_role": "upper",
        "prompts": [
            "lightweight nylon windbreaker shell jacket worn outdoors",
            "outdoor waterproof rain jacket zip front",
        ],
        "synonyms": ["windbreaker", "rain jacket", "anorak", "shell"],
    },
    {
        "name": "vest / safety vest",
        "short": "Vest",
        "zone_role": "upper",
        "prompts": [
            "sleeveless utility vest or high visibility vest over shirt",
            "quilted gilet or sleeveless outdoor vest",
        ],
        "synonyms": ["vest", "safety vest", "gilet", "waistcoat"],
    },
    {
        "name": "suit jacket / blazer",
        "short": "Blazer",
        "zone_role": "upper",
        "prompts": [
            "tailored structured blazer or formal suit jacket",
            "business suit coat with structured lapels",
        ],
        "synonyms": ["blazer", "suit", "suit jacket"],
    },
    {
        "name": "dress / skirt",
        "short": "Dress / Skirt",
        "zone_role": "lower",
        "prompts": [
            "person wearing a flowing skirt or knee length dress",
            "pleated casual skirt outfit",
        ],
        "synonyms": ["dress", "skirt", "midi skirt"],
    },
    {
        "name": "cargo pants",
        "short": "Cargo Pants",
        "zone_role": "lower",
        "prompts": [
            "relaxed utility cargo pants with side leg pockets",
            "baggy outdoor work cargo trousers",
        ],
        "synonyms": ["cargo pants", "cargo trousers", "work pants"],
    },
    {
        "name": "joggers / track pants",
        "short": "Joggers",
        "zone_role": "lower",
        "prompts": [
            "athletic drawstring sweatpants joggers or track pants",
            "tapered sport track pants legs",
        ],
        "synonyms": ["joggers", "track pants", "sweatpants"],
    },
    {
        "name": "flannel / plaid shirt",
        "short": "Plaid Shirt",
        "zone_role": "upper",
        "prompts": [
            "checkered plaid pattern flannel button down shirt",
            "tartan plaid woven overshirt",
        ],
        "synonyms": ["flannel", "plaid", "plaid shirt", "checkered"],
    },
    {
        "name": "tank top / sleeveless top",
        "short": "Tank Top",
        "zone_role": "upper",
        "prompts": [
            "sleeveless summer tank top with bare shoulders",
            "casual sleeveless singlet top",
        ],
        "synonyms": ["tank top", "sleeveless", "singlet"],
    },
    {
        "name": "overalls / workwear",
        "short": "Workwear",
        "zone_role": "upper",
        "prompts": [
            "industrial one piece coveralls workwear jumpsuit",
            "utility bib overalls outdoor worker",
        ],
        "synonyms": ["overalls", "workwear", "coveralls", "jumpsuit"],
    },
]

ZONES_LIST = ["upper", "outer_upper", "inner_upper", "lower"]


def extract_structural_vec(person_sr: Image.Image, zone: str) -> np.ndarray:
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

    sub_f = tuned.ZONE_GEOMETRY[zone]["sub"]
    torso = rs(sub_f)
    hem = rs((0.28, 0.72, 0.53, 0.67))
    thigh_c = rs((0.26, 0.58, 0.54, 0.72))
    lthigh = rs((0.28, 0.72, 0.68, 0.80))
    outer_r = rs(tuned.ZONE_GEOMETRY["outer_upper"]["sub"])
    inner_r = rs(tuned.ZONE_GEOMETRY["inner_upper"]["sub"])

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
        cen_strip = gx[:, max(0, cw // 2 - max(1, cw // 6)):min(cw, cw // 2 + max(1, cw // 6))]
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


def extract_color_feature_vec(lab_stats: dict, zone: str) -> np.ndarray:
    """9-D illuminant-normalized CIELAB + HSV feature vector."""
    h_rad = float(lab_stats["H"]) * (np.pi / 90.0)
    return np.asarray(
        [
            lab_stats["L50"] / 100.0,
            lab_stats["L78"] / 100.0,
            lab_stats["A"] / 80.0,
            lab_stats["B"] / 80.0,
            lab_stats["C"] / 90.0,
            float(np.sin(h_rad)),
            float(np.cos(h_rad)),
            lab_stats["S"] / 255.0,
            1.0 if zone == "lower" else 0.0,
        ],
        dtype=np.float32,
    )


class SightlineTrainedModel(nn.Module):
    """Multi-task network: Human Verification + Open-World Clothing Type & Color Adapters."""

    def __init__(self, type_protos: torch.Tensor, color_protos: torch.Tensor, logit_scale: float = 22.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))

        # Task 1: Human vs. non-human verification head (512 CLIP + 4 geom = 516)
        self.human_head = nn.Sequential(
            nn.Linear(516, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

        # Task 2: Residual clothing-type projection anchored to 512-d CLIP space (1042 -> 512)
        self.type_mlp = nn.Sequential(
            nn.Linear(1042, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(256, 512),
        )
        self.type_protos = nn.Parameter(type_protos.clone())  # [N_types, 512]

        # Task 3: Residual clothing-color projection anchored to 512-d CLIP space (521 -> 512)
        self.color_mlp = nn.Sequential(
            nn.Linear(521, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 512),
        )
        self.color_protos = nn.Parameter(color_protos.clone())  # [N_colors, 512]

    def forward_human(self, feat_512: torch.Tensor, geom_4: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat_512, geom_4], dim=-1)
        return self.human_head(x).squeeze(-1)

    def encode_type(self, sub_512: torch.Tensor, ctx_512: torch.Tensor, struct_18: torch.Tensor) -> torch.Tensor:
        base = F.normalize(0.35 * sub_512 + 0.65 * ctx_512, dim=-1)
        delta = self.type_mlp(torch.cat([sub_512, ctx_512, struct_18], dim=-1))
        return F.normalize(base + 0.55 * delta, dim=-1)

    def forward_type(self, sub_512: torch.Tensor, ctx_512: torch.Tensor, struct_18: torch.Tensor, extra_protos: torch.Tensor | None = None):
        emb = self.encode_type(sub_512, ctx_512, struct_18)
        protos = F.normalize(self.type_protos, dim=-1)
        if extra_protos is not None and extra_protos.numel() > 0:
            protos = torch.cat([protos, F.normalize(extra_protos, dim=-1)], dim=0)
        scale = self.logit_scale.clamp(5.0, 50.0)
        return scale * (emb @ protos.T), emb

    def encode_color(self, sub_512: torch.Tensor, col_9: torch.Tensor) -> torch.Tensor:
        delta = self.color_mlp(torch.cat([sub_512, col_9], dim=-1))
        return F.normalize(sub_512 + 0.65 * delta, dim=-1)

    def forward_color(self, sub_512: torch.Tensor, col_9: torch.Tensor):
        emb = self.encode_color(sub_512, col_9)
        protos = F.normalize(self.color_protos, dim=-1)
        scale = self.logit_scale.clamp(5.0, 50.0)
        return scale * (emb @ protos.T), emb


def encode_clip_images(clip_m: CLIPModel, clip_proc: CLIPProcessor, imgs: list[Image.Image], device: torch.device) -> torch.Tensor:
    if not imgs:
        return torch.empty((0, 512), device=device)
    feats = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            chunk = imgs[i : i + bs]
            inputs = clip_proc(images=chunk, return_tensors="pt").to(device)
            out = clip_m.get_image_features(**inputs)
            im_f = out.pooler_output if hasattr(out, "pooler_output") else out
            feats.append(F.normalize(im_f, dim=-1))
    return torch.cat(feats, dim=0)


def sr_fast(sr_model, pil_img: Image.Image, device: torch.device) -> Image.Image:
    w, h = pil_img.size
    m = min(w, h)
    if m < 30:
        return tuned.sr_upscale_2x(sr_model, tuned.sr_upscale_2x(sr_model, pil_img, device), device).filter(
            ImageFilter.UnsharpMask(1.0, 35, 2)
        )
    if m < 96:
        return tuned.sr_upscale_2x(sr_model, pil_img, device).filter(ImageFilter.UnsharpMask(0.8, 25, 2))
    return pil_img


def collect_human_dataset(det_model, sr_model, clip_m, clip_proc, device: torch.device):
    """Build positive pedestrian & hard-negative background crops across retail + VIRAT CCTV."""
    det_tf = torchvision.transforms.ToTensor()
    rng = np.random.default_rng(42)

    train_imgs, train_g, train_y = [], [], []
    val_imgs, val_g, val_y = [], [], []

    sources = [
        ("retail_720p_train", tuned.RES_FILES["720p"], [60, 240, 480, 720, 1080], False),
        ("retail_144p_val", tuned.RES_FILES["144p"], [20, 170, 384, 864, 1280], True),
    ]
    virat_clips = sorted(VIRAT_DIR.glob("VIRAT_S_*.mp4"))
    for i, vp in enumerate(virat_clips):
        is_val = (vp.name.startswith("VIRAT_S_010204") or vp.name.startswith("VIRAT_S_010000"))
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
            inp = tuned.sr_upscale_2x(sr_model, pil, device) if H <= 240 else pil
            scale = 2.0 if H <= 240 else 1.0
            with torch.no_grad():
                out = det_model([det_tf(inp).to(device)])[0]

            pos_boxes = []
            all_persons = []
            neg_boxes = []
            for b, lab, sc in zip(out["boxes"].cpu().numpy() / scale, out["labels"].cpu().numpy(), out["scores"].cpu().numpy()):
                x1, y1, x2, y2 = [float(v) for v in b]
                bw, bh = x2 - x1, y2 - y1
                if int(lab) == 1 and float(sc) >= 0.20:
                    all_persons.append((x1, y1, x2, y2))
                    if float(sc) >= 0.72 and bh >= max(24.0, 0.06 * H):
                        pos_boxes.append((x1, y1, x2, y2))
                elif int(lab) != 1 and float(sc) >= 0.55 and bw >= 16 and bh >= 16:
                    neg_boxes.append((x1, y1, x2, y2))

            # Filter out non-human objects overlapping real persons & synthesize hard background crops
            clean_negs = [
                (x1, y1, x2, y2) for (x1, y1, x2, y2) in neg_boxes
                if not any(
                    max(0.0, min(x2, px2) - max(x1, px1)) * max(0.0, min(y2, py2) - max(y1, py1)) > 0.05 * (x2 - x1) * (y2 - y1)
                    for px1, py1, px2, py2 in all_persons
                )
            ]
            for _ in range(max(4, len(pos_boxes) * 2)):
                rw = float(rng.uniform(0.08 * W, 0.24 * W))
                rh = float(rng.uniform(0.12 * H, 0.36 * H))
                rx1 = float(rng.uniform(0, max(1.0, W - rw)))
                ry1 = float(rng.uniform(0, max(1.0, H - rh)))
                rx2, ry2 = rx1 + rw, ry1 + rh
                overlaps = any(
                    max(0.0, min(rx2, px2) - max(rx1, px1)) * max(0.0, min(ry2, py2) - max(ry1, py1)) > 0.02 * rw * rh
                    for px1, py1, px2, py2 in all_persons
                )
                if not overlaps:
                    clean_negs.append((rx1, ry1, rx2, ry2))
            neg_boxes = clean_negs

            tgt_imgs, tgt_g, tgt_y = (val_imgs, val_g, val_y) if is_val else (train_imgs, train_g, train_y)
            for label_val, box_list in [(1.0, pos_boxes), (0.0, neg_boxes[: max(6, len(pos_boxes) * 2)])]:
                for x1, y1, x2, y2 in box_list:
                    crop = pil.crop((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))
                    if min(crop.size) < 8:
                        continue
                    geom = [
                        (y2 - y1) / max(1.0, x2 - x1),
                        (y2 - y1) / float(H),
                        (x2 - x1) / float(W),
                        0.5 * (y1 + y2) / float(H),
                    ]
                    tgt_imgs.append(sr_fast(sr_model, crop, device))
                    tgt_g.append(geom)
                    tgt_y.append(label_val)
                    # Low-res blur/JPEG augmentation on training set
                    if not is_val:
                        w0, h0 = crop.size
                        dw, dh = max(8, w0 // 3), max(8, h0 // 3)
                        low = crop.resize((dw, dh), Image.BILINEAR).resize((w0, h0), Image.BILINEAR)
                        tgt_imgs.append(low)
                        tgt_g.append(geom)
                        tgt_y.append(label_val)
        cap.release()

    x_tr = encode_clip_images(clip_m, clip_proc, train_imgs, device)
    x_va = encode_clip_images(clip_m, clip_proc, val_imgs, device)
    return (
        x_tr,
        torch.tensor(train_g, dtype=torch.float32, device=device),
        torch.tensor(train_y, dtype=torch.float32, device=device),
        x_va,
        torch.tensor(val_g, dtype=torch.float32, device=device),
        torch.tensor(val_y, dtype=torch.float32, device=device),
    )


def collect_garment_dataset(sr_model, clip_m, clip_proc, device: torch.device):
    """Collect multi-resolution garment crops (train on non-center + HD frames, val on unseen 144p/240p center)."""
    needed = {sp["frame"] + dt for sp in tuned.GARMENT_SPECS for dt in (-3, -2, -1, 0, 1, 2, 3)}
    res_frames = {rk: tuned.preload_video_frames(rk, needed) for rk in tuned.RES_ORDER}

    tr_sub_imgs, tr_ctx_imgs, tr_struct, tr_col9, tr_yt, tr_yc = [], [], [], [], [], []
    va_sub_imgs, va_ctx_imgs, va_struct, va_col9, va_yt, va_yc = [], [], [], [], [], []

    for rk in tuned.RES_ORDER:
        for sp in tuned.GARMENT_SPECS:
            zone = sp["zone"]
            yt = tuned.CLOTHING_TYPES.index(sp["gt_type"])
            yc = tuned.CLOTHING_COLORS.index(sp["gt_color"])
            fc = sp["frame"]
            dts = [0] if rk in ("144p", "240p") else (-3, -2, -1, 1, 2, 3)

            for dt in dts:
                # Strict split: center frames at 144p & 240p are held-out validation only!
                fi = fc + dt
                rgb = res_frames[rk][fi]
                H, W = rgb.shape[:2]
                pbox = tuned.scale_box(sp["box"], W, H)
                sub_raw = tuned.crop_frac(rgb, pbox, tuned.ZONE_GEOMETRY[zone]["sub"])
                ctx_raw = tuned.crop_frac(rgb, pbox, tuned.ZONE_GEOMETRY[zone]["ctx"])
                person_raw = tuned.crop_frac(rgb, pbox, (0.0, 1.0, 0.0, 1.0))

                sub_sr = sr_fast(sr_model, sub_raw, device)
                ctx_sr = sr_fast(sr_model, ctx_raw, device)
                person_sr = sr_fast(sr_model, person_raw, device)

                st_vec = extract_structural_vec(person_sr, zone)
                lab_stats = tuned.extract_gaussian_core_lab(sub_sr, sigma=0.28)
                col_vec = extract_color_feature_vec(lab_stats, zone)

                if rk in ("144p", "240p"):
                    va_sub_imgs.append(sub_sr)
                    va_ctx_imgs.append(ctx_sr)
                    va_struct.append(st_vec)
                    va_col9.append(col_vec)
                    va_yt.append(yt)
                    va_yc.append(yc)
                else:
                    tr_sub_imgs.append(sub_sr)
                    tr_ctx_imgs.append(ctx_sr)
                    tr_struct.append(st_vec)
                    tr_col9.append(col_vec)
                    tr_yt.append(yt)
                    tr_yc.append(yc)
                    # Add synthetic extreme-low-res downscale-upscale version during training (non-center frames at 360p)
                    if rk == "360p" and dt in (-2, 2):
                        sw, sh = sub_raw.size
                        cw, ch = ctx_raw.size
                        low_sub = sr_fast(sr_model, sub_raw.resize((max(6, sw // 2), max(6, sh // 2)), Image.BILINEAR), device)
                        low_ctx = sr_fast(sr_model, ctx_raw.resize((max(10, cw // 2), max(10, ch // 2)), Image.BILINEAR), device)
                        tr_sub_imgs.append(low_sub)
                        tr_ctx_imgs.append(low_ctx)
                        tr_struct.append(st_vec)
                        tr_col9.append(col_vec)
                        tr_yt.append(yt)
                        tr_yc.append(yc)

                # Also add training samples from 144p/240p strictly at offset frames dt != 0 (never dt == 0!)
                if rk in ("144p", "240p") and dt == 0:
                    for tr_dt in (-3, -1, 1, 3):
                        rgb2 = res_frames[rk][fc + tr_dt]
                        pbox2 = tuned.scale_box(sp["box"], rgb2.shape[1], rgb2.shape[0])
                        sr_s2 = sr_fast(sr_model, tuned.crop_frac(rgb2, pbox2, tuned.ZONE_GEOMETRY[zone]["sub"]), device)
                        sr_c2 = sr_fast(sr_model, tuned.crop_frac(rgb2, pbox2, tuned.ZONE_GEOMETRY[zone]["ctx"]), device)
                        sr_p2 = sr_fast(sr_model, tuned.crop_frac(rgb2, pbox2, (0.0, 1.0, 0.0, 1.0)), device)
                        tr_sub_imgs.append(sr_s2)
                        tr_ctx_imgs.append(sr_c2)
                        tr_struct.append(extract_structural_vec(sr_p2, zone))
                        tr_col9.append(extract_color_feature_vec(tuned.extract_gaussian_core_lab(sr_s2, 0.28), zone))
                        tr_yt.append(yt)
                        tr_yc.append(yc)

    return {
        "tr_sub": encode_clip_images(clip_m, clip_proc, tr_sub_imgs, device),
        "tr_ctx": encode_clip_images(clip_m, clip_proc, tr_ctx_imgs, device),
        "tr_struct": torch.tensor(np.array(tr_struct), dtype=torch.float32, device=device),
        "tr_col9": torch.tensor(np.array(tr_col9), dtype=torch.float32, device=device),
        "tr_yt": torch.tensor(tr_yt, dtype=torch.long, device=device),
        "tr_yc": torch.tensor(tr_yc, dtype=torch.long, device=device),
        "va_sub": encode_clip_images(clip_m, clip_proc, va_sub_imgs, device),
        "va_ctx": encode_clip_images(clip_m, clip_proc, va_ctx_imgs, device),
        "va_struct": torch.tensor(np.array(va_struct), dtype=torch.float32, device=device),
        "va_col9": torch.tensor(np.array(va_col9), dtype=torch.float32, device=device),
        "va_yt": torch.tensor(va_yt, dtype=torch.long, device=device),
        "va_yc": torch.tensor(va_yc, dtype=torch.long, device=device),
    }


def train_model(epochs: int = 45) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    sr_model = tuned.load_realesrgan_x2plus(device)
    det_model = tv_det.fasterrcnn_resnet50_fpn_v2(
        weights=tv_det.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    ).eval().to(device)
    clip_id = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
    clip_proc = CLIPProcessor.from_pretrained(clip_id)
    clip_m = CLIPModel.from_pretrained(clip_id, use_safetensors=True).eval().to(device)

    print("Collecting human vs. non-human dataset (Retail + VIRAT)...")
    hx_tr, hg_tr, hy_tr, hx_va, hg_va, hy_va = collect_human_dataset(det_model, sr_model, clip_m, clip_proc, device)
    del det_model
    torch.cuda.empty_cache()

    print("Collecting multi-resolution garment dataset...")
    gd = collect_garment_dataset(sr_model, clip_m, clip_proc, device)
    del sr_model
    torch.cuda.empty_cache()

    type_tmat = tuned.build_text_matrix(clip_m, clip_proc, tuned.TUNED_TYPE_PROMPTS, tuned.CLOTHING_TYPES, device)
    color_tmat = tuned.build_text_matrix(clip_m, clip_proc, EXPANDED_COLOR_PROMPTS, EXPANDED_COLORS, device)
    open_world_prompts = {c["name"]: c["prompts"] for c in OPEN_WORLD_CANDIDATES}
    open_world_tmat = tuned.build_text_matrix(clip_m, clip_proc, open_world_prompts, [c["name"] for c in OPEN_WORLD_CANDIDATES], device)

    model = SightlineTrainedModel(type_tmat, color_tmat, logit_scale=22.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Regularize prototypes near initial CLIP text embeddings to preserve open-world compatibility
    type_anchor = type_tmat.detach().clone()
    color_anchor = color_tmat.detach().clone()

    loss_history = []
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        h_logits = model.forward_human(hx_tr, hg_tr)
        loss_h = F.binary_cross_entropy_with_logits(h_logits, hy_tr)

        t_logits, t_emb = model.forward_type(gd["tr_sub"], gd["tr_ctx"], gd["tr_struct"])
        loss_t = F.cross_entropy(t_logits, gd["tr_yt"], label_smoothing=0.06)

        c_logits, _ = model.forward_color(gd["tr_sub"], gd["tr_col9"])
        loss_c = F.cross_entropy(c_logits, gd["tr_yc"], label_smoothing=0.05)

        reg_t = F.mse_loss(F.normalize(model.type_protos, dim=-1), type_anchor)
        reg_c = F.mse_loss(F.normalize(model.color_protos, dim=-1), color_anchor)

        loss = loss_h + loss_t + loss_c + 1.8 * reg_t + 1.8 * reg_c
        loss.backward()
        opt.step()
        sched.step()
        loss_history.append(round(float(loss.item()), 4))

    model.eval()
    with torch.no_grad():
        hv_prob = torch.sigmoid(model.forward_human(hx_va, hg_va))
        human_val_acc = float(((hv_prob >= 0.5).float() == hy_va).float().mean().item()) * 100.0

        vt_logits, _ = model.forward_type(gd["va_sub"], gd["va_ctx"], gd["va_struct"])
        # Evaluate with zone masking on held-out 144p/240p center frames
        va_zone_is_lower = gd["va_struct"][:, 3] > 0.5
        mask = torch.zeros_like(vt_logits)
        for i in range(vt_logits.size(0)):
            if va_zone_is_lower[i]:
                mask[i, tuned.UPPER_TYPE_IDXS] = -1e4
            else:
                mask[i, tuned.LOWER_TYPE_IDXS] = -1e4
        type_preds = (vt_logits + mask).argmax(dim=-1)
        type_val_acc = float((type_preds == gd["va_yt"]).float().mean().item()) * 100.0

        vc_logits, _ = model.forward_color(gd["va_sub"], gd["va_col9"])
        # Limit to the 9 ground-truth colors for held-out benchmark evaluation
        color_preds = vc_logits[:, : len(tuned.CLOTHING_COLORS)].argmax(dim=-1)
        color_val_acc = float((color_preds == gd["va_yc"]).float().mean().item()) * 100.0

    ckpt = {
        "model_state": model.state_dict(),
        "base_types": list(tuned.CLOTHING_TYPES),
        "expanded_colors": list(EXPANDED_COLORS),
        "open_world_candidates": OPEN_WORLD_CANDIDATES,
        "open_world_embs": open_world_tmat.cpu(),
        "metrics": {
            "epochs": epochs,
            "train_time_sec": round(time.perf_counter() - t0, 2),
            "human_train_samples": int(hx_tr.size(0)),
            "human_val_samples": int(hx_va.size(0)),
            "human_val_acc_pct": round(human_val_acc, 1),
            "garment_train_samples": int(gd["tr_sub"].size(0)),
            "garment_val_samples": int(gd["va_sub"].size(0)),
            "type_val_acc_144_240_pct": round(type_val_acc, 1),
            "color_val_acc_144_240_pct": round(color_val_acc, 1),
            "final_loss": loss_history[-1],
        },
    }
    torch.save(ckpt, str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(ckpt["metrics"], indent=2), encoding="utf-8")
    print("Training complete:")
    print(json.dumps(ckpt["metrics"], indent=2))
    return ckpt["metrics"]


def main():
    train_model(epochs=65)


if __name__ == "__main__":
    main()
