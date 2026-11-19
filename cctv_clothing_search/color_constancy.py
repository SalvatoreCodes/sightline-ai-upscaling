"""Scene-level Shades-of-Gray white balance, Retinex shadow stabilization, and 16-D chromaticity features."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT_DIR = Path(r"D:\projects\sightline-ai-upscaling")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_cctv_clothing_tuning_report as tuned  # noqa: E402


def crop_mask_frac(person_mask: np.ndarray, fracs: tuple[float, float, float, float]) -> np.ndarray:
    """Crop normalized sub-region (x0, x1, y0, y1) from a person foreground mask."""
    mh, mw = person_mask.shape[:2]
    x0, x1, y0, y1 = fracs
    c0 = max(0, min(mw - 2, int(round(mw * x0))))
    c1 = max(c0 + 2, min(mw, int(round(mw * x1))))
    r0 = max(0, min(mh - 2, int(round(mh * y0))))
    r1 = max(r0 + 2, min(mh, int(round(mh * y1))))
    return person_mask[r0:r1, c0:c1]


def extract_flank_mask_composite(
    person_mask: np.ndarray,
    left_frac: tuple[float, float, float, float] = (0.18, 0.36, 0.16, 0.54),
    right_frac: tuple[float, float, float, float] = (0.64, 0.82, 0.16, 0.54),
) -> np.ndarray:
    """Stitch left and right torso flank foreground masks vertically aligned."""
    left = crop_mask_frac(person_mask, left_frac)
    right = crop_mask_frac(person_mask, right_frac)
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


def compute_frame_illuminant_gains(
    rgb: np.ndarray, p: float = 6.0, strength: float = 0.55
) -> dict:
    """Estimate Minkowski p-norm Shades-of-Gray illuminant gains and ambient lightness on the full scene frame."""
    H, W = rgb.shape[:2]
    scale = min(1.0, 192.0 / max(1, max(H, W)))
    if scale < 1.0:
        sw, sh = max(16, int(round(W * scale))), max(16, int(round(H * scale)))
        small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small = rgb

    arr = small.astype(np.float32) / 255.0
    lum = arr.mean(axis=2)
    valid = (lum > 0.04) & (lum < 0.96)
    pix = arr[valid] if int(valid.sum()) >= 32 else arr.reshape(-1, 3)

    # Finlayson-Trezzi Shades-of-Gray Minkowski p-norm per RGB channel
    eps = 1e-4
    norms = np.power(np.mean(np.power(np.clip(pix, eps, 1.0), p), axis=0), 1.0 / p)
    gray_ref = float(np.mean(norms))
    raw_gains = gray_ref / np.maximum(norms, eps)
    # Clamp and blend toward neutral to prevent extreme color shifts on single-color scenes
    clamped = np.clip(raw_gains, 0.75, 1.35)
    gains = 1.0 + strength * (clamped - 1.0)
    gains = np.clip(gains, 0.75, 1.35).astype(np.float32)

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    frame_L50 = float(np.percentile(L, 50))
    frame_L80 = float(np.percentile(L, 80))

    return {
        "gains": gains,
        "frame_L50": frame_L50,
        "frame_L80": frame_L80,
    }


def apply_color_constancy(
    rgb: np.ndarray,
    frame_stats: dict | None,
    retinex_strength: float = 0.22,
) -> np.ndarray:
    """Apply full-frame Shades-of-Gray white balance + shadow-safe Multi-Scale Retinex chroma stabilization."""
    if frame_stats is None:
        return rgb

    gains = np.asarray(frame_stats["gains"], dtype=np.float32).reshape(1, 1, 3)
    wb = np.clip(rgb.astype(np.float32) * gains, 0.0, 255.0)

    if retinex_strength <= 0.0:
        return wb.round().astype(np.uint8)

    # Soft multi-scale illumination map on the greyscale scene (downsample large blurs if big)
    gray = (0.299 * wb[:, :, 0] + 0.587 * wb[:, :, 1] + 0.114 * wb[:, :, 2]).astype(np.float32)
    blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=7.0)
    blur2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=21.0)
    illum = 0.5 * blur1 + 0.5 * blur2

    frame_L80_lum = max(60.0, float(frame_stats.get("frame_L80", 65.0)) * 2.55)
    ratio = frame_L80_lum / np.maximum(illum, 18.0)
    # Lift shadows gently while fading out above midtones (gray > 140) so bright tan/beige never blows out to white
    shadow_weight = (
        np.clip((gray - 22.0) / 36.0, 0.0, 1.0)
        * np.clip((140.0 - gray) / 50.0, 0.0, 1.0)
        * retinex_strength
    )
    gain_field = 1.0 + shadow_weight * np.clip(ratio - 1.0, 0.0, 0.85)
    # Cap total per-pixel luminance multiplier to protect black/charcoal garments from washing to grey
    gain_field = np.clip(gain_field, 1.0, 1.24)[:, :, None]

    out = np.clip(wb * gain_field, 0.0, 255.0)
    return out.round().astype(np.uint8)


def extract_shadow_invariant_color_vec(
    roi_img: Image.Image,
    fg_mask: np.ndarray | None,
    lab_stats: dict,
    frame_stats: dict | None,
    zone: str,
) -> np.ndarray:
    """Compute 16-D shadow-invariant CIELAB + log-chromaticity + scene-relative color feature vector."""
    arr = np.asarray(roi_img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]

    w_map = lab_stats.get("w_map")
    if isinstance(w_map, np.ndarray) and w_map.shape == (h, w):
        weights = w_map.astype(np.float32)
    elif fg_mask is not None and fg_mask.size > 0:
        m = cv2.resize(fg_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        weights = np.clip(m, 0.05, 1.0)
    else:
        weights = np.ones((h, w), dtype=np.float32)
    weights = weights / (float(weights.sum()) + 1e-6)

    arr01 = arr / 255.0
    eps = 0.015
    R = arr01[:, :, 0]
    G = arr01[:, :, 1]
    B = arr01[:, :, 2]

    log_rg = float((np.log((R + eps) / (G + eps)) * weights).sum())
    log_bg = float((np.log((B + eps) / (G + eps)) * weights).sum())

    s_rgb = R + G + B + 1e-3
    rg_opp = float((((R - G) / s_rgb) * weights).sum())
    yb_opp = float((((0.5 * (R + G) - B) / s_rgb) * weights).sum())

    lab_pix = cv2.cvtColor(arr.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L_pix = lab_pix[:, :, 0] * (100.0 / 255.0)
    mean_L = float((L_pix * weights).sum())
    std_L = float(np.sqrt(np.maximum(0.0, (((L_pix - mean_L) ** 2) * weights).sum())))

    L50 = float(lab_stats["L50"])
    L78 = float(lab_stats["L78"])
    A_val = float(lab_stats["A"])
    B_val = float(lab_stats["B"])
    C_val = float(lab_stats["C"])
    H_rad = float(lab_stats["H"]) * (np.pi / 90.0)
    S_val = float(lab_stats["S"])

    frame_L80 = float(frame_stats.get("frame_L80", 65.0)) if frame_stats else 65.0
    L_rel = float(np.clip(L50 / max(28.0, frame_L80), 0.0, 1.5))
    retinex_L_ratio = float(np.clip(L78 / max(15.0, L50), 0.5, 2.0))

    return np.asarray(
        [
            L50 / 100.0,
            L78 / 100.0,
            L_rel,
            A_val / 80.0,
            B_val / 80.0,
            C_val / 90.0,
            float(np.sin(H_rad)),
            float(np.cos(H_rad)),
            S_val / 255.0,
            float(np.clip(log_rg, -2.0, 2.0)),
            float(np.clip(log_bg, -2.0, 2.0)),
            float(np.clip(rg_opp, -1.0, 1.0)),
            float(np.clip(yb_opp, -1.0, 1.0)),
            retinex_L_ratio,
            min(2.0, std_L / 40.0),
            1.0 if zone == "lower" else 0.0,
        ],
        dtype=np.float32,
    )
