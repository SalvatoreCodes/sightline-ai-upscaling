"""COCO-17 pose-guided garment ROI alignment and anatomical silhouette mask helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(r"D:\projects\sightline-ai-upscaling")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_cctv_clothing_tuning_report as tuned  # noqa: E402


def compute_pose_stats(kpts: np.ndarray | None, conf_thr: float = 0.35) -> tuple[float, float]:
    """Return (visible_joint_fraction, has_torso_keypoints_flag) from COCO-17 keypoints."""
    if kpts is None or kpts.shape[0] < 17:
        return 0.0, 0.0
    vis = kpts[:, 2] >= conf_thr
    vis_frac = float(np.mean(vis))
    has_sh = bool(vis[5] or vis[6])
    has_hip = bool(vis[11] or vis[12])
    return vis_frac, 1.0 if (has_sh and has_hip) else 0.0


def compute_pose_zone_fracs(
    pbox: tuple[float, float, float, float],
    kpts: np.ndarray | None,
    conf_thr: float = 0.35,
) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    """Compute pose-aligned normalized crop fractions blended with stable anatomical defaults."""
    defaults = {
        "upper": {
            "sub": tuned.ZONE_GEOMETRY["upper"]["sub"],
            "ctx": tuned.ZONE_GEOMETRY["upper"]["ctx"],
        },
        "outer_upper": {
            "sub": (0.18, 0.82, 0.16, 0.54),
            "left": (0.18, 0.36, 0.16, 0.54),
            "right": (0.64, 0.82, 0.16, 0.54),
            "ctx": tuned.ZONE_GEOMETRY["outer_upper"]["ctx"],
        },
        "inner_upper": {
            "sub": tuned.ZONE_GEOMETRY["inner_upper"]["sub"],
            "ctx": tuned.ZONE_GEOMETRY["inner_upper"]["ctx"],
        },
        "lower": {
            "sub": tuned.ZONE_GEOMETRY["lower"]["sub"],
            "ctx": tuned.ZONE_GEOMETRY["lower"]["ctx"],
        },
    }
    if kpts is None or kpts.shape[0] < 17:
        return defaults

    bx1, by1, bx2, by2 = pbox
    bw = max(1.0, float(bx2 - bx1))
    bh = max(1.0, float(by2 - by1))

    def pts(indices: list[int]) -> list[tuple[float, float]]:
        out = []
        for idx in indices:
            if kpts[idx, 2] >= conf_thr:
                nx = float(np.clip((kpts[idx, 0] - bx1) / bw, 0.02, 0.98))
                ny = float(np.clip((kpts[idx, 1] - by1) / bh, 0.02, 0.98))
                out.append((nx, ny))
        return out

    sh = pts([5, 6])
    hips = pts([11, 12])
    knees = pts([13, 14])
    ankles = pts([15, 16])

    if not sh and not hips:
        return defaults

    y_sh_pose = float(np.clip(np.mean([p[1] for p in sh]) - 0.02, 0.06, 0.38)) if sh else 0.16
    y_hip_pose = float(np.clip(np.mean([p[1] for p in hips]) + 0.01, 0.35, 0.72)) if hips else 0.54
    if y_hip_pose < y_sh_pose + 0.16:
        y_hip_pose = min(0.75, y_sh_pose + 0.24)

    if ankles:
        y_low_end_pose = float(np.clip(max(p[1] for p in ankles) + 0.03, y_hip_pose + 0.18, 0.96))
    elif knees:
        y_low_end_pose = float(np.clip(max(p[1] for p in knees) + 0.16, y_hip_pose + 0.20, 0.96))
    else:
        y_low_end_pose = 0.94

    torso_pts = sh + hips
    if len(torso_pts) >= 2:
        txs = [p[0] for p in torso_pts]
        tx_c = 0.5 * (min(txs) + max(txs))
        span = max(0.36, (max(txs) - min(txs)) * 1.48)
        ux0_pose = float(np.clip(tx_c - 0.5 * span, 0.06, 0.45))
        ux1_pose = float(np.clip(tx_c + 0.5 * span, max(ux0_pose + 0.34, 0.55), 0.94))
    else:
        ux0_pose, ux1_pose = 0.14, 0.86

    leg_pts = hips + knees + ankles
    if len(leg_pts) >= 2:
        lxs = [p[0] for p in leg_pts]
        lx_c = 0.5 * (min(lxs) + max(lxs))
        lspan = max(0.34, (max(lxs) - min(lxs)) * 1.45)
        lx0_pose = float(np.clip(lx_c - 0.5 * lspan, 0.08, 0.45))
        lx1_pose = float(np.clip(lx_c + 0.5 * lspan, max(lx0_pose + 0.34, 0.55), 0.92))
    else:
        lx0_pose, lx1_pose = 0.18, 0.82

    def blend(p_val: float, d_val: float, alpha: float = 0.70) -> float:
        return round(alpha * p_val + (1.0 - alpha) * d_val, 4)

    ux0 = blend(ux0_pose, 0.14)
    ux1 = blend(ux1_pose, 0.86)
    uy0 = blend(y_sh_pose, 0.14)
    uy1 = blend(y_hip_pose, 0.56)
    uw = max(0.24, ux1 - ux0)

    lx0 = blend(lx0_pose, 0.18)
    lx1 = blend(lx1_pose, 0.82)
    ly0 = blend(max(0.34, y_hip_pose - 0.02), 0.52)
    ly1 = blend(y_low_end_pose, 0.94)

    def make_ctx(x0: float, x1: float, y0: float, y1: float) -> tuple[float, float, float, float]:
        return (
            round(max(0.0, x0 - 0.14), 4),
            round(min(1.0, x1 + 0.14), 4),
            round(max(0.02, y0 - 0.10), 4),
            round(min(1.0, y1 + 0.12), 4),
        )

    out_y0 = round(uy0 + 0.02, 4)
    out_y1 = round(max(out_y0 + 0.14, uy1 - 0.02), 4)
    left_x1 = round(ux0 + 0.28 * uw, 4)
    right_x0 = round(ux1 - 0.28 * uw, 4)
    in_x0 = round(ux0 + 0.28 * uw, 4)
    in_x1 = round(max(in_x0 + 0.18, ux1 - 0.28 * uw), 4)

    return {
        "upper": {"sub": (ux0, ux1, uy0, uy1), "ctx": make_ctx(ux0, ux1, uy0, uy1)},
        "outer_upper": {
            "sub": (ux0, ux1, out_y0, out_y1),
            "left": (ux0, left_x1, out_y0, out_y1),
            "right": (right_x0, ux1, out_y0, out_y1),
            "ctx": make_ctx(ux0, ux1, out_y0, out_y1),
        },
        "inner_upper": {"sub": (in_x0, in_x1, out_y0, out_y1), "ctx": make_ctx(in_x0, in_x1, out_y0, out_y1)},
        "lower": {"sub": (lx0, lx1, ly0, ly1), "ctx": make_ctx(lx0, lx1, ly0, ly1)},
    }


def build_pose_silhouette_mask(
    pbox: tuple[float, float, float, float],
    kpts: np.ndarray | None,
    conf_thr: float = 0.30,
) -> np.ndarray | None:
    """Rasterize an anatomical foreground human mask from COCO-17 skeleton joints."""
    if kpts is None or kpts.shape[0] < 17:
        return None
    bx1, by1, bx2, by2 = pbox
    bw = max(1.0, float(bx2 - bx1))
    bh = max(1.0, float(by2 - by1))

    mw, mh = 64, 128
    mask = np.zeros((mh, mw), dtype=np.float32)

    def to_px(idx: int) -> tuple[int, int] | None:
        if kpts[idx, 2] < conf_thr:
            return None
        px = int(round(np.clip((kpts[idx, 0] - bx1) / bw, 0.0, 1.0) * (mw - 1)))
        py = int(round(np.clip((kpts[idx, 1] - by1) / bh, 0.0, 1.0) * (mh - 1)))
        return (px, py)

    torso_pts = [p for p in (to_px(5), to_px(6), to_px(12), to_px(11)) if p is not None]
    if len(torso_pts) >= 3:
        cv2.fillConvexPoly(mask, np.array(torso_pts, dtype=np.int32), 1.0)
        # Widen torso fill slightly to cover jacket/shirt sleeves and waist width
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    limbs = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (5, 11), (6, 12),
    ]
    any_drawn = len(torso_pts) >= 3
    for a, b in limbs:
        pa, pb = to_px(a), to_px(b)
        if pa is not None and pb is not None:
            cv2.line(mask, pa, pb, color=1.0, thickness=8)
            any_drawn = True

    if not any_drawn:
        return None
    return cv2.GaussianBlur(mask, (5, 5), sigmaX=1.5).clip(0.0, 1.0)
