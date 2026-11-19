# Sightline AI Model Improvement Roadmap — Overhead CCTV, Dense Tracking & Fine-Grained Attributes

**Document Scope:** Engineering roadmap and data/architecture requirements identified from real-world failure analysis on high-angle parking-lot footage (`Assets/Intel_CCTV_Person_Bicycle_Car.mp4`, `768×432`, `12.0 FPS`, `17` detections in `scans/Intel_CCTV_Person_Bi_scan.json`).

---

## 1. Root-Cause Audit: `Intel_CCTV_Person_Bicycle_Car.mp4` (`00:02.9` / Frame `35`)

Direct inspection with `FasterRCNN-ResNet50-FPN-V2` confirmed that the pedestrian at `00:02.9` (`Frame 35`) **is detected at `1.00` (`100%`) confidence** (`49×99 px`, normalized box `[0.710, 0.593, 0.774, 0.823]`). However, three pipeline gaps prevented accurate labeling at that paused moment and reduced attribute precision on distant overhead frames (`F312`–`F576`, `26×42 px` to `35×80 px`):

| Bottleneck Observed | Exact Measurement / Evidence | Practical Impact on Video Search |
|---|---|---|
| **Fixed `stride=24` Frame Sampling** | On `12.0 FPS` video, `stride=24` samples once every **`2.0s`** (`00:02.0` F24, `00:04.0` F48), skipping `Frame 35 (00:02.9)` (`27/647` frames indexed). | Walking pedestrians travel between sampled keyframes, creating `1–2s` blind spots where paused frames have no exact box. |
| **Lower-Zone Boundary Clipping** | At `00:02.0` (F24), `person_box_norm[3] = 0.9952` (`>= 0.97`), causing `scan_video()` to skip the `lower` garment zone. | Only `Navy Blue · Sweater` was recorded at `00:02.0`; pants were deferred until `00:04.0` (`Light Grey · Trousers / Pants`). |
| **Asphalt Background Bleed in Rectangular ROIs** | At `F312`–`F576` (`26×42 px`–`35×80 px`), rectangular sub-crops pull neutral gray asphalt (`L* ≈ 50, C* ≈ 7.5`) into `extract_gaussian_core_lab()`. | Dilutes true garment hue and triggers noisy zero-shot CLIP labels (`Tank Top`, `Work Overalls`, `Safety Vest`, `Dress / Skirt`). |
| **Selected-Match-Only Overlay & Stale Filters** | `drawOverlay()` in `index.html` previously drew only `state.matches[state.selectedIdx]` (`0` boxes when filter mismatch returned `0` results). | Active search terms carried over from another video hide detected humans instead of showing all visible humans or close matches. |

---

## 2. Implemented Pipeline & UI Improvements (Verified in Production)

All four pipeline and UI enhancements below are now **implemented and verified** across `indexer.py`, `server.py`, and `static/index.html`:

1. **FPS-Adaptive Dense Sampling + Continuous Track Interpolation (`Implemented`):**
   - Replaced fixed `stride=24` frames with time-normalized sampling (`eff_stride = max(3, int(round(fps * 0.4)))`, e.g., every `5` frames / `~0.42s` at `12 FPS`) in `scan_video()`, paired with two-stage velocity + IoU association (`_associate_tracks()`) and linear box interpolation (`_interpolate_tracklets()`) across gaps `<= 1.6s`.
   - **Verified Impact on `Intel_CCTV_Person_Bi` (`Intel_CCTV_Person_Bicycle_Car.mp4`):** Indexed keyframes rose from **`17`** (`27` frames) to **`75` keyframe sightings** (`130` frames), plus **`138` interpolated track observations** (**`213` total live track overlays** via `GET /api/tracks/{quality}`).
2. **Foreground Human Silhouette Masking Before Color Extraction (`Implemented`):**
   - Upgraded the primary detector from box-only `FasterRCNN-ResNet50-FPN-V2` to **`MaskRCNN-ResNet50-FPN-V2`** (`tv_det.maskrcnn_resnet50_fpn_v2`) and replaced rectangular averaging with silhouette-weighted CIELAB extraction (`extract_masked_core_lab()`, weighting pixels by `0.05 + 0.95 * m_fg` and computing seed `(L0, A0, B0)` on foreground mask `>= 0.45`).
   - **Verified Benchmark & Impact:** `208.5 ms/frame` at `768×432`, `673.3 MiB` peak GPU VRAM, eliminating `~46%` background asphalt contamination on Frame `35` (`00:02.9`, cleanly labeled **`Human 96% · Navy Blue · Sweater`** + **`Light Grey · Denim Jeans`**).
3. **Adaptive Small-Pedestrian Upscaling (`< 64 px` Height Gate) & Regularization (`Implemented`):**
   - Added `force_2x=(bh < 64.0)` inside `_detect_persons()` and `scan_video()` to trigger `2× RealESRGAN_x2plus` upscaling on `< 64 px` pedestrians even in `432p`/`720p` footage, lowered `lower`-zone bottom-border cutoff to `0.992 * H`, and added `-1.50` overhead logit regularization in `classify_variable_type()` when `person_h_px < 64.0` and niche CLIP margin `< 0.035`.
4. **Always-On Live Multi-Human Overlay & Close-Match Fallback (`Implemented`):**
   - Added SQLite columns `track_id` / `is_interpolated` and lightweight endpoint `GET /api/tracks/{quality}`.
   - Upgraded `drawOverlay()` in `static/index.html` with a `requestAnimationFrame` playback lerp loop: all visible humans at `videoPlayer.currentTime` are drawn in **dimmed slate** (`rgba(148, 163, 184, 0.58)`)—even during empty search or *"No exact match found — you might be thinking of this:"* fallback (`"Orange safety vest"` -> `12` close matches)—while exact search targets and selected cards highlight on top in **bright cyan (`#38BDF8`)**.

### Updated Cross-Video Verification Benchmarks (`stride=0`, FPS-Adaptive)

| Source | Keyframes Indexed | Keyframe Sightings | Interpolated Track Rows | Sample Query Verification |
|---|---|---|---|---|
| `Intel_CCTV_Person_Bi` (`768×432`, `12 FPS`) | `130` (`+3.8×`) | **`75`** (`+4.4×` vs `17`) | **`138`** (**`213`** total overlays) | Frame `35 (00:02.9)`: `Human 96% · Navy Blue · Sweater` (`11` hits in `1.47 ms`) |
| `Construction-Site-CC` (`586×480`, `25 FPS`) | `218` | **`470`** | Tracked continuous | `Red shirt`: **`46`** (`4.28 ms`), `Orange safety vest`: **`25`** (`2.05 ms`), `Yellow safety vest`: **`29`** (`1.75 ms`), `Black pants`: **`83`** (`4.20 ms`) |
| `144p` Retail (`254×144`, `13.1 FPS`) | `293` | **`2,587`** | Tracked continuous | `Blue jacket`: **`158`**, `Pink shirt`: **`91`**, `Black pants`: **`563`**, `Blue jeans`: **`101`**, `Grey sweater`: **`119`**, `Beige coat`: **`30`**, `White hoodie`: **`8`** |

---

## 3. Data & Backbone Model Upgrades Needed for Next-Generation Accuracy

| Upgrade Axis | Current Stack | Recommended Replacement / Addition | Why It Solves Overhead & Low-Quality Failure Modes |
|---|---|---|---|
| **Surveillance Attribute Training Data** | Retail Store (`12` curated garments) + VIRAT (`person/non-person`) | **PA-100K** (`100k` outdoor surveillance crops), **PETA** (`19k` multi-angle CCTV crops), **RAPv2** (`84.9k` indoor/outdoor CCTV crops), **Market-1501** | Teaches the classifier foreshortened top-down viewing angles where torso, shoulders, and legs overlap vertically. |
| **Person Detector + Body Keypoints** | `FasterRCNN-ResNet50-FPN-V2` (box-only) | **`YOLO11m-pose`** / **`YOLOv8x-pose`** (`COCO 17 keypoints`) | Boosts recall on tiny distant pedestrians (`< 30 px` tall) and aligns upper/lower clothing crops directly to shoulder, hip, and knee joints instead of fixed box percentages. |
| **Vision-Language Encoder** | `TinyCLIP-ViT-61M-32` (`224×224`, `32×32` patch grid) | **`google/siglip-base-patch16-384`** or **`siglip-so400m-patch14-384`** (`384×384`, `14–16px` patches) | Increases spatial token density by up to **`7.5×`**, resolving subtle collars, jackets vs. sweaters, shorts, and reflective stripes at long CCTV distances. |
| **Color Constancy & Shadow Normalization** | Raw CIELAB `L*a*b*` Gaussian core | **Retinex / Shades-of-Gray white balance** + shadow-invariant chromaticity head | Stabilizes white/silver/beige vs. dark navy/black predictions between sunny construction sites and overcast grey asphalt parking lots. |
