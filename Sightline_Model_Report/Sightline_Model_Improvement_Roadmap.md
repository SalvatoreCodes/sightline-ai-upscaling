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

## 2. Immediate Pipeline & UI Improvements (Software Only — No Extra Datasets Required)

1. **FPS-Adaptive Dense Sampling + Continuous Track Interpolation (`ByteTrack` / IoU Tracker):**
   - Replace fixed `stride=24` frames with time-normalized sampling (`stride = max(3, round(fps * 0.4))`, e.g., every `5` frames at `12 FPS` or `~0.4s` intervals).
   - Linearly interpolate person and garment boxes across intermediate frames between matched tracklets so bounding boxes follow walking subjects smoothly without `2s` gaps.
2. **Foreground Human Silhouette Masking Before Color Extraction:**
   - Replace rectangular `crop_frac` averaging inside `extract_gaussian_core_lab()` with a lightweight pixel-level foreground mask (`Mask R-CNN` or GrabCut / saliency matting).
   - Excluding pavement, grass, and metal railings outside the human silhouette eliminates gray-asphalt color contamination on `26–60 px` pedestrians.
3. **Adaptive Small-Pedestrian Upscaling (`< 64 px` Height Gate):**
   - Trigger `2× RealESRGAN_x2plus` upscaling whenever any detected person bounding box is `< 64 px` tall—even when the full video resolution is `432p` or `720p` (`Intel_CCTV_Person_Bicycle_Car`, `VIRAT_0100`).
   - Apply scale-aware attribute regularization on overhead crops (`< 64 px`) so niche indoor/specialty labels (`Tank Top`, `Dress / Skirt`) require high CLIP margin before overriding standard jackets, shirts, and trousers.
4. **Always-On Live Overlay & Close-Match Fallback (`Implemented`):**
   - **Implemented in `indexer.py` & `index.html`:** When a search yields `0` exact matches, `ClothingSearchDB.search()` now ranks perceptual color/garment-family neighbors (`close_matches`) and surfaces *"No exact match found — you might be thinking of this:"* in the UI.
   - Next UI enhancement: render all humans present at `videoPlayer.currentTime` in dimmed slate boxes even while a specific query highlights its target matches in cyan.

---

## 3. Data & Backbone Model Upgrades Needed for Next-Generation Accuracy

| Upgrade Axis | Current Stack | Recommended Replacement / Addition | Why It Solves Overhead & Low-Quality Failure Modes |
|---|---|---|---|
| **Surveillance Attribute Training Data** | Retail Store (`12` curated garments) + VIRAT (`person/non-person`) | **PA-100K** (`100k` outdoor surveillance crops), **PETA** (`19k` multi-angle CCTV crops), **RAPv2** (`84.9k` indoor/outdoor CCTV crops), **Market-1501** | Teaches the classifier foreshortened top-down viewing angles where torso, shoulders, and legs overlap vertically. |
| **Person Detector + Body Keypoints** | `FasterRCNN-ResNet50-FPN-V2` (box-only) | **`YOLO11m-pose`** / **`YOLOv8x-pose`** (`COCO 17 keypoints`) | Boosts recall on tiny distant pedestrians (`< 30 px` tall) and aligns upper/lower clothing crops directly to shoulder, hip, and knee joints instead of fixed box percentages. |
| **Vision-Language Encoder** | `TinyCLIP-ViT-61M-32` (`224×224`, `32×32` patch grid) | **`google/siglip-base-patch16-384`** or **`siglip-so400m-patch14-384`** (`384×384`, `14–16px` patches) | Increases spatial token density by up to **`7.5×`**, resolving subtle collars, jackets vs. sweaters, shorts, and reflective stripes at long CCTV distances. |
| **Color Constancy & Shadow Normalization** | Raw CIELAB `L*a*b*` Gaussian core | **Retinex / Shades-of-Gray white balance** + shadow-invariant chromaticity head | Stabilizes white/silver/beige vs. dark navy/black predictions between sunny construction sites and overcast grey asphalt parking lots. |
