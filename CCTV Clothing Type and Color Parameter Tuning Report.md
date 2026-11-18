# Parameter Tuning Report: CCTV Clothing Type & Color Classification at Lowest Video Quality (`720p` to `144p`)

**Dataset Tested:** 5 real retail-store CCTV footage qualities in `D:\Projects\affine-ai-upscaling\Assets` (`720p` `1270x720`, `480p` `846x480`, `360p` `634x360`, `240p` `424x240`, `144p` `254x144`; `13.09 FPS`, `1452 frames`).
**Local Hardware & Models:** NVIDIA GeForce RTX 4050 Laptop GPU (`6,141 MiB VRAM`) + `wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M` + `RealESRGAN_x2plus` + Gaussian `CIELAB/HSV` Fusion + `FasterRCNN-ResNet50-FPN-V2`.

---

## 1. Executive Summary — Minimum Quality Floor & Accuracy Across Resolutions

| Video Quality | 1. Untuned Baseline (Type / Color / **Joint**) | 2. + Sub-ROI (Type / Color) | 3. + 4x SR & Priors (Type / Color) | 4. Full Tuned Stack (**Type / Color / Joint**) | Person Recall (`Raw -> 2x SR`) |
|---|---|---|---|---|---|
| **720p** | 41.7% / 41.7% (**8.3%**) | 33.3% / 58.3% | 91.7% / 50.0% | **91.7% / 100.0% (91.7%)** | 100.0% -> **100.0%** |
| **480p** | 41.7% / 41.7% (**8.3%**) | 41.7% / 50.0% | 91.7% / 41.7% | **91.7% / 100.0% (91.7%)** | 100.0% -> **98.2%** |
| **360p** | 33.3% / 41.7% (**8.3%**) | 41.7% / 66.7% | 91.7% / 58.3% | **91.7% / 100.0% (91.7%)** | 94.6% -> **96.4%** |
| **240p** | 25.0% / 50.0% (**8.3%**) | 41.7% / 75.0% | 91.7% / 58.3% | **91.7% / 91.7% (83.3%)** | 82.1% -> **94.6%** |
| **144p** | 33.3% / 50.0% (**25.0%**) | 41.7% / 66.7% | 100.0% / 58.3% | **91.7% / 100.0% (91.7%)** | 58.9% -> **76.8%** |

- **Absolute Lowest Usable Floor (`144p`, `254x144`):** Tuning raises joint Clothing Type + Color accuracy from **25.0%** (`3/12`) to **91.7%** (`11/12`) — hitting **91.7% Type** and **100.0% Color** even on `11x29 px` to `28x30 px` sub-ROIs.
- **Recommended Production Minimum (`240p`, `424x240`):** Achieves **91.7% Type**, **91.7% Color**, and **83.3% Joint** while boosting all-scene shopper detection recall from **82.1%** to **94.6%**.

---

## 2. Tuned Parameter Settings for Low-Quality CCTV

| Parameter | Untuned Default | Tuned Optimal Value | Why It Matters at `144p`–`240p` |
|---|---|---|---|
| `roi_crop_mode` | `full_person` `[0..1]` | **Anatomical 4-Zone Sub-ROI** (`upper`, `outer_upper`, `inner_upper`, `lower`) | Strips carried bags, red checkout stands, wood floors, and neighboring shoppers from color/type embeddings. |
| `sr_mode` / `unsharp_amount` | `none` (`1x` raw) | **Adaptive `4x` `RealESRGAN_x2plus`** (`144p`/`240p`) + `Unsharp(35%)` | Restores shoulder contours, buttons/plackets, and denim weave before ViT `224x224` patch tokenization. |
| `zone_mask` & `prior_zone_weight` | `False` / `0.00` | **`True` / `0.45`** (hood ΔE, hem extension ΔE, placket gradient, denim chroma) | Eliminates impossible cross-zone labels (e.g. `pants` on upper torso) and separates coats from sweaters/hoodies. |
| `logit_debias_lambda` / `softmax_temp` | `0.00` / `1.0` | **`0.40` (centered null-blur) / `1.5`** | Cancels LAION prior dominance (`shirt/blouse` bias on blurry patches). |
| `color_lab_weight` & `center_sigma` | `0.00` (CLIP only) | **`0.68` Gaussian CIELAB/HSV (`σ=0.28`) + `0.32` CLIP** | Separates jet-black vs charcoal, light-grey vs warm beige vs off-white, and navy vs royal blue without chroma bleed. |
| `temporal_window_T` / `ema_alpha` | `1` / `1.00` | **`9` frames (`±4`) / `0.35` EMA** | Suppresses H.264 `4:2:0` macroblock chroma flicker across consecutive `13.09 FPS` frames. |
| `detector_conf` & `det_sr` | Raw `0.50` | **`2x RealESRGAN` + `conf=0.30`** | Lifts all-scene person recall at `144p` (`58.9% -> 76.8%`). |

---

## 3. Visual Inspection — Resolution Degradation & Gaussian Color Isolation

![Visual 1: 720p to 144p Resolution Degradation & Sub-ROI Tuning](CCTV_Clothing_Visual_1_Resolution_Degradation_720p_to_144p.png)

---

## 4. Per-Garment Scorecard Across All 12 Shopper Garments (`144p` & `240p`)

![Visual 2: 12-Garment Grid at 144p Untuned vs Tuned](CCTV_Clothing_Visual_2_12Garment_Grid_Untuned_vs_Tuned_144p.png)

| ID (Zone, `144p` Size) | Ground Truth | `144p` Untuned Baseline | `144p` Full Tuned (`T=9`) | `240p` Full Tuned (`T=9`) |
|---|---|---|---|---|
| **G01** (`upper`, `18x22`) | `puffer jacket / quilted coat` + **navy blue** | `puffer jacket / quilted coat` + `navy blue` | **PASS** `puffer jacket / quilted coat` (66.2%) + **navy blue** (62.7%) | **PASS** `puffer jacket / quilted coat` + **navy blue** |
| **G02** (`upper`, `15x16`) | `hoodie / hooded sweatshirt` + **white** | `jeans / denim pants` + `navy blue` | **PASS** `hoodie / hooded sweatshirt` (96.7%) + **white** (50.0%) | **PASS** `hoodie / hooded sweatshirt` + **white** |
| **G03** (`lower`, `14x12`) | `jeans / denim pants` + **blue** | `jeans / denim pants` + `navy blue` | **PASS** `jeans / denim pants` (97.4%) + **blue** (58.8%) | **PASS** `jeans / denim pants` + **blue** |
| **G04** (`upper`, `24x27`) | `sweater / sweatshirt` + **dark grey / charcoal** | `pants / trousers` + `black` | **PASS** `sweater / sweatshirt` (32.6%) + **dark grey / charcoal** (49.6%) | **PASS** `sweater / sweatshirt` + **dark grey / charcoal** |
| **G05** (`lower`, `22x22`) | `pants / trousers` + **black** | `pants / trousers` + `black` | **PASS** `pants / trousers` (96.5%) + **black** (45.9%) | **PASS** `pants / trousers` + **black** |
| **G06** (`upper`, `21x24`) | `jacket / sweater` + **dark grey / charcoal** | `sweater / sweatshirt` + `navy blue` | **PASS** `jacket / sweater` (91.8%) + **dark grey / charcoal** (38.3%) | PARTIAL `jacket / sweater` + **blue** |
| **G07** (`lower`, `20x19`) | `pants / trousers` + **beige / khaki** | `sweater / sweatshirt` + `navy blue` | **PASS** `pants / trousers` (94.8%) + **beige / khaki** (68.5%) | **PASS** `pants / trousers` + **beige / khaki** |
| **G08** (`outer_upper`, `9x26`) | `open jacket / coat` + **royal blue** | `shirt / blouse` + `pink / magenta` | **PASS** `open jacket / coat` (97.2%) + **royal blue** (85.4%) | **PASS** `open jacket / coat` + **royal blue** |
| **G09** (`inner_upper`, `12x23`) | `shirt / blouse` + **pink / magenta** | `shirt / blouse` + `pink / magenta` | **PASS** `shirt / blouse` (97.9%) + **pink / magenta** (97.8%) | **PASS** `shirt / blouse` + **pink / magenta** |
| **G10** (`upper`, `27x24`) | `long coat / overcoat` + **white** | `puffer jacket / quilted coat` + `white` | **PASS** `long coat / overcoat` (95.9%) + **white** (76.4%) | **PASS** `long coat / overcoat` + **white** |
| **G11** (`upper`, `12x20`) | `sweater / sweatshirt` + **light grey** | `pants / trousers` + `light grey` | COLOR ONLY `hoodie / hooded sweatshirt` (36.7%) + **light grey** (54.2%) | PARTIAL `hoodie / hooded sweatshirt` + **light grey** |
| **G12** (`upper`, `14x19`) | `buttoned coat / jacket` + **beige / khaki** | `jacket / sweater` + `beige / khaki` | **PASS** `buttoned coat / jacket` (96.9%) + **beige / khaki** (79.0%) | **PASS** `buttoned coat / jacket` + **beige / khaki** |

---

## 5. Ablation Curves, Person Recall & RTX 4050 Compute Profile

![Visual 3: Parameter Tuning Ablation and Quality Floor](CCTV_Clothing_Visual_3_Parameter_Tuning_Ablation_And_Quality_Floor.png)

- **`RealESRGAN_x2plus` (`4x` cascaded on `144p`/`240p` ROI):** `134.2 ms/ROI` (`77 MiB` isolated VRAM)
- **`TinyCLIP-ViT-61M-32` (`115.4M`):** `27.1 ms` per multi-crop garment pass (`456 MiB` isolated VRAM)
- **Full Tuned Stack Amortized Cost:** **`298.9 ms/frame`** per active garment tracklet (`~265 ms` keyframe `2x-SR FasterRCNN` at `144p`)