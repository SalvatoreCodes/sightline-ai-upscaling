# KEEP Architecture — CCTV Footage Upscaling Test Report

**Model Tested:** Official `jnjaby/KEEP` (*Kalman-Inspired Feature Propagation for Video Face Super-Resolution*, ECCV 2024, checkpoint `KEEP-b76feb75.pth`, 634.1 MB) paired with `RealESRGAN_x2plus.pth` (67.1 MB), `detection_mobilenet0.25_Final.pth` (1.79 MB), and `parsing_parsenet.pth` (85.3 MB)
**Test Hardware:** Local Windows laptop — **NVIDIA GeForce RTX 4050 Laptop GPU** (6,141 MiB VRAM)
**Pipeline Settings:** FP32 precision, `--max_length 8` temporal chunks, `-s 2` background scale (`--bg_upsampler realesrgan --face_upsample --draw_box`), 16-frame clips

---

## 1. Executive Summary

We installed and executed the official **KEEP** repository locally against three 16-frame surveillance-style test clips spanning medium-distance CCTV (`~54 px` face height, H.264 CRF 34), real compressed low-quality handheld/field video (`390x512`), and extreme-distance parking-lot CCTV (`~24 px` face height, H.264 CRF 37).

**Bottom Line:**
- **Where KEEP shines (`~50+ px` face height):** On both `CAM-01 LOBBY` (`~54 px` face) and real compressed low-resolution video (`390x512`), KEEP recovers crisp eyes, eyebrows, forehead lines, teeth, earrings, and individual hair strands while keeping identity coherent frame-to-frame via Kalman latent propagation.
- **Where generic Real-ESRGAN fails vs KEEP:** Generic `RealESRGAN_x2plus` (even across two passes = 4x) only sharpens compression edges and leaves the `~54 px` CCTV face devoid of reconstructed irises or teeth (`LapVar = 32.5` vs KEEP `LapVar = 43.4`).
- **Critical CCTV failure floor (`~24 px` face height):** At extreme distance (`CAM-04 PARKING`, `~24 px` face, CRF 37), RetinaFace *still detects the head*, but KEEP generates **photorealistic hallucinations** — notably synthesizing one open eye while the subject is blinking in ground truth (`t=08`) and failing to reopen the heavily blurred left eye later (`t=15`).
- **Bottleneck on 6 GB VRAM:** Full-frame `--bg_upsampler realesrgan` drives total GPU usage to **5,829–5,881 MiB / 6,141 MiB (~95–96%)**, slowing throughput to **6.67–10.68 s/frame** cleanly showing why production CCTV deployments should crop person/head ROIs before running KEEP rather than super-resolving empty hallways.

---

## 2. Test Clips & Measured RTX 4050 Benchmarks

| Test Run | Input Spec | Output Spec | Elapsed Time | Pure Speed | Peak GPU Memory | Notes |
|---|---|---|---:|---:|---:|---|
| **Clip A (`lobby_med_48px`)** | `640x360`, 16f @ 15 fps, CRF 34, ~54 px face | `1280x720` (`-s 2`) | 162.18 s *(incl. weight downloads)* | — | 5877 MiB | First-run downloaded ~752 MB of model weights |
| **Clip C (`field_real_lq`)** | `390x512`, 16f @ 20 fps, real `real_3.mp4` | `780x1024` (`-s 2`) | **106.70 s** | **6.67 s/frame** | **5850 MiB** | Zero downloads; clean full-pipeline timing |
| **Clip B (`parking_far_22px_whole`)** | `640x360`, 16f @ 15 fps, CRF 37, ~24 px face | `1280x720` (`-s 2`) | **170.80 s** | **10.68 s/frame** | **5829 MiB** | Zero downloads; 2x tiling overhead on `640x360` bg |
| **Clip B Aligned (`parking_far_22px_aligned`)** | `512x512` aligned H.264 face crops | `512x512` (`--has_aligned -s 1`) | 168.69 s *(incl. 104 MB ResNet50 dl)* | — | 5881 MiB | Pre-aligned crop path; same ~24 px eye loss |

> [!NOTE]
> All `peak_gpu_total_mb` values are host-wide GPU memory sampled via `nvidia-smi` during active inference (`6,141 MiB` total card capacity). Reducing `-s 1` or omitting `--bg_upsampler realesrgan` dramatically lowers memory pressure when only the subject's face needs enhancement.

---

## 3. Visual Result 1: Medium-Range Lobby CCTV (`~54 px` Face Height)

![KEEP Medium-Range Lobby CCTV Comparison](KEEP_CCTV_Visual_1_Lobby_Med_Comparison.png)

**What we observed on `CAM-01 LOBBY` (`t=08`):**
- **Sharpness vs Reality Gap:** Aligned 512x512 crop Laplacian variance jumped from **`2.1`** (raw CCTV warp) to **`43.4`** (KEEP) — compared to **`32.5`** for two passes of generic `RealESRGAN_x2plus` and **`28.6`** for the pre-CCTV ground truth.
- **Detail Recovery:** KEEP reconstructed crisp irises, upper/lower eyelashes, individual eyebrows, forehead lines, natural teeth/lips, earring gleam, and flowing hair strands from just ~54 pixels of vertical face height.
- **Paste-Back Blending Caveat:** When `--bg_upsampler realesrgan` is active alongside a synthetic low-detail background, the rectangular Region-of-Interest around the head sharpens differently than the flat hallway walls, creating a visible seam around the head/shoulders in `final_results`.

---

## 4. Visual Result 2: Temporal Consistency Across Motion & Speech

![KEEP Temporal Consistency Strip](KEEP_CCTV_Visual_2_Temporal_Tracking_Strip.png)

**Measured 16-Frame Alignment & Stability Metrics:**

| Sequence | Raw Aligned `LapVar` | KEEP `LapVar` | Aligned Frame-to-Frame Delta (`[0..255]`) | Visual Stability Finding |
|---|---:|---:|---:|---|
| **Clip C (`cctv_field_real_lq`)** | 9.1 | **49.1** | 1.79 -> 2.23 | Sharp skin/eyes/hair added consistently without I-frame flicker |
| **Clip A (`cctv_lobby_med_48px`)** | 2.1 | **35.7** | 6.70 -> 10.11 | Kalman landmark smoothing locks earring, hairline & expression lines |

**Why KEEP beats single-frame restorers (GFPGAN / CodeFormer) on video:**
Single-frame restorers re-hallucinate high-frequency pore, tooth, and hair detail independently every frame, causing distracting texture shimmer on CCTV keyframes. KEEP's Kalman-inspired latent propagation + cross-frame attention carries facial geometry continuously across the 8-frame temporal windows (`--max_length 8`), keeping the face locked to one coherent person across speaking frames (`t=0, 5, 10, 15`).

---

## 5. Visual Result 3: Extreme-Distance Parking CCTV (`~24 px` Face) — Forensic Failure Floor

![KEEP Far Parking CCTV Forensic Test](KEEP_CCTV_Visual_3_Far_CCTV_Forensic_Test.png)

**What happens when CCTV subjects are too far (`~24 px`, H.264 CRF 37):**
1. **Detection Still Fires:** Even with `retinaface_mobile0.25`, the helper localized the `~24 px` head across all 16 frames of `640x360` video (`cropped_faces` + `final_results` completed cleanly).
2. **Sharpness Looks Convincing, Identity Is Compromised:** Laplacian variance rose from **`1.9`** to **`14.2`**, yet direct comparison against Pre-CCTV Ground Truth reveals severe semantic errors:
   - At **`t=08`**, the true subject is **blinking (both eyes closed)**. KEEP propagated partial open-eye cues from nearby frames into the subject's right eye while keeping the left eye shut — fabricating a false half-wink expression.
   - At **`t=15`**, the true subject opens **both eyes**, yet the `~24 px` CRF 37 input lacks enough contrast around the left eye socket, so KEEP leaves the left eye closed and alters the eyebrow ridge.
3. **Pre-Aligned Crop vs Whole-Frame:** Running `--has_aligned -s 1` on extracted `512x512` H.264 face crops reproduces the same left-eye hallucination, proving the bottleneck is **missing information bits at ~24 px**, not face detector alignment noise.

---

## 6. Practical CCTV Deployment Recommendations

1. **Enforce a Minimum Face-Height Threshold (`>= 45–50 px`):** Use KEEP as an **enhancement & investigator aid** for medium-to-close CCTV clips (`>= 50 px` faces). Treat outputs below `~30 px` strictly as unverified generative reconstructions — **never** as standalone biometric proof.
2. **Always Archive & Compare Side-by-Side with Raw CCTV Frames:** Inspect the raw `cropped_faces` / original H.264 stream beside KEEP's `restored_faces` output so analysts can distinguish recovered pixels from VQGAN prior fills (such as asymmetrical eyes or smoothed skin).
3. **Crop Subject Tracklets Before Running KEEP:** On consumer 6–8 GB GPUs (RTX 4050/4060), drop `--bg_upsampler realesrgan` (or crop the tracked person bbox first with `ffmpeg` / YOLO + ByteTrack), run KEEP only on the aligned face tracklet, and avoid spending ~95% of VRAM and 7–11 s/frame upscaling static walls and parking asphalt.
4. **Tune `--max_length` to GPU Headroom:** `--max_length 8` kept peak memory at **5,829–5,881 MiB** inside 6 GB VRAM without OOM errors. On 12–24 GB GPUs, increasing `--max_length` (e.g., `16–20`) improves long-range temporal consistency and reduces chunk-boundary resets.
