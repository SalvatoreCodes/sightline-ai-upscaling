# Sightline — Low-Resolution CCTV Human & Clothing Search

**Sightline** is a lightweight multi-task vision and SQLite search system for low-resolution CCTV footage (`144p`–`720p` and VIRAT clips). It verifies human detections against background fixtures, classifies clothing type and color (using Real-ESRGAN x2+ upscaling, TinyCLIP visual-text prototypes, and CIELAB statistics), automatically discovers unseen clothing categories into a persistent category registry, and indexes every sighting with frame timestamps for millisecond search.

## Features

- **Human Verification (`SightlineTrainedModel.human_head`)**: Filters false positives from non-human background crops and outputs per-detection human confidence (`96.1%` validation accuracy across retail + VIRAT footage).
- **Low-Resolution Clothing Type & Color Classification**: Fine-tuned adapters (`type_mlp` + cosine `type_protos`, `color_mlp` + `color_protos` fused with CIELAB likelihoods) achieving `100.0%` validation accuracy on held-out `144p` / `240p` frames.
- **Open-World Category Discovery**: Evaluates unseen garment candidates (`OPEN_WORLD_CANDIDATES`) in shared 512-D CLIP space and automatically registers confident new categories (e.g., `joggers / track pants`, `shorts / cargo shorts`) into SQLite (`clothing_categories`), immediately updating search synonyms and UI filters.
- **Fast Web UI & HTTP Range Video Streaming**: Dark-mode FastAPI + SQLite + vanilla HTML/JS interface (`http://127.0.0.1:8766/`) supporting natural queries like `"Blue jacket"`, interactive timeline markers, bounding-box overlays, and video upload/scanning (`POST /api/upload`, `POST /api/scan`).

## Project Structure

- `cctv_clothing_search/trainer.py` — Multi-task PyTorch training pipeline (`human_head`, `type_mlp`, `color_mlp`, open-world text embeddings) saving `data/sightline_trained_model.pt` and `data/training_metrics.json`.
- `cctv_clothing_search/indexer.py` — Video scanner (`CCTVClothingIndexer`), open-world category discovery, temporal EMA smoothing, and SQLite/JSON index (`ClothingSearchDB`).
- `cctv_clothing_search/server.py` — FastAPI backend (`/api/search`, `/api/stats`, `/api/categories`, `/api/scan`, `/api/upload`, `/api/train`, `/video/{quality}`).
- `cctv_clothing_search/static/index.html` — Subtitle-free interactive search & video inspection UI.
- `download_virat_dataset.py` — Downloader for VIRAT Release 2.0 sample `.mp4` clips into `Assets/VIRAT/`.
- `generate_cctv_clothing_tuning_report.py` — Resolution ablation & parameter tuning benchmark across `144p`–`720p`.

## Quick Start

1. **Train the multi-task classifier**:
   ```bash
   python -m cctv_clothing_search.trainer
   ```

2. **Index CCTV videos with open-world discovery**:
   ```bash
   python -m cctv_clothing_search.indexer --qualities 144p 240p --discover
   python -m cctv_clothing_search.indexer --file "Assets/VIRAT/VIRAT_S_010204_05_000856_000890.mp4" --label "VIRAT_0102" --stride 24 --discover
   ```

3. **Start the web UI**:
   ```bash
   python -m cctv_clothing_search.server
   ```
   Open `http://127.0.0.1:8766/` in your browser.
