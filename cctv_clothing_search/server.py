"""FastAPI server for Sightline CCTV clothing timestamp search & HTTP Range video streaming."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from .indexer import SCANS_DIR, UPLOADS_DIR, ClothingSearchDB, resolve_video_path

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Sightline Clothing Search", docs_url=None, redoc_url=None)
db = ClothingSearchDB()
_indexer_singleton = None


def get_indexer():
    global _indexer_singleton
    if _indexer_singleton is None:
        from .indexer import CCTVClothingIndexer
        _indexer_singleton = CCTVClothingIndexer()
    return _indexer_singleton


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI missing")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    stats = db.get_stats("144p")
    return {"status": "ok", "indexed_qualities": stats["available_qualities"]}


@app.get("/api/stats")
def api_stats(quality: str = Query("latest")):
    return db.get_stats(quality)


@app.get("/api/scan-file/{quality}")
def api_scan_file(quality: str, fmt: str = Query("json")):
    stats = db.get_stats(quality)
    resolved_q = stats["quality"]
    if not stats.get("video"):
        raise HTTPException(status_code=404, detail=f"No scan data found for {quality}")
    safe_q = re.sub(r"[^A-Za-z0-9._-]", "_", resolved_q) or "scan"
    if fmt.lower() == "csv":
        target = SCANS_DIR / f"{safe_q}_scan.csv"
        media_t = "text/csv"
    else:
        target = SCANS_DIR / f"{safe_q}_scan.json"
        media_t = "application/json"
    if not target.exists():
        db.export_video_scan_files(resolved_q)
    return FileResponse(target, media_type=media_t, filename=target.name)


@app.get("/api/categories")
def api_categories():
    return {"categories": db.get_categories()}


@app.post("/api/train")
def api_train(epochs: int = Query(65, ge=5, le=200)):
    global _indexer_singleton
    from . import trainer
    metrics = trainer.train_model(epochs=epochs)
    _indexer_singleton = None
    return {"ok": True, "metrics": metrics}


@app.get("/api/search")
def api_search(
    q: str = Query("", max_length=120),
    quality: str = Query("latest"),
    color: str = Query(""),
    garment_type: str = Query("", alias="type"),
    min_conf: float = Query(0.0, ge=0.0, le=100.0),
    sort: str = Query("time"),
):
    if not quality or quality == "latest":
        quality = db.get_stats("latest")["quality"]
    return db.search(
        query=q,
        quality=quality,
        color_filter=color,
        type_filter=garment_type,
        min_conf=min_conf,
        sort_by=sort,
    )


@app.post("/api/scan")
def api_scan(quality: str = Query("144p"), stride: int = Query(24, ge=8, le=120)):
    try:
        vpath = resolve_video_path(quality, db)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    indexer = get_indexer()
    res = indexer.scan_video(quality=quality, stride_frames=stride, db=db, video_path=vpath)
    db.export_json_snapshot()
    return {"ok": True, "scan": res, "stats": db.get_stats(quality)}


@app.post("/api/upload")
async def api_upload(
    request: Request,
    filename: str = Query("footage.mp4"),
    stride: int = Query(24, ge=8, le=120),
):
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name) or f"clip_{int(time.time())}.mp4"
    dest_path = UPLOADS_DIR / safe_name
    with open(dest_path, "wb") as out_f:
        async for chunk in request.stream():
            if chunk:
                out_f.write(chunk)
    if dest_path.stat().st_size == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload payload")

    vid_key = re.sub(r"[^A-Za-z0-9_-]", "_", Path(safe_name).stem)[:20] or f"clip_{int(time.time())}"
    indexer = get_indexer()
    res = indexer.scan_video(quality=vid_key, stride_frames=stride, db=db, video_path=dest_path)
    db.export_json_snapshot()
    return {"ok": True, "quality": vid_key, "scan": res, "stats": db.get_stats(vid_key)}


def _file_iterator(path: Path, start: int, length: int, chunk_size: int = 1024 * 256):
    with open(path, "rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            data = f.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


@app.get("/video/{quality}")
def stream_video(quality: str, request: Request):
    try:
        path = resolve_video_path(quality, db)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file missing on disk")

    file_size = os.path.getsize(path)
    range_header: Optional[str] = request.headers.get("range")
    if range_header:
        try:
            byte_range = range_header.strip().lower().replace("bytes=", "")
            parts = byte_range.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                return JSONResponse(status_code=416, content={"detail": "Range Not Satisfiable"})
        except ValueError:
            return FileResponse(path, media_type="video/mp4")

        content_len = end - start + 1
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_len),
            "Cache-Control": "public, max-age=3600",
        }
        return StreamingResponse(
            _file_iterator(path, start, content_len),
            status_code=206,
            headers=headers,
            media_type="video/mp4",
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


def main():
    port = int(os.environ.get("PORT", "8766"))
    uvicorn.run("cctv_clothing_search.server:app", host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
