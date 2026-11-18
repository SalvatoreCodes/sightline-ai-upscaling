"""Download official VIRAT Video Dataset Release 2.0 sequences & full Kitware catalog."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import cv2

VIRAT_DIR = Path(r"D:\Projects\sightline-ai-upscaling\Assets\VIRAT")
VIRAT_DIR.mkdir(parents=True, exist_ok=True)

KITWARE_VIDEOS_FOLDER = "56f581ce8d777f753209ca43"
KITWARE_RELEASE_LIST_ITEM = "56f581c88d777f753209c9cf"

DEFAULT_CLIPS = [
    ("VIRAT_S_010204_05_000856_000890.mp4", "https://viratdata.org/video/VIRAT_S_010204_05_000856_000890.mp4"),
    ("VIRAT_S_010003_07_000608_000636.mp4", "https://data.kitware.com/api/v1/item/56f5879c8d777f753209cb34/download"),
    ("VIRAT_S_010004_00_000064_000100.mp4", "https://data.kitware.com/api/v1/item/56f587a08d777f753209cb4c/download"),
    ("VIRAT_S_010003_08_000739_000778.mp4", "https://data.kitware.com/api/v1/item/56f5879c8d777f753209cb37/download"),
    ("VIRAT_S_010000_07_000827_000860.mp4", "https://data.kitware.com/api/v1/item/56f587818d777f753209cae3/download"),
    ("VIRAT_S_010002_02_000174_000204.mp4", "https://data.kitware.com/api/v1/item/56f587958d777f753209cb10/download"),
]


def download_file(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out_f:
        shutil.copyfileobj(resp, out_f, length=1024 * 256)
    tmp.replace(dest)
    return dest


def fetch_catalog() -> tuple[list[dict], Path, Path]:
    items = []
    offset = 0
    while True:
        u = f"https://data.kitware.com/api/v1/item?folderId={KITWARE_VIDEOS_FOLDER}&limit=200&offset={offset}"
        batch = json.loads(urllib.request.urlopen(u, timeout=30).read().decode())
        if not batch:
            break
        items.extend(batch)
        if len(batch) < 200:
            break
        offset += 200

    catalog = [
        {
            "name": it["name"],
            "id": it["_id"],
            "size_mb": round(it.get("size", 0) / (1024 * 1024), 2),
            "download_url": f"https://data.kitware.com/api/v1/item/{it['_id']}/download",
        }
        for it in sorted(items, key=lambda x: x["name"])
    ]
    cat_path = VIRAT_DIR / "virat_catalog.json"
    cat_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    list_path = VIRAT_DIR / "list_release2.0.txt"
    download_file(f"https://data.kitware.com/api/v1/item/{KITWARE_RELEASE_LIST_ITEM}/download", list_path)
    return catalog, cat_path, list_path


def verify_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to decode downloaded video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    cap.release()
    if not ok or frames <= 0:
        raise RuntimeError(f"Video has no readable frames: {path}")
    return {
        "name": path.name,
        "path": str(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "resolution": f"{w}x{h}",
        "fps": fps,
        "frames": frames,
        "duration_sec": round(frames / max(1.0, fps), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Download VIRAT Release 2.0 videos & catalog.")
    parser.add_argument("--all", action="store_true", help="Download all 329 official videos (35.9 GB)")
    parser.add_argument("--count", type=int, default=0, help="Download N smallest catalog clips")
    args = parser.parse_args()

    catalog, cat_path, list_path = fetch_catalog()

    if args.all:
        targets = [(it["name"], it["download_url"]) for it in catalog]
    elif args.count > 0:
        smallest = sorted(catalog, key=lambda x: x["size_mb"])[: args.count]
        targets = [(it["name"], it["download_url"]) for it in smallest]
    else:
        targets = DEFAULT_CLIPS

    verified = []
    for fname, url in targets:
        dest = VIRAT_DIR / fname
        print(f"Downloading {fname} ...")
        download_file(url, dest)
        info = verify_video(dest)
        verified.append(info)

    print(
        json.dumps(
            {
                "virat_dir": str(VIRAT_DIR),
                "catalog_entries": len(catalog),
                "catalog_total_gb": round(sum(x["size_mb"] for x in catalog) / 1024.0, 2),
                "catalog_path": str(cat_path),
                "release_list_path": str(list_path),
                "downloaded_count": len(verified),
                "downloaded_videos": verified,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
