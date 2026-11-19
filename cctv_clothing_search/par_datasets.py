"""Harmonized CCTV Pedestrian Attribute Recognition (PAR) dataset loaders (PA-100K, Market-1501, Simuletic CCTV-1K + PETA/RAPv2 local hooks)."""

from __future__ import annotations

import json
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
from PIL import Image

from .vocab import COLOR_NAMES, TYPE_NAMES

ROOT_DIR = Path(r"D:\projects\sightline-ai-upscaling")
DATASET_CACHE_DIR = ROOT_DIR / "datasets_cache"
DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

COLOR_IDX = {n: i for i, n in enumerate(COLOR_NAMES)}
TYPE_IDX = {n: i for i, n in enumerate(TYPE_NAMES)}

PA_LONG_SLEEVE_TYPES = [
    "dress shirt / blouse",
    "sweater / knitwear",
    "hoodie / sweatshirt",
    "zip-up jacket / bomber jacket",
    "open jacket / coat",
    "puffer jacket",
    "windbreaker / rain jacket",
    "suit jacket / blazer",
]
PA_TROUSERS_TYPES = [
    "trousers / pants",
    "blue denim jeans",
    "cargo pants",
    "joggers / track pants",
]


def _idxs(names: list[str], mapping: dict[str, int]) -> list[int]:
    return [mapping[n] for n in names if n in mapping]


def load_pa100k_records(max_train: int = 2400, max_val: int = 400) -> tuple[list[dict], list[dict]]:
    """Load stratified training/validation samples from tuandunghcmut/PA-100K on Hugging Face."""
    from datasets import load_dataset

    def collect_split(split_name: str, limit: int) -> list[dict]:
        ds = load_dataset("tuandunghcmut/PA-100K", split=split_name, streaming=True)
        # Encourage balance across sparse outdoor attributes instead of taking only dominant trousers/long-sleeve rows
        bucket_caps = {
            "Plaid": int(limit * 0.20),
            "LongCoat": int(limit * 0.20),
            "Shorts": int(limit * 0.20),
            "Skirt": int(limit * 0.20),
            "ShortSleeve": int(limit * 0.35),
            "Default": int(limit * 0.35),
        }
        bucket_counts = {k: 0 for k in bucket_caps}
        out: list[dict] = []
        seen_persons = 0

        for row in ds:
            if seen_persons >= limit:
                break
            is_plaid = int(row.get("UpperPlaid", 0)) == 1
            is_coat = int(row.get("LongCoat", 0)) == 1
            is_short_slv = int(row.get("ShortSleeve", 0)) == 1
            is_shorts = int(row.get("Shorts", 0)) == 1
            is_skirt = int(row.get("Skirt&Dress", 0)) == 1

            if is_plaid:
                bkey = "Plaid"
            elif is_coat:
                bkey = "LongCoat"
            elif is_shorts:
                bkey = "Shorts"
            elif is_skirt:
                bkey = "Skirt"
            elif is_short_slv:
                bkey = "ShortSleeve"
            else:
                bkey = "Default"

            if bucket_counts[bkey] >= bucket_caps[bkey] and seen_persons < int(limit * 0.85):
                continue

            img = row.get("image")
            if img is None:
                continue
            pil = img.convert("RGB")
            if min(pil.size) < 16:
                continue

            if is_plaid:
                up_types = _idxs(["flannel / plaid shirt"], TYPE_IDX)
            elif is_coat:
                up_types = _idxs(["long overcoat", "buttoned coat"], TYPE_IDX)
            elif is_short_slv:
                up_types = _idxs(["t-shirt / polo shirt", "tank top / sleeveless top"], TYPE_IDX)
            elif int(row.get("LongSleeve", 0)) == 1:
                up_types = _idxs(PA_LONG_SLEEVE_TYPES, TYPE_IDX)
            else:
                up_types = []

            if is_shorts:
                low_types = _idxs(["shorts / cargo shorts"], TYPE_IDX)
            elif is_skirt:
                low_types = _idxs(["dress / skirt"], TYPE_IDX)
            elif int(row.get("Trousers", 0)) == 1:
                low_types = _idxs(PA_TROUSERS_TYPES, TYPE_IDX)
            else:
                low_types = []

            if not up_types and not low_types:
                continue

            bucket_counts[bkey] += 1
            seen_persons += 1
            pid = f"pa100k_{split_name}_{row.get('image_name', seen_persons)}"
            if up_types:
                out.append({
                    "source": f"pa100k_{split_name}",
                    "split": "train" if split_name == "train" else "val",
                    "person_id": pid,
                    "pil": pil,
                    "zone": "upper",
                    "type_idxs": up_types,
                    "color_idxs": [],
                })
            if low_types:
                out.append({
                    "source": f"pa100k_{split_name}",
                    "split": "train" if split_name == "train" else "val",
                    "person_id": pid,
                    "pil": pil,
                    "zone": "lower",
                    "type_idxs": low_types,
                    "color_idxs": [],
                })
        return out

    return collect_split("train", max_train), collect_split("val", max_val)


def load_market1501_records(max_train: int = 2000, max_val: int = 400) -> tuple[list[dict], list[dict]]:
    """Load Market-1501 images + 27 identity-level clothing color & type annotations."""
    from huggingface_hub import hf_hub_download

    m_dir = DATASET_CACHE_DIR / "market1501"
    m_dir.mkdir(parents=True, exist_ok=True)
    mat_path = m_dir / "market_attribute.mat"
    if not mat_path.exists():
        url = "https://raw.githubusercontent.com/vana77/Market-1501_Attribute/master/market_attribute.mat"
        mat_path.write_bytes(urllib.request.urlopen(url, timeout=30).read())

    root_extracted = m_dir / "Market-1501-v15.09.15"
    if not (root_extracted / "bounding_box_train").exists():
        tar_path = hf_hub_download(
            repo_id="Lekim89/market1501",
            filename="Market-1501-v15.09.15.tar.gz",
            repo_type="dataset",
        )
        with tarfile.open(tar_path, "r:gz") as tf:
            # Extract only bounding_box_train and bounding_box_test JPGs cleanly
            members = [
                m for m in tf.getmembers()
                if m.isfile() and ("bounding_box_train/" in m.name or "bounding_box_test/" in m.name) and m.name.lower().endswith(".jpg")
            ]
            tf.extractall(path=m_dir, members=members)

    mat = scipy.io.loadmat(str(mat_path))["market_attribute"][0, 0]

    up_color_map = {
        "upblack": ["black"],
        "upwhite": ["white"],
        "upred": ["red"],
        "uppurple": ["purple / violet"],
        "upyellow": ["yellow", "orange", "beige / khaki"],
        "upgray": ["dark grey / charcoal", "light grey / silver"],
        "upblue": ["blue", "royal blue", "navy blue", "light blue", "teal / cyan"],
        "upgreen": ["green"],
    }
    low_color_map = {
        "downblack": ["black"],
        "downwhite": ["white"],
        "downpink": ["pink / magenta"],
        "downpurple": ["purple / violet"],
        "downyellow": ["yellow", "beige / khaki"],
        "downgray": ["dark grey / charcoal", "light grey / silver"],
        "downblue": ["blue", "royal blue", "navy blue", "light blue", "teal / cyan"],
        "downgreen": ["green"],
        "downbrown": ["brown", "beige / khaki"],
    }

    def build_split(split_key: str, folder: Path, per_id_cap: int, total_cap: int) -> list[dict]:
        st = mat[split_key][0, 0]
        names = st.dtype.names
        id_attr: dict[str, dict[str, int]] = {}
        raw_ids = st["image_index"][0]
        for idx_pos, raw_id in enumerate(raw_ids):
            pid_str = str(raw_id[0]).zfill(4)
            id_attr[pid_str] = {k: int(st[k][0][idx_pos]) for k in names if k != "image_index"}

        rng = np.random.default_rng(101 if split_key == "train" else 202)
        by_id: dict[str, list[Path]] = {}
        for img_p in sorted(folder.glob("*.jpg")):
            pid = img_p.name.split("_", 1)[0]
            if pid in id_attr:
                by_id.setdefault(pid, []).append(img_p)

        chosen_files: list[tuple[str, Path]] = []
        id_keys = list(by_id.keys())
        rng.shuffle(id_keys)
        for pid in id_keys:
            paths = list(by_id[pid])
            rng.shuffle(paths)
            for p in paths[:per_id_cap]:
                chosen_files.append((pid, p))
                if len(chosen_files) >= total_cap:
                    break
            if len(chosen_files) >= total_cap:
                break

        records: list[dict] = []
        for idx_i, (pid, img_p) in enumerate(chosen_files):
            attr = id_attr[pid]
            with Image.open(img_p) as im:
                pil = im.convert("RGB")

            up_colors: list[str] = []
            for k, targets in up_color_map.items():
                if attr.get(k, 1) == 2:
                    up_colors.extend(targets)
            low_colors: list[str] = []
            for k, targets in low_color_map.items():
                if attr.get(k, 1) == 2:
                    low_colors.extend(targets)

            # up: 1=long sleeve, 2=short sleeve
            if attr.get("up", 1) == 2:
                up_types = _idxs(["t-shirt / polo shirt", "tank top / sleeveless top"], TYPE_IDX)
            else:
                up_types = _idxs(PA_LONG_SLEEVE_TYPES + ["flannel / plaid shirt", "long overcoat", "buttoned coat"], TYPE_IDX)

            # clothes: 1=dress, 2=pants ; down: 1=long lower, 2=short lower
            if attr.get("clothes", 2) == 1:
                low_types = _idxs(["dress / skirt"], TYPE_IDX)
            elif attr.get("down", 1) == 2:
                low_types = _idxs(["shorts / cargo shorts"], TYPE_IDX)
            else:
                low_types = _idxs(PA_TROUSERS_TYPES, TYPE_IDX)

            split_out = "train" if split_key == "train" else "val"
            rec_id = f"market_{split_key}_{pid}_{idx_i}"
            records.append({
                "source": f"market_{split_key}",
                "split": split_out,
                "person_id": rec_id,
                "pil": pil,
                "zone": "upper",
                "type_idxs": up_types,
                "color_idxs": _idxs(sorted(set(up_colors)), COLOR_IDX),
            })
            records.append({
                "source": f"market_{split_key}",
                "split": split_out,
                "person_id": rec_id,
                "pil": pil,
                "zone": "lower",
                "type_idxs": low_types,
                "color_idxs": _idxs(sorted(set(low_colors)), COLOR_IDX),
            })
        return records

    tr_recs = build_split("train", root_extracted / "bounding_box_train", per_id_cap=3, total_cap=max_train)
    va_recs = build_split("test", root_extracted / "bounding_box_test", per_id_cap=1, total_cap=max_val)
    return tr_recs, va_recs


def load_simuletic_records(views_per_id: int = 2) -> tuple[list[dict], list[dict]]:
    """Load Simuletic CCTV-Pedestrian-1K fine-grained color & type records with identity-disjoint split."""
    from huggingface_hub import hf_hub_download

    repo = "Simuletic/CCTV-Pedestrian-1K-Person-Attribute-Dataset"
    meta_file = hf_hub_download(repo_id=repo, filename="pedestrian_1k/metadata.jsonl", repo_type="dataset")
    lines = Path(meta_file).read_text(encoding="utf-8").splitlines()
    raw_rows = [json.loads(l) for l in lines if l.strip()]

    by_ped: dict[int, list[dict]] = {}
    for r in raw_rows:
        fn = r["image"]
        try:
            ped_num = int(fn.split("_")[1])
        except Exception:
            continue
        by_ped.setdefault(ped_num, []).append(r)

    selected: list[tuple[str, dict]] = []
    for ped_num in sorted(by_ped.keys()):
        split_name = "train" if ped_num <= 160 else "val"
        # Pick front/side/back diverse views per identity (capped cleanly for disk/time budget)
        for item in by_ped[ped_num][:views_per_id]:
            selected.append((split_name, item))

    def dl_one(item_tuple: tuple[str, dict]) -> tuple[str, dict, Image.Image] | None:
        split_name, item = item_tuple
        try:
            fp = hf_hub_download(repo_id=repo, filename=f"pedestrian_1k/{item['image']}", repo_type="dataset")
            with Image.open(fp) as im:
                rgb = im.convert("RGB")
                # Downsample huge 1024px+ PNGs slightly while retaining generous detail for SigLIP/YOLO
                if max(rgb.size) > 512:
                    rgb.thumbnail((512, 512), Image.BICUBIC)
                else:
                    rgb = rgb.copy()
            return (split_name, item, rgb)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        loaded = [x for x in ex.map(dl_one, selected) if x is not None]

    top_type_map = {
        "knit sweater": ["sweater / knitwear"],
        "tank top": ["tank top / sleeveless top"],
        "hoodie": ["hoodie / sweatshirt"],
        "t-shirt": ["t-shirt / polo shirt"],
        "puffer jacket": ["puffer jacket"],
        "leather jacket": ["zip-up jacket / bomber jacket"],
    }
    bottom_type_map = {
        "sweatpants": ["joggers / track pants"],
        "leggings": ["trousers / pants", "joggers / track pants"],
        "denim jeans": ["blue denim jeans", "trousers / pants"],
        "cargo pants": ["cargo pants"],
        "formal trousers": ["trousers / pants"],
        "shorts": ["shorts / cargo shorts"],
    }
    top_color_map = {
        "white": ["white"],
        "forest green": ["green"],
        "bright red": ["red"],
        "black": ["black"],
        "yellow": ["yellow"],
        "navy blue": ["navy blue"],
        "grey": ["light grey / silver", "dark grey / charcoal"],
    }
    bottom_color_map = {
        "khaki": ["beige / khaki"],
        "dark grey": ["dark grey / charcoal"],
        "black": ["black"],
        "blue": ["blue", "navy blue", "royal blue", "light blue"],
        "white": ["white"],
    }

    tr_recs: list[dict] = []
    va_recs: list[dict] = []
    for split_name, item, pil in loaded:
        attr = item["attributes"]
        ut = _idxs(top_type_map.get(attr.get("top_type", ""), []), TYPE_IDX)
        uc = _idxs(top_color_map.get(attr.get("top_color", ""), []), COLOR_IDX)
        bt_key = attr.get("bottom_type", "")
        bc_key = attr.get("bottom_color", "")
        if bt_key == "denim jeans" and bc_key == "blue":
            lt = _idxs(["blue denim jeans"], TYPE_IDX)
        else:
            lt = _idxs(bottom_type_map.get(bt_key, []), TYPE_IDX)
        lc = _idxs(bottom_color_map.get(bc_key, []), COLOR_IDX)

        target_list = tr_recs if split_name == "train" else va_recs
        pid = f"sim_{item['image']}"
        if ut and uc:
            target_list.append({
                "source": f"simuletic_{split_name}",
                "split": split_name,
                "person_id": pid,
                "pil": pil,
                "zone": "upper",
                "type_idxs": ut,
                "color_idxs": uc,
            })
        if lt and lc:
            target_list.append({
                "source": f"simuletic_{split_name}",
                "split": split_name,
                "person_id": pid,
                "pil": pil,
                "zone": "lower",
                "type_idxs": lt,
                "color_idxs": lc,
            })
    return tr_recs, va_recs


def check_local_peta_rapv2_hooks() -> dict[str, bool]:
    """Documented drop-in hooks if PETA.mat or RAP_annotation.mat are placed in datasets_cache/."""
    peta_ready = (DATASET_CACHE_DIR / "PETA" / "PETA.mat").exists()
    rap_ready = (DATASET_CACHE_DIR / "RAPv2" / "RAP_annotation.mat").exists()
    return {"PETA_local": peta_ready, "RAPv2_local": rap_ready}


def load_all_par_datasets() -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Fetch/cache open-access CCTV PAR datasets and return unified train/val record lists."""
    hooks = check_local_peta_rapv2_hooks()
    print(f"Loading open-access surveillance PAR datasets (optional local hooks: {hooks})...")
    pa_tr, pa_va = load_pa100k_records(max_train=2400, max_val=400)
    mk_tr, mk_va = load_market1501_records(max_train=2000, max_val=400)
    sim_tr, sim_va = load_simuletic_records(views_per_id=2)

    train_recs = pa_tr + mk_tr + sim_tr
    val_recs = pa_va + mk_va + sim_va
    summary = {
        "pa100k_train_zones": len(pa_tr),
        "pa100k_val_zones": len(pa_va),
        "market1501_train_zones": len(mk_tr),
        "market1501_val_zones": len(mk_va),
        "simuletic_train_zones": len(sim_tr),
        "simuletic_val_zones": len(sim_va),
        "optional_local_hooks": hooks,
    }
    print("PAR dataset summary:", json.dumps(summary))
    return train_recs, val_recs, summary
