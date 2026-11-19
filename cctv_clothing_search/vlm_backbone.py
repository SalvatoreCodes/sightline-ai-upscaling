"""Unified SigLIP-384 vision-language backbone with calibrated similarity temperatures."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

VLM_MODEL_ID = "google/siglip-base-patch16-384"

# Empirically calibrated logit temperatures for SigLIP-384 normalized cosine spreads
SIGLIP_HUMAN_TEMP = 45.0
SIGLIP_TYPE_TEMP = 52.0
SIGLIP_COLOR_TEMP = 45.0


def _extract_tensor(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return out[0]


def load_vlm(device: torch.device):
    """Load google/siglip-base-patch16-384 (fp16 on CUDA) and return (model, processor, embed_dim)."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = AutoModel.from_pretrained(VLM_MODEL_ID, dtype=dtype).eval().to(device)
    embed_dim = int(getattr(model.config.vision_config, "hidden_size", 768))
    return model, processor, embed_dim


def encode_vlm_images(
    model,
    processor,
    imgs: list[Image.Image],
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Encode RGB images into L2-normalized float32 embeddings on device."""
    embed_dim = int(getattr(model.config.vision_config, "hidden_size", 768))
    if not imgs:
        return torch.empty((0, embed_dim), dtype=torch.float32, device=device)

    feats = []
    with torch.inference_mode():
        for i in range(0, len(imgs), batch_size):
            chunk = [im.convert("RGB") for im in imgs[i : i + batch_size]]
            inputs = processor(images=chunk, return_tensors="pt").to(device)
            if device.type == "cuda" and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)
            out = model.get_image_features(**inputs)
            emb = _extract_tensor(out).float()
            feats.append(F.normalize(emb, dim=-1))
    return torch.cat(feats, dim=0)


def encode_vlm_texts(
    model,
    processor,
    prompts: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Encode text prompts with SigLIP max_length padding into L2-normalized float32 embeddings."""
    with torch.inference_mode():
        inputs = processor(
            text=prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        out = model.get_text_features(**inputs)
        emb = _extract_tensor(out).float()
        return F.normalize(emb, dim=-1)


def build_vlm_text_matrix(
    model,
    processor,
    prompt_dict: dict[str, list[str]],
    names: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Average multi-prompt SigLIP embeddings per class name into [len(names), embed_dim]."""
    all_prompts: list[str] = []
    slices: list[tuple[int, int]] = []
    for name in names:
        ps = prompt_dict[name]
        start = len(all_prompts)
        all_prompts.extend(ps)
        slices.append((start, len(all_prompts)))

    # Batch text prompts cleanly to keep tokenizer/memory compact
    chunks = []
    bs = 64
    for i in range(0, len(all_prompts), bs):
        chunks.append(encode_vlm_texts(model, processor, all_prompts[i : i + bs], device))
    full_emb = torch.cat(chunks, dim=0)

    class_vecs = []
    for start, end in slices:
        mv = full_emb[start:end].mean(dim=0)
        class_vecs.append(F.normalize(mv, dim=-1))
    return torch.stack(class_vecs, dim=0)
