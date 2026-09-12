"""Shared helpers for the Klein9 RefMod pack: media loading and models/refmods-klein9 folder."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def refmods_klein9_dir() -> str:
    """ComfyUI models/refmods-klein9 — the mod storage folder for Flux2 / Klein9.

    Sits next to loras/, unet/, checkpoints/, and refmods/.
    """
    import folder_paths
    d = os.path.join(folder_paths.models_dir, "refmods-klein9")
    os.makedirs(d, exist_ok=True)
    return d


def list_media_files(folder: str) -> List[str]:
    """List image files directly under folder (top level only), sorted by name."""
    images = []
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            ext = os.path.splitext(fn)[1].lower()
            p = os.path.join(folder, fn)
            if os.path.isfile(p) and ext in IMAGE_EXTS:
                images.append(p)
    return images


def load_image_file(path: str, max_edge: Optional[int] = None) -> torch.Tensor:
    """Load one image file -> [1, H, W, 3] float32 in [0, 1]."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max_edge is not None:
            scale = min(1.0, max_edge / max(w, h))
            if scale < 1.0:
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                 Image.LANCZOS)
        arr = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
    return arr.unsqueeze(0)  # [1, H, W, 3]
