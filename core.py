"""
core.py — "RefMod" reference adapter engine for FLUX.2 and Klein9

FLUX.2 / Klein9 injects reference image tokens directly into the joint sequence:
the reference image is VAE-encoded, patchified (patch_size=2), and positioned on
the 2D RoPE coordinate grid alongside text tokens and noisy latent tokens.

Standard reference workflows encode raw high-resolution images live on every run,
which generates thousands of tokens (e.g. 1024x1024 -> 4096 tokens) per reference image,
causing high VRAM consumption and quadratic self-attention latency.

A Klein9 RefMod pre-encodes reference imagery once into a compact, pre-computed
latent adapter (.safetensors) stored in models/refmods-klein9/:
  * The reference is VAE-encoded to its latent representation [B, C, H, W].
  * Optionally, the latent is spatial-pooled/compressed to a targeted token budget
    (e.g., 512, 1024, 2048 tokens, or native full resolution).
  * At inference time, the pre-computed latents are injected into CONDITIONING
    under the model's native reference_latents key instantly with zero VAE latency.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

META_KEY = "klein9_refmod_meta"

CONCEPT_TYPES = (
    "identity",      # a specific person/character face/body
    "style",         # artistic style, color grading, illustration aesthetic
    "character",     # fictional character, costume, anatomy
    "clothing",      # specific outfit, garment, textures
    "background",    # environment, lighting, backdrop plate
    "generic",       # mixed / multi-concept
)

TOKEN_BUDGETS = (
    "native",        # full uncompressed VAE resolution
    "4096",          # ~1024x1024 equivalent
    "2048",          # ~724x724 equivalent
    "1024",          # ~512x512 equivalent (fast & high fidelity)
    "512",           # ~362x362 equivalent (ultra lightweight)
    "256",           # ~256x256 equivalent
)


@dataclass
class Klein9RefMod:
    name: str
    samples: torch.Tensor                     # [B, C, H, W]
    concept_type: str = "identity"
    mode: str = "encode"                      # "encode" (native) or "pooled"
    resolution: int = 1024
    token_budget: int = 1024
    meta: Dict = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        """Approximate patchified token count (assuming patch_size=2)."""
        if self.samples.dim() == 4:
            b, c, h, w = self.samples.shape
            return (h // 2) * (w // 2) * b
        return self.samples.numel() // (16 * 4)


def _blur_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Spatial low-pass filter (downsample then upsample) on the latent manifold.

    Used when applying reference strength < 1.0. Mixing toward a blurred copy
    preserves smooth latent correlations rather than injecting Gaussian noise artifacts.
    """
    if z.dim() != 4:
        return z
    b, c, h, w = z.shape
    sh, sw = max(1, h // factor), max(1, w // factor)
    down = F.adaptive_avg_pool2d(z.float(), (sh, sw))
    up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
    return up.to(z.dtype)


def pool_latent_to_budget(z: torch.Tensor, target_tokens: int, patch_size: int = 2) -> torch.Tensor:
    """Adaptively downsample latent spatial dimensions to match a target token budget."""
    if z.dim() != 4:
        return z
    b, c, h, w = z.shape
    current_tokens = (h // patch_size) * (w // patch_size)
    if current_tokens <= target_tokens or target_tokens <= 0:
        return z

    aspect = w / max(1, h)
    # target_tokens = (th / 2) * (tw / 2) = (th / 2) * (th * aspect / 2) = th^2 * aspect / 4
    th = max(patch_size, int(round(math.sqrt((target_tokens * (patch_size ** 2)) / aspect))))
    tw = max(patch_size, int(round(th * aspect)))

    # Ensure divisible by patch_size
    th = max(patch_size, (th // patch_size) * patch_size)
    tw = max(patch_size, (tw // patch_size) * patch_size)

    down = F.adaptive_avg_pool2d(z.float(), (th, tw))
    return down.to(z.dtype)


def fit_token_budget(z: torch.Tensor, budget_str: str, patch_size: int = 2) -> torch.Tensor:
    """Pool or keep latent according to the specified token budget setting."""
    if budget_str == "native" or not budget_str:
        return z
    try:
        budget_int = int(budget_str)
        return pool_latent_to_budget(z, budget_int, patch_size=patch_size)
    except (ValueError, TypeError):
        return z


def read_refmod_meta(path_no_ext: str) -> Optional[Dict]:
    """Read metadata block from a .safetensors file without loading tensor payloads."""
    path = path_no_ext if path_no_ext.endswith(".safetensors") else path_no_ext + ".safetensors"
    if not os.path.isfile(path):
        return None
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            raw_meta = f.metadata()
            if raw_meta and META_KEY in raw_meta:
                try:
                    return json.loads(raw_meta[META_KEY])
                except Exception:
                    pass
    except Exception:
        pass
    return None


def save_refmod(mod: Klein9RefMod, out_path: str) -> str:
    """Save a Klein9RefMod to a .safetensors file with embedded JSON metadata."""
    if not out_path.endswith(".safetensors"):
        out_path += ".safetensors"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    header_meta = {
        "name": mod.name,
        "concept_type": mod.concept_type,
        "mode": mod.mode,
        "resolution": mod.resolution,
        "token_budget": mod.token_budget,
        "token_count": mod.token_count,
        "channels": mod.samples.shape[1] if mod.samples.dim() >= 2 else 16,
        "shape": list(mod.samples.shape),
        "generator": "ComfyUI-Flux2Klein9Mod",
    }
    if mod.meta:
        header_meta.update(mod.meta)

    tensors = {
        "samples": mod.samples.contiguous().to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    }

    save_file(tensors, out_path, metadata={META_KEY: json.dumps(header_meta)})
    return out_path


def load_refmod(path: str, device: str = "cpu") -> Klein9RefMod:
    """Load a Klein9RefMod from a .safetensors file."""
    if not path.endswith(".safetensors"):
        path += ".safetensors"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Klein9 RefMod file not found: {path}")

    meta = read_refmod_meta(path) or {}
    tensors = load_file(path, device=device)
    samples = tensors.get("samples")
    if samples is None:
        raise ValueError(f"Invalid RefMod: 'samples' tensor missing in {path}")

    name = meta.get("name", os.path.splitext(os.path.basename(path))[0])
    concept_type = meta.get("concept_type", "identity")
    mode = meta.get("mode", "encode")
    resolution = meta.get("resolution", 1024)
    token_budget = meta.get("token_budget", 1024)

    return Klein9RefMod(
        name=name,
        samples=samples,
        concept_type=concept_type,
        mode=mode,
        resolution=resolution,
        token_budget=token_budget,
        meta=meta
    )
