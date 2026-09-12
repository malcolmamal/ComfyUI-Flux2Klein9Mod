"""
nodes.py — ComfyUI nodes for FLUX.2 & Klein9 "RefMod" reference adapters

Nodes:
  Klein9RefModsLoader       — Load 1-8 pre-extracted RefMods from models/refmods-klein9/
  Klein9RefModApply         — Inject RefMods bundle into standard ComfyUI CONDITIONING
  Klein9RefModExtract       — In-graph VAE encode & compression -> saved .safetensors in models/refmods-klein9/
  Klein9RefModFolderLoader  — Batch load reference images from a disk folder
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import folder_paths
import node_helpers
from comfy_api.latest import io

from .common import (
    list_media_files,
    load_image_file,
    refmods_klein9_dir,
)
from .core import (
    CONCEPT_TYPES,
    TOKEN_BUDGETS,
    Klein9RefMod,
    _blur_latent,
    fit_token_budget,
    load_refmod,
    read_refmod_meta,
    save_refmod,
)

# Register models/refmods-klein9 with ComfyUI's model path system
try:
    folder_paths.add_model_folder_path("refmods-klein9", refmods_klein9_dir())
except Exception:
    pass

_MOD_CACHE: Dict[str, Klein9RefMod] = {}
_MOD_CACHE_MAX = 32


def _list_klein9_mod_names() -> List[str]:
    """List all .safetensors mod files available in models/refmods-klein9/."""
    mods_dir = refmods_klein9_dir()
    if not os.path.isdir(mods_dir):
        return ["none"]
    files = []
    for fn in sorted(os.listdir(mods_dir)):
        if fn.lower().endswith(".safetensors"):
            files.append(os.path.splitext(fn)[0])
    return ["none"] + files if files else ["none"]


def _get_or_load_mod(mod_name: str) -> Optional[Klein9RefMod]:
    """Retrieve mod from in-memory cache or load from models/refmods-klein9/."""
    if not mod_name or mod_name == "none":
        return None
    if mod_name in _MOD_CACHE:
        return _MOD_CACHE[mod_name]

    mods_dir = refmods_klein9_dir()
    path = os.path.join(mods_dir, mod_name + ".safetensors")
    if not os.path.isfile(path):
        return None

    mod = load_refmod(path)
    if len(_MOD_CACHE) >= _MOD_CACHE_MAX:
        # Simple FIFO eviction
        oldest_key = next(iter(_MOD_CACHE))
        _MOD_CACHE.pop(oldest_key, None)

    _MOD_CACHE[mod_name] = mod
    return mod


# ═══════════════════════════════════════════════════════════════════════════
# 1. Klein9RefModsLoader
# ═══════════════════════════════════════════════════════════════════════════

class Klein9RefModsLoader:
    """Load up to 8 pre-encoded FLUX.2 / Klein9 RefMods with individual strength dials."""

    @classmethod
    def INPUT_TYPES(cls):
        mod_choices = _list_klein9_mod_names()
        inputs = {
            "required": {
                "mod_1": (mod_choices, {"default": mod_choices[0] if mod_choices else "none"}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "copies_1": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {}
        }
        for i in range(2, 9):
            inputs["optional"][f"mod_{i}"] = (mod_choices, {"default": "none"})
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05})
            inputs["optional"][f"copies_{i}"] = ("INT", {"default": 1, "min": 1, "max": 8, "step": 1})

        return inputs

    RETURN_TYPES = ("KLEIN9_REFMODS", "STRING")
    RETURN_NAMES = ("klein9_refmods", "prompt_hint")
    FUNCTION = "load_mods"
    CATEGORY = "model/conditioning/klein9"

    def load_mods(self, **kwargs):
        bundle = []
        hint_parts = []

        for i in range(1, 9):
            mod_name = kwargs.get(f"mod_{i}", "none")
            strength = float(kwargs.get(f"strength_{i}", 1.0))
            copies = int(kwargs.get(f"copies_{i}", 1))

            if mod_name and mod_name != "none":
                mod = _get_or_load_mod(mod_name)
                if mod is not None:
                    bundle.append((mod, strength, copies))
                    hint_parts.append(f"[{mod.concept_type}: {mod.name} @ {strength:.2f}x ({mod.token_count} tokens)]")

        prompt_hint = " • ".join(hint_parts) if hint_parts else "No Klein9 RefMods loaded"
        return (bundle, prompt_hint)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Klein9RefModApply
# ═══════════════════════════════════════════════════════════════════════════

class Klein9RefModApply:
    """Inject pre-encoded Klein9 RefMod reference latents into standard ComfyUI CONDITIONING."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "klein9_refmods": ("KLEIN9_REFMODS",),
                "retention": ([
                    "fully_preserved (1.0)",
                    "partially_preserved (0.7)",
                    "attribute_transfer (0.4)",
                    "weak_reference (0.2)",
                    "custom_multiplier",
                ], {"default": "fully_preserved (1.0)"}),
                "custom_retention": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "ref_method": (["default", "index", "uxo", "index_timestep_zero"], {"default": "default"}),
                "blend_mode": (["manifold_blur", "direct_scale"], {"default": "manifold_blur"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply_mods"
    CATEGORY = "model/conditioning/klein9"

    def apply_mods(self, conditioning, klein9_refmods, retention, custom_retention=1.0,
                   ref_method="default", blend_mode="manifold_blur"):
        if not klein9_refmods:
            return (conditioning,)

        # Parse master retention factor
        if "1.0" in retention:
            master_mult = 1.0
        elif "0.7" in retention:
            master_mult = 0.7
        elif "0.4" in retention:
            master_mult = 0.4
        elif "0.2" in retention:
            master_mult = 0.2
        else:
            master_mult = float(custom_retention)

        extracted_latents = []

        for mod, row_strength, copies in klein9_refmods:
            effective_strength = float(row_strength) * master_mult
            if effective_strength <= 0.001:
                continue

            raw_samples = mod.samples.clone() # [B, C, H, W]

            # Apply strength weighting along latent manifold
            if effective_strength < 0.999:
                if blend_mode == "manifold_blur":
                    # Blur factor proportional to weakening
                    blur_factor = max(2, int(round(8.0 * (1.0 - effective_strength) + 2.0)))
                    blurred = _blur_latent(raw_samples, factor=blur_factor)
                    scaled_samples = (effective_strength * raw_samples) + ((1.0 - effective_strength) * blurred)
                else:
                    scaled_samples = raw_samples * effective_strength
            elif effective_strength > 1.001:
                scaled_samples = raw_samples * effective_strength
            else:
                scaled_samples = raw_samples

            # Append samples (unroll batch if multiple images inside one mod)
            for _ in range(copies):
                if scaled_samples.dim() == 4:
                    for b in range(scaled_samples.shape[0]):
                        extracted_latents.append(scaled_samples[b:b+1])
                else:
                    extracted_latents.append(scaled_samples)

        if not extracted_latents:
            return (conditioning,)

        # Inject into conditioning dictionary
        cond_values = {"reference_latents": extracted_latents}
        if ref_method != "default":
            cond_values["reference_latents_method"] = ref_method

        updated_cond = node_helpers.conditioning_set_values(conditioning, cond_values, append=True)
        return (updated_cond,)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Klein9RefModExtract
# ═══════════════════════════════════════════════════════════════════════════

class Klein9RefModExtract:
    """Encode reference images with FLUX.2/Klein VAE and save pre-computed RefMod (.safetensors)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "vae": ("VAE",),
                "mod_name": ("STRING", {"default": "fk9_character_v1_refmod"}),
                "concept_type": (list(CONCEPT_TYPES), {"default": "identity"}),
                "token_budget": (list(TOKEN_BUDGETS), {"default": "1024"}),
                "mode": (["encode", "pooled"], {"default": "encode"}),
            },
            "optional": {
                "save_to_disk": ("BOOLEAN", {"default": True}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("KLEIN9_REFMODS", "LATENT", "STRING")
    RETURN_NAMES = ("klein9_refmods", "latent", "saved_path")
    FUNCTION = "extract_mod"
    CATEGORY = "model/conditioning/klein9"

    def extract_mod(self, images, vae, mod_name, concept_type="identity",
                    token_budget="1024", mode="encode", save_to_disk=True, overwrite=True):
        clean_name = mod_name.strip()
        if not clean_name:
            clean_name = f"fk9_refmod_{int(time.time())}"

        # 1. VAE Encode
        # images is [B, H, W, 3] in [0, 1]
        latent_dict = vae.encode(images[:, :, :, :3])
        samples = latent_dict["samples"] # [B, C, H, W]

        # 2. Token Budget Compression / Pooling
        if mode == "pooled" or token_budget != "native":
            samples = fit_token_budget(samples, token_budget, patch_size=2)

        mod = Klein9RefMod(
            name=clean_name,
            samples=samples,
            concept_type=concept_type,
            mode=mode,
            resolution=max(images.shape[1], images.shape[2]),
            token_budget=int(token_budget) if token_budget != "native" else samples.shape[-2] * samples.shape[-1] // 4,
            meta={"extracted_with": "Klein9RefModExtract", "source_batch_size": images.shape[0]}
        )

        saved_path = ""
        if save_to_disk:
            out_dir = refmods_klein9_dir()
            target_path = os.path.join(out_dir, f"{clean_name}.safetensors")
            if os.path.exists(target_path) and not overwrite:
                target_path = os.path.join(out_dir, f"{clean_name}_{int(time.time())}.safetensors")

            saved_path = save_refmod(mod, target_path)
            # Update cache
            _MOD_CACHE[clean_name] = mod
            print(f"[Klein9RefModExtract] Successfully saved RefMod to {saved_path} ({mod.token_count} tokens)")

        bundle = [(mod, 1.0, 1)]
        return (bundle, {"samples": samples}, saved_path)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Klein9RefModFolderLoader
# ═══════════════════════════════════════════════════════════════════════════

class Klein9RefModFolderLoader:
    """Load an entire folder of reference stills as a batch for RefMod extraction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "C:/Development/ai-toolkit/datasets/aneta"}),
                "max_images": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                "max_edge": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "load_folder"
    CATEGORY = "model/conditioning/klein9"

    def load_folder(self, folder_path, max_images=16, max_edge=1024):
        image_paths = list_media_files(folder_path)
        if not image_paths:
            raise ValueError(f"No image files found in folder: {folder_path}")

        image_paths = image_paths[:max_images]
        loaded_tensors = []
        for p in image_paths:
            t = load_image_file(p, max_edge=max_edge)
            loaded_tensors.append(t)

        # Pad / batch images if dimensions vary
        max_h = max(t.shape[1] for t in loaded_tensors)
        max_w = max(t.shape[2] for t in loaded_tensors)

        padded = []
        for t in loaded_tensors:
            h, w = t.shape[1], t.shape[2]
            if h != max_h or w != max_w:
                pad_h = max_h - h
                pad_w = max_w - w
                t_pad = F.pad(t.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode="replicate").permute(0, 2, 3, 1)
                padded.append(t_pad)
            else:
                padded.append(t)

        batch = torch.cat(padded, dim=0)
        return (batch, len(padded))


# ═══════════════════════════════════════════════════════════════════════════
# Class & Display Mappings
# ═══════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "Klein9RefModsLoader": Klein9RefModsLoader,
    "Klein9RefModApply": Klein9RefModApply,
    "Klein9RefModExtract": Klein9RefModExtract,
    "Klein9RefModFolderLoader": Klein9RefModFolderLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Klein9RefModsLoader": "Load Klein9 RefMods (FLUX.2)",
    "Klein9RefModApply": "Apply Klein9 RefMods (FLUX.2)",
    "Klein9RefModExtract": "Extract Klein9 RefMod (FLUX.2)",
    "Klein9RefModFolderLoader": "Load RefMod Folder (Images)",
}
