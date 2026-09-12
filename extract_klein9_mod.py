#!/usr/bin/env python3
"""
extract_klein9_mod.py — CLI tool to extract FLUX.2 / Klein9 RefMod (.safetensors) from reference images.

Usage:
  python extract_klein9_mod.py \
    --vae path/to/flux2_vae.safetensors \
    --name fk9_aneta_v1_refmod \
    --dataset-dir C:/Development/ai-toolkit/datasets/aneta \
    --concept-type identity \
    --token-budget 1024 \
    --output-dir C:/Development/ComfyUI/models/refmods-klein9/
"""

import argparse
import glob
import os
import sys
import time

# Ensure ComfyUI and custom_nodes in sys.path
COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

import torch
from PIL import Image

try:
    import comfy.sd
    import comfy.utils
    import comfy.model_management
except ImportError:
    print("[Warning] Running outside ComfyUI root; ensure ComfyUI dependencies are installed.")

from core import (
    CONCEPT_TYPES,
    TOKEN_BUDGETS,
    Klein9RefMod,
    fit_token_budget,
    save_refmod,
)
from common import load_image_file, list_media_files, refmods_klein9_dir


def main():
    parser = argparse.ArgumentParser(description="Extract FLUX.2 / Klein9 RefMod reference adapter from images")
    parser.add_argument("--vae", type=str, default=None, help="Path to FLUX.2 / Klein VAE .safetensors file")
    parser.add_argument("--name", type=str, required=True, help="RefMod output name (e.g. fk9_aneta_v1_refmod)")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Directory containing reference images")
    parser.add_argument("--images", nargs="+", default=None, help="Explicit list of image paths")
    parser.add_argument("--concept-type", type=str, default="identity", choices=CONCEPT_TYPES, help="Mod concept type")
    parser.add_argument("--token-budget", type=str, default="1024", choices=TOKEN_BUDGETS, help="Target token budget per image")
    parser.add_argument("--resolution", type=int, default=1024, help="Max image edge resolution")
    parser.add_argument("--max-images", type=int, default=16, help="Max images to encode from dataset")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder (defaults to models/refmods-klein9/)")
    args = parser.parse_args()

    # 1. Collect images
    image_paths = []
    if args.images:
        image_paths.extend(args.images)
    if args.dataset_dir:
        image_paths.extend(list_media_files(args.dataset_dir))

    if not image_paths:
        print(f"[Error] No images found. Provide --dataset-dir or --images.")
        sys.exit(1)

    image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} reference images for RefMod '{args.name}'.")

    # 2. Setup output path
    out_dir = args.output_dir or refmods_klein9_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.name}.safetensors")

    # 3. Load VAE
    device = comfy.model_management.get_torch_device() if hasattr(comfy, "model_management") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_path = args.vae

    if not vae_path:
        # Search for default Flux2 / Klein VAE in models/vae
        default_vae_dir = os.path.join(COMFY_ROOT, "models", "vae")
        candidates = glob.glob(os.path.join(default_vae_dir, "*flux*.safetensors")) + glob.glob(os.path.join(default_vae_dir, "*klein*.safetensors")) + glob.glob(os.path.join(default_vae_dir, "*.safetensors"))
        if candidates:
            vae_path = candidates[0]
            print(f"Auto-selected VAE: {vae_path}")
        else:
            print("[Error] No VAE path provided and none found in models/vae/.")
            sys.exit(1)

    print(f"Loading VAE from {vae_path}...")
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

    # 4. Load and process images
    print("Loading and preprocessing images...")
    loaded_tensors = []
    for p in image_paths:
        t = load_image_file(p, max_edge=args.resolution)
        loaded_tensors.append(t)

    # 5. VAE Encode (One by one to avoid VRAM OOM)
    print(f"Encoding {len(loaded_tensors)} images through VAE on {device}...")
    encoded_latents = []
    for idx, img_t in enumerate(loaded_tensors):
        with torch.no_grad():
            encoded = vae.encode(img_t[:, :, :, :3])
            lat = encoded["samples"] if isinstance(encoded, dict) else encoded
            # Token Budget Compression per image
            if args.token_budget != "native":
                lat = fit_token_budget(lat, args.token_budget, patch_size=2)
            encoded_latents.append(lat)
    
    samples = torch.cat(encoded_latents, dim=0)

    # 7. Construct & Save RefMod
    mod = Klein9RefMod(
        name=args.name,
        samples=samples,
        concept_type=args.concept_type,
        mode="pooled" if args.token_budget != "native" else "encode",
        resolution=args.resolution,
        token_budget=int(args.token_budget) if args.token_budget != "native" else samples.shape[-2] * samples.shape[-1] // 4,
        meta={
            "source_images_count": len(image_paths),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vae_source": os.path.basename(vae_path)
        }
    )

    final_path = save_refmod(mod, out_path)
    file_size_kb = os.path.getsize(final_path) / 1024
    print(f"\n=======================================================")
    print(f"FLUX.2 / Klein9 RefMod Extracted Successfully!")
    print(f"File: {final_path}")
    print(f"Size: {file_size_kb:.1f} KB")
    print(f"Latent Shape: {list(mod.samples.shape)}")
    print(f"Tokens: {mod.token_count} tokens")
    print(f"Concept Type: {mod.concept_type}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
