"""
ComfyUI-Flux2Klein9Mod — No-training "RefMod" reference adapters for FLUX.2 & Klein9

FLUX.2 and Klein9 accept reference image tokens directly within the packed sequence.
RefMods compress reference datasets/images into compact, pre-encoded latents (.safetensors)
stored in models/refmods-klein9/.

Nodes:
  Klein9RefModsLoader       — Load 1-8 pre-extracted RefMods from models/refmods-klein9/
  Klein9RefModApply         — Inject RefMods bundle into standard ComfyUI CONDITIONING
  Klein9RefModExtract       — In-graph VAE encode & compression -> saved .safetensors in models/refmods-klein9/
  Klein9RefModFolderLoader  — Batch load reference images from a disk folder
"""

__author__ = "malcolmrey"
__version__ = "1.0.0"

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
