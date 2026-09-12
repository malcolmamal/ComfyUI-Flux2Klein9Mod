# ComfyUI-Flux2Klein9Mod

**No-training "RefMod" reference adapters for FLUX.2 and Klein9.**

RefMods pre-encode reference images and datasets into compact, reusable `.safetensors` adapters stored in `models/refmods-klein9/`. Rather than loading and VAE-encoding raw 4K/1024px image files on every generation run (which adds 4096+ tokens and heavy attention latency), RefMods inject pre-computed reference latents directly into the FLUX.2 / Klein9 2D RoPE sequence with instant loading, zero VAE overhead, and full multi-character scaling.

---

## 📦 Features

- **Pre-encoded Latent Storage:** Stored in `ComfyUI/models/refmods-klein9/`.
- **Zero VAE Generation Latency:** Latents are loaded directly from disk without runtime VAE encoding.
- **Configurable Token Budgets:** Native resolution or spatial pooling to target budgets (e.g. 512, 1024, 2048 tokens).
- **Multi-Subject Scaling:** Load up to 8 distinct RefMods simultaneously (`mod_1` through `mod_8`) with independent strengths.
- **Manifold Blending:** Smooth weakening and strength attenuation using spatial low-pass filtering along the latent manifold instead of noise degradation.
- **Universal Compatibility:** Works with any standard ComfyUI `CONDITIONING` graph for FLUX.2, Klein 4B, Klein 8B, and Klein 9B models.

---

## 🚀 Installation

1. Navigate to your ComfyUI `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone or place this repository:
   ```bash
   git clone https://github.com/malcolmrey/ComfyUI-Flux2Klein9Mod.git
   ```
3. Ensure the model folder exists:
   ```text
   ComfyUI/models/refmods-klein9/
   ```
4. Restart ComfyUI.

---

## 🧩 Nodes

| Node Name | Category | Description |
| :--- | :--- | :--- |
| **`Load Klein9 RefMods (FLUX.2)`** (`Klein9RefModsLoader`) | `model/conditioning/klein9` | Loads 1–8 RefMod `.safetensors` files from `models/refmods-klein9/` with individual strengths (0.0–3.0) and repeat counts. |
| **`Apply Klein9 RefMods (FLUX.2)`** (`Klein9RefModApply`) | `model/conditioning/klein9` | Injects the loaded RefMod bundle into standard ComfyUI `CONDITIONING` (`reference_latents`). |
| **`Extract Klein9 RefMod (FLUX.2)`** (`Klein9RefModExtract`) | `model/conditioning/klein9` | In-graph VAE encode & token compression of input images directly into a saved `.safetensors` file in `models/refmods-klein9/`. |
| **`Load RefMod Folder (Images)`** (`Klein9RefModFolderLoader`) | `model/conditioning/klein9` | Batch loads an entire folder of reference images for extraction. |

---

## 🛠️ Graph Wiring

```text
[ CLIPTextEncode ] (Prompt) 
         │
         ▼
[ Apply Klein9 RefMods (FLUX.2) ] ◄── [ Load Klein9 RefMods (FLUX.2) ]
         │ (conditioning out)          (Selects models from models/refmods-klein9/)
         ▼
[ CFGGuider / SamplerCustomAdvanced ]
```

---

## 💻 CLI Extraction Tool

You can also pre-encode image datasets using the CLI:

```bash
python custom_nodes/ComfyUI-Flux2Klein9Mod/extract_klein9_mod.py \
  --name fk9_aneta_v1_refmod \
  --dataset-dir C:/Development/ai-toolkit/datasets/aneta \
  --concept-type identity \
  --token-budget 1024 \
  --output-dir C:/Development/ComfyUI/models/refmods-klein9/
```
