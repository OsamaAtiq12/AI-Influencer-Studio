#!/usr/bin/env python3
"""Download Comfy-Org Krea 2 FP8 stack for local 12GB GPUs (no hosted API)."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "checkpoints" / "comfy"

FILES = [
    "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
    "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
    "vae/qwen_image_vae.safetensors",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = OUT / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"[skip] {dest}")
            continue
        print(f"[download] {name}")
        path = hf_hub_download("Comfy-Org/Krea-2", name, local_dir=str(OUT))
        print(f"[done] {path}")
    print("Local FP8 stack ready under checkpoints/comfy/")
    print("Next: python scripts/run_comfyui.py")


if __name__ == "__main__":
    main()
