#!/usr/bin/env python3
"""CLI: generate influencer images into a scene with Krea 2 Turbo + character LoRA."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from core.pipeline import GenerateRequest, InfluencerPipeline, batch_generate
from core.vram import print_startup_banner


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        text = Path(args.prompts_file).read_text(encoding="utf-8")
        prompts = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if not prompts:
            raise SystemExit(f"No prompts found in {args.prompts_file}")
        return prompts
    if not args.prompt:
        raise SystemExit("Provide --prompt or --prompts-file")
    return [args.prompt]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scene composites with a trained influencer LoRA.")
    parser.add_argument("--slug", required=True, help="Trained influencer slug")
    parser.add_argument("--background", required=True, type=Path, help="Background / scene image")
    parser.add_argument("--prompt", default=None, help="Pose / action / mood prompt")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Text file with one prompt per line (batch mode)",
    )
    parser.add_argument("--creativity", type=float, default=0.55, help="0–1; higher = freer from background")
    parser.add_argument("--lora-strength", type=float, default=0.85)
    parser.add_argument("--resolution", default="1K", help="1K | 1.5K | 2K | WxH")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=1, help="Variations per prompt")
    parser.add_argument("--steps", type=int, default=None, help="Default Turbo=8")
    parser.add_argument("--cfg", type=float, default=None, help="Default Turbo=0.0")
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use hosted Krea 2 API (KREA_API_KEY) instead of local GPU",
    )
    args = parser.parse_args()
    print_startup_banner()

    prompts = _load_prompts(args)
    background = Image.open(args.background).convert("RGB")

    if len(prompts) == 1 and args.num_images >= 1 and not args.prompts_file:
        pipe = InfluencerPipeline(use_api=args.use_api)
        pipe.load()
        result = pipe.generate(
            GenerateRequest(
                slug=args.slug,
                prompt=prompts[0],
                background=background,
                creativity=args.creativity,
                lora_strength=args.lora_strength,
                resolution=args.resolution,
                width=args.width,
                height=args.height,
                seed=args.seed,
                num_images=args.num_images,
                steps=args.steps,
                cfg=args.cfg,
                use_api=args.use_api,
            )
        )
        for p in result.paths:
            print(f"[generate] saved {p}")
        return

    results = batch_generate(
        args.slug,
        background,
        prompts,
        use_api=args.use_api,
        creativity=args.creativity,
        lora_strength=args.lora_strength,
        resolution=args.resolution,
        width=args.width,
        height=args.height,
        seed=args.seed,
        num_images=args.num_images,
        steps=args.steps,
        cfg=args.cfg,
    )
    for result in results:
        for p in result.paths:
            print(f"[generate] saved {p}")


if __name__ == "__main__":
    main()
