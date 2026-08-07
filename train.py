#!/usr/bin/env python3
"""CLI: train a character LoRA on Krea 2 RAW (or prepare dataset + trainer configs)."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.lora_trainer import TrainConfig, train_character
from core.vram import print_startup_banner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train (or prepare) a character LoRA for AI Influencer Studio. "
        "Official workflow: train on Krea 2 RAW, run on Turbo."
    )
    parser.add_argument("--slug", required=True, help="Influencer slug, e.g. jane")
    parser.add_argument("--name", default=None, help="Display name (defaults to slug)")
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Folder with 1–10 character sheet images",
    )
    parser.add_argument("--trigger", default=None, help="LoRA trigger word")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=16, help="LoRA rank / network_dim")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-blip",
        action="store_true",
        help="Skip BLIP and use templated captions",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "diffusers", "ai-toolkit", "musubi", "prepare-only"],
        default="auto",
        help="Training backend. 'prepare-only' always succeeds and writes Ostris/musubi configs.",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Train a hosted Krea style via API instead of local RAW LoRA (requires KREA_API_KEY).",
    )
    args = parser.parse_args()
    print_startup_banner()

    if args.use_api:
        from core.api_backend import KreaAPIClient
        from core.captioning import list_images
        from core.profiles import CharacterProfile

        images = list_images(args.images)
        profile = CharacterProfile.create(
            name=args.name or args.slug,
            slug=args.slug,
            trigger_word=args.trigger,
            backend="krea-api",
        )
        client = KreaAPIClient()
        style_id = client.train_style(
            profile.name,
            images,
            trigger_word=profile.trigger_word,
        )
        profile.training["api_style_id"] = style_id
        profile.training["status"] = "api_trained"
        # Copy references for the GUI
        dataset = profile.dir / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        import shutil

        for i, src in enumerate(images):
            shutil.copy2(src, dataset / f"{profile.slug}_{i:02d}{src.suffix.lower()}")
        profile.sample_images = [f"dataset/{p.name}" for p in sorted(dataset.iterdir()) if p.suffix]
        profile.save()
        print(f"[train] Hosted style id={style_id} saved to {profile.dir / 'profile.json'}")
        return

    cfg = TrainConfig(
        slug=args.slug,
        name=args.name or args.slug,
        images_dir=args.images,
        trigger_word=args.trigger,
        steps=args.steps,
        learning_rate=args.lr,
        network_dim=args.dim,
        network_alpha=args.alpha,
        resolution=args.resolution,
        batch_size=args.batch_size,
        use_blip=not args.no_blip,
        seed=args.seed,
        backend=args.backend,
    )
    profile = train_character(cfg)
    print(f"[train] Character dir: {profile.dir}")
    print(f"[train] Expected LoRA path: {profile.lora_file}")
    if not profile.lora_file.exists():
        print(
            "[train] LoRA weights not produced yet. Finish training with Ostris AI Toolkit "
            f"or musubi-tuner using configs in {profile.dir / 'training'}"
        )


if __name__ == "__main__":
    main()
