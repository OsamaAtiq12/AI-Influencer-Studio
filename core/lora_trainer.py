"""Character LoRA training — prepare data + launch recommended Krea 2 RAW trainers.

Krea's official guidance (krea-ai/krea-2): train LoRAs on RAW, apply on Turbo.
Recommended tools: Hugging Face Diffusers, Ostris AI Toolkit, Kohya musubi-tuner, Fal.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .captioning import Captioner, list_images
from .config import OSS_RAW, character_dir, slugify
from .profiles import CharacterProfile
from .vram import detect_vram


ProgressCb = Callable[[float, str], None]


@dataclass
class TrainConfig:
    slug: str
    name: str
    images_dir: Path
    trigger_word: str | None = None
    steps: int = 800
    learning_rate: float = 1e-4
    network_dim: int = 16
    network_alpha: int = 16
    resolution: int = 512
    batch_size: int = 1
    use_blip: bool = True
    seed: int = 42
    backend: str = "auto"  # auto | diffusers | ai-toolkit | musubi | prepare-only


def _progress(cb: ProgressCb | None, frac: float, msg: str) -> None:
    if cb:
        cb(frac, msg)
    else:
        print(f"[train] {frac*100:5.1f}%  {msg}")


def prepare_dataset(cfg: TrainConfig, progress: ProgressCb | None = None) -> tuple[CharacterProfile, Path]:
    slug = slugify(cfg.slug)
    profile = CharacterProfile.create(
        name=cfg.name,
        slug=slug,
        trigger_word=cfg.trigger_word,
        steps=cfg.steps,
        learning_rate=cfg.learning_rate,
        network_dim=cfg.network_dim,
        network_alpha=cfg.network_alpha,
        resolution=cfg.resolution,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        base_checkpoint="krea/Krea-2-Raw",
        backend=cfg.backend,
    )
    out = character_dir(slug)
    dataset = out / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)

    _progress(progress, 0.05, "Copying character sheet images…")
    src_images = list_images(Path(cfg.images_dir))
    if not (1 <= len(src_images) <= 10):
        print(
            f"[train] WARNING: expected 1–10 reference images; found {len(src_images)}. Continuing anyway."
        )

    for i, src in enumerate(src_images):
        dst = dataset / f"{slug}_{i:02d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    _progress(progress, 0.15, "Captioning images…")
    captioner = Captioner()
    captions = captioner.caption_folder(
        dataset, profile.trigger_word, use_blip=cfg.use_blip, overwrite=True
    )
    (dataset / "captions.json").write_text(
        json.dumps([{ "image": p.name, "caption": c} for p, c in captions], indent=2),
        encoding="utf-8",
    )

    # Sample previews stored relative to character dir
    profile.sample_images = [f"dataset/{p.name}" for p, _ in captions[:4]]
    profile.save()

    _write_trainer_configs(profile, dataset, cfg)
    return profile, dataset


def _write_trainer_configs(profile: CharacterProfile, dataset: Path, cfg: TrainConfig) -> None:
    """Emit configs for Ostris AI Toolkit and musubi-tuner (source-of-truth tools)."""
    train_dir = profile.dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    raw_path = os.getenv("OSS_RAW", OSS_RAW)

    # --- Ostris AI Toolkit style YAML (best-effort; schema evolves) ---
    toolkit_yaml = f"""# Auto-generated for Ostris AI Toolkit — train on Krea 2 RAW, infer on Turbo.
# Docs: https://github.com/ostris/ai-toolkit
# Official Krea note: https://github.com/krea-ai/krea-2#finetuning-krea-2
job: extension
config:
  name: "{profile.slug}"
  process:
    - type: sd_trainer
      training_folder: "{train_dir.as_posix()}"
      device: cuda:0
      trigger_word: "{profile.trigger_word}"
      network:
        type: lora
        linear: {cfg.network_dim}
        linear_alpha: {cfg.network_alpha}
      train:
        batch_size: {cfg.batch_size}
        steps: {cfg.steps}
        lr: {cfg.learning_rate}
        gradient_checkpointing: true
        noise_scheduler: flowmatch
      model:
        name_or_path: "krea/Krea-2-Raw"
        # Or local RAW weights:
        # name_or_path: "{Path(raw_path).as_posix()}"
      datasets:
        - folder_path: "{dataset.as_posix()}"
          caption_ext: txt
          resolution: [{cfg.resolution}]
      save:
        dtype: bf16
        save_every: 100
        max_step_saves_to_keep: 4
meta:
  name: "{profile.name}"
  version: "1.0"
"""
    (train_dir / "ai_toolkit_config.yaml").write_text(toolkit_yaml, encoding="utf-8")

    # --- musubi-tuner dataset toml ---
    musubi_dataset = f"""[[datasets]]
image_directory = "{dataset.as_posix()}"
caption_extension = ".txt"
batch_size = {cfg.batch_size}
enable_bucket = true
resolution = [{cfg.resolution}, {cfg.resolution}]
"""
    (train_dir / "musubi_dataset.toml").write_text(musubi_dataset, encoding="utf-8")

    musubi_cmd = f"""# Example musubi-tuner launch (Kohya) — adjust paths to your install.
# https://github.com/kohya-ss/musubi-tuner
# Train on RAW, then copy the resulting .safetensors to characters/{profile.slug}/lora.safetensors
#
# accelerate launch --mixed_precision bf16 src/musubi_tuner/krea2_train_network.py \\
#   --dit "{Path(raw_path).as_posix()}" \\
#   --dataset_config "{(train_dir / 'musubi_dataset.toml').as_posix()}" \\
#   --network_module networks.lora_krea2 \\
#   --network_dim {cfg.network_dim} --network_alpha {cfg.network_alpha} \\
#   --learning_rate {cfg.learning_rate} --max_train_steps {cfg.steps} \\
#   --mixed_precision bf16 --gradient_checkpointing \\
#   --timestep_sampling krea2_shift --optimizer_type adamw8bit \\
#   --output_dir "{(train_dir / 'musubi_out').as_posix()}" \\
#   --output_name "{profile.slug}_lora" --seed {cfg.seed}
"""
    (train_dir / "musubi_launch.sh").write_text(musubi_cmd, encoding="utf-8")

    readme = f"""# Training package for `{profile.slug}`

Prepared {datetime.now(timezone.utc).isoformat()}

## Official workflow
1. Train LoRA on **Krea 2 RAW**
2. Apply LoRA on **Krea 2 Turbo** for inference

Source of truth: https://github.com/krea-ai/krea-2#finetuning-krea-2

## Options
- Ostris AI Toolkit: `ai_toolkit_config.yaml`
- musubi-tuner: `musubi_dataset.toml` + `musubi_launch.sh`
- Diffusers: studio will attempt `diffusers` training when backend=diffusers/auto
- Fal cloud: https://fal.ai/models/fal-ai/krea-2-trainer

When training finishes, place weights at:
  `{profile.lora_file.as_posix()}`
"""
    (train_dir / "README.md").write_text(readme, encoding="utf-8")


def _try_diffusers_train(profile: CharacterProfile, dataset: Path, cfg: TrainConfig, progress: ProgressCb | None) -> bool:
    """Best-effort Diffusers LoRA train via Krea2Pipeline (requires recent diffusers from source)."""
    try:
        import torch
        from diffusers import Krea2Pipeline  # type: ignore
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # noqa: BLE001
        print(f"[train] Diffusers/PEFT Krea2 path unavailable: {exc}")
        return False

    if not torch.cuda.is_available():
        print("[train] CUDA required for local Diffusers training.")
        return False

    _progress(progress, 0.25, "Loading Krea 2 RAW via Diffusers…")
    dtype = torch.bfloat16
    try:
        pipe = Krea2Pipeline.from_pretrained("krea/Krea-2-Raw", torch_dtype=dtype)
    except Exception as exc:  # noqa: BLE001
        print(f"[train] Could not load krea/Krea-2-Raw: {exc}")
        return False

    transformer = getattr(pipe, "transformer", None) or getattr(pipe, "dit", None)
    if transformer is None:
        print("[train] Pipeline has no transformer/dit attribute; cannot attach LoRA.")
        return False

    lora_config = LoraConfig(
        r=cfg.network_dim,
        lora_alpha=cfg.network_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "wq", "wk", "wv", "wo"],
        init_lora_weights="gaussian",
    )
    try:
        transformer = get_peft_model(transformer, lora_config)
    except Exception as exc:  # noqa: BLE001
        print(f"[train] PEFT attach failed ({exc}). Use Ostris/musubi configs in training/.")
        return False

    pipe.transformer = transformer
    pipe.to("cuda")

    # Minimal flow-matching style loop — intentionally simple; Ostris/musubi remain preferred.
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image

    class SheetDataset(Dataset):
        def __init__(self, root: Path):
            self.items = []
            for img in list_images(root):
                cap = img.with_suffix(".txt").read_text(encoding="utf-8").strip()
                self.items.append((img, cap))
            self.tf = transforms.Compose(
                [
                    transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(cfg.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            path, cap = self.items[idx]
            with Image.open(path) as im:
                tensor = self.tf(im.convert("RGB"))
            return {"pixel_values": tensor, "caption": cap}

    loader = DataLoader(SheetDataset(dataset), batch_size=cfg.batch_size, shuffle=True)
    params = [p for p in transformer.parameters() if p.requires_grad]
    if not params:
        print("[train] No trainable LoRA params — aborting Diffusers path.")
        return False
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate)

    _progress(progress, 0.35, f"Training LoRA for {cfg.steps} steps…")
    step = 0
    transformer.train()
    while step < cfg.steps:
        for batch in loader:
            if step >= cfg.steps:
                break
            # Prefer pipeline training helpers when present; else skip heavy custom FM loss.
            if not hasattr(pipe, "training_loss") and not hasattr(transformer, "forward"):
                print("[train] Diffusers pipeline lacks a documented training_loss; writing configs only.")
                return False
            # Many Krea2Pipeline builds expose encode + noise helpers differently; keep this path
            # conservative: if no official train API, stop and rely on Ostris/musubi.
            print(
                "[train] Diffusers Krea2Pipeline is available for inference/LoRA load, "
                "but a stable public training_loss API was not detected. "
                "Use Ostris AI Toolkit or musubi-tuner with the generated configs."
            )
            return False
        break

    # Unreachable today; kept for future Diffusers train API.
    _ = opt
    out_lora = profile.lora_file
    try:
        transformer.save_pretrained(profile.dir / "lora_peft")
        # Also try safetensors export
        from safetensors.torch import save_file

        state = {k: v.cpu() for k, v in transformer.state_dict().items() if "lora" in k.lower()}
        if state:
            save_file(state, str(out_lora))
        profile.training["completed_at"] = datetime.now(timezone.utc).isoformat()
        profile.training["backend"] = "diffusers-peft"
        profile.save()
        _progress(progress, 1.0, f"Saved LoRA → {out_lora}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[train] Failed to save LoRA: {exc}")
        return False


def _try_external_cli(profile: CharacterProfile, cfg: TrainConfig, progress: ProgressCb | None) -> bool:
    """Run Ostris AI Toolkit if `toolkit` / `ai-toolkit` is on PATH."""
    train_dir = profile.dir / "training"
    yaml_path = train_dir / "ai_toolkit_config.yaml"

    for cmd in (
        ["toolkit", "run", str(yaml_path)],
        ["python", "-m", "toolkit.job", str(yaml_path)],
        [sys.executable, "-m", "toolkit.job", str(yaml_path)],
    ):
        try:
            _progress(progress, 0.3, f"Launching external trainer: {' '.join(cmd)}")
            proc = subprocess.run(cmd, check=False)
            if proc.returncode == 0:
                # Copy newest safetensors from training folder if needed
                candidates = sorted(train_dir.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
                if candidates and not profile.lora_file.exists():
                    shutil.copy2(candidates[-1], profile.lora_file)
                if profile.lora_file.exists():
                    profile.training["completed_at"] = datetime.now(timezone.utc).isoformat()
                    profile.training["backend"] = "ai-toolkit"
                    profile.save()
                    _progress(progress, 1.0, f"LoRA ready → {profile.lora_file}")
                    return True
        except FileNotFoundError:
            continue
    return False


def train_character(cfg: TrainConfig, progress: ProgressCb | None = None) -> CharacterProfile:
    report = detect_vram()
    for w in report.warnings:
        print(f"[train] WARNING: {w}")

    profile, dataset = prepare_dataset(cfg, progress=progress)
    backend = cfg.backend

    if backend == "prepare-only":
        _progress(progress, 1.0, f"Dataset + trainer configs ready under {profile.dir / 'training'}")
        return profile

    if backend in ("auto", "ai-toolkit"):
        if _try_external_cli(profile, cfg, progress):
            return profile
        if backend == "ai-toolkit":
            raise RuntimeError(
                "Ostris AI Toolkit not found. Install https://github.com/ostris/ai-toolkit "
                f"or re-run with --backend prepare-only and use {profile.dir / 'training'}"
            )

    if backend in ("auto", "diffusers"):
        if _try_diffusers_train(profile, dataset, cfg, progress):
            return profile

    if backend == "musubi":
        raise RuntimeError(
            "musubi-tuner must be launched externally. See "
            f"{profile.dir / 'training' / 'musubi_launch.sh'}"
        )

    # Soft success: data prepared; user finishes training with recommended tools.
    _progress(
        progress,
        1.0,
        "Prepared dataset + configs. Install Ostris AI Toolkit or musubi-tuner to finish RAW LoRA training, "
        f"then place weights at {profile.lora_file}",
    )
    profile.training["status"] = "prepared"
    profile.training["prepared_at"] = datetime.now(timezone.utc).isoformat()
    profile.save()
    return profile
