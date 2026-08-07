"""Shared configuration and path helpers for AI Influencer Studio."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(env_key: str, default: str) -> Path:
    raw = os.getenv(env_key, default)
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p).resolve()


CHARACTERS_DIR = _path("CHARACTERS_DIR", "./characters")
OUTPUTS_DIR = _path("OUTPUTS_DIR", "./outputs")
KREA2_REPO = _path("KREA2_REPO", "./vendor/krea-2")
CHECKPOINTS_DIR = _path("CHECKPOINTS_DIR", "./checkpoints")

OSS_RAW = os.getenv("OSS_RAW", str(CHECKPOINTS_DIR / "krea2_raw.safetensors"))
OSS_TURBO = os.getenv("OSS_TURBO", str(CHECKPOINTS_DIR / "krea2_turbo.safetensors"))

KREA_API_KEY = os.getenv("KREA_API_KEY") or os.getenv("KREA_API_TOKEN") or ""
KREA_API_BASE = os.getenv("KREA_API_BASE", "https://api.krea.ai")

# Approximate VRAM floors (GB). Krea 2 is a ~12B DiT + Qwen3-VL text encoder.
VRAM_WARN_TURBO_INFER_GB = float(os.getenv("VRAM_WARN_TURBO_INFER_GB", "16"))
VRAM_WARN_RAW_TRAIN_GB = float(os.getenv("VRAM_WARN_RAW_TRAIN_GB", "24"))
VRAM_COMFORTABLE_RAW_TRAIN_GB = float(os.getenv("VRAM_COMFORTABLE_RAW_TRAIN_GB", "40"))

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        raise ValueError("Influencer name/slug resolved to empty string")
    if not SLUG_RE.match(s):
        raise ValueError(f"Invalid slug '{s}'. Use lowercase letters, digits, _ or -.")
    return s


def character_dir(slug: str) -> Path:
    return CHARACTERS_DIR / slugify(slug)


def output_dir(slug: str) -> Path:
    d = OUTPUTS_DIR / slugify(slug)
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class ResolutionPreset:
    label: str
    width: int
    height: int


RESOLUTIONS: dict[str, ResolutionPreset] = {
    "1K": ResolutionPreset("1K", 1024, 1024),
    "1.5K": ResolutionPreset("1.5K", 1536, 1536),
    "2K": ResolutionPreset("2K", 2048, 2048),
}


def resolve_size(resolution: str | int, width: int | None = None, height: int | None = None) -> tuple[int, int]:
    if width and height:
        return int(width), int(height)
    if isinstance(resolution, int):
        return resolution, resolution
    key = str(resolution).upper().replace(" ", "")
    if key in RESOLUTIONS:
        p = RESOLUTIONS[key]
        return p.width, p.height
    # allow "1280" or "1280x720"
    if "x" in key.lower():
        w, h = key.lower().split("x", 1)
        return int(w), int(h)
    n = int(key.replace("K", "000") if key.endswith("K") and key[0].isdigit() else key)
    return n, n
