#!/usr/bin/env python3
"""Resume a partial Hugging Face download via HTTP Range (avoids duplicate .incomplete files)."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_url
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "checkpoints" / "comfy"
CACHE = OUT / ".cache" / "huggingface" / "download" / "diffusion_models"
FINAL = OUT / "diffusion_models" / "krea2_turbo_fp8_scaled.safetensors"
REPO = "Comfy-Org/Krea-2"
FILE = "diffusion_models/krea2_turbo_fp8_scaled.safetensors"
EXPECTED = 13_141_730_784  # from HF API


def largest_incomplete() -> Path | None:
    if not CACHE.exists():
        return None
    files = sorted(CACHE.glob("*.incomplete"), key=lambda p: p.stat().st_size, reverse=True)
    return files[0] if files else None


def main() -> None:
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    if FINAL.exists() and FINAL.stat().st_size == EXPECTED:
        print(f"[ok] already complete: {FINAL}")
        return

    partial = largest_incomplete()
    start = partial.stat().st_size if partial else 0
    if start >= EXPECTED:
        print("[ok] partial already full size; moving into place")
        assert partial is not None
        partial.replace(FINAL)
        return

    url = hf_hub_url(REPO, FILE, repo_type="model")
    print(f"[resume] {start}/{EXPECTED} bytes ({start/1e9:.2f}/{EXPECTED/1e9:.2f} GB)")
    print(f"[url] {url}")

    headers = {"Range": f"bytes={start}-"} if start else {}
    dest = partial or (CACHE / "resume.incomplete")
    CACHE.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=None) as resp:
        if resp.status_code not in (200, 206):
            raise SystemExit(f"HTTP {resp.status_code}: {resp.text[:200]}")
        mode = "ab" if start and resp.status_code == 206 else "wb"
        if mode == "wb" and start:
            print("[warn] server ignored Range; restarting from 0")
            start = 0
        total = EXPECTED
        with open(dest, mode) as fh, tqdm(
            total=total,
            initial=start,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="fp8",
        ) as bar:
            for chunk in resp.iter_bytes(1024 * 1024):
                fh.write(chunk)
                bar.update(len(chunk))

    size = dest.stat().st_size
    if size != EXPECTED:
        raise SystemExit(f"Incomplete size {size} != {EXPECTED}. Re-run to resume.")
    dest.replace(FINAL)
    print(f"[done] {FINAL} ({size} bytes)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[paused] re-run to resume", file=sys.stderr)
        raise SystemExit(130)
