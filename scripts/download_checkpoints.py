#!/usr/bin/env python3
"""Download Krea 2 RAW / Turbo safetensors from Hugging Face (gated — requires access + token)."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _download(repo_id: str, preferred_name: str, dest: Path, token: str | None) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(repo_id=repo_id, filename=preferred_name, token=token)
    except Exception:
        files = [f for f in list_repo_files(repo_id, token=token) if f.endswith(".safetensors")]
        if not files:
            raise SystemExit(
                f"No safetensors found in {repo_id}. Accept the license on Hugging Face and set HF_TOKEN."
            )
        print(f"[download] '{preferred_name}' not found; using {files[0]}")
        path = hf_hub_download(repo_id=repo_id, filename=files[0], token=token)
    shutil.copy2(path, dest)
    print(f"[download] → {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--raw", action="store_true", help="Download RAW (training)")
    parser.add_argument("--turbo", action="store_true", help="Download Turbo (inference)")
    parser.add_argument("--all", action="store_true", help="Download both")
    args = parser.parse_args()
    if not (args.raw or args.turbo or args.all):
        args.all = True

    try:
        from huggingface_hub import login
    except ImportError as exc:
        raise SystemExit("pip install huggingface_hub") from exc

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    if args.raw or args.all:
        print("[download] krea/Krea-2-Raw")
        _download("krea/Krea-2-Raw", "raw.safetensors", args.out / "krea2_raw.safetensors", token)
    if args.turbo or args.all:
        print("[download] krea/Krea-2-Turbo")
        _download("krea/Krea-2-Turbo", "turbo.safetensors", args.out / "krea2_turbo.safetensors", token)

    print("Set in .env:")
    print(f"  OSS_RAW={(args.out / 'krea2_raw.safetensors').as_posix()}")
    print(f"  OSS_TURBO={(args.out / 'krea2_turbo.safetensors').as_posix()}")


if __name__ == "__main__":
    main()
