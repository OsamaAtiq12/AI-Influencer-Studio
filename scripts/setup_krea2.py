#!/usr/bin/env python3
"""Clone official krea-ai/krea-2 inference code into vendor/krea-2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "vendor" / "krea-2"
REPO = "https://github.com/krea-ai/krea-2.git"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if (DEST / "inference.py").exists():
        print(f"[setup] krea-2 already present at {DEST}")
        return
    if DEST.exists():
        # incomplete clone
        import shutil

        shutil.rmtree(DEST)
    print(f"[setup] Cloning {REPO} -> {DEST}")
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(DEST)])
    print("[setup] Done. Next: download checkpoints (see README) and set OSS_RAW / OSS_TURBO.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("git is required to clone krea-ai/krea-2", file=sys.stderr)
        sys.exit(1)
