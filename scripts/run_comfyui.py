#!/usr/bin/env python3
"""Start local ComfyUI pointed at the studio's FP8 Krea 2 checkpoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMFY = next(
    (
        p
        for p in (
            ROOT / "vendor" / "ComfyUI",
            ROOT / "vendor" / "ComfyUI-tmp",
        )
        if (p / "main.py").exists()
    ),
    ROOT / "vendor" / "ComfyUI",
)


def main() -> None:
    if not (COMFY / "main.py").exists():
        raise SystemExit(f"ComfyUI missing at {COMFY}. Re-run git clone into vendor/ComfyUI")

    sys.path.insert(0, str(ROOT))
    from core.comfy_backend import link_models_into_comfy, missing_models

    missing = missing_models()
    if missing:
        raise SystemExit(
            "Missing FP8 models:\n  - "
            + "\n  - ".join(missing)
            + "\nRun: python scripts/download_comfy_fp8.py"
        )
    link_models_into_comfy()

    # Prefer the studio venv's Python (CUDA torch already installed).
    py = Path(sys.executable)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        str(py),
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        "8188",
        "--preview-method",
        "auto",
    ]
    print("[comfy] starting:", " ".join(cmd))
    print("[comfy] UI: http://127.0.0.1:8188")
    os.chdir(COMFY)
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
