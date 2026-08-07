"""Clothes change: person photo + brush mask → Identity Edit inpaint (pose-safe)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageFilter

from .comfy_backend import ComfyUIClient, identity_edit_ready
from .config import OUTPUTS_DIR, resolve_size
from .oneshot import OneshotResult, _to_pil


ProgressCb = Callable[[float, str], None]


def extract_image_and_mask(editor_value: Any) -> tuple[Image.Image | None, Image.Image | None]:
    """Parse Gradio ImageEditor → (clean RGB person, binary L mask). White = edit."""
    if editor_value is None:
        return None, None

    if isinstance(editor_value, Image.Image):
        return editor_value.convert("RGB"), None

    if isinstance(editor_value, dict):
        bg = editor_value.get("background")
        composite = editor_value.get("composite")
        layers = editor_value.get("layers") or []

        person_src = bg if bg is not None else composite
        if person_src is None and layers:
            person_src = layers[0]
        if person_src is None:
            return None, None
        person = person_src if isinstance(person_src, Image.Image) else _to_pil(person_src)
        person = person.convert("RGB")

        mask = None
        for layer in layers:
            if layer is None:
                continue
            layer_im = layer if isinstance(layer, Image.Image) else _to_pil(layer)
            if layer_im.mode == "RGBA":
                alpha = np.array(layer_im.split()[-1])
                if alpha.max() > 10:
                    mask = Image.fromarray(alpha, mode="L")
                    break
            else:
                arr = np.array(layer_im.convert("L"))
                if arr.max() > 10:
                    mask = Image.fromarray(arr, mode="L")
                    break

        if mask is None and composite is not None and bg is not None:
            c = np.array(
                composite.convert("RGB")
                if isinstance(composite, Image.Image)
                else _to_pil(composite)
            )
            b = np.array(person)
            if c.shape == b.shape:
                diff = np.abs(c.astype(np.int16) - b.astype(np.int16)).max(axis=2)
                if diff.max() > 12:
                    mask = Image.fromarray(
                        np.clip(diff * 10, 0, 255).astype(np.uint8), mode="L"
                    )

        if mask is None and composite is not None:
            c_im = composite if isinstance(composite, Image.Image) else _to_pil(composite)
            c = np.array(c_im.convert("RGB"))
            white = (c[:, :, 0] > 230) & (c[:, :, 1] > 230) & (c[:, :, 2] > 230)
            if white.mean() > 0.01:
                mask = Image.fromarray((white.astype(np.uint8) * 255), mode="L")

        if mask is not None:
            arr = np.array(mask.convert("L"))
            mask = Image.fromarray(np.where(arr > 20, 255, 0).astype(np.uint8), mode="L")
            if mask.size != person.size:
                mask = mask.resize(person.size, Image.Resampling.NEAREST)
            # Slight dilate so seams aren't harsh (MaxFilter = grow white).
            mask = mask.filter(ImageFilter.MaxFilter(7))

        return person, mask

    return _to_pil(editor_value), None


def compose_clothes_instruction(outfit: str) -> str:
    outfit = (outfit or "").strip()
    if not outfit:
        outfit = "a stylish casual outfit"
    return (
        f"Edit only the masked clothing and headwear region. "
        f"Replace what they are wearing with: {outfit}. "
        f"Remove the previous clothes, vest, robe, and any turban — "
        f"put on {outfit} instead (including a properly shaped hat if requested). "
        "Keep the exact same face, body pose, hands, and background outside the edit. "
        "One coherent body, correct anatomy, photorealistic."
    )


def generate_clothes_change(
    editor_value: Any,
    outfit: str,
    *,
    creativity: float = 0.55,
    resolution: str = "1K",
    seed: int = 0,
    progress: ProgressCb | None = None,
) -> OneshotResult:
    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    person, mask = extract_image_and_mask(editor_value)
    if person is None:
        raise ValueError("Upload a person photo, then paint white over the clothes to change.")
    if mask is None:
        raise ValueError(
            "Paint a white mask over the clothes (and turban/hat if replacing headwear). "
            "Without a mask the model regenerates the whole body and anatomy breaks."
        )

    report(0.05, "Checking Identity Edit…")
    if not identity_edit_ready():
        raise RuntimeError("Krea2 Identity Edit is not ready.")
    client = ComfyUIClient()
    if not client.alive():
        raise RuntimeError("ComfyUI is not running. Start: python scripts/run_comfyui.py")

    width, height = resolve_size(resolution)
    instruction = compose_clothes_instruction(outfit)
    # Keep structure from the photo; only resample masked clothes.
    # Higher creativity → slightly more denoise / lower appearance lock.
    ref_boost = max(1.5, min(3.0, 3.0 - float(creativity) * 1.5))
    denoise = min(0.95, max(0.78, 0.78 + float(creativity) * 0.2))

    report(0.35, "Inpainting new outfit (masked, pose locked)…")
    image = client.generate_clothes_edit(
        instruction,
        person=person,
        mask=mask,
        identity_ref=None,
        width=width,
        height=height,
        seed=int(seed),
        steps=12,
        cfg=1.0,
        denoise=denoise,
        ref_boost=ref_boost,
        grounding_px=576,
    )

    report(0.9, "Saving…")
    out_dir = OUTPUTS_DIR / "clothes"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{ts}.png"
    image.save(path)
    mask.save(out_dir / f"{ts}_mask.png")

    meta = {
        "mode": "clothes_inpaint",
        "outfit": outfit,
        "prompt_final": instruction,
        "has_mask": True,
        "ref_boost": ref_boost,
        "denoise": denoise,
        "composite": False,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": path.name,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report(1.0, "Done")
    return OneshotResult(image=image, path=path, prompt=instruction, metadata=meta)
