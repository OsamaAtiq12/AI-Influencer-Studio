"""One-shot: character sheet + background → influencer image via Krea2 Identity Edit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .comfy_backend import ComfyUIClient, identity_edit_ready, models_ready, missing_models
from .config import OUTPUTS_DIR, resolve_size


ProgressCb = Callable[[float, str], None]


@dataclass
class OneshotResult:
    image: Image.Image
    path: Path
    prompt: str
    metadata: dict[str, Any]


def _to_pil(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, (str, Path)):
        return Image.open(item).convert("RGB")
    name = getattr(item, "name", None) or getattr(item, "path", None)
    if name:
        return Image.open(name).convert("RGB")
    raise TypeError(f"Cannot read image from {type(item)}")


def _pick_person_reference(sheets: list[Image.Image]) -> Image.Image:
    """Prefer a close-up-ish crop among uploads (face panels win)."""
    if len(sheets) == 1:
        return sheets[0]
    scored = []
    for im in sheets:
        w, h = im.size
        area = w * h
        aspect = w / max(h, 1)
        score = area / max(aspect, 1.0)
        scored.append((score, im))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _face_identity_crop(image: Image.Image) -> Image.Image:
    """Keep head + upper torso only so pants/skirts in the ref can't lock clothing."""
    w, h = image.size
    # Top ~58% — enough for face/hair/shoulders, cuts off most bottoms.
    top = 0
    bottom = max(int(h * 0.58), int(h * 0.4))
    left = int(w * 0.08)
    right = int(w * 0.92)
    if right <= left:
        left, right = 0, w
    return image.crop((left, top, right, bottom))


def _prompt_overrides_clothing(prompt: str) -> bool:
    p = prompt.lower()
    keys = (
        "nude",
        "naked",
        "topless",
        "bottomless",
        "bare breast",
        "bare breasts",
        "bare legs",
        "no pants",
        "no jeans",
        "no underwear",
        "without pants",
        "without clothes",
        "undressed",
        "lingerie",
        "bikini",
        "wearing",
        "outfit",
        "change clothes",
        "remove clothes",
        "remove pants",
        "remove jeans",
        "full nude",
        "fully nude",
    )
    return any(k in p for k in keys)


def _prompt_wants_full_nude(prompt: str) -> bool:
    p = prompt.lower()
    return any(
        k in p
        for k in (
            "fully nude",
            "full nude",
            "completely nude",
            "totally nude",
            "naked",
            "no clothes",
            "without clothes",
            "nude, no",
            "nude no",
        )
    ) or ("nude" in p and ("no pants" in p or "bare legs" in p or "no jeans" in p))


def _prompt_wants_no_bottoms(prompt: str) -> bool:
    """True when the user wants pants/panties/jeans removed (keep top optional)."""
    p = prompt.lower()
    return any(
        k in p
        for k in (
            "no panties",
            "no panty",
            "without panties",
            "no underwear",
            "bottomless",
            "no pants",
            "no jeans",
            "without pants",
            "bare below",
            "only a bra",
            "bra only",
            "wearing only a",
            "wearing only an",
        )
    ) or _prompt_wants_full_nude(p)


def compose_edit_instruction(user_prompt: str = "") -> str:
    pose = (user_prompt or "").strip()
    if not pose:
        pose = "standing naturally in the scene, candid photo"

    if _prompt_wants_full_nude(pose):
        return (
            "CRITICAL CLOTHING RULE: the person must be COMPLETELY NUDE — zero clothing. "
            "No pants, no jeans, no skirt, no underwear, no top, no fishnets, no jacket, no belt, no chains. "
            "Bare breasts and bare legs from hips down. Do not copy any garments from the person reference. "
            f"Scene action: {pose}. "
            "Place this exact person into this exact background scene. "
            "Keep the background unchanged. "
            "Preserve ONLY facial identity, hair, and body shape from the person reference — not clothing. "
            "Photorealistic adult photo, natural skin, coherent anatomy."
        )

    if _prompt_wants_no_bottoms(pose):
        return (
            "CRITICAL CLOTHING RULE: NO BOTTOMS. "
            "Do not generate pants, jeans, shorts, skirts, pantyhose, fishnets, or panties. "
            "Legs and hips must be bare below the waist unless the prompt names a specific bottom garment. "
            "Do not copy jeans or pants from the person reference. "
            f"Follow this outfit exactly: {pose}. "
            "Place this exact person into this exact background scene. "
            "Keep the background unchanged. "
            "Preserve ONLY facial identity, hair, and body shape from the person reference — not pants or jeans. "
            "Photorealistic adult photo, natural skin, coherent anatomy."
        )

    if _prompt_overrides_clothing(pose):
        clothing = (
            "Follow the clothing instructions exactly. "
            "Do NOT keep jeans, pants, shirts, or other garments from the person reference "
            "unless the prompt asks for them."
        )
    else:
        clothing = "Clothing can match the reference or fit the scene naturally."

    return (
        f"Create a candid photorealistic photo of this person: {pose}. "
        "Place this exact person into this exact background scene. "
        "Keep the background unchanged — same location, lighting, color grade, and camera angle. "
        "Preserve exact facial identity, hair, body proportions, and skin details from the person reference. "
        f"{clothing} "
        "Natural skin texture, coherent anatomy, high detail, not illustration, not CGI."
    )


def generate_oneshot(
    character_images: list[Any],
    background: Any,
    *,
    prompt: str = "",
    creativity: float = 0.65,
    resolution: str = "1K",
    seed: int = 0,
    steps: int = 12,
    enable_nsfw_lora: bool = False,
    nsfw_strength: float = 1.0,
    progress: ProgressCb | None = None,
) -> OneshotResult:
    """Generate one influencer image using the same method as the Reference Influencer pack stills."""

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    if not character_images:
        raise ValueError("Upload at least one character-sheet / person photo.")
    if background is None:
        raise ValueError("Upload a background / scene image.")

    sheets = [_to_pil(x) for x in character_images]
    bg = _to_pil(background)
    person = _pick_person_reference(sheets)
    clothing_override = _prompt_overrides_clothing(prompt or "")
    full_nude = _prompt_wants_full_nude(prompt or "")
    no_bottoms = _prompt_wants_no_bottoms(prompt or "")
    # Face/upper crop so Identity Edit cannot lock onto reference pants.
    if clothing_override or no_bottoms:
        person = _face_identity_crop(person)

    width, height = resolve_size(resolution)

    report(0.05, "Checking local ComfyUI + Identity Edit…")
    if not models_ready():
        raise RuntimeError(f"Krea FP8 models missing: {', '.join(missing_models())}")
    if not identity_edit_ready():
        raise RuntimeError(
            "Identity Edit LoRA missing. Place krea2_identity_edit_v1_2.safetensors "
            "under ComfyUI models/loras/Krea2/ and install custom_nodes/comfyui-krea2edit."
        )
    client = ComfyUIClient()
    if not client.alive():
        raise RuntimeError("ComfyUI is not running. Start: python scripts/run_comfyui.py")

    instruction = compose_edit_instruction(prompt)
    ref_boost = max(1.5, min(6.0, 5.5 - float(creativity) * 3.0))
    grounding_px = 640
    if full_nude or no_bottoms:
        ref_boost = max(1.2, min(2.0, ref_boost - 1.5))
        grounding_px = 512
    elif clothing_override:
        ref_boost = max(1.4, ref_boost - 1.0)
        grounding_px = 576

    # Cap extreme NSFW LoRA strength — very high values can hurt adherence oddly.
    nsfw_str = float(nsfw_strength)
    if enable_nsfw_lora and (full_nude or no_bottoms):
        nsfw_str = max(nsfw_str, 2.0)
        nsfw_str = min(nsfw_str, 4.0)

    report(0.2, "Uploading scene + person references…")
    report(0.45, "Generating with Krea2 Identity Edit…")
    image = client.generate_edit(
        instruction,
        scene=bg,
        person=person,
        width=width,
        height=height,
        seed=int(seed),
        steps=max(1, int(steps)),
        cfg=1.0,
        ref_boost=ref_boost,
        grounding_px=grounding_px,
        enable_nsfw_lora=bool(enable_nsfw_lora),
        nsfw_strength=nsfw_str,
    )

    report(0.9, "Saving…")
    out_dir = OUTPUTS_DIR / "oneshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{ts}.png"
    image.save(path)
    meta = {
        "mode": "krea2_identity_edit",
        "prompt_user": prompt,
        "prompt_final": instruction,
        "ref_boost": ref_boost,
        "grounding_px": grounding_px,
        "face_crop_for_clothing": clothing_override,
        "full_nude": full_nude,
        "creativity": creativity,
        "seed": seed,
        "steps": int(steps),
        "resolution": resolution,
        "nsfw_lora": bool(enable_nsfw_lora),
        "nsfw_strength": nsfw_str if enable_nsfw_lora else 0.0,
        "backend": "comfy-krea2edit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": path.name,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report(1.0, "Done")
    return OneshotResult(image=image, path=path, prompt=instruction, metadata=meta)
