"""Local ComfyUI backend for Krea 2 Turbo FP8 (fits ~12GB VRAM). No hosted API."""

from __future__ import annotations

import io
import json
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from .config import ROOT

COMFY_ROOT_CANDIDATES = (
    ROOT / "vendor" / "ComfyUI",
    ROOT / "vendor" / "ComfyUI-tmp",
)
COMFY_ROOT = next((p for p in COMFY_ROOT_CANDIDATES if (p / "main.py").exists()), COMFY_ROOT_CANDIDATES[0])
COMFY_MODELS = ROOT / "checkpoints" / "comfy"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188

UNET_CANDIDATES = (
    "krea2_turbo_fp8_scaled.safetensors",
    "krea2_turbo_nvfp4.safetensors",
    "krea2_turbo_int8_convrot.safetensors",
)
CLIP_NAME = "qwen3vl_4b_fp8_scaled.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"


def resolve_unet_name() -> str | None:
    for name in UNET_CANDIDATES:
        if (COMFY_MODELS / "diffusion_models" / name).exists():
            return name
    return None


def models_ready() -> bool:
    return resolve_unet_name() is not None and all(
        (COMFY_MODELS / sub / name).exists()
        for sub, name in (
            ("text_encoders", CLIP_NAME),
            ("vae", VAE_NAME),
        )
    )


def missing_models() -> list[str]:
    missing = []
    if resolve_unet_name() is None:
        missing.append(str(COMFY_MODELS / "diffusion_models" / UNET_CANDIDATES[0]))
    for sub, name in (
        ("text_encoders", CLIP_NAME),
        ("vae", VAE_NAME),
    ):
        p = COMFY_MODELS / sub / name
        if not p.exists():
            missing.append(str(p))
    return missing


def link_models_into_comfy() -> None:
    """Point ComfyUI model folders at our downloaded checkpoints (junctions/symlinks)."""
    if not COMFY_ROOT.exists():
        raise FileNotFoundError(f"ComfyUI not found at {COMFY_ROOT}. Run scripts/setup_comfyui.py")
    mapping = {
        COMFY_ROOT / "models" / "diffusion_models": COMFY_MODELS / "diffusion_models",
        COMFY_ROOT / "models" / "text_encoders": COMFY_MODELS / "text_encoders",
        COMFY_ROOT / "models" / "vae": COMFY_MODELS / "vae",
        COMFY_ROOT / "models" / "loras": ROOT / "characters",
    }
    for dest, src in mapping.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            continue
        try:
            dest.symlink_to(src, target_is_directory=True)
        except OSError:
            # Windows without symlink privilege: copy junction via mklink /J
            import subprocess

            subprocess.check_call(["cmd", "/c", "mklink", "/J", str(dest), str(src)])


IDENTITY_LORA_CANDIDATES = (
    "Krea2/krea2_identity_edit_v1_2.safetensors",
    "Krea2/krea2_identity_edit_v1_2_r64.safetensors",
    "Krea2/krea2_identity_edit_v1_2_r128.safetensors",
)

# CivitAI: Krea2FilterBypass (S1LV3RC01N) — unlocks Krea 2 NSFW / filter refusal.
NSFW_LORA_CANDIDATES = (
    "Krea2/krea2filterbypass3.safetensors",
    "Krea2/krea2filterbypass.safetensors",
)


def _lora_exists(name: str) -> Path | None:
    roots = (
        COMFY_ROOT / "models" / "loras",
        COMFY_MODELS / "loras",
        ROOT / "checkpoints" / "comfy" / "loras",
    )
    for root in roots:
        p = root / name
        if p.exists():
            return p
    return None


def resolve_identity_lora() -> str | None:
    """Return relative LoRA path under ComfyUI models/loras if present."""
    for name in IDENTITY_LORA_CANDIDATES:
        if _lora_exists(name) is not None:
            return name.replace("/", "\\")
    return None


def resolve_nsfw_lora() -> str | None:
    for name in NSFW_LORA_CANDIDATES:
        if _lora_exists(name) is not None:
            return name.replace("/", "\\")
    return None


def identity_edit_ready() -> bool:
    return models_ready() and resolve_identity_lora() is not None


def nsfw_lora_ready() -> bool:
    return resolve_nsfw_lora() is not None


def build_krea2_edit_prompt(
    instruction: str,
    *,
    scene_image_name: str,
    person_image_name: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    steps: int = 12,
    cfg: float = 1.0,
    ref_boost: float = 3.5,
    grounding_px: int = 640,
    identity_lora: str | None = None,
    lora_strength: float = 1.0,
    nsfw_lora: str | None = None,
    nsfw_strength: float = 1.0,
) -> dict[str, Any]:
    """Two-reference Krea2 Edit graph (same idea as the Reference Influencer pack still path).

    IMAGE 1 = scene/background, IMAGE 2 = person. Order is required by the edit LoRA.
    Optional NSFW LoRA (Filter Bypass) stacks after the identity LoRA when enabled.
    """
    unet_name = resolve_unet_name()
    if not unet_name:
        raise FileNotFoundError("Krea 2 Turbo UNET missing. Run scripts/download_comfy_fp8.py")
    lora = identity_lora or resolve_identity_lora()
    if not lora:
        raise FileNotFoundError(
            "Krea2 identity edit LoRA missing. Download "
            "krea2_identity_edit_v1_2_r64.safetensors into models/loras/Krea2/"
        )

    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP_NAME, "type": "krea2", "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
        "4": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": lora,
                "strength_model": float(lora_strength),
            },
        },
    }

    model_for_patch = ["4", 0]
    if nsfw_lora:
        graph["16"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["4", 0],
                "lora_name": nsfw_lora,
                "strength_model": float(nsfw_strength),
            },
        }
        model_for_patch = ["16", 0]

    graph.update(
        {
            "5": {
                "class_type": "LoadImage",
                "inputs": {"image": scene_image_name},
            },
            "6": {
                "class_type": "LoadImage",
                "inputs": {"image": person_image_name},
            },
            "7": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["5", 0], "vae": ["3", 0]},
            },
            "8": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["6", 0], "vae": ["3", 0]},
            },
            "9": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
            },
            "10": {
                "class_type": "Krea2EditModelPatch",
                "inputs": {
                    "model": model_for_patch,
                    "source_latent": ["7", 0],
                    "source_latent_b": ["8", 0],
                    "ref_boost": float(ref_boost),
                    "ref_boost_a": 1.0,
                    "fit_mode": "fit",
                    "vae": ["3", 0],
                    "source_image": ["5", 0],
                    "source_image_b": ["6", 0],
                    "target_latent": ["9", 0],
                },
            },
            "11": {
                "class_type": "Krea2EditGroundedEncode",
                "inputs": {
                    "clip": ["2", 0],
                    "prompt": instruction,
                    "image": ["5", 0],
                    "image_b": ["6", 0],
                    "grounding_px": int(grounding_px),
                    "system_prompt": "",
                },
            },
            "12": {
                "class_type": "Krea2EditGroundedEncode",
                "inputs": {
                    "clip": ["2", 0],
                    "prompt": "",
                    "image": ["5", 0],
                    "image_b": ["6", 0],
                    "grounding_px": int(grounding_px),
                    "system_prompt": "",
                },
            },
            "13": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": int(seed),
                    "steps": int(steps),
                    "cfg": float(cfg),
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["10", 0],
                    "positive": ["11", 0],
                    "negative": ["12", 0],
                    "latent_image": ["9", 0],
                },
            },
            "14": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["13", 0], "vae": ["3", 0]},
            },
            "15": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "ais_krea2_edit", "images": ["14", 0]},
            },
        }
    )
    return graph


def build_krea2_clothes_edit_prompt(
    instruction: str,
    *,
    person_image_name: str,
    mask_image_name: str | None = None,
    identity_image_name: str | None = None,
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    steps: int = 10,
    cfg: float = 1.0,
    denoise: float = 0.92,
    ref_boost: float = 4.0,
    grounding_px: int = 640,
    identity_lora: str | None = None,
    lora_strength: float = 1.0,
) -> dict[str, Any]:
    """Single-image Identity Edit for outfit change, optional brush mask (inpaint region).

    Person photo is IMAGE 1. Optional identity_image_name (face crop) is IMAGE 2.
    When mask_image_name is set (white = change clothes), only that region is resampled.
    """
    unet_name = resolve_unet_name()
    if not unet_name:
        raise FileNotFoundError("Krea 2 Turbo UNET missing.")
    lora = identity_lora or resolve_identity_lora()
    if not lora:
        raise FileNotFoundError("Krea2 identity edit LoRA missing.")

    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP_NAME, "type": "krea2", "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
        "4": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": lora,
                "strength_model": float(lora_strength),
            },
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {"image": person_image_name},
        },
        "16": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["5", 0],
                "upscale_method": "lanczos",
                "width": int(width),
                "height": int(height),
                "crop": "center",
            },
        },
        "7": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["16", 0], "vae": ["3", 0]},
        },
    }

    # Optional second identity reference (face / sheet crop).
    if identity_image_name:
        graph["6"] = {"class_type": "LoadImage", "inputs": {"image": identity_image_name}}
        graph["8"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["6", 0], "vae": ["3", 0]},
        }
        patch_inputs: dict[str, Any] = {
            "model": ["4", 0],
            "source_latent": ["7", 0],
            "source_latent_b": ["8", 0],
            "ref_boost": float(ref_boost),
            "ref_boost_a": 1.0,
            "fit_mode": "fit",
            "vae": ["3", 0],
            "source_image": ["16", 0],
            "source_image_b": ["6", 0],
        }
        encode_img = ["16", 0]
        encode_img_b = ["6", 0]
    else:
        patch_inputs = {
            "model": ["4", 0],
            "source_latent": ["7", 0],
            "ref_boost": float(ref_boost),
            "ref_boost_a": 1.0,
            "fit_mode": "fit",
            "vae": ["3", 0],
            "source_image": ["16", 0],
        }
        encode_img = ["16", 0]
        encode_img_b = None

    # Latent for sampler: masked inpaint region or full-frame empty latent.
    if mask_image_name:
        graph["20"] = {
            "class_type": "LoadImageMask",
            "inputs": {"image": mask_image_name, "channel": "red"},
        }
        graph["21"] = {
            "class_type": "SetLatentNoiseMask",
            "inputs": {"samples": ["7", 0], "mask": ["20", 0]},
        }
        latent_ref = ["21", 0]
        denoise_val = float(denoise)
    else:
        graph["9"] = {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
        }
        latent_ref = ["9", 0]
        denoise_val = 1.0

    patch_inputs["target_latent"] = latent_ref
    graph["10"] = {"class_type": "Krea2EditModelPatch", "inputs": patch_inputs}

    pos_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "prompt": instruction,
        "image": encode_img,
        "grounding_px": int(grounding_px),
        "system_prompt": "",
    }
    neg_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "prompt": "",
        "image": encode_img,
        "grounding_px": int(grounding_px),
        "system_prompt": "",
    }
    if encode_img_b is not None:
        pos_inputs["image_b"] = encode_img_b
        neg_inputs["image_b"] = encode_img_b

    graph["11"] = {"class_type": "Krea2EditGroundedEncode", "inputs": pos_inputs}
    graph["12"] = {"class_type": "Krea2EditGroundedEncode", "inputs": neg_inputs}
    graph["13"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": denoise_val,
            "model": ["10", 0],
            "positive": ["11", 0],
            "negative": ["12", 0],
            "latent_image": latent_ref,
        },
    }
    graph["14"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["13", 0], "vae": ["3", 0]},
    }
    graph["15"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ais_clothes_edit", "images": ["14", 0]},
    }
    return graph


def build_txt2img_prompt(
    prompt: str,
    *,
    negative: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    steps: int = 8,
    cfg: float = 1.0,
    lora_name: str | None = None,
    lora_strength: float = 0.85,
    init_image_name: str | None = None,
    denoise: float = 1.0,
) -> dict[str, Any]:
    """API-format graph for native Krea 2 (ComfyUI >= 0.26, CLIP type=krea2).

    When ``init_image_name`` is set (Comfy input folder filename), runs img2img
    so the background plate anchors lighting / layout.
    """
    unet_name = resolve_unet_name()
    if not unet_name:
        raise FileNotFoundError(
            f"No Turbo UNET found under {COMFY_MODELS / 'diffusion_models'}. "
            "Run: python scripts/download_comfy_fp8.py"
        )
    model_node = "1"
    if lora_name:
        graph: dict[str, Any] = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
            },
            "15": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": [model_node, 0],
                    "lora_name": lora_name,
                    "strength_model": float(lora_strength),
                },
            },
        }
        model_ref = ["15", 0]
    else:
        graph = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
            },
        }
        model_ref = ["1", 0]

    if init_image_name:
        latent_src = ["11", 0]
        graph.update(
            {
                "10": {
                    "class_type": "LoadImage",
                    "inputs": {"image": init_image_name},
                },
                "12": {
                    "class_type": "ImageScale",
                    "inputs": {
                        "image": ["10", 0],
                        "upscale_method": "lanczos",
                        "width": int(width),
                        "height": int(height),
                        "crop": "center",
                    },
                },
                "11": {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": ["12", 0], "vae": ["3", 0]},
                },
            }
        )
        denoise_val = float(denoise)
    else:
        latent_src = ["6", 0]
        graph["6"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
        }
        denoise_val = 1.0

    graph.update(
        {
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": CLIP_NAME,
                    "type": "krea2",
                    "device": "default",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": VAE_NAME},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["2", 0]},
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative, "clip": ["2", 0]},
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": int(seed),
                    "steps": int(steps),
                    "cfg": float(cfg),
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": denoise_val,
                    "model": model_ref,
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": latent_src,
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "ais_krea2", "images": ["8", 0]},
            },
        }
    )
    return graph


class ComfyUIClient:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.base = f"http://{host}:{port}"

    def alive(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/system_stats", timeout=2) as resp:
                return resp.status == 200
        except Exception:  # noqa: BLE001
            return False

    def queue_prompt(self, prompt: dict[str, Any]) -> str:
        data = json.dumps({"prompt": prompt}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ComfyUI rejected prompt ({exc.code}): {detail[:1200]}") from exc
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI queue failed: {body}")
        return prompt_id

    def wait_image(self, prompt_id: str, timeout_s: float = 600.0) -> Image.Image:
        start = time.time()
        while time.time() - start < timeout_s:
            with urllib.request.urlopen(f"{self.base}/history/{prompt_id}", timeout=30) as resp:
                hist = json.loads(resp.read().decode("utf-8"))
            item = hist.get(prompt_id)
            if item:
                outputs = item.get("outputs") or {}
                for node_out in outputs.values():
                    images = node_out.get("images") or []
                    if images:
                        info = images[0]
                        return self._fetch_image(info["filename"], info.get("subfolder", ""), info.get("type", "output"))
            time.sleep(0.75)
        raise TimeoutError(f"ComfyUI job {prompt_id} timed out")

    def _fetch_image(self, filename: str, subfolder: str, folder_type: str) -> Image.Image:
        from urllib.parse import urlencode
        import io

        qs = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        with urllib.request.urlopen(f"{self.base}/view?{qs}", timeout=120) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")

    def upload_image(self, image: Image.Image, name: str | None = None) -> str:
        """Upload a PIL image into ComfyUI's input folder; return the filename."""
        import mimetypes

        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("requests is required to upload images to ComfyUI") from exc

        fname = name or f"ais_{uuid.uuid4().hex}.png"
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        mime = mimetypes.guess_type(fname)[0] or "image/png"
        resp = requests.post(
            f"{self.base}/upload/image",
            files={"image": (fname, buf, mime)},
            data={"overwrite": "true"},
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
        # Comfy returns {"name": "...", "subfolder": "", "type": "input"}
        return body.get("name") or fname

    def generate_edit(
        self,
        instruction: str,
        *,
        scene: Image.Image,
        person: Image.Image,
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        steps: int = 12,
        cfg: float = 1.0,
        ref_boost: float = 3.5,
        grounding_px: int = 640,
        enable_nsfw_lora: bool = False,
        nsfw_strength: float = 1.0,
    ) -> Image.Image:
        """Place person into scene using Krea2 Identity Edit (Reference Influencer still path)."""
        if not self.alive():
            raise RuntimeError(
                f"ComfyUI is not running at {self.base}. Start: python scripts/run_comfyui.py"
            )
        if not identity_edit_ready():
            raise RuntimeError(
                "Krea2 Identity Edit is not ready. Need Turbo FP8 models + "
                "models/loras/Krea2/krea2_identity_edit_v1_2.safetensors and "
                "custom_nodes/comfyui-krea2edit."
            )
        nsfw_name = None
        if enable_nsfw_lora:
            nsfw_name = resolve_nsfw_lora()
            if not nsfw_name:
                raise RuntimeError(
                    "NSFW LoRA enabled but file missing. Expected "
                    "models/loras/Krea2/krea2filterbypass3.safetensors"
                )
        scene_name = self.upload_image(scene, name=f"ais_scene_{uuid.uuid4().hex}.png")
        person_name = self.upload_image(person, name=f"ais_person_{uuid.uuid4().hex}.png")
        graph = build_krea2_edit_prompt(
            instruction,
            scene_image_name=scene_name,
            person_image_name=person_name,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            ref_boost=ref_boost,
            grounding_px=grounding_px,
            nsfw_lora=nsfw_name,
            nsfw_strength=nsfw_strength,
        )
        pid = self.queue_prompt(graph)
        return self.wait_image(pid)

    def generate_clothes_edit(
        self,
        instruction: str,
        *,
        person: Image.Image,
        mask: Image.Image | None = None,
        identity_ref: Image.Image | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        steps: int = 10,
        cfg: float = 1.0,
        denoise: float = 0.92,
        ref_boost: float = 4.0,
        grounding_px: int = 640,
    ) -> Image.Image:
        """Change clothes on a person photo via Identity Edit (+ optional brush mask)."""
        if not self.alive():
            raise RuntimeError(
                f"ComfyUI is not running at {self.base}. Start: python scripts/run_comfyui.py"
            )
        if not identity_edit_ready():
            raise RuntimeError("Krea2 Identity Edit is not ready.")

        person_name = self.upload_image(person, name=f"ais_clothes_{uuid.uuid4().hex}.png")
        mask_name = None
        if mask is not None:
            # White = edit clothes. Store as RGB so LoadImageMask(channel=red) works.
            m = mask.convert("L")
            if m.size != person.size:
                m = m.resize(person.size, Image.Resampling.NEAREST)
            mask_rgb = Image.merge("RGB", (m, m, m))
            mask_name = self.upload_image(mask_rgb, name=f"ais_mask_{uuid.uuid4().hex}.png")

        identity_name = None
        if identity_ref is not None:
            identity_name = self.upload_image(
                identity_ref, name=f"ais_id_{uuid.uuid4().hex}.png"
            )

        graph = build_krea2_clothes_edit_prompt(
            instruction,
            person_image_name=person_name,
            mask_image_name=mask_name,
            identity_image_name=identity_name,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            ref_boost=ref_boost,
            grounding_px=grounding_px,
        )
        pid = self.queue_prompt(graph)
        return self.wait_image(pid)

    def generate(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        seed: int = 0,
        steps: int = 8,
        cfg: float = 1.0,
        lora_name: str | None = None,
        lora_strength: float = 0.85,
        negative: str = "",
        init_image: Image.Image | None = None,
        denoise: float = 1.0,
    ) -> Image.Image:
        if not self.alive():
            raise RuntimeError(
                f"ComfyUI is not running at {self.base}. Start it with: "
                "python scripts/run_comfyui.py"
            )
        init_name = None
        if init_image is not None:
            init_name = self.upload_image(init_image)
        graph = build_txt2img_prompt(
            prompt,
            negative=negative,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            lora_name=lora_name,
            lora_strength=lora_strength,
            init_image_name=init_name,
            denoise=denoise,
        )
        try:
            pid = self.queue_prompt(graph)
            return self.wait_image(pid)
        except Exception as exc:
            if init_name is None:
                raise
            # If img2img graph fails (e.g. LoadImage/PyAV), fall back to text-to-image.
            print(f"[comfy] img2img failed ({exc}); falling back to text-to-image")
            graph = build_txt2img_prompt(
                prompt,
                negative=negative,
                width=width,
                height=height,
                seed=seed,
                steps=steps,
                cfg=cfg,
                lora_name=lora_name,
                lora_strength=lora_strength,
            )
            pid = self.queue_prompt(graph)
            return self.wait_image(pid)
