"""Krea 2 generation pipeline — local Turbo (+ LoRA) with optional API fallback.

Local OSS inference (krea-ai/krea-2) is text-to-image only. Background matching is
implemented as VAE img2img noise blending + scene-aware prompt enrichment.
True `image_style_references` are used when --use-api hits the hosted Krea API.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .api_backend import KreaAPIClient
from .captioning import caption_scene
from .config import (
    KREA2_REPO,
    OSS_RAW,
    OSS_TURBO,
    output_dir,
    resolve_size,
)
from .lora_utils import apply_lora_to_diffusers_pipe, inject_lora_modules, load_lora_state_into_injected
from .profiles import CharacterProfile
from .vram import print_startup_banner


@dataclass
class GenerateRequest:
    slug: str
    prompt: str
    background: Image.Image | Path | None = None
    creativity: float = 0.55  # higher = more deviation from background (img2img strength)
    lora_strength: float = 0.85
    resolution: str | int = "1K"
    width: int | None = None
    height: int | None = None
    seed: int = 0
    num_images: int = 1
    steps: int | None = None
    cfg: float | None = None
    use_api: bool = False
    style_ref_strength: float | None = None  # API style strength; default derived from creativity


@dataclass
class GenerateResult:
    images: list[Image.Image]
    paths: list[Path]
    meta_paths: list[Path]
    metadata: list[dict[str, Any]] = field(default_factory=list)


class InfluencerPipeline:
    def __init__(self, use_api: bool = False):
        self.use_api = use_api
        self._diffusers_pipe = None
        self._oss = None  # dit, ae, encoder, injected
        self._comfy = None
        self._backend = "uninitialized"
        self.vram = print_startup_banner()

    # ------------------------------------------------------------------ setup
    def _ensure_env_checkpoints(self) -> None:
        if OSS_RAW and Path(OSS_RAW).exists():
            os.environ.setdefault("OSS_RAW", str(Path(OSS_RAW).resolve()))
        if OSS_TURBO and Path(OSS_TURBO).exists():
            os.environ.setdefault("OSS_TURBO", str(Path(OSS_TURBO).resolve()))

    def _try_load_comfy(self) -> bool:
        """Prefer local ComfyUI + FP8 Turbo on <=16GB cards (no hosted API)."""
        try:
            from .comfy_backend import ComfyUIClient, models_ready, missing_models
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] Comfy backend import failed: {exc}")
            return False
        if not models_ready():
            miss = missing_models()
            print("[pipeline] Comfy FP8 models not ready yet.")
            for m in miss:
                print(f"  missing: {m}")
            print("  Run: python scripts/download_comfy_fp8.py")
            return False
        client = ComfyUIClient()
        if not client.alive():
            print(
                "[pipeline] ComfyUI FP8 models found but server is not running. "
                "Start: python scripts/run_comfyui.py"
            )
            return False
        self._comfy = client
        self._backend = "comfy"
        print("[pipeline] Using local ComfyUI + Krea 2 Turbo FP8")
        return True

    def _try_load_diffusers(self) -> bool:
        try:
            import torch
            from diffusers import Krea2Pipeline  # type: ignore
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] Diffusers Krea2Pipeline unavailable: {exc}")
            return False
        if not torch.cuda.is_available():
            return False
        self._ensure_env_checkpoints()
        print("[pipeline] Loading Krea 2 Turbo via Diffusers…")
        try:
            pipe = Krea2Pipeline.from_pretrained("krea/Krea-2-Turbo", torch_dtype=torch.bfloat16)
            # 12–16GB cards: keep weights off GPU until needed.
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_gb < 20 and hasattr(pipe, "enable_model_cpu_offload"):
                print(f"[pipeline] Enabling CPU offload for {total_gb:.1f} GB GPU")
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            self._diffusers_pipe = pipe
            self._backend = "diffusers"
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] Diffusers load failed: {exc}")
            return False

    def _try_load_oss(self) -> bool:
        """Load official krea-ai/krea-2 inference stack from vendor checkout."""
        repo = Path(KREA2_REPO)
        if not repo.exists():
            print(
                f"[pipeline] Official krea-2 repo not found at {repo}. "
                "Run: python scripts/setup_krea2.py"
            )
            return False
        turbo = Path(os.environ.get("OSS_TURBO", OSS_TURBO))
        if not turbo.exists():
            print(f"[pipeline] OSS_TURBO checkpoint missing: {turbo}")
            return False

        self._ensure_env_checkpoints()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        try:
            import torch
            from safetensors.torch import load_file
            from autoencoder import QwenAutoencoder
            from encoder import Qwen3VLConditioner, TextEncoderConfig
            from mmdit import SingleMMDiTConfig, SingleStreamDiT
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] Failed importing krea-2 modules: {exc}")
            return False

        print("[pipeline] Loading Krea 2 Turbo via official OSS inference code…")
        # Match inference.py::_pipeline defaults from krea-ai/krea-2
        cfg = SingleMMDiTConfig(
            features=6144,
            tdim=256,
            txtdim=2560,
            heads=48,
            kvheads=12,
            multiplier=4,
            layers=28,
            patch=2,
            channels=16,
            txtheads=20,
            txtkvheads=20,
            txtlayers=12,
        )
        device, dtype = "cuda", torch.bfloat16
        tec = TextEncoderConfig(model_id="Qwen/Qwen3-VL-4B-Instruct")
        ae = QwenAutoencoder()
        encoder = Qwen3VLConditioner(
            tec.model_id, tec.max_length, select_layers=tec.select_layers
        )

        with torch.device("meta"):
            dit = SingleStreamDiT(cfg)
        dit.load_state_dict(load_file(str(turbo)), strict=True, assign=True)
        dit = dit.to(device=device, dtype=dtype).eval().requires_grad_(False)
        ae = ae.to(device=device, dtype=dtype).eval().requires_grad_(False)
        encoder = encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)

        self._oss = {
            "dit": dit,
            "ae": ae,
            "encoder": encoder,
            "injected": {},
            "device": device,
            "dtype": dtype,
        }
        self._backend = "oss"
        return True

    def load(self) -> str:
        if self.use_api:
            self._backend = "api"
            return self._backend
        # On <=16GB prefer ComfyUI FP8 path (local, no API).
        total = self.vram.total_gb or 0
        if total and total < 20:
            if self._try_load_comfy():
                return self._backend
        if self._try_load_diffusers():
            return self._backend
        if self._try_load_comfy():
            return self._backend
        if self._try_load_oss():
            return self._backend
        raise RuntimeError(
            "Could not load a local Krea 2 backend.\n"
            "For 12GB GPUs (recommended):\n"
            "  1) python scripts/download_comfy_fp8.py\n"
            "  2) python scripts/run_comfyui.py\n"
            "Or accept the HF license + set HF_TOKEN and use Diffusers "
            "(krea/Krea-2-Turbo) with CPU offload.\n"
            "Hosted API is optional via --use-api (not required)."
        )

    # ---------------------------------------------------------------- prompts
    def _compose_prompt(self, profile: CharacterProfile, user_prompt: str, background: Image.Image | None) -> str:
        parts = [f"{profile.trigger_word}", user_prompt.strip()]
        if background is not None:
            try:
                scene = caption_scene(background)
                parts.append(
                    f"in this scene: {scene}. match the lighting, color grade, camera perspective, and atmosphere of the background reference"
                )
            except Exception:  # noqa: BLE001
                parts.append(
                    "match the lighting, color grade, and perspective of the background reference image"
                )
        parts.append("photorealistic, coherent anatomy, natural skin, high detail")
        return ", ".join(p for p in parts if p)

    # -------------------------------------------------------------- generation
    def generate(self, req: GenerateRequest) -> GenerateResult:
        if self._backend == "uninitialized":
            self.use_api = req.use_api or self.use_api
            self.load()

        profile = CharacterProfile.load(req.slug)
        width, height = resolve_size(req.resolution, req.width, req.height)

        background = req.background
        if isinstance(background, (str, Path)):
            background = Image.open(background).convert("RGB")
        elif background is not None:
            background = background.convert("RGB")

        prompt = self._compose_prompt(profile, req.prompt, background)
        creativity = float(req.creativity)
        # Map creativity → img2img denoise strength (higher creativity = more freedom).
        img2img_strength = min(0.95, max(0.25, creativity))
        style_strength = (
            req.style_ref_strength
            if req.style_ref_strength is not None
            else max(0.2, min(1.0, 1.0 - creativity * 0.5))
        )

        if self._backend == "api" or req.use_api:
            images = self._generate_api(
                profile,
                prompt,
                background,
                style_strength=style_strength,
                resolution=req.resolution if isinstance(req.resolution, str) else "1K",
                creativity=creativity,
                lora_strength=req.lora_strength,
                seed=req.seed,
                num_images=req.num_images,
                width=width,
                height=height,
            )
        elif self._backend == "comfy":
            images = self._generate_comfy(
                profile,
                prompt,
                width=width,
                height=height,
                seed=req.seed,
                num_images=req.num_images,
                lora_strength=req.lora_strength,
                steps=req.steps or 8,
                cfg=1.0 if req.cfg is None else req.cfg,
            )
        elif self._backend == "diffusers":
            images = self._generate_diffusers(
                profile,
                prompt,
                background,
                width=width,
                height=height,
                seed=req.seed,
                num_images=req.num_images,
                lora_strength=req.lora_strength,
                steps=req.steps,
                cfg=req.cfg,
                img2img_strength=img2img_strength,
            )
        else:
            images = self._generate_oss(
                profile,
                prompt,
                background,
                width=width,
                height=height,
                seed=req.seed,
                num_images=req.num_images,
                lora_strength=req.lora_strength,
                steps=req.steps or 8,
                cfg=0.0 if req.cfg is None else req.cfg,
                img2img_strength=img2img_strength,
            )

        return self._save(profile, images, req, prompt, img2img_strength, style_strength)

    def _generate_comfy(
        self,
        profile: CharacterProfile,
        prompt: str,
        *,
        width: int,
        height: int,
        seed: int,
        num_images: int,
        lora_strength: float,
        steps: int,
        cfg: float,
    ) -> list[Image.Image]:
        assert self._comfy is not None
        lora_name = None
        if profile.lora_file.exists():
            # Comfy loads LoRAs from models/loras; we junction characters/ there.
            # Expect relative path like demo/lora.safetensors
            lora_name = f"{profile.slug}/lora.safetensors"
        else:
            print(
                f"[pipeline] WARNING: no LoRA at {profile.lora_file}; "
                "generating without identity lock (local T2I only)."
            )
        images: list[Image.Image] = []
        for i in range(num_images):
            images.append(
                self._comfy.generate(
                    prompt,
                    width=width,
                    height=height,
                    seed=seed + i,
                    steps=steps,
                    cfg=cfg,
                    lora_name=lora_name,
                    lora_strength=lora_strength,
                )
            )
        return images

    def _generate_api(
        self,
        profile: CharacterProfile,
        prompt: str,
        background: Image.Image | None,
        **kwargs: Any,
    ) -> list[Image.Image]:
        client = KreaAPIClient()
        style_id = profile.training.get("api_style_id")
        images: list[Image.Image] = []
        n = int(kwargs.get("num_images", 1))
        seed = int(kwargs.get("seed", 0))
        for i in range(n):
            img = client.generate(
                prompt,
                background=background,
                style_strength=float(kwargs.get("style_strength", 0.6)),
                resolution=str(kwargs.get("resolution", "1K")),
                creativity=float(kwargs.get("creativity", 0.55)),
                style_id=style_id,
                lora_strength=float(kwargs.get("lora_strength", 0.85)),
                seed=seed + i,
                aspect_ratio=_aspect_ratio(int(kwargs["width"]), int(kwargs["height"])),
            )
            images.append(img)
        return images

    def _generate_diffusers(
        self,
        profile: CharacterProfile,
        prompt: str,
        background: Image.Image | None,
        *,
        width: int,
        height: int,
        seed: int,
        num_images: int,
        lora_strength: float,
        steps: int | None,
        cfg: float | None,
        img2img_strength: float,
    ) -> list[Image.Image]:
        import torch

        pipe = self._diffusers_pipe
        if profile.lora_file.exists():
            apply_lora_to_diffusers_pipe(pipe, profile.lora_file, strength=lora_strength)
        else:
            print(f"[pipeline] WARNING: LoRA missing at {profile.lora_file}; identity may drift.")

        steps = 8 if steps is None else steps
        guidance = 0.0 if cfg is None else cfg
        generator = torch.Generator(device="cuda").manual_seed(seed)

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "num_images_per_prompt": num_images,
            "generator": generator,
        }

        # Prefer native img2img / style APIs when the pipeline exposes them.
        if background is not None:
            if hasattr(pipe, "image"):
                pass
            bg = background.resize((width, height), Image.Resampling.LANCZOS)
            if "image" in getattr(pipe, "__call__").__code__.co_varnames or hasattr(pipe, "img2img"):
                call_kwargs["image"] = bg
                call_kwargs["strength"] = img2img_strength
            else:
                # Prompt already enriched; keep T2I.
                print(
                    "[pipeline] Diffusers pipe has no image/strength args; "
                    "using scene-enriched text prompt (OSS style-ref not in open weights CLI)."
                )

        result = pipe(**call_kwargs)
        return list(result.images)

    def _generate_oss(
        self,
        profile: CharacterProfile,
        prompt: str,
        background: Image.Image | None,
        *,
        width: int,
        height: int,
        seed: int,
        num_images: int,
        lora_strength: float,
        steps: int,
        cfg: float,
        img2img_strength: float,
    ) -> list[Image.Image]:
        import torch
        from sampling import sample  # type: ignore  # from vendor on sys.path

        assert self._oss is not None
        dit, ae, encoder = self._oss["dit"], self._oss["ae"], self._oss["encoder"]
        device, dtype = self._oss["device"], self._oss["dtype"]

        if profile.lora_file.exists():
            if not self._oss["injected"]:
                self._oss["injected"] = inject_lora_modules(dit)
            load_lora_state_into_injected(self._oss["injected"], profile.lora_file, strength=lora_strength)
        else:
            print(f"[pipeline] WARNING: LoRA missing at {profile.lora_file}")

        # Fast path: pure T2I via official sampler when no background.
        if background is None:
            return sample(
                dit,
                ae,
                encoder,
                [prompt] * num_images,
                width=width,
                height=height,
                steps=steps,
                guidance=cfg,
                seed=seed,
                mu=1.15,
                device=device,
                dtype=dtype,
            )

        images: list[Image.Image] = []
        for i in range(num_images):
            images.append(
                self._oss_img2img(
                    dit,
                    ae,
                    encoder,
                    prompt,
                    background,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance=cfg,
                    seed=seed + i,
                    strength=img2img_strength,
                    device=device,
                    dtype=dtype,
                )
            )
        return images

    def _oss_img2img(
        self,
        dit,
        ae,
        encoder,
        prompt: str,
        background: Image.Image,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        strength: float,
        device: str,
        dtype,
    ) -> Image.Image:
        """Flow-matching img2img using AE encode when available."""
        import torch
        from einops import rearrange
        from sampling import prepare, timesteps  # type: ignore

        patch = dit.config.patch
        align = ae.compression * patch
        width = ((width + align - 1) // align) * align
        height = ((height + align - 1) // align) * align

        import numpy as np

        bg = background.resize((width, height), Image.Resampling.LANCZOS)
        np_img = np.array(bg).astype("float32") / 255.0
        pixel = torch.from_numpy(np_img).to(device=device, dtype=dtype)
        pixel = rearrange(pixel, "h w c -> 1 c h w") * 2 - 1

        init_latents = _ae_encode(ae, pixel)
        noise = torch.randn(
            init_latents.shape,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        # strength=1 → pure noise; strength=0 → pure image
        latents = (1.0 - float(strength)) * init_latents + float(strength) * noise

        txt, txtmask = encoder([prompt])
        x, pos, mask = prepare(latents, txt.shape[1], patch, txtmask)
        use_cfg = guidance > 0
        if use_cfg:
            untxt, untxtmask = encoder([""])
            _, unpos, unmask = prepare(latents, untxt.shape[1], patch, untxtmask)

        x1 = (256 // (ae.compression * patch)) ** 2
        x2 = (1280 // (ae.compression * patch)) ** 2
        full_ts = timesteps(x.shape[1], steps, x1, x2, mu=1.15)
        start_idx = 0
        for idx, tval in enumerate(full_ts[:-1]):
            if tval <= float(strength) + 1e-5:
                start_idx = idx
                break
        ts = full_ts[start_idx:]
        if len(ts) < 2:
            ts = full_ts

        img = x
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
            cond = dit(img=img, context=txt, t=t, pos=pos, mask=mask)
            if use_cfg:
                uncond = dit(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                v = cond + guidance * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v

        img = rearrange(
            img,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            ph=patch,
            pw=patch,
            h=height // (ae.compression * patch),
            w=width // (ae.compression * patch),
        )
        img = ae.decode(img.to(torch.bfloat16))
        img = img.clamp(-1, 1) * 0.5 + 0.5
        img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
        return Image.fromarray(img[0])

    def _save(
        self,
        profile: CharacterProfile,
        images: list[Image.Image],
        req: GenerateRequest,
        prompt: str,
        img2img_strength: float,
        style_strength: float,
    ) -> GenerateResult:
        out = output_dir(profile.slug)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        paths: list[Path] = []
        metas: list[Path] = []
        metadata: list[dict[str, Any]] = []
        for i, image in enumerate(images):
            stem = f"{ts}_{i:02d}" if len(images) > 1 else ts
            img_path = out / f"{stem}.png"
            meta_path = out / f"{stem}.json"
            image.save(img_path)
            meta = {
                "slug": profile.slug,
                "trigger_word": profile.trigger_word,
                "prompt_user": req.prompt,
                "prompt_final": prompt,
                "seed": req.seed + i,
                "creativity": req.creativity,
                "lora_strength": req.lora_strength,
                "resolution": req.resolution,
                "backend": self._backend,
                "img2img_strength": img2img_strength,
                "style_ref_strength": style_strength,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "image": img_path.name,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            paths.append(img_path)
            metas.append(meta_path)
            metadata.append(meta)
        return GenerateResult(images=images, paths=paths, meta_paths=metas, metadata=metadata)


def _ae_encode(ae, pixel: "torch.Tensor"):
    """Encode RGB [-1,1] image tensor with Qwen VAE when possible."""
    import torch
    from einops import rearrange

    # Official wrapper only exposes decode; use underlying diffusers VAE.
    inner = getattr(ae, "ae", None)
    if inner is None or not hasattr(inner, "encode"):
        raise RuntimeError("Autoencoder encode() unavailable; cannot run local img2img.")
    x = rearrange(pixel, "b c h w -> b c 1 h w")
    posterior = inner.encode(x)
    latents = posterior.latent_dist.sample() if hasattr(posterior, "latent_dist") else posterior.latents
    # Match decode normalization
    latents = (latents - ae.latents_mean) / ae.latents_std
    return rearrange(latents, "b c 1 h w -> b c h w").to(dtype=pixel.dtype)


def _aspect_ratio(width: int, height: int) -> str:
    from math import gcd

    g = gcd(width, height)
    return f"{width // g}:{height // g}"


def batch_generate(
    slug: str,
    background: Image.Image | Path,
    prompts: list[str],
    *,
    use_api: bool = False,
    creativity: float = 0.55,
    lora_strength: float = 0.85,
    resolution: str | int = "1K",
    width: int | None = None,
    height: int | None = None,
    seed: int = 0,
    num_images: int = 1,
    steps: int | None = None,
    cfg: float | None = None,
) -> list[GenerateResult]:
    pipe = InfluencerPipeline(use_api=use_api)
    pipe.load()
    results: list[GenerateResult] = []
    for i, prompt in enumerate(prompts):
        req = GenerateRequest(
            slug=slug,
            prompt=prompt,
            background=background,
            use_api=use_api,
            seed=seed + i,
            creativity=creativity,
            lora_strength=lora_strength,
            resolution=resolution,
            width=width,
            height=height,
            num_images=num_images,
            steps=steps,
            cfg=cfg,
        )
        results.append(pipe.generate(req))
    return results
