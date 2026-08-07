"""LoRA load/apply helpers for Diffusers pipelines and the official OSS MMDiT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper around nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scaling * self.scale


def _find_linears(module: nn.Module, prefix: str = "") -> dict[str, nn.Linear]:
    found: dict[str, nn.Linear] = {}
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            found[full] = child
        else:
            found.update(_find_linears(child, full))
    return found


def inject_lora_modules(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    target_substrings: tuple[str, ...] = ("wq", "wk", "wv", "wo", "gate", "up", "down", "first", "linear"),
) -> dict[str, LoRALinear]:
    """Replace matching Linear layers with LoRALinear wrappers. Returns name->module map."""
    linears = _find_linears(model)
    injected: dict[str, LoRALinear] = {}
    for name, linear in linears.items():
        if not any(t in name for t in target_substrings):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        lora = LoRALinear(linear, rank=rank, alpha=alpha)
        setattr(parent, parts[-1], lora)
        injected[name] = lora
    return injected


def load_lora_state_into_injected(
    injected: dict[str, LoRALinear],
    lora_path: str | Path,
    strength: float = 1.0,
) -> None:
    """Load PEFT / kohya-style LoRA weights into previously injected modules when keys match.

    Key formats attempted:
      - {name}.lora_down.weight / {name}.lora_up.weight
      - lora_A.{name}.weight / lora_B.{name}.weight
      - {name}.lora_A.weight / {name}.lora_B.weight
    """
    from safetensors.torch import load_file

    state = load_file(str(lora_path))
    # Normalize keys
    flat = {k.replace("module.", ""): v for k, v in state.items()}

    loaded = 0
    for name, module in injected.items():
        candidates = [
            (f"{name}.lora_down.weight", f"{name}.lora_up.weight"),
            (f"{name}.lora_A.weight", f"{name}.lora_B.weight"),
            (f"lora_A.{name}.weight", f"lora_B.{name}.weight"),
            (f"{name}.lora_down", f"{name}.lora_up"),
        ]
        for down_k, up_k in candidates:
            if down_k in flat and up_k in flat:
                module.lora_down.weight.data.copy_(flat[down_k].to(module.lora_down.weight.dtype))
                module.lora_up.weight.data.copy_(flat[up_k].to(module.lora_up.weight.dtype))
                module.scale = float(strength)
                loaded += 1
                break
    if loaded == 0:
        # Still set strength so Diffusers-loaded LoRAs aren't confused with OSS path.
        for module in injected.values():
            module.scale = float(strength)
        print(
            f"[lora] No matching Linear keys found in {lora_path} for OSS MMDiT injection. "
            "If this LoRA was trained for Diffusers/ComfyUI, prefer the Diffusers backend."
        )
    else:
        print(f"[lora] Applied {loaded} LoRA layers from {lora_path} (strength={strength})")


def apply_lora_to_diffusers_pipe(pipe: Any, lora_path: str | Path, strength: float = 1.0) -> None:
    path = Path(lora_path)
    if not path.exists():
        raise FileNotFoundError(path)
    # unload previous adapters if present
    if hasattr(pipe, "unload_lora_weights"):
        try:
            pipe.unload_lora_weights()
        except Exception:  # noqa: BLE001
            pass
    pipe.load_lora_weights(str(path.parent), weight_name=path.name)
    if hasattr(pipe, "set_adapters"):
        # single anonymous adapter
        try:
            pipe.set_adapters(["default"], adapter_weights=[strength])
            return
        except Exception:  # noqa: BLE001
            pass
    if hasattr(pipe, "fuse_lora"):
        pipe.fuse_lora(lora_scale=strength)
