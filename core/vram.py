"""GPU / VRAM detection and capability warnings."""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    VRAM_COMFORTABLE_RAW_TRAIN_GB,
    VRAM_WARN_RAW_TRAIN_GB,
    VRAM_WARN_TURBO_INFER_GB,
)


@dataclass
class GpuReport:
    cuda_available: bool
    device_name: str | None
    total_gb: float | None
    free_gb: float | None
    warnings: list[str]
    can_turbo_infer: bool
    can_raw_train: bool


def detect_vram() -> GpuReport:
    warnings: list[str] = []
    try:
        import torch
    except ImportError:
        return GpuReport(
            cuda_available=False,
            device_name=None,
            total_gb=None,
            free_gb=None,
            warnings=["PyTorch is not installed. Local Krea 2 inference/training will not work."],
            can_turbo_infer=False,
            can_raw_train=False,
        )

    if not torch.cuda.is_available():
        return GpuReport(
            cuda_available=False,
            device_name=None,
            total_gb=None,
            free_gb=None,
            warnings=[
                "No CUDA GPU detected. Use --use-api with KREA_API_KEY, or install CUDA PyTorch.",
            ],
            can_turbo_infer=False,
            can_raw_train=False,
        )

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    free, _total = torch.cuda.mem_get_info(0)
    free_gb = free / (1024**3)
    name = props.name

    can_turbo = total_gb >= VRAM_WARN_TURBO_INFER_GB
    can_train = total_gb >= VRAM_WARN_RAW_TRAIN_GB

    if total_gb < VRAM_WARN_TURBO_INFER_GB:
        warnings.append(
            f"GPU has ~{total_gb:.1f} GB VRAM; Turbo inference typically needs "
            f">={VRAM_WARN_TURBO_INFER_GB:.0f} GB (or quantized/ComfyUI setups). "
            "Consider --use-api if generation OOMs."
        )
    if total_gb < VRAM_WARN_RAW_TRAIN_GB:
        warnings.append(
            f"GPU has ~{total_gb:.1f} GB VRAM; RAW LoRA training usually needs "
            f">={VRAM_WARN_RAW_TRAIN_GB:.0f} GB (comfortable ~{VRAM_COMFORTABLE_RAW_TRAIN_GB:.0f} GB). "
            "Low-VRAM paths: Ostris AI Toolkit / musubi-tuner with FP8 + block swap, or Fal cloud trainer."
        )
    elif total_gb < VRAM_COMFORTABLE_RAW_TRAIN_GB:
        warnings.append(
            f"~{total_gb:.1f} GB VRAM may train RAW LoRAs with memory optimizations "
            f"(gradient checkpointing / FP8 / block swap). Full-precision training prefers "
            f">={VRAM_COMFORTABLE_RAW_TRAIN_GB:.0f} GB."
        )

    return GpuReport(
        cuda_available=True,
        device_name=name,
        total_gb=total_gb,
        free_gb=free_gb,
        warnings=warnings,
        can_turbo_infer=can_turbo,
        can_raw_train=can_train,
    )


def print_startup_banner(prefix: str = "[studio]") -> GpuReport:
    report = detect_vram()
    if report.cuda_available:
        print(
            f"{prefix} GPU: {report.device_name} | "
            f"{report.total_gb:.1f} GB total, {report.free_gb:.1f} GB free"
        )
    else:
        print(f"{prefix} GPU: none (CUDA unavailable)")
    for w in report.warnings:
        print(f"{prefix} WARNING: {w}")
    return report
