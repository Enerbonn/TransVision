# Adapted from FrameRead; modified in2026 for TransVision. See NOTICE and LICENSE.
"""
hardware.py
GPU/CPU hardware detection and optimal-config computation.

torch is now OPTIONAL: the v2 manual-review pipeline (frame_extractor.py)
only ever reads `.device` off the returned HardwareConfig, so if torch
isn't installed (it's no longer in requirements.txt -- see
PROJECT_SPEC.md), detect_hardware() reports device="cpu" instead of
crashing on import. Full GPU/VRAM detection still works exactly as
before if torch *is* installed (e.g. if you're still using the legacy
Qwen2-VL pipeline in src/llm/ + analyzer.py).
"""

from __future__ import annotations
from dataclasses import dataclass
from src.utils.logger import log

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class HardwareConfig:
    device: str
    gpu_name: str | None
    vram_gb: float
    whisper_dtype: str
    torch_dtype: object | None  # a torch.dtype when torch is installed, else None


def detect_hardware() -> HardwareConfig:
    """
    Detects hardware specifications and builds the configuration object.
    Applies the GPU tier optimization model to select proper dtypes.
    Falls back to CPU cleanly if torch isn't installed at all.
    """
    device = "cpu"
    gpu_name = None
    vram_gb = 0.0
    whisper_dtype = "int8"
    torch_dtype = None

    if _TORCH_AVAILABLE and torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        # High end: >= 20GB (A100, H100, RTX 4090) -> use bfloat16
        if vram_gb >= 20.0:
            torch_dtype = torch.bfloat16
            whisper_dtype = "float16"
        # Mid/low end: < 20GB -> use float16
        else:
            torch_dtype = torch.float16
            whisper_dtype = "float16"

    config = HardwareConfig(
        device=device,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        whisper_dtype=whisper_dtype,
        torch_dtype=torch_dtype,
    )

    log("HARDWARE", "── Hardware Survey ──────────────────────────")
    if config.device == "cuda":
        log("HARDWARE", f"Device:       CUDA (GPU)")
        log("HARDWARE", f"GPU:          {config.gpu_name}")
        log("HARDWARE", f"VRAM:         {config.vram_gb:.1f} GB")
        log("HARDWARE", f"Torch dtype:  {str(config.torch_dtype).split('.')[-1]}")
    else:
        if not _TORCH_AVAILABLE:
            log(
                "HARDWARE",
                "Device:       CPU  (torch not installed -- fine for the v2 pipeline)",
            )
        else:
            log("HARDWARE", f"Device:       CPU")
        log("HARDWARE", f"Whisper dtype: {config.whisper_dtype}")
    log("HARDWARE", "─────────────────────────────────────────────")

    return config
