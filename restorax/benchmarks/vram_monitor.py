"""VRAMMonitor — context manager that records peak GPU memory during a block."""

from __future__ import annotations

import torch


class VRAMMonitor:
    """
    Records peak VRAM allocated (in MB) during the enclosed block.

    Usage:
        with VRAMMonitor() as mon:
            model(input)
        print(f"Peak VRAM: {mon.peak_mb:.1f} MB")

    Returns 0.0 on CPU or when CUDA is unavailable.
    """

    def __init__(self) -> None:
        self._peak_bytes: int = 0
        self._base_bytes: int = 0

    def __enter__(self) -> VRAMMonitor:
        if torch.cuda.is_available():
            self._base_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *_: object) -> None:
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            self._peak_bytes = max(0, peak - self._base_bytes)

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024**2)
