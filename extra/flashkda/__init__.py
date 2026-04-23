"""
FlashKDA-Metal: Kimi Delta Attention kernels for Apple Silicon
Port of MoonshotAI/FlashKDA CUTLASS kernels to Metal via tinygrad

Original: https://github.com/MoonshotAI/FlashKDA
Port by: @DeadByDawn101 (RavenX)
Backend: tinygrad Metal (Apple M-series GPU)

This module implements the KDA (Kimi Delta Attention) forward pass
using tinygrad's Metal backend, enabling high-performance gated
delta attention on Apple Silicon hardware.

Tested on:
  - M3 Ultra 96GB (60-core GPU) — Star Platinum Brain A
  - M4 Max 128GB (40-core GPU) — Star Platinum Brain B
  - M1 Max 64GB (32-core GPU) — Star Platinum Worker
"""

from extra.flashkda.kda_metal import flash_kda_fwd

__all__ = ["flash_kda_fwd"]
__version__ = "0.1.0"
