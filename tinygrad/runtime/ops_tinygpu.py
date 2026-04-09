from __future__ import annotations

"""RavenX TinyGPU backend.

Adds a first-class `DEV=TINYGPU` entry point for macOS external GPUs attached over
USB4/Thunderbolt through TinyGPU.app. This backend is intentionally thin: the real
compute path still goes through tinygrad's proven AMD/NV runtimes, while this layer
handles backend selection, user ergonomics, and Apple-Silicon-specific messaging.

Examples:
  DEV=TINYGPU python3 tinygrad/apps/llm.py
  DEV=TINYGPU:AMD python3 tinygrad/apps/llm.py
  DEV=TINYGPU:NV python3 tinygrad/apps/llm.py
  TINYGPU_BACKEND=NV DEV=TINYGPU python3 tinygrad/apps/llm.py
"""

import os
from dataclasses import dataclass

from tinygrad.helpers import OSX, getenv
from tinygrad.device import Compiled, Device


@dataclass(frozen=True)
class TinyGPUSelection:
  requested: str
  chosen: str
  reason: str


def _parse_requested_backend(ix:str) -> str|None:
  parts = ix.split(":")
  if len(parts) > 1 and parts[1].upper() in ("AMD", "NV"):
    return parts[1].upper()
  env_backend = getenv("TINYGPU_BACKEND", "")
  return env_backend.upper() if env_backend.upper() in ("AMD", "NV") else None


def _candidate_backends(ix:str) -> list[str]:
  explicit = _parse_requested_backend(ix)
  return [explicit] if explicit else ["AMD", "NV"]


class TinyGPUDevice(Compiled):
  """Proxy device that routes `TINYGPU` to the best available external AMD/NV backend."""

  def __init__(self, ix:str):
    if not OSX:
      raise RuntimeError("TINYGPU is macOS-only and requires Apple Silicon + TinyGPU.app")

    self.selection = self._select_backend(ix)
    self.delegate = Device[self.selection.chosen]

    super().__init__(
      self.delegate.device,
      self.delegate.allocator,
      self.delegate.renderers,
      self.delegate.runtime,
      getattr(self.delegate, "graph", None),
      getattr(self.delegate, "arch", None),
    )

  def _select_backend(self, ix:str) -> TinyGPUSelection:
    errors:list[str] = []
    for backend in _candidate_backends(ix):
      try:
        dev = Device[backend]
        return TinyGPUSelection(requested=ix, chosen=dev.device, reason=f"selected {backend} through TinyGPU bridge")
      except Exception as e:  # pragma: no cover - hardware dependent
        errors.append(f"{backend}: {e}")
    raise RuntimeError(
      "No TinyGPU-compatible external GPU backend is available. "
      "Expected AMD RDNA3+ or NVIDIA Ampere+ over TinyGPU.app. "
      f"Tried: {'; '.join(errors) if errors else 'none'}"
    )

  def synchronize(self): return self.delegate.synchronize()
  def _at_profile_finalize(self): return self.delegate._at_profile_finalize()
  def finalize(self): return self.delegate.finalize()

  def __getattr__(self, name): return getattr(self.delegate, name)
