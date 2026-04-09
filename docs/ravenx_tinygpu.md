# RavenX TinyGPU Kernel Plan

## Why this matters

TinyGPU already gives tinygrad a working Apple-side bridge for external AMD/NVIDIA cards on macOS via `APLRemotePCIDevice` and TinyGPU.app. The winning RavenX move is to turn that bridge into a **first-class Apple-Silicon external GPU runtime** instead of treating it like a hidden implementation detail.

## What we added

- `tinygrad/runtime/ops_tinygpu.py`
  - new `DEV=TINYGPU` backend
  - auto-routes to AMD or NV runtime on macOS
  - supports explicit `DEV=TINYGPU:AMD` or `DEV=TINYGPU:NV`
  - supports `TINYGPU_BACKEND={AMD|NV}` env override
- `extra/setup_ravenx_tinygpu_osx.sh`
  - ensures TinyGPU.app is installed
  - prints the correct next-step compiler setup

## Architectural insight

This is not a brand-new PCIe or GPU driver. The Apple/Tiny Corp side is already present:

1. `APLRemotePCIDevice.ensure_app()` installs TinyGPU.app
2. On macOS, `System.pci_devices(...)` returns `APLRemotePCIDevice`
3. `APLRemotePCIDevice` launches TinyGPU.app as a local server over a Unix socket
4. Existing `AMDDevice` and `NVDevice` runtimes operate on top of that remote PCI bridge

So the RavenX kernel seam is:

```text
DEV=TINYGPU
  -> ops_tinygpu.py
  -> select AMD or NV external device
  -> existing HCQ runtime (ops_amd.py / ops_nv.py)
  -> TinyGPU.app bridge via APLRemotePCIDevice
  -> external GPU over USB4 / Thunderbolt on Apple Silicon
```

## Why this is useful

- clean UX: one device name for external cards on Mac
- easier benchmarking and automation in RavenX flows
- creates a place for future RavenX-specific scheduling, memory policy, and fused kernels
- lets us add Apple-Silicon-specific heuristics without patching AMD/NV runtime logic directly

## Next kernel moves

### Phase 1 — ergonomic runtime
- [x] First-class `TINYGPU` backend
- [ ] device enumeration CLI
- [ ] benchmark harness for LLM/token throughput on external GPUs
- [ ] automatic compiler path validation (HIP/NVCC)

### Phase 2 — RavenX fused runtime
- [ ] auto-select backend by model size / compiler availability
- [ ] prefetch + pinned-host-memory policy for USB4 workloads
- [ ] external-GPU KV cache streaming experiments
- [ ] fused launch pipeline for long-context inference workloads

### Phase 3 — universe dent
- [ ] integrate TriAttention + TurboQuant on top of `DEV=TINYGPU`
- [ ] add RavenX kernel presets for external-card inference
- [ ] multi-device dispatch across internal Metal + external NV/AMD

## Usage

```bash
DEV=TINYGPU python3 tinygrad/apps/llm.py
DEV=TINYGPU:AMD python3 tinygrad/apps/llm.py
DEV=TINYGPU:NV python3 tinygrad/apps/llm.py
```

## Caveat

This is a strategic first kernel, not the final fused universe-denter. Real performance work starts after hardware is online and we can profile bandwidth, compile latency, and queue behavior on the attached cards.
