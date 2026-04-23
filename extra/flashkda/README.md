# FlashKDA-Metal

**Flash Kimi Delta Attention kernels for Apple Silicon via tinygrad**

Port of [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) CUTLASS kernels to Metal, enabling high-performance gated delta attention on M-series Macs.

## Why

FlashKDA implements Kimi Delta Attention (KDA) — a gated linear attention mechanism with recurrent state, used in Kimi/Moonshot models. The original requires SM90+ NVIDIA GPUs and CUDA 12.9+. This port brings KDA to Apple Silicon using tinygrad's Metal backend.

## Architecture

```
Original FlashKDA          This Port
─────────────────          ──────────────────
CUTLASS templates    →     tinygrad Tensor ops
CUDA SM90 kernels    →     Metal GPU shaders
cuBLAS matmul        →     tinygrad lazy matmul
NVIDIA H100/B100     →     Apple M1/M2/M3/M4/M5
```

## Usage

```python
from tinygrad import Tensor
from extra.flashkda import flash_kda_fwd

B, T, H, K, V = 1, 1024, 16, 128, 128

q = Tensor.randn(B, T, H, K)
k = Tensor.randn(B, T, H, K)
v = Tensor.randn(B, T, H, V)
g = Tensor.randn(B, T, H, K)
beta = Tensor.randn(B, T, H)
A_log = Tensor.randn(H) * 0.01
dt_bias = Tensor.randn(H, K) * 0.01
scale = 1.0 / (K ** 0.5)

out, final_state = flash_kda_fwd(q, k, v, g, beta, scale, A_log, dt_bias)
```

## API

### flash_kda_fwd

```
flash_kda_fwd(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound=-5.0, initial_state=None, chunk_size=64)
```

| Parameter | Dtype | Shape | Description |
|-----------|-------|-------|-------------|
| q | bf16/fp32 | [B, T, H, K] | Query |
| k | bf16/fp32 | [B, T, H, K] | Key |
| v | bf16/fp32 | [B, T, H, V] | Value |
| g | bf16/fp32 | [B, T, H, K] | Gate (pre-activation) |
| beta | bf16/fp32 | [B, T, H] | Beta logits (sigmoid applied internally) |
| scale | float | scalar | Scaling factor |
| A_log | fp32 | [H] | Log-gate parameter |
| dt_bias | fp32 | [H, K] | Gate bias |
| lower_bound | float | scalar | Gate lower bound (default: -5.0) |
| initial_state | optional | [B, H, V, K] | Initial recurrent state |
| chunk_size | int | scalar | Chunk size for tiled computation (default: 64) |

Returns: `(out, final_state)` where out is [B, T, H, V] and final_state is [B, H, V, K].

## Testing

```bash
cd star-platinum-tinygrad
python extra/flashkda/test_kda.py
```

## Hardware Tested

| Device | GPU Cores | Memory | Status |
|--------|-----------|--------|--------|
| M3 Ultra | 60 | 96 GB | Star Platinum Brain A |
| M4 Max | 40 | 128 GB | Star Platinum Brain B |
| M1 Max | 32 | 64 GB | Star Platinum Worker |
| M3 | 10 | 24 GB | Star Platinum Worker |

## Roadmap

- [x] KDA forward pass (chunked)
- [x] Reference implementation for correctness testing
- [x] Benchmark suite
- [ ] KDA backward pass (training)
- [ ] Fused Metal kernel (bypass tinygrad scheduler for peak performance)
- [ ] Distributed KDA across Star Platinum cluster via RDMA
- [ ] ANE offload for gate/decay computation
- [ ] PR to MoonshotAI/FlashKDA upstream

## Credits

- Original FlashKDA: [MoonshotAI](https://github.com/MoonshotAI/FlashKDA)
- Metal port: [@DeadByDawn101](https://github.com/DeadByDawn101)
- Backend: [tinygrad](https://github.com/tinygrad/tinygrad)
- Cluster: Star Platinum (312 GB Apple Silicon)
