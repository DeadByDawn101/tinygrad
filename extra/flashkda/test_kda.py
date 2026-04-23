#!/usr/bin/env python3
"""
FlashKDA-Metal: Correctness and performance tests

Tests the Metal KDA implementation against the reference
implementation for numerical correctness, then benchmarks
on the current Apple Silicon hardware.
"""

import time
import sys
sys.path.insert(0, '/Users/ravenx/Developer/star-platinum-tinygrad')

from tinygrad import Tensor, Device
from extra.flashkda.kda_metal import flash_kda_fwd, flash_kda_fwd_reference

def test_correctness():
    """Test Metal KDA against reference implementation."""
    print(f"Device: {Device.DEFAULT}")
    print("=" * 60)
    print("FlashKDA-Metal Correctness Test")
    print("=" * 60)

    B, T, H, K, V = 2, 64, 8, 128, 128

    # Random inputs
    Tensor.manual_seed(42)
    q = Tensor.randn(B, T, H, K) * 0.1
    k = Tensor.randn(B, T, H, K) * 0.1
    v = Tensor.randn(B, T, H, V) * 0.1
    g = Tensor.randn(B, T, H, K) * 0.1
    beta = Tensor.randn(B, T, H)
    A_log = Tensor.randn(H) * 0.01
    dt_bias = Tensor.randn(H, K) * 0.01
    scale = 1.0 / (K ** 0.5)

    print(f"  Shape: B={B}, T={T}, H={H}, K={K}, V={V}")
    print(f"  Scale: {scale:.6f}")

    # Reference (token-by-token)
    print("  Running reference implementation...")
    out_ref, state_ref = flash_kda_fwd_reference(q, k, v, g, beta, scale, A_log, dt_bias)
    out_ref_np = out_ref.numpy()

    # Metal chunked
    print("  Running Metal chunked implementation...")
    out_metal, state_metal = flash_kda_fwd(q, k, v, g, beta, scale, A_log, dt_bias, chunk_size=16)
    out_metal_np = out_metal.numpy()

    # Compare
    import numpy as np
    max_diff = np.max(np.abs(out_ref_np - out_metal_np))
    mean_diff = np.mean(np.abs(out_ref_np - out_metal_np))

    print(f"  Max diff:  {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")

    if max_diff < 1e-2:
        print("  PASS — outputs match within tolerance")
        return True
    else:
        print("  FAIL — outputs diverge")
        return False


def benchmark(B=1, T=1024, H=16, K=128, V=128, warmup=3, runs=10):
    """Benchmark KDA forward pass on Metal."""
    print("=" * 60)
    print(f"FlashKDA-Metal Benchmark")
    print(f"  Device: {Device.DEFAULT}")
    print(f"  Shape: B={B}, T={T}, H={H}, K={K}, V={V}")
    print("=" * 60)

    Tensor.manual_seed(0)
    q = Tensor.randn(B, T, H, K)
    k = Tensor.randn(B, T, H, K)
    v = Tensor.randn(B, T, H, V)
    g = Tensor.randn(B, T, H, K)
    beta = Tensor.randn(B, T, H)
    A_log = Tensor.randn(H) * 0.01
    dt_bias = Tensor.randn(H, K) * 0.01
    scale = 1.0 / (K ** 0.5)

    # Warmup
    print(f"  Warming up ({warmup} runs)...")
    for _ in range(warmup):
        out, _ = flash_kda_fwd(q, k, v, g, beta, scale, A_log, dt_bias, chunk_size=64)
        out.realize()

    # Benchmark
    print(f"  Benchmarking ({runs} runs)...")
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        out, _ = flash_kda_fwd(q, k, v, g, beta, scale, A_log, dt_bias, chunk_size=64)
        out.realize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    best = min(times)
    worst = max(times)

    tokens_per_sec = (B * T) / avg
    print(f"  Avg:  {avg*1000:.2f} ms")
    print(f"  Best: {best*1000:.2f} ms")
    print(f"  Worst: {worst*1000:.2f} ms")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    print(f"  Memory: B*T*H*K = {B*T*H*K*4/1024/1024:.1f} MB (fp32)")


if __name__ == "__main__":
    Tensor.training = False

    passed = test_correctness()
    print()

    if passed:
        benchmark(B=1, T=512, H=16, K=128, V=128)
        print()
        benchmark(B=1, T=1024, H=16, K=128, V=128)
        print()
        benchmark(B=1, T=2048, H=8, K=128, V=128)
    else:
        print("Skipping benchmark — correctness test failed")
