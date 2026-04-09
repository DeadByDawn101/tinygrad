#!/bin/bash
# RavenX eGPU Activation Script
# ================================
# Run this on M3 the first time after connecting an external GPU
# over USB4/Thunderbolt 4.
#
# Requirements:
#   AMD: RDNA3+ (RX 7000 series) — recommended: RX 7900 XTX
#   NV:  Ampere+ (RTX 3000+)     — requires Docker for NVCC
#
# Usage:
#   bash extra/ravenx_egpu_activate.sh        # auto-detect
#   bash extra/ravenx_egpu_activate.sh amd    # force AMD path
#   bash extra/ravenx_egpu_activate.sh nv     # force NV path

set -euo pipefail
eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true

CARD=${1:-auto}
TINYGRAD_DIR="$HOME/Projects/tinygrad"
VENV="$HOME/venv/ravenx/bin/python"

echo "╔════════════════════════════════════════════════════╗"
echo "║  RavenX eGPU Activation — M3 Node (M4 Max 128GB) ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

echo "[1/4] Verify TinyGPU.app is running..."
cd "$TINYGRAD_DIR"
PYTHONPATH=. $VENV -c "
from tinygrad.runtime.support.system import APLRemotePCIDevice
APLRemotePCIDevice.ensure_app()
print('  ✅ TinyGPU.app ready')
"

echo ""
echo "[2/4] Probe for connected GPU..."
PROBE_OUT=$(PYTHONPATH=. $VENV extra/ravenx_tinygpu_probe.py 2>&1)
echo "$PROBE_OUT"

# Detect which backend responded
if echo "$PROBE_OUT" | grep -q '"ok": true'; then
  echo ""
  echo "  ✅ GPU detected! Skipping compiler install (already working)"
  SKIP_COMPILER=1
else
  SKIP_COMPILER=0
fi

echo ""
echo "[3/4] Compiler setup..."
if [ "$SKIP_COMPILER" = "0" ]; then
  if [ "$CARD" = "amd" ] || [ "$CARD" = "auto" ]; then
    echo "  Installing AMD HIP compiler (no Docker needed)..."
    curl -fsSL https://raw.githubusercontent.com/tinygrad/tinygrad/master/extra/setup_hipcomgr_osx.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "  ✅ AMD HIP compiler ready"
  elif [ "$CARD" = "nv" ]; then
    echo "  NV path requires Docker Desktop."
    docker --version 2>/dev/null || { echo "  ❌ Docker not installed. Get it at docker.com/products/docker-desktop"; exit 1; }
    curl -fsSL https://raw.githubusercontent.com/tinygrad/tinygrad/master/extra/setup_nvcc_osx.sh | sh
    echo "  ✅ NVCC compiler ready"
  fi
else
  echo "  Skipped (GPU already working)"
fi

echo ""
echo "[4/4] Final probe + smoke test..."
export PATH="$HOME/.local/bin:$PATH"
PYTHONPATH=. $VENV extra/ravenx_tinygpu_probe.py

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  If probe shows ok:true above — YOU'RE LIVE 🖤    ║"
echo "╠════════════════════════════════════════════════════╣"
echo "║  Test run:                                         ║"
echo "║  DEV=TINYGPU PYTHONPATH=. python3 -c \            ║"
echo "║    \"from tinygrad import Tensor;              \\   ║"
echo "║     print((Tensor([1,2,3])*2).tolist())\"          ║"
echo "╚════════════════════════════════════════════════════╝"
