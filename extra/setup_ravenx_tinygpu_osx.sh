#!/bin/sh
set -eu

python3 - <<'PY'
from tinygrad.runtime.support.system import APLRemotePCIDevice
APLRemotePCIDevice.ensure_app()
print('TinyGPU.app ensured.')
PY

cat <<'EOF'

RavenX TinyGPU bootstrap complete.

Next steps:
  AMD path:
    sh extra/setup_hipcomgr_osx.sh

  NVIDIA path:
    sh extra/setup_nvcc_osx.sh

Usage:
  DEV=TINYGPU python3 tinygrad/apps/llm.py
  DEV=TINYGPU:AMD python3 tinygrad/apps/llm.py
  DEV=TINYGPU:NV python3 tinygrad/apps/llm.py
EOF
