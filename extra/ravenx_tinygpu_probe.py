#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from tinygrad import Tensor, Device
from tinygrad.helpers import Context


def try_device(dev:str):
  out = {"requested": dev, "ok": False}
  try:
    d = Device[dev]
    with Context(DEV=dev):
      test = (Tensor([1, 2, 3], device=dev) * 2).tolist()
    out.update({
      "ok": test == [2, 4, 6],
      "resolved_device": getattr(d, "device", None),
      "selection": getattr(getattr(d, "selection", None), "__dict__", None),
      "test": test,
    })
  except Exception as e:
    out["error"] = str(e)
  return out


def main():
  parser = argparse.ArgumentParser(description="Probe RavenX TinyGPU backends")
  parser.add_argument("--devices", nargs="*", default=["TINYGPU", "TINYGPU:AMD", "TINYGPU:NV"])
  args = parser.parse_args()
  print(json.dumps([try_device(dev) for dev in args.devices], indent=2))


if __name__ == "__main__":
  main()
