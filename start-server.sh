#!/bin/bash
# Local Qwen server. Defaults keep the Mac usable:
#   QWEN_QUANT=4-bit  (~15 GB; the 8-bit build was deleted 2026-10-07)
#   QWEN_KV=16384     context cap in tokens (prompts ~12k + answer + headroom)
#   QWEN_PREFILL=512  smaller prefill chunks = lower peak memory
#   QWEN_CACHE_GB=2   cap on MLX's cache of freed GPU buffers (uncapped: ~95% of RAM)
# Runs at low CPU/IO priority so your apps win when they compete.
# The python -c shim runs `mlx_vlm server` as before (keeping "mlx_vlm server --model"
# in the command line for porthole down) with two memory fixes. mlx_vlm leaves each
# finished request's GPU arrays in Python reference cycles, and Python rarely collects
# them because to it they are a few small objects: the server grew from 15 to 37 GB.
# A gc.collect() every 30 s frees them; the cache cap hands the memory back to macOS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec taskpolicy -c utility nice -n 10 "$ROOT/.venv/bin/python" -c '
import gc, os, runpy, sys, threading, time
import mlx.core as mx
mx.set_cache_limit(int(float(os.environ.get("QWEN_CACHE_GB") or "2") * 2**30))
def collect():
    while True:
        time.sleep(30)
        gc.collect()
threading.Thread(target=collect, daemon=True).start()
sys.argv = sys.argv[1:]
runpy.run_module("mlx_vlm", run_name="__main__", alter_sys=True)
' mlx_vlm server \
  --model "$ROOT/Qwen3.8-27B-Uncensored-MLX/${QWEN_QUANT:-4-bit}" \
  --host 127.0.0.1 \
  --port 8080 \
  --max-kv-size ${QWEN_KV:-16384} \
  --max-num-seqs 1 \
  --prefill-step-size ${QWEN_PREFILL:-512} \
  --log-level INFO
