#!/bin/bash
# Local Qwen server. Defaults keep the Mac usable:
#   QWEN_QUANT=4-bit  (~15 GB; the 8-bit build was deleted 2026-10-07)
#   QWEN_KV=16384     context cap in tokens (prompts ~12k + answer + headroom)
#   QWEN_PREFILL=512  smaller prefill chunks = lower peak memory
# Runs at low CPU/IO priority so your apps win when they compete.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec taskpolicy -c utility nice -n 10 "$ROOT/.venv/bin/python" -m mlx_vlm server \
  --model "$ROOT/Qwen3.8-27B-Uncensored-MLX/${QWEN_QUANT:-4-bit}" \
  --host 127.0.0.1 \
  --port 8080 \
  --max-kv-size ${QWEN_KV:-16384} \
  --max-num-seqs 1 \
  --prefill-step-size ${QWEN_PREFILL:-512} \
  --log-level INFO
