#!/usr/bin/env python3
"""Re-quantize an MLX 8-bit model to 4-bit one shard at a time (low peak memory).

Usage: requant.py SRC_DIR DST_DIR [--bits 4] [--group-size 64]
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--group-size", type=int, default=64)
a = ap.parse_args()
src, dst = Path(a.src), Path(a.dst)
dst.mkdir(parents=True, exist_ok=True)
mx.set_memory_limit(8 * 1024**3)  # keep the GPU allocator small; this is a background job

cfg = json.loads((src / "config.json").read_text())
old = cfg["quantization"]
for f in src.iterdir():
    if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("config.json", "model.safetensors.index.json"):
        shutil.copy2(f, dst / f.name)

index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
shards = sorted(set(index.values()))
new_map, total = {}, 0
for shard in shards:
    out_name = shard
    tensors = mx.load(str(src / shard))
    out = {}
    for k, v in tensors.items():
        if k.endswith((".scales", ".biases")) and k.rsplit(".", 1)[0] + ".weight" in index:
            continue  # handled with their weight
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base and base + ".scales" in index:
            def get(name):
                return tensors[name] if name in tensors else mx.load(str(src / index[name]))[name]
            w = mx.dequantize(v, get(base + ".scales"), get(base + ".biases"),
                              group_size=old["group_size"], bits=old["bits"])
            wq, s, b = mx.quantize(w, group_size=a.group_size, bits=a.bits)
            mx.eval(wq, s, b)
            out[k], out[base + ".scales"], out[base + ".biases"] = wq, s, b
            del w
        else:
            out[k] = v
    mx.eval(list(out.values()))
    mx.save_safetensors(str(dst / out_name), out, metadata={"format": "mlx"})
    for k in out:
        new_map[k] = out_name
    total += (dst / out_name).stat().st_size
    print(f"{shard}: {len(out)} tensors -> {(dst / out_name).stat().st_size / 1e9:.2f} GB", flush=True)
    del tensors, out
    gc.collect()
    mx.clear_cache()

q = {"bits": a.bits, "group_size": a.group_size, "mode": old.get("mode", "affine")}
cfg["quantization"] = q
cfg["quantization_config"] = dict(q)
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
(dst / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {"total_size": total}, "weight_map": new_map}, indent=2)
)
print(f"done: {total / 1e9:.1f} GB in {dst}")
