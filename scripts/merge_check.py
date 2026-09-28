#!/usr/bin/env python3
"""Algebraic merge-correctness check for PiSSA-style adapters.

For one probe layer this computes

    r = ||W_merged - W_base||_F / (scaling * ||B A||_F)

A correct merge yields r well below 1, because the learned update is a
perturbation of the initialization. A ratio near 1.0 indicates that the
principal components were double-counted during export (the failure
signature this check guards against). In the paper's audit all PiSSA
exports passed with r between 0.081 and 0.246.

Example:
    python scripts/merge_check.py \
        --base Qwen/Qwen2.5-1.5B-Instruct \
        --merged exports/pilot_math_pissa_s42 \
        --adapter saves/pilot/math/pissa \
        --layer model.layers.0.self_attn.q_proj
"""
import argparse
import glob
import json
import os

import torch
from safetensors import safe_open


def find_tensor(model_dir, suffix):
    """Return the first tensor whose key ends with `suffix`, searching every
    safetensors shard of the model directory."""
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.endswith(suffix):
                    return f.get_tensor(key)
    raise KeyError(f"no tensor ending with {suffix!r} under {model_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model directory")
    ap.add_argument("--merged", required=True, help="merged export directory")
    ap.add_argument("--adapter", required=True, help="adapter directory")
    ap.add_argument("--layer", default="model.layers.0.self_attn.q_proj",
                    help="probe layer prefix (default: first q_proj)")
    args = ap.parse_args()

    with open(os.path.join(args.adapter, "adapter_config.json")) as f:
        cfg = json.load(f)
    scaling = cfg["lora_alpha"] / cfg["r"]

    w_base = find_tensor(args.base, args.layer + ".weight").double()
    w_merged = find_tensor(args.merged, args.layer + ".weight").double()
    a = find_tensor(args.adapter, args.layer + ".lora_A.weight").double()
    b = find_tensor(args.adapter, args.layer + ".lora_B.weight").double()

    update = torch.linalg.matrix_norm(w_merged - w_base)          # Frobenius
    adapter = scaling * torch.linalg.matrix_norm(b @ a)
    r = (update / adapter).item()

    print(f"layer            : {args.layer}")
    print(f"||W_merged-W_base||_F          = {update.item():.4f}")
    print(f"scaling * ||BA||_F             = {adapter.item():.4f}")
    print(f"ratio r                        = {r:.3f}")
    if r < 0.5:
        print("verdict          : PASS (well below 1, merge is a perturbation)")
    elif r < 1.0:
        print("verdict          : CHECK (unusually large, inspect the export)")
    else:
        print("verdict          : FAIL (near or above 1, principal components "
              "were probably double-counted)")


if __name__ == "__main__":
    main()
