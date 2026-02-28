#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to HF-style checkpoint dir (has config.json, weights, tokenizer files)")
    ap.add_argument("--out", default="./results/mmlu.json", help="Output JSON path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch_size", default="auto", help='Integer, or "auto", or "auto:N"')
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--limit", type=str, default=None, help="Optional: int count or float fraction (e.g., 100 or 0.1)")
    ap.add_argument("--log_samples", action="store_true")
    args = ap.parse_args()

    ckpt = str(Path(args.ckpt).resolve())
    out = str(Path(args.out).resolve())
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lm-eval", "run",
        "--model", "hf",
        "--model_args", f"pretrained={ckpt},dtype={args.dtype}",
        "--tasks", "hellaswag",
        "--device", args.device,
        "--batch_size", args.batch_size,
        "--num_fewshot", str(args.num_fewshot),
        "--output_path", out,
    ]
    if args.limit is not None:
        cmd += ["--limit", args.limit]
    if args.log_samples:
        cmd += ["--log_samples"]

    print("Running:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    sys.exit(main())