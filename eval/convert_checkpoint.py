#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

def infer_gpt2_config_from_state_dict(sd: dict) -> GPT2Config:
    # vocab + hidden
    wte = sd.get("transformer.wte.weight")
    if wte is None:
        raise ValueError("Can't find transformer.wte.weight in state_dict. Not a GPT-2 style checkpoint?")
    vocab_size, n_embd = wte.shape

    # positions
    wpe = sd.get("transformer.wpe.weight")
    if wpe is None:
        raise ValueError("Can't find transformer.wpe.weight in state_dict (positional embeddings).")
    n_positions = wpe.shape[0]

    # layers
    layer_idxs = []
    pat = re.compile(r"^transformer\.h\.(\d+)\.")
    for k in sd.keys():
        m = pat.match(k)
        if m:
            layer_idxs.append(int(m.group(1)))
    if not layer_idxs:
        raise ValueError("Can't infer n_layer from keys transformer.h.<i>.*")
    n_layer = max(layer_idxs) + 1

    # heads (best-effort inference)
    # Prefer head_dim ~ 64/80/96/128 when possible.
    candidate_head_dims = [64, 80, 96, 128]
    n_head = None
    for hd in candidate_head_dims:
        if n_embd % hd == 0:
            n_head = n_embd // hd
            break
    if n_head is None:
        # fallback: pick a reasonable divisor
        for h in [32, 24, 20, 16, 12, 10, 8, 6, 4, 2, 1]:
            if n_embd % h == 0:
                n_head = h
                break
    if n_head is None:
        n_head = 12  # final fallback

    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_embd=int(n_embd),
        n_layer=int(n_layer),
        n_head=int(n_head),
        n_positions=int(n_positions),
        n_ctx=int(n_positions),
    )
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Input checkpoint dir (contains pytorch_model.bin)")
    ap.add_argument("--out_dir", required=True, help="Output HF dir to write (config.json + tokenizer + weights)")
    ap.add_argument("--tokenizer", default="gpt2", help="Tokenizer name/path to save alongside (default: gpt2)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    w_path = in_dir / "pytorch_model.bin"
    if not w_path.exists():
        raise FileNotFoundError(f"Missing {w_path}")

    sd = torch.load(w_path, map_location="cpu")

    # Some trainers save {"state_dict": ...}
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    cfg = infer_gpt2_config_from_state_dict(sd)
    model = GPT2LMHeadModel(cfg)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:10]:
        print("  missing examples:", missing[:10])
    if unexpected[:10]:
        print("  unexpected examples:", unexpected[:10])

    # Save HF artifacts
    model.save_pretrained(out_dir)
    cfg.save_pretrained(out_dir)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tok.save_pretrained(out_dir)

    # Write a tiny note
    (out_dir / "export_note.json").write_text(json.dumps({
        "source_dir": str(in_dir),
        "tokenizer_source": args.tokenizer,
    }, indent=2))

    print(f"[OK] Exported HF checkpoint to: {out_dir}")
    print("     You should now see: config.json + tokenizer files + model weights.")

if __name__ == "__main__":
    main()