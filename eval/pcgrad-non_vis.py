#!/usr/bin/env python3
"""
Compare a PCGrad checkpoint vs a non-PCGrad checkpoint in *weight space*.

Produces:
  1) Heatmap: per-layer/module RMS of (Δ_pcgrad - Δ_nonpcgrad) where Δ = (θ_ckpt - θ_base)
  2) Line plot: per-layer cosine similarity between Δ_pcgrad and Δ_nonpcgrad (how aligned the updates are)
  3) Line plot: per-layer log10 RMS ratio log10( RMS(Δ_pcgrad) / RMS(Δ_nonpcgrad) )
  4) Table (csv): numeric summaries

Example:
  python compare_pcgrad_vs_nonpcgrad.py \
    --base_model gpt2 \
    --non_ckpt ./ckpts/GradDiff \
    --pc_ckpt  ./ckpts/GA_PCGRAD \
    --out_dir  ./viz/GA_vs_GA_PCGRAD

Run it again for your other pair(s), e.g. idkDPO vs idkdpo-pcgrad (or whichever is your intended non/pc pairing).
"""

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt


# ----------------------------
# Grouping / parsing utilities
# ----------------------------


@dataclass(frozen=True)
class GroupKey:
    layer: int  # -1 for non-block params (embeddings, ln_f, lm_head)
    module: str  # coarse bucket


_LAY_RE = re.compile(r"^transformer\.h\.(\d+)\.(.*)$")


def bucket_param(name: str) -> GroupKey:
    """
    Buckets GPT-2 style parameter names into (layer_idx, module_bucket).
    Adjust if your naming differs.
    """
    m = _LAY_RE.match(name)
    if m:
        layer = int(m.group(1))
        rest = m.group(2)

        # Attention buckets
        if rest.startswith("attn.c_attn"):
            return GroupKey(layer, "attn_qkv")
        if rest.startswith("attn.c_proj"):
            return GroupKey(layer, "attn_out")

        # MLP buckets
        if rest.startswith("mlp.c_fc"):
            return GroupKey(layer, "mlp_in")
        if rest.startswith("mlp.c_proj"):
            return GroupKey(layer, "mlp_out")

        # LayerNorm buckets
        if rest.startswith("ln_1"):
            return GroupKey(layer, "ln_1")
        if rest.startswith("ln_2"):
            return GroupKey(layer, "ln_2")

        return GroupKey(layer, "other")

    # Non-block params
    if name.startswith("transformer.wte"):
        return GroupKey(-1, "emb_wte")
    if name.startswith("transformer.wpe"):
        return GroupKey(-1, "emb_wpe")
    if name.startswith("transformer.ln_f"):
        return GroupKey(-1, "ln_f")
    if name.startswith("lm_head"):
        return GroupKey(-1, "lm_head")

    return GroupKey(-1, "other_global")


def sorted_modules_present(keys: List[GroupKey]) -> List[str]:
    mods = sorted(set(k.module for k in keys))
    # nicer ordering if present
    preferred = [
        "emb_wte",
        "emb_wpe",
        "ln_f",
        "attn_qkv",
        "attn_out",
        "mlp_in",
        "mlp_out",
        "ln_1",
        "ln_2",
        "other",
        "lm_head",
        "other_global",
    ]
    out = [m for m in preferred if m in mods] + [m for m in mods if m not in preferred]
    return out


# ----------------------------
# Core math (streaming)
# ----------------------------


@dataclass
class Accum:
    # For RMS of (Δdiff): sumsq and count
    sumsq_diff: float = 0.0
    count: int = 0

    # For cosine between Δ_pc and Δ_non: dot, norms
    dot: float = 0.0
    sumsq_pc: float = 0.0
    sumsq_non: float = 0.0

    # For RMS ratio: sumsq of each delta
    sumsq_delta_pc: float = 0.0
    sumsq_delta_non: float = 0.0


def safe_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def load_state_dict(model_name_or_path: str) -> Dict[str, torch.Tensor]:
    """
    Loads a HF causal LM and returns its state_dict on CPU.
    Works for HF model names or checkpoint directories.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float32, device_map=None
    )
    model.eval()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    return sd


def compare_checkpoints(
    base_sd: Dict[str, torch.Tensor],
    non_sd: Dict[str, torch.Tensor],
    pc_sd: Dict[str, torch.Tensor],
    ignore_regex: str = r"^lm_head\.weight$",
    eps: float = 1e-12,
) -> Tuple[Dict[GroupKey, Accum], List[GroupKey]]:
    """
    Streaming accumulation over parameters:
      Δ_non = non - base
      Δ_pc  = pc  - base
      Δdiff = Δ_pc - Δ_non

    Accumulate per group key:
      RMS(Δdiff)
      cosine(Δ_pc, Δ_non)
      log ratio of RMS(Δ_pc)/RMS(Δ_non)
    """
    ig = re.compile(ignore_regex) if ignore_regex else None

    acc: Dict[GroupKey, Accum] = defaultdict(Accum)

    # Only compare intersection of keys to avoid surprises
    keys = sorted(set(base_sd.keys()) & set(non_sd.keys()) & set(pc_sd.keys()))

    for k in keys:
        if ig and ig.search(k):
            continue

        b = base_sd[k].float()
        n = non_sd[k].float()
        p = pc_sd[k].float()

        dn = (n - b).view(-1)
        dp = (p - b).view(-1)
        dd = dp - dn

        gk = bucket_param(k)
        a = acc[gk]

        # RMS diff
        a.sumsq_diff += safe_float((dd * dd).sum())
        a.count += dd.numel()

        # cosine(Δ_pc, Δ_non)
        a.dot += safe_float((dp * dn).sum())
        a.sumsq_pc += safe_float((dp * dp).sum())
        a.sumsq_non += safe_float((dn * dn).sum())

        # ratio components
        a.sumsq_delta_pc += safe_float((dp * dp).sum())
        a.sumsq_delta_non += safe_float((dn * dn).sum())

    gkeys = sorted(acc.keys(), key=lambda x: (x.layer, x.module))
    return acc, gkeys


def rms(sumsq: float, count: int, eps: float = 1e-12) -> float:
    return math.sqrt(max(sumsq / max(count, 1), eps))


def cosine(dot: float, sumsq_a: float, sumsq_b: float, eps: float = 1e-12) -> float:
    denom = math.sqrt(max(sumsq_a, eps)) * math.sqrt(max(sumsq_b, eps))
    return float(dot / denom) if denom > 0 else 0.0


# ----------------------------
# Plotting
# ----------------------------


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def plot_heatmap(matrix, xlabels, ylabels, title, outpath):
    plt.figure(figsize=(max(8, 0.6 * len(xlabels)), max(6, 0.35 * len(ylabels))))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    plt.yticks(range(len(ylabels)), ylabels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_lines(x, ys, labels, title, xlabel, ylabel, outpath):
    plt.figure(figsize=(10, 5))
    for y, lab in zip(ys, labels):
        plt.plot(x, y, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def write_csv(rows: List[Dict[str, str]], outpath: str) -> None:
    if not rows:
        return
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ----------------------------
# Main
# ----------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_model",
        required=True,
        help="HF model name or path for the *common base* (e.g., gpt2)",
    )
    ap.add_argument(
        "--non_ckpt", required=True, help="Non-PCGrad checkpoint (HF dir or model id)"
    )
    ap.add_argument(
        "--pc_ckpt", required=True, help="PCGrad checkpoint (HF dir or model id)"
    )
    ap.add_argument("--out_dir", required=True, help="Output directory for figures/csv")
    ap.add_argument(
        "--ignore_regex",
        default=r"^lm_head\.weight$",
        help="Regex of params to ignore (default skips tied lm_head)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    print("Loading base...")
    base_sd = load_state_dict(args.base_model)
    print("Loading non-PCGrad...")
    non_sd = load_state_dict(args.non_ckpt)
    print("Loading PCGrad...")
    pc_sd = load_state_dict(args.pc_ckpt)

    print("Computing grouped stats...")
    acc, gkeys = compare_checkpoints(
        base_sd, non_sd, pc_sd, ignore_regex=args.ignore_regex
    )

    # Build layer list and module list
    layers = sorted(set(k.layer for k in gkeys))
    modules = sorted_modules_present(gkeys)

    # Heatmap matrix: log10 RMS(Δdiff)
    heat = [[None for _ in modules] for __ in layers]

    # Per-layer cosine + per-layer ratio (aggregate over all modules in that layer)
    cos_per_layer = []
    ratio_per_layer = []  # log10 RMS(dp) / RMS(dn)

    rows = []

    for li, layer in enumerate(layers):
        # aggregate accum for layer
        layer_dot = layer_sumsq_pc = layer_sumsq_non = 0.0
        layer_sumsq_dp = layer_sumsq_dn = 0.0
        layer_count_dp = layer_count_dn = 0  # use counts from diff accumulator as proxy

        for mi, mod in enumerate(modules):
            gk = GroupKey(layer, mod)
            if gk not in acc:
                heat[li][mi] = float("nan")
                continue

            a = acc[gk]
            r = rms(a.sumsq_diff, a.count)
            heat[li][mi] = math.log10(r + 1e-12)

            rows.append(
                {
                    "layer": str(layer),
                    "module": mod,
                    "rms_delta_diff": f"{r:.6e}",
                    "log10_rms_delta_diff": f"{math.log10(r + 1e-12):.6f}",
                    "cos_delta_pc_vs_non": f"{cosine(a.dot, a.sumsq_pc, a.sumsq_non):.6f}",
                    "rms_delta_pc": f"{math.sqrt(max(a.sumsq_delta_pc / max(a.count,1), 1e-12)):.6e}",
                    "rms_delta_non": f"{math.sqrt(max(a.sumsq_delta_non / max(a.count,1), 1e-12)):.6e}",
                }
            )

            layer_dot += a.dot
            layer_sumsq_pc += a.sumsq_pc
            layer_sumsq_non += a.sumsq_non
            layer_sumsq_dp += a.sumsq_delta_pc
            layer_sumsq_dn += a.sumsq_delta_non
            layer_count_dp += a.count
            layer_count_dn += a.count

        cosL = cosine(layer_dot, layer_sumsq_pc, layer_sumsq_non)
        cos_per_layer.append(cosL)

        rms_dp = math.sqrt(max(layer_sumsq_dp / max(layer_count_dp, 1), 1e-12))
        rms_dn = math.sqrt(max(layer_sumsq_dn / max(layer_count_dn, 1), 1e-12))
        ratio_per_layer.append(math.log10((rms_dp + 1e-12) / (rms_dn + 1e-12)))

    # Save CSV
    csv_path = os.path.join(args.out_dir, "pcgrad_vs_nonpcgrad_weightspace_summary.csv")
    write_csv(rows, csv_path)
    print(f"Wrote: {csv_path}")

    # Plots
    heat_path = os.path.join(
        args.out_dir, "heatmap_log10_rms_deltaDiff_perLayerModule.png"
    )
    plot_heatmap(
        matrix=heat,
        xlabels=modules,
        ylabels=[str(l) for l in layers],
        title="log10 RMS( (Δ_pcgrad - Δ_nonpcgrad) ) per layer/module",
        outpath=heat_path,
    )
    print(f"Wrote: {heat_path}")

    # Layerwise line plots (skip layer=-1 in x if you want; we keep it)
    x = list(range(len(layers)))
    layer_labels = [str(l) for l in layers]

    cos_path = os.path.join(args.out_dir, "layerwise_cosine_deltaPC_vs_deltaNon.png")
    plot_lines(
        x=x,
        ys=[cos_per_layer],
        labels=["cos(Δ_pcgrad, Δ_nonpcgrad)"],
        title="Layerwise alignment of updates",
        xlabel="Layer index in this plot (see tick labels)",
        ylabel="Cosine similarity",
        outpath=cos_path,
    )
    # add tick labels by saving a second figure with labels for readability
    plt.figure(figsize=(10, 5))
    plt.plot(x, cos_per_layer, label="cos(Δ_pcgrad, Δ_nonpcgrad)")
    plt.xticks(x, layer_labels, rotation=45, ha="right")
    plt.title("Layerwise alignment of updates")
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity")
    plt.legend()
    plt.tight_layout()
    labeled_cos_path = os.path.join(
        args.out_dir, "layerwise_cosine_deltaPC_vs_deltaNon_labeled.png"
    )
    plt.savefig(labeled_cos_path, dpi=200)
    plt.close()
    print(f"Wrote: {cos_path}")
    print(f"Wrote: {labeled_cos_path}")

    ratio_path = os.path.join(args.out_dir, "layerwise_log10_rmsRatio_pc_over_non.png")
    plt.figure(figsize=(10, 5))
    plt.plot(x, ratio_per_layer, label="log10( RMS(Δ_pcgrad) / RMS(Δ_nonpcgrad) )")
    plt.xticks(x, layer_labels, rotation=45, ha="right")
    plt.title("Layerwise relative update size (PCGrad vs non-PCGrad)")
    plt.xlabel("Layer")
    plt.ylabel("log10 RMS ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ratio_path, dpi=200)
    plt.close()
    print(f"Wrote: {ratio_path}")

    print("Done.")


if __name__ == "__main__":
    main()
