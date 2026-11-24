#!/usr/bin/env python3
"""Turn results/benchmark.csv into the report figures.

Usage:  python3 scripts/plot.py results/benchmark.csv [--outdir results/plots]
                                [--batch N] [--size N]

Figures written to --outdir:
  1_speedup_vs_size.png      kernel-only and full-run speedup vs image size (fixed batch)
  2_speedup_vs_batch.png     same speedups vs batch size (fixed image size)
  3_gpu_time_breakdown.png   share of GPU time spent in H2D copy / kernel / D2H copy
  4_time_per_image.png       absolute time per image: CPU vs GPU (kernel and full run)
  speedup_table.csv          the numbers behind the plots
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
import pandas as pd

FILTERS = ["gray", "blur", "sobel"]
COLORS = {"gray": "#1f77b4", "blur": "#ff7f0e", "sobel": "#2ca02c"}
PART_COLORS = {"h2d_ms": "#d62728", "kernel_ms": "#2ca02c", "d2h_ms": "#9467bd"}
PART_LABELS = {"h2d_ms": "Host to device copy", "kernel_ms": "Kernel", "d2h_ms": "Device to host copy"}


def nearest(values, target):
    return min(values, key=lambda v: abs(v - target))


def log2_axis(ax, values, labeler):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.set_xticklabels([labeler(v) for v in values], rotation=0)
    ax.minorticks_off()


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)


def fig_speedup_vs_size(df, batch, outdir, sub):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sizes = sorted(df["width"].unique())
    for ax, col, title in [(axes[0], "speedup_kernel", "Kernel only"),
                           (axes[1], "speedup_total", "Full run (copies + kernel)")]:
        for f in FILTERS:
            d = df[(df["filter"] == f) & (df["batch"] == batch)].sort_values("width")
            if len(d):
                ax.plot(d["width"], d[col], marker="o", label=f, color=COLORS[f])
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)   # break-even line
        log2_axis(ax, sizes, lambda s: f"{s}x{s}")
        style(ax, "Image size", "Speedup over CPU (x)", title)
        ax.legend(title="Filter")
    fig.suptitle(f"Speedup vs image size (batch = {batch})" + sub, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "1_speedup_vs_size.png"), dpi=150)
    plt.close(fig)


def fig_speedup_vs_batch(df, size, outdir, sub):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    batches = sorted(df["batch"].unique())
    for ax, col, title in [(axes[0], "speedup_kernel", "Kernel only"),
                           (axes[1], "speedup_total", "Full run (copies + kernel)")]:
        for f in FILTERS:
            d = df[(df["filter"] == f) & (df["width"] == size)].sort_values("batch")
            if len(d):
                ax.plot(d["batch"], d[col], marker="o", label=f, color=COLORS[f])
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        log2_axis(ax, batches, str)
        style(ax, "Batch size (images)", "Speedup over CPU (x)", title)
        ax.legend(title="Filter")
    fig.suptitle(f"Speedup vs batch size (image {size}x{size})" + sub, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "2_speedup_vs_batch.png"), dpi=150)
    plt.close(fig)


def fig_breakdown(df, batch, outdir, sub):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    sizes = sorted(df["width"].unique())
    for ax, f in zip(axes, FILTERS):
        d = df[(df["filter"] == f) & (df["batch"] == batch)].sort_values("width")
        if d.empty:
            ax.set_visible(False)
            continue
        parts = d[list(PART_COLORS)]
        pct = parts.div(parts.sum(axis=1), axis=0) * 100
        bottom = [0.0] * len(d)
        labels = [f"{s}x{s}" for s in d["width"]]
        for col in PART_COLORS:
            ax.bar(labels, pct[col], bottom=bottom, color=PART_COLORS[col], label=PART_LABELS[col])
            bottom = [b + v for b, v in zip(bottom, pct[col])]
        ax.set_title(f, fontsize=10)
        ax.set_xlabel("Image size")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Share of GPU time (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"Where GPU time goes: transfers vs kernel (batch = {batch})" + sub, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(outdir, "3_gpu_time_breakdown.png"), dpi=150)
    plt.close(fig)


def fig_time_per_image(df, batch, outdir, sub):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    sizes = sorted(df["width"].unique())
    for ax, f in zip(axes, FILTERS):
        d = df[(df["filter"] == f) & (df["batch"] == batch)].sort_values("width")
        if d.empty:
            ax.set_visible(False)
            continue
        ax.plot(d["width"], d["cpu_ms"] / batch, marker="o", label="CPU", color="#444444")
        ax.plot(d["width"], d["gpu_total_ms"] / batch, marker="s", label="GPU full run", color="#d62728")
        ax.plot(d["width"], d["kernel_ms"] / batch, marker="^", label="GPU kernel only", color="#2ca02c")
        ax.set_yscale("log")
        log2_axis(ax, sizes, lambda s: f"{s}")
        style(ax, "Image side (pixels)", "Time per image (ms, log)", f)
        ax.legend(fontsize=8)
    fig.suptitle(f"Time per image (batch = {batch})" + sub, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "4_time_per_image.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="results/plots")
    ap.add_argument("--batch", type=int, help="batch size for the size-sweep plots (default: largest)")
    ap.add_argument("--size", type=int, help="image width for the batch-sweep plot (default: largest)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"CSV not found: {args.csv} (run ./benchmark first)")
    df = pd.read_csv(args.csv)
    if df.empty:
        sys.exit("CSV has no rows")
    os.makedirs(args.outdir, exist_ok=True)

    sub = ""
    info = args.csv + ".info.txt"
    if os.path.exists(info):
        with open(info) as fh:
            sub = "\n" + fh.readline().strip()          # first line = GPU name

    batch = nearest(sorted(df["batch"].unique()), args.batch) if args.batch else int(df["batch"].max())
    size = nearest(sorted(df["width"].unique()), args.size) if args.size else int(df["width"].max())

    fig_speedup_vs_size(df, batch, args.outdir, sub)
    fig_speedup_vs_batch(df, size, args.outdir, sub)
    fig_breakdown(df, batch, args.outdir, sub)
    fig_time_per_image(df, batch, args.outdir, sub)

    table = df[["filter", "width", "height", "batch", "cpu_ms", "gpu_total_ms", "kernel_ms",
                "speedup_kernel", "speedup_total", "max_diff"]]
    table.to_csv(os.path.join(args.outdir, "speedup_table.csv"), index=False)
    print(table.to_string(index=False))

    if (df["max_diff"] > 0).any():
        print("\nWARNING: some GPU outputs differ from the CPU reference (max_diff > 0)."
              " Check correctness before trusting the speedups.")
    # Small inputs where the GPU does not pay off after copies:
    slow = df[df["speedup_total"] < 1]
    if len(slow):
        print(f"\nNote: {len(slow)} configuration(s) have full-run speedup < 1 "
              "(transfer overhead outweighs the GPU gain).")
    print(f"\nFigures saved in {args.outdir}/")


if __name__ == "__main__":
    main()
