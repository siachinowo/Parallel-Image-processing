import sys, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
os.makedirs("plots", exist_ok=True)

df = pd.read_csv(CSV)
df["size"] = df["width"]
FIXED_BATCH = 10 if 10 in df.batch.unique() else int(df.batch.max())
FIXED_SIZE  = 1024 if 1024 in df["size"].unique() else int(df["size"].max())

df["speedup_kernel"] = df.cpu_ms / df.kernel_ms
df["speedup_full"]   = df.cpu_ms / df.gpu_total_ms
df["mpix_s_gpu"]     = df.width * df.height * df.batch / 1e6 / (df.gpu_total_ms / 1000)
df["mpix_s_cpu"]     = df.width * df.height * df.batch / 1e6 / (df.cpu_ms / 1000)

agg = df.groupby(["filter", "size", "batch"]).agg(
    cpu=("cpu_ms", "mean"), cpu_sd=("cpu_ms", "std"),
    h2d=("h2d_ms", "mean"), kernel=("kernel_ms", "mean"), d2h=("d2h_ms", "mean"),
    sp_kernel=("speedup_kernel", "mean"), sp_kernel_sd=("speedup_kernel", "std"),
    sp_full=("speedup_full", "mean"),     sp_full_sd=("speedup_full", "std"),
    tp_gpu=("mpix_s_gpu", "mean"), tp_cpu=("mpix_s_cpu", "mean"),
    max_diff=("max_diff", "max"),
).reset_index()
agg.to_csv("results/summary.csv", index=False)

def speedup_plot(xcol, subset, xlabel, fname, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for flt, g in subset.groupby("filter"):
        g = g.sort_values(xcol)
        ln, = ax.plot(g[xcol], g.sp_kernel, "o-", label=f"{flt} (kernel only)")
        ax.plot(g[xcol], g.sp_full, "s--", color=ln.get_color(), label=f"{flt} (full run)")
    ax.axhline(1, color="gray", lw=1, ls=":")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("Speedup (CPU time / GPU time)")
    ax.set_title(title); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"plots/{fname}", dpi=200); plt.close(fig)

speedup_plot("size", agg[agg.batch == FIXED_BATCH], "Image width (px, square)",
             "speedup_vs_size.png", f"Speedup vs image size (batch = {FIXED_BATCH})")
speedup_plot("batch", agg[agg["size"] == FIXED_SIZE], "Batch size (images)",
             "speedup_vs_batch.png", f"Speedup vs batch size ({FIXED_SIZE}x{FIXED_SIZE})")

for flt, g in agg[agg.batch == FIXED_BATCH].groupby("filter"):
    g = g.sort_values("size")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = [str(s) for s in g["size"]]
    h2d, ker, d2h = g.h2d / FIXED_BATCH, g.kernel / FIXED_BATCH, g.d2h / FIXED_BATCH
    ax.bar(x, h2d, label="H2D copy")
    ax.bar(x, ker, bottom=h2d, label="Kernel")
    ax.bar(x, d2h, bottom=h2d + ker, label="D2H copy")
    ax.set_xlabel("Image width (px)"); ax.set_ylabel("GPU time per image (ms)")
    ax.set_title(f"{flt}: GPU time breakdown"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"plots/breakdown_{flt}.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
for flt, g in agg[agg.batch == FIXED_BATCH].groupby("filter"):
    g = g.sort_values("size")
    ln, = ax.plot(g["size"], g.tp_gpu, "o-", label=f"{flt} GPU (full run)")
    ax.plot(g["size"], g.tp_cpu, "x--", color=ln.get_color(), label=f"{flt} CPU")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("Image width (px)"); ax.set_ylabel("Throughput (Mpixels/s)")
ax.set_title("Throughput"); ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("plots/throughput.png", dpi=200)

print("Wrote plots/*.png and results/summary.csv")
print("Max GPU-vs-CPU difference per filter:\n", agg.groupby("filter").max_diff.max())
