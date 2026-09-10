import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# -----------------------------
# Command-line arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true",
                    help="Save figures as PNG")
parser.add_argument(
    "--build-type",
    type=str,
    default="cuda",
    choices=["cuda", "cuvs"],
    help="GPU build configuration used for the benchmark results"
)
args = parser.parse_args()

# -----------------------------
# Read CSV
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / f"../results/param-sweep/gpu-ivf-flat-{args.build_type}"
df = pd.read_csv(RESULTS_DIR / "gpu-ivf-flat-param-sweep.csv")

# ---------------------------------------------------------
# Slice CSV according to sweep order (see
# scripts/gpu-ivf-flat-param-sweep.sh). Slicing by position
# rather than filtering avoids duplicating the shared
# nlist=2048 / nprobe=32 baseline point into both sweeps.
# ---------------------------------------------------------
NNLIST  = 8
NNPROBE = 9

offset = 0

df_nlist = df.iloc[offset:offset + NNLIST].sort_values("nlist")
offset += NNLIST

df_nprobe = df.iloc[offset:offset + NNPROBE].sort_values("nprobe")

sweeps = [
    (df_nlist,  "nlist",  "nlist"),
    (df_nprobe, "nprobe", "nprobe"),
]


def _save(name):
    plt.tight_layout()
    if args.save:
        plt.savefig(RESULTS_DIR / name, dpi=300)


# ---------------------------------------------------------
# Latency (with min/max error bars and P50 / P90 lines)
# ---------------------------------------------------------
for sub, param, prefix in sweeps:
    fig, ax = plt.subplots(figsize=(7, 5))

    y = sub["avglat"]
    ymin = y - sub["minlat"]
    ymax = sub["maxlat"] - y

    ax.errorbar(
        sub[param],
        y,
        yerr=[ymin, ymax],
        marker="o",
        linewidth=2,
        capsize=4,
        label="avg (min/max)",
    )

    ax.plot(sub[param], sub["p50lat"], linestyle="--", linewidth=1.2,
            alpha=0.35, label="p50")
    ax.plot(sub[param], sub["p90lat"], linestyle=":", linewidth=1.2,
            alpha=0.35, label="p90")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sub[param])
    ax.set_xticklabels(sub[param])
    ax.set_xlabel(param)
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"GPU IVF-Flat ({args.build_type}): Latency vs {param}")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()

    _save(f"{prefix}_latency.png")

# ---------------------------------------------------------
# Throughput
# ---------------------------------------------------------
for sub, param, prefix in sweeps:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(sub[param], sub["avgQPS"], marker="o", linewidth=2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sub[param])
    ax.set_xticklabels(sub[param])
    ax.set_xlabel(param)
    ax.set_ylabel("Average Throughput (QPS)")
    ax.set_title(f"GPU IVF-Flat ({args.build_type}): Throughput vs {param}")
    ax.grid(True)

    _save(f"{prefix}_throughput.png")

# ---------------------------------------------------------
# Recall (with min/max error bars)
# ---------------------------------------------------------
for sub, param, prefix in sweeps:
    fig, ax = plt.subplots(figsize=(7, 5))

    y = sub["avgrecall"]
    ymin = y - sub["minrecall"]
    ymax = sub["maxrecall"] - y

    ax.errorbar(
        sub[param],
        y,
        yerr=[ymin, ymax],
        marker="o",
        linewidth=2,
        capsize=4,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(sub[param])
    ax.set_xticklabels(sub[param])
    ax.set_xlabel(param)
    ax.set_ylabel("Recall@10")
    ax.set_title(f"GPU IVF-Flat ({args.build_type}): Recall@10 vs {param}")
    ax.grid(True)

    _save(f"{prefix}_recall.png")

# ---------------------------------------------------------
# Recall vs latency trade-off (the operating curve)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for sub, param, _ in sweeps:
    ax.plot(sub["avglat"], sub["avgrecall"], marker="o", linewidth=2,
            label=f"{param} sweep")
    for _, row in sub.iterrows():
        ax.annotate(f"{int(row[param])}",
                    (row["avglat"], row["avgrecall"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)

ax.set_xscale("log")
ax.set_xlabel("Average Latency (s)")
ax.set_ylabel("Average Recall@10")
ax.set_title(f"GPU IVF-Flat ({args.build_type}): Recall vs Latency")
ax.grid(True, which="both", alpha=0.4)
ax.legend()

_save("recall_vs_latency.png")

if not args.save:
    plt.show()
