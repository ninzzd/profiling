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
RESULTS_DIR = SCRIPT_DIR / f"../results/param-sweep/gpu-ivf-pq-{args.build_type}"
df = pd.read_csv(RESULTS_DIR / "gpu-ivf-pq-param-sweep.csv")

# ---------------------------------------------------------
# Slice CSV according to sweep order (see
# scripts/gpu-ivf-pq-param-sweep.sh). The nbits sweep only
# runs on the cuvs build -- the non-cuVS path hard-requires
# nbits == 8 -- so it is picked up only if rows remain.
# ---------------------------------------------------------
NNLIST  = 8
NNPROBE = 9
NM      = 6
NNBITS  = 5

offset = 0

df_nlist = df.iloc[offset:offset + NNLIST].sort_values("nlist")
offset += NNLIST

df_nprobe = df.iloc[offset:offset + NNPROBE].sort_values("nprobe")
offset += NNPROBE

df_m = df.iloc[offset:offset + NM].sort_values("m")
offset += NM

# (df, param, log2 x-axis?)
sweeps = [
    (df_nlist,  "nlist",  True),
    (df_nprobe, "nprobe", True),
    (df_m,      "m",      False),
]

if len(df) > offset:
    df_nbits = df.iloc[offset:offset + NNBITS].sort_values("nbits")
    sweeps.append((df_nbits, "nbits", False))


def _style_x(ax, sub, param, log2):
    if log2:
        ax.set_xscale("log", base=2)
    ax.set_xticks(sub[param])
    ax.set_xticklabels(sub[param])
    ax.set_xlabel(param)


def _save(name):
    plt.tight_layout()
    if args.save:
        plt.savefig(RESULTS_DIR / name, dpi=300)


# ---------------------------------------------------------
# Latency (with min/max error bars and P50 / P90 lines)
# ---------------------------------------------------------
for sub, param, log2 in sweeps:
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

    ax.set_yscale("log")
    _style_x(ax, sub, param, log2)
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"GPU IVF-PQ ({args.build_type}): Latency vs {param}")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()

    _save(f"{param}_latency.png")

# ---------------------------------------------------------
# Throughput
# ---------------------------------------------------------
for sub, param, log2 in sweeps:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(sub[param], sub["avgQPS"], marker="o", linewidth=2)

    _style_x(ax, sub, param, log2)
    ax.set_ylabel("Average Throughput (QPS)")
    ax.set_title(f"GPU IVF-PQ ({args.build_type}): Throughput vs {param}")
    ax.grid(True)

    _save(f"{param}_throughput.png")

# ---------------------------------------------------------
# Recall (with min/max error bars)
# ---------------------------------------------------------
for sub, param, log2 in sweeps:
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

    _style_x(ax, sub, param, log2)
    ax.set_ylabel("Recall@10")
    ax.set_title(f"GPU IVF-PQ ({args.build_type}): Recall@10 vs {param}")
    ax.grid(True)

    _save(f"{param}_recall.png")

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
ax.set_title(f"GPU IVF-PQ ({args.build_type}): Recall vs Latency")
ax.grid(True, which="both", alpha=0.4)
ax.legend()

_save("recall_vs_latency.png")

if not args.save:
    plt.show()
