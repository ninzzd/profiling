import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# -----------------------------
# Command-line arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--build-type",
    type=str,
    default="generic",
    choices=["generic", "dd", "avx2", "avx512", "cuda", "cuvs"],
    help="Build configuration used for the benchmark results"
)
args = parser.parse_args()

# Read CSV (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / f"../results/param-sweep/ivf-pq-{args.build_type}"
df = pd.read_csv(RESULTS_DIR / "ivf-pq-param-sweep.csv")

SAVE_GRAPHS = True

plots = [
    ("avglat", "Average Latency (s)", "latency"),
    ("avgQPS", "Average Throughput (QPS)", "throughput"),
    ("avgrecall", "Average Recall@10", "recall"),
]

# ------------------------------------------------------------------
# Slice the CSV by sweep order to avoid duplicate baseline points
# ------------------------------------------------------------------
NNLIST  = 8
NNPROBE = 9
NM       = 8
NNBITS   = 5

offset = 0

df_nlist = df.iloc[offset:offset + NNLIST].sort_values("nlist")
offset += NNLIST

df_nprobe = df.iloc[offset:offset + NNPROBE].sort_values("nprobe")
offset += NNPROBE

df_m = df.iloc[offset:offset + NM].sort_values("m")
offset += NM

df_nbits = df.iloc[offset:offset + NNBITS].sort_values("nbits")

# ------------------------------------------------------------------
# nlist sweep
# ------------------------------------------------------------------
for metric, ylabel, suffix in plots:
    plt.figure(figsize=(7, 5))

    plt.plot(
        df_nlist["nlist"],
        df_nlist[metric],
        marker="o",
        linewidth=2
    )

    plt.xscale("log", base=2)
    plt.xticks(df_nlist["nlist"], df_nlist["nlist"])

    plt.xlabel("nlist")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs nlist")
    plt.grid(True)

    plt.tight_layout()
    if SAVE_GRAPHS:
        plt.savefig(f"nlist_{suffix}.png", dpi=300)

# ------------------------------------------------------------------
# nprobe sweep
# ------------------------------------------------------------------
for metric, ylabel, suffix in plots:
    plt.figure(figsize=(7, 5))

    plt.plot(
        df_nprobe["nprobe"],
        df_nprobe[metric],
        marker="o",
        linewidth=2
    )

    plt.xscale("log", base=2)
    plt.xticks(df_nprobe["nprobe"], df_nprobe["nprobe"])

    plt.xlabel("nprobe")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs nprobe")
    plt.grid(True)

    plt.tight_layout()
    if SAVE_GRAPHS:
        plt.savefig(f"nprobe_{suffix}.png", dpi=300)

# ------------------------------------------------------------------
# m sweep
# ------------------------------------------------------------------
for metric, ylabel, suffix in plots:
    plt.figure(figsize=(7, 5))

    plt.plot(
        df_m["m"],
        df_m[metric],
        marker="o",
        linewidth=2
    )

    plt.xticks(df_m["m"], df_m["m"])

    plt.xlabel("m")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs m")
    plt.grid(True)

    plt.tight_layout()
    if SAVE_GRAPHS:
        plt.savefig(f"m_{suffix}.png", dpi=300)

# ------------------------------------------------------------------
# nbits sweep
# ------------------------------------------------------------------
for metric, ylabel, suffix in plots:
    plt.figure(figsize=(7, 5))

    plt.plot(
        df_nbits["nbits"],
        df_nbits[metric],
        marker="o",
        linewidth=2
    )

    plt.xticks(df_nbits["nbits"], df_nbits["nbits"])

    plt.xlabel("nbits")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs nbits")
    plt.grid(True)

    plt.tight_layout()
    if SAVE_GRAPHS:
        plt.savefig(f"nbits_{suffix}.png", dpi=300)

plt.show()