import pandas as pd
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
RESULTS_DIR = SCRIPT_DIR / f"../results/param-sweep/ivf-flat-{args.build_type}"
df = pd.read_csv(RESULTS_DIR / "ivf-flat-param-sweep.csv")

# -----------------------------
# NLIST SWEEP
# -----------------------------
df_nlist = df[df["nprobe"] == 32].sort_values("nlist")

plots = [
    ("avglat", "Average Latency (s)", "nlist_latency.png"),
    ("avgQPS", "Average Throughput (QPS)", "nlist_throughput.png"),
    ("avgrecall", "Average Recall@10", "nlist_recall.png"),
]

for metric, ylabel, outfile in plots:
    plt.figure(figsize=(7,5))

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
    plt.savefig(outfile, dpi=300)

# -----------------------------
# NPROBE SWEEP
# -----------------------------
df_nprobe = df[df["nlist"] == 2048].sort_values("nprobe")

plots = [
    ("avglat", "Average Latency (s)", "nprobe_latency.png"),
    ("avgQPS", "Average Throughput (QPS)", "nprobe_throughput.png"),
    ("avgrecall", "Average Recall@10", "nprobe_recall.png"),
]

for metric, ylabel, outfile in plots:
    plt.figure(figsize=(7,5))

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
    plt.savefig(outfile, dpi=300)

plt.show()