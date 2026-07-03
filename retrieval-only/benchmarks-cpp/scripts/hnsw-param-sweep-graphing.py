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
args = parser.parse_args()

# -----------------------------
# Read CSV
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "../results/param-sweep/hnsw"
df = pd.read_csv(RESULTS_DIR / "hnsw-param-sweep.csv")

plots = [
    ("avglat", "Average Latency (s)", "latency"),
    ("avgQPS", "Average Throughput (QPS)", "throughput"),
    ("avgrecall", "Average Recall@10", "recall"),
]

# ---------------------------------------------------------
# Slice CSV according to sweep order
# ---------------------------------------------------------
NM = 6
NEFCONSTRUCTION = 7
NEFSEARCH = 7

offset = 0

df_M = df.iloc[offset:offset + NM].sort_values("M")
offset += NM

df_efConstruction = (
    df.iloc[offset:offset + NEFCONSTRUCTION]
    .sort_values("efConstruction")
)
offset += NEFCONSTRUCTION

df_efSearch = (
    df.iloc[offset:offset + NEFSEARCH]
    .sort_values("efSearch")
)

# ---------------------------------------------------------
# M sweep
# ---------------------------------------------------------
for metric, ylabel, suffix in plots:
    plt.figure(figsize=(7, 5))

    plt.plot(
        df_M["M"],
        df_M[metric],
        marker="o",
        linewidth=2,
    )

    plt.xticks(df_M["M"], df_M["M"])
    plt.xlabel("M")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs M")
    plt.grid(True)

    plt.tight_layout()

    if args.save:
        plt.savefig(f"M_{suffix}.png", dpi=300)

# ---------------------------------------------------------
# efConstruction sweep
# ---------------------------------------------------------
for metric, ylabel, suffix in plots:
    plt.figure(figsize=(7, 5))

    plt.plot(
        df_efConstruction["efConstruction"],
        df_efConstruction[metric],
        marker="o",
        linewidth=2,
    )

    plt.xticks(
        df_efConstruction["efConstruction"],
        df_efConstruction["efConstruction"],
    )

    plt.xlabel("efConstruction")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs efConstruction")
    plt.grid(True)

    plt.tight_layout()

    if args.save:
        plt.savefig(f"efConstruction_{suffix}.png", dpi=300)

# ---------------------------------------------------------
# efSearch sweep
# ---------------------------------------------------------
for metric, ylabel, suffix in plots:
    plt.figure(figsize=(7, 5))

    plt.plot(
        df_efSearch["efSearch"],
        df_efSearch[metric],
        marker="o",
        linewidth=2,
    )

    plt.xscale("log", base=2)
    plt.xticks(
        df_efSearch["efSearch"],
        df_efSearch["efSearch"],
    )

    plt.xlabel("efSearch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs efSearch")
    plt.grid(True)

    plt.tight_layout()

    if args.save:
        plt.savefig(RESULTS_DIR / f"efSearch_{suffix}.png", dpi=300)

plt.show()