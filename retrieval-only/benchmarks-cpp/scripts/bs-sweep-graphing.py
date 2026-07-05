import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--save",
    action="store_true",
    help="Save figures as PNG"
)
args = parser.parse_args()

# ---------------------------------------------------------
# Read CSV
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(SCRIPT_DIR / "../results/workload-sweep/bs-sweep-stats.csv")

# ---------------------------------------------------------
# Index display names
# ---------------------------------------------------------
labels = {
    "cpu-baseline": "Flat",
    "cpu-hnsw": "HNSW",
    "cpu-ivf-flat": "IVF-Flat",
    "cpu-ivf-pq": "IVF-PQ",
}

# Filter out baseline for the "without baseline" plots
df_no_baseline = df[df["index"] != "cpu-baseline"]

# ---------------------------------------------------------
# Shared axis configuration
# ---------------------------------------------------------
XTICKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
XTICK_LABELS = [str(v) for v in XTICKS]


def _label(idx):
    return labels.get(idx, idx)


def _style_plot(ax, xlabel, ylabel, title):
    ax.set_xscale("log", base=2)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XTICK_LABELS)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


# ---------------------------------------------------------
# 1. Average Latency vs Batch Size (all indexes, with error
#    bars and P50 / P90 dotted lines)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for idx in df["index"].unique():
    subset = df[df["index"] == idx].sort_values("nq")

    y = subset["avglat"]
    ymin = y - subset["minlat"]
    ymax = subset["maxlat"] - y

    ax.errorbar(
        subset["nq"],
        y,
        yerr=[ymin, ymax],
        marker="o",
        linewidth=2,
        capsize=4,
        label=_label(idx),
    )

    ax.plot(
        subset["nq"],
        subset["p50lat"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.35,
    )

    ax.plot(
        subset["nq"],
        subset["p90lat"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.35,
    )

_style_plot(ax, "Query Batch Size", "Average Latency (s)", "Average Latency vs Query Batch Size")
plt.tight_layout()
if args.save:
    plt.savefig("latency_vs_batch_size.png", dpi=300)

# ---------------------------------------------------------
# 2. Average Latency vs Batch Size (without baseline, with
#    error bars and P50 / P90 dotted lines)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for idx in df_no_baseline["index"].unique():
    subset = df_no_baseline[df_no_baseline["index"] == idx].sort_values("nq")

    y = subset["avglat"]
    ymin = y - subset["minlat"]
    ymax = subset["maxlat"] - y

    ax.errorbar(
        subset["nq"],
        y,
        yerr=[ymin, ymax],
        marker="o",
        linewidth=2,
        capsize=4,
        label=_label(idx),
    )

    ax.plot(
        subset["nq"],
        subset["p50lat"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.35,
    )

    ax.plot(
        subset["nq"],
        subset["p90lat"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.35,
    )

_style_plot(ax, "Query Batch Size", "Average Latency (s)", "Average Latency vs Batch Size (without Flat)")
plt.tight_layout()
if args.save:
    plt.savefig("latency_vs_batch_size_no_baseline.png", dpi=300)

# ---------------------------------------------------------
# 3. Throughput vs Batch Size (all indexes)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for idx in df["index"].unique():
    subset = df[df["index"] == idx].sort_values("nq")

    ax.plot(
        subset["nq"],
        subset["avgQPS"],
        marker="o",
        linewidth=2,
        label=_label(idx),
    )

_style_plot(ax, "Query Batch Size", "Average Throughput (QPS)", "Average Throughput vs Query Batch Size")
plt.tight_layout()
if args.save:
    plt.savefig("throughput_vs_batch_size.png", dpi=300)

# ---------------------------------------------------------
# 4. Throughput vs Batch Size (without baseline)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for idx in df_no_baseline["index"].unique():
    subset = df_no_baseline[df_no_baseline["index"] == idx].sort_values("nq")

    ax.plot(
        subset["nq"],
        subset["avgQPS"],
        marker="o",
        linewidth=2,
        label=_label(idx),
    )

_style_plot(ax, "Query Batch Size", "Average Throughput (QPS)", "Average Throughput vs Batch Size (without Flat)")
plt.tight_layout()
if args.save:
    plt.savefig("throughput_vs_batch_size_no_baseline.png", dpi=300)

# ---------------------------------------------------------
# 5. Recall vs Batch Size (with error bars)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for idx in df["index"].unique():
    subset = df[df["index"] == idx].sort_values("nq")

    y = subset["avgrecall"]
    ymin = y - subset["minrecall"]
    ymax = subset["maxrecall"] - y

    ax.errorbar(
        subset["nq"],
        y,
        yerr=[ymin, ymax],
        marker="o",
        linewidth=2,
        capsize=4,
        label=_label(idx),
    )

_style_plot(ax, "Query Batch Size", "Average Recall@10", "Average Recall vs Query Batch Size")
plt.tight_layout()
if args.save:
    plt.savefig("recall_vs_batch_size.png", dpi=300)

# ---------------------------------------------------------
# 6. Latency Standard Deviation vs Batch Size
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for idx in df["index"].unique():
    subset = df[df["index"] == idx].sort_values("nq")

    ax.plot(
        subset["nq"],
        subset["stdlat"],
        marker="o",
        linewidth=2,
        label=_label(idx),
    )

_style_plot(ax, "Query Batch Size", "Latency Standard Deviation (s)", "Latency Standard Deviation vs Query Batch Size")
plt.tight_layout()
if args.save:
    plt.savefig("latency_std_vs_batch_size.png", dpi=300)

# ---------------------------------------------------------
# 7. Recall Standard Deviation vs Batch Size
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for idx in df["index"].unique():
    subset = df[df["index"] == idx].sort_values("nq")

    ax.plot(
        subset["nq"],
        subset["stdrcl"],
        marker="o",
        linewidth=2,
        label=_label(idx),
    )

_style_plot(ax, "Query Batch Size", "Recall Standard Deviation", "Recall Standard Deviation vs Query Batch Size")
plt.tight_layout()
if args.save:
    plt.savefig("recall_std_vs_batch_size.png", dpi=300)

plt.show()