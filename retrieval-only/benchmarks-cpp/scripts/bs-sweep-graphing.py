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
df = pd.read_csv(SCRIPT_DIR / "bs-sweep-stats.csv")

# ---------------------------------------------------------
# Index display names
# ---------------------------------------------------------
labels = {
    "cpu-baseline": "Flat",
    "cpu-hnsw": "HNSW",
    "cpu-ivf-flat": "IVF-Flat",
    "cpu-ivf-pq": "IVF-PQ",
}

# ---------------------------------------------------------
# 1. Average latency with error bars + P50/P90
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

for idx in df["index"].unique():
    subset = df[df["index"] == idx].sort_values("nq")

    y = subset["avglat"]
    ymin = y - subset["minlat"]
    ymax = subset["maxlat"] - y

    plt.errorbar(
        subset["nq"],
        y,
        yerr=[ymin, ymax],
        marker="o",
        linewidth=2,
        capsize=4,
        label=labels.get(idx, idx),
    )

    plt.plot(
        subset["nq"],
        subset["p50lat"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.35,
    )

    plt.plot(
        subset["nq"],
        subset["p90lat"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.35,
    )

plt.xscale("log", base=2)
plt.xticks(
    [1,2,4,8,16,32,64,128,256,512,1024],
    [1,2,4,8,16,32,64,128,256,512,1024]
)

plt.xlabel("Query Batch Size")
plt.ylabel("Average Latency (s)")
plt.title("Average Latency vs Query Batch Size")
plt.grid(True)
plt.legend()
plt.tight_layout()

if args.save:
    plt.savefig("latency_vs_batch_size.png", dpi=300)

# ---------------------------------------------------------
# 2. Average Recall with error bars
# ---------------------------------------------------------
plt.figure(figsize=(8,5))

for idx in df["index"].unique():
    subset = df[df["index"] == idx].sort_values("nq")

    y = subset["avgrecall"]
    ymin = y - subset["minrecall"]
    ymax = subset["maxrecall"] - y

    plt.errorbar(
        subset["nq"],
        y,
        yerr=[ymin, ymax],
        marker="o",
        linewidth=2,
        capsize=4,
        label=labels.get(idx, idx),
    )

plt.xscale("log", base=2)
plt.xticks(
    [1,2,4,8,16,32,64,128,256,512,1024],
    [1,2,4,8,16,32,64,128,256,512,1024]
)

plt.xlabel("Query Batch Size")
plt.ylabel("Average Recall@10")
plt.title("Average Recall vs Query Batch Size")
plt.grid(True)
plt.legend()
plt.tight_layout()

if args.save:
    plt.savefig("recall_vs_batch_size.png", dpi=300)

# ---------------------------------------------------------
# 3. Throughput
# ---------------------------------------------------------
plt.figure(figsize=(8,5))

for idx in df["index"].unique():
    subset = df[df["index"] == idx].sort_values("nq")

    plt.plot(
        subset["nq"],
        subset["avgQPS"],
        marker="o",
        linewidth=2,
        label=labels.get(idx, idx),
    )

plt.xscale("log", base=2)
plt.xticks(
    [1,2,4,8,16,32,64,128,256,512,1024],
    [1,2,4,8,16,32,64,128,256,512,1024]
)

plt.xlabel("Query Batch Size")
plt.ylabel("Average Throughput (QPS)")
plt.title("Average Throughput vs Query Batch Size")
plt.grid(True)
plt.legend()
plt.tight_layout()

if args.save:
    plt.savefig("throughput_vs_batch_size.png", dpi=300)

# ---------------------------------------------------------
# 4. Latency standard deviation
# ---------------------------------------------------------
plt.figure(figsize=(8,5))

for idx in df["index"].unique():
    subset = df[df["index"] == idx].sort_values("nq")

    plt.plot(
        subset["nq"],
        subset["stdlat"],
        marker="o",
        linewidth=2,
        label=labels.get(idx, idx),
    )

plt.xscale("log", base=2)
plt.xticks(
    [1,2,4,8,16,32,64,128,256,512,1024],
    [1,2,4,8,16,32,64,128,256,512,1024]
)

plt.xlabel("Query Batch Size")
plt.ylabel("Latency Standard Deviation (s)")
plt.title("Latency Standard Deviation vs Query Batch Size")
plt.grid(True)
plt.legend()
plt.tight_layout()

if args.save:
    plt.savefig("latency_std_vs_batch_size.png", dpi=300)

# ---------------------------------------------------------
# 5. Recall standard deviation
# ---------------------------------------------------------
plt.figure(figsize=(8,5))

for idx in df["index"].unique():
    subset = df[df["index"] == idx].sort_values("nq")

    plt.plot(
        subset["nq"],
        subset["stdrcl"],
        marker="o",
        linewidth=2,
        label=labels.get(idx, idx),
    )

plt.xscale("log", base=2)
plt.xticks(
    [1,2,4,8,16,32,64,128,256,512,1024],
    [1,2,4,8,16,32,64,128,256,512,1024]
)

plt.xlabel("Query Batch Size")
plt.ylabel("Recall Standard Deviation")
plt.title("Recall Standard Deviation vs Query Batch Size")
plt.grid(True)
plt.legend()
plt.tight_layout()

if args.save:
    plt.savefig("recall_std_vs_batch_size.png", dpi=300)

plt.show()