import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
csv_path = script_dir / "bs-sweep-stats.csv"

# Read CSV
df = pd.read_csv(csv_path)

# Index display names
labels = {
    "cpu-baseline": "Flat",
    "cpu-hnsw": "HNSW",
    "cpu-ivf-flat": "IVF-Flat",
    "cpu-ivf-pq": "IVF-PQ",
}

# Metrics to plot
plots = [
    ("avglat", "Average Latency (s)", script_dir / "latency_vs_batch_size.png"),
    ("avgQPS", "Average Throughput (QPS)", script_dir / "throughput_vs_batch_size.png"),
    ("avgrecall", "Average Recall@10", script_dir / "recall_vs_batch_size.png"),
]

for metric, ylabel, outfile in plots:
    plt.figure(figsize=(7, 5))

    for idx in df["index"].unique():
        subset = df[df["index"] == idx].sort_values("nq")

        plt.plot(
            subset["nq"],
            subset[metric],
            marker="o",
            linewidth=2,
            label=labels.get(idx, idx)
        )

    plt.xscale("log", base=2)
    plt.xticks(
        [1,2,4,8,16,32,64,128,256,512,1024],
        [1,2,4,8,16,32,64,128,256,512,1024]
    )

    plt.xlabel("Query Batch Size")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs Query Batch Size")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)

plt.show()