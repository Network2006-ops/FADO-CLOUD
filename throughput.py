import os
import matplotlib.pyplot as plt
from data import results, scenario_x, scenario_labels, OUTPUT_DIR

# ================================================================
# GRAPH 7: THROUGHPUT
# ================================================================

plt.figure(figsize=(12, 6.5))

tp = results["Throughput_tasks_per_s"]

plt.plot(
    scenario_x,
    tp,
    marker="o",
    linewidth=3,
    markersize=8,
    color="black"
)

# Text annotations above each point
for x, y in zip(scenario_x, tp):
    plt.annotate(
        f"{y:.1f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

avg_val = tp.mean()
plt.axhline(
    avg_val,
    linestyle="--",
    linewidth=1.5,
    color="gray",
    label=f"Overall Average = {avg_val:.1f} tasks/s"
)

plt.xticks(
    scenario_x,
    scenario_labels,
    fontsize=11
)

plt.yticks(
    range(420, 490, 10),
    fontsize=11
)

plt.ylim(415, 495)

plt.xlabel(
    "Simulation Time",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel(
    "Throughput (tasks/s) ↑",
    fontsize=15,
    fontweight="bold"
)

plt.title(
    "Proposed FADO-CLOUD Throughput Under Dynamic Scenarios",
    fontsize=18,
    fontweight="bold"
)

plt.legend(
    loc="upper right",
    fontsize=10
)

plt.grid(
    True,
    linestyle=":",
    alpha=0.35
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "07_FADO_CLOUD_Throughput.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

if __name__ == "__main__":
    plt.show()