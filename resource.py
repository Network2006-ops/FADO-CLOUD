import os
import matplotlib.pyplot as plt
from data import results, scenario_x, scenario_labels, OUTPUT_DIR

# ================================================================
# GRAPH 2: RESOURCE STATUS OVER TIME
# ================================================================

plt.figure(figsize=(12, 6.5))

plt.plot(
    scenario_x,
    results["CPU_Availability"],
    marker="o",
    linewidth=2.5,
    markersize=6,
    label="CPU Availability",
    color="#1f77b4"
)

plt.plot(
    scenario_x,
    results["RAM_Availability"],
    marker="s",
    linewidth=2.5,
    markersize=6,
    label="RAM Availability",
    color="#ff7f0e"
)

plt.plot(
    scenario_x,
    results["Disk_Availability"],
    marker="^",
    linewidth=2.5,
    markersize=6,
    label="Disk Availability",
    color="#2ca02c"
)

plt.plot(
    scenario_x,
    results["Network_Availability"],
    marker="d",
    linewidth=2.5,
    markersize=6,
    label="Network Availability",
    color="#d62728"
)

plt.xticks(
    scenario_x,
    scenario_labels,
    fontsize=11
)

plt.yticks(
    range(0, 120, 20),
    fontsize=11
)

plt.ylim(0, 100)

plt.xlabel(
    "Simulation Time",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel(
    "Resource Availability (%)",
    fontsize=15,
    fontweight="bold"
)

plt.title(
    "Dynamic Scenario: Resource Status Over Time",
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
        "02_Resource_Status.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

if __name__ == "__main__":
    plt.show()