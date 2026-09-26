import os
import matplotlib.pyplot as plt
from data import results, scenario_x, scenario_labels, OUTPUT_DIR

# ================================================================
# GRAPH 1: NUMBER OF TASKS OVER TIME
# ================================================================

plt.figure(figsize=(12, 6.5))

tasks = results["Tasks_Count"]

plt.plot(
    scenario_x,
    tasks,
    marker="o",
    linewidth=3,
    markersize=8,
    color="#800080"
)

# Text annotations above each point
for x, y in zip(scenario_x, tasks):
    plt.annotate(
        f"{y} tasks",
        (x, y),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.xticks(
    scenario_x,
    scenario_labels,
    fontsize=11
)

plt.yticks(
    range(100, 600, 100),
    fontsize=11
)

plt.ylim(50, 520)

plt.xlabel(
    "Simulation Time",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel(
    "Number of Tasks",
    fontsize=15,
    fontweight="bold"
)

plt.title(
    "Dynamic Scenario: Number of Tasks Over Time",
    fontsize=18,
    fontweight="bold"
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
        "01_Task_Workload.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

if __name__ == "__main__":
    plt.show()
