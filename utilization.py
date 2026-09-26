import os
import matplotlib.pyplot as plt
from data import results, scenario_x, scenario_labels, OUTPUT_DIR

# ================================================================
# GRAPH 8: RESOURCE UTILIZATION
# ================================================================

plt.figure(figsize=(12, 6.5))

util = results["Utilization_percent"]

plt.plot(
    scenario_x,
    util,
    marker="o",
    linewidth=3,
    markersize=8,
    color="black"
)

# Text annotations above each point
for x, y in zip(scenario_x, util):
    plt.annotate(
        f"{y:.1f}%",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

avg_val = util.mean()
plt.axhline(
    avg_val,
    linestyle="--",
    linewidth=1.5,
    color="gray",
    label=f"Overall Average = {avg_val:.1f}%"
)

plt.xticks(
    scenario_x,
    scenario_labels,
    fontsize=11
)

plt.yticks(
    range(86, 102, 2),
    fontsize=11
)

plt.ylim(86, 100.5)

plt.xlabel(
    "Simulation Time",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel(
    "Resource Utilization (%) ↑",
    fontsize=15,
    fontweight="bold"
)

plt.title(
    "Proposed FADO-CLOUD Resource Utilization",
    fontsize=18,
    fontweight="bold"
)

plt.legend(
    loc="upper left",
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
        "08_FADO_CLOUD_Utilization.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

if __name__ == "__main__":
    plt.show()
