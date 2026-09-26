import os
import matplotlib.pyplot as plt
from data import results, scenario_x, scenario_labels, OUTPUT_DIR

# ================================================================
# GRAPH 4: ENERGY CONSUMPTION
# ================================================================

plt.figure(figsize=(12, 6.5))

energy = results["Energy_J"]

plt.plot(
    scenario_x,
    energy,
    marker="o",
    linewidth=3,
    markersize=8,
    color="black"
)

# Text annotations above each point
for x, y in zip(scenario_x, energy):
    plt.annotate(
        f"{y:.1f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

avg_val = energy.mean()
plt.axhline(
    avg_val,
    linestyle="--",
    linewidth=1.5,
    color="gray",
    label=f"Overall Average = {avg_val:.1f} J"
)

plt.xticks(
    scenario_x,
    scenario_labels,
    fontsize=11
)

plt.yticks(
    range(440, 540, 20),
    fontsize=11
)

plt.ylim(420, 525)

plt.xlabel(
    "Simulation Time",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel(
    "Energy Consumption (J) ↓",
    fontsize=15,
    fontweight="bold"
)

plt.title(
    "Proposed FADO-CLOUD Energy Consumption Under Dynamic Scenarios",
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
        "04_FADO_CLOUD_Energy.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

if __name__ == "__main__":
    plt.show()