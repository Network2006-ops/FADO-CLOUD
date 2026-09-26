import os
import matplotlib.pyplot as plt
from data import results, scenario_x, scenario_labels, OUTPUT_DIR

# ================================================================
# GRAPH 5: SLA VIOLATION
# ================================================================

plt.figure(figsize=(12, 6.5))

sla_val = results["SLA_Violation_percent"]

plt.plot(
    scenario_x,
    sla_val,
    marker="o",
    linewidth=3,
    markersize=8,
    color="black"
)

# Text annotations above each point
for x, y in zip(scenario_x, sla_val):
    plt.annotate(
        f"{y:.2f}%",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

avg_val = sla_val.mean()
plt.axhline(
    avg_val,
    linestyle="--",
    linewidth=1.5,
    color="gray",
    label=f"Overall Average = {avg_val:.2f}%"
)

plt.xticks(
    scenario_x,
    scenario_labels,
    fontsize=11
)

plt.yticks(
    [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
    [f"{v:.2f}" for v in [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]],
    fontsize=11
)

plt.ylim(0.00, 2.00)

plt.xlabel(
    "Simulation Time",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel(
    "SLA Violation (%) ↓",
    fontsize=15,
    fontweight="bold"
)

plt.title(
    "Proposed FADO-CLOUD SLA Violation Under Dynamic Scenarios",
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
        "05_FADO_CLOUD_SLA.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

if __name__ == "__main__":
    plt.show()
