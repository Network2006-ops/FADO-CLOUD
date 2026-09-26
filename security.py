import os
import matplotlib.pyplot as plt
from data import results, scenario_x, scenario_labels, OUTPUT_DIR

# ================================================================
# GRAPH 9: SECURITY
# ================================================================

plt.figure(figsize=(12, 6.5))

sec = results["Security_percent"]

plt.plot(
    scenario_x,
    sec,
    marker="o",
    linewidth=3,
    markersize=8,
    color="black"
)

# Text annotations above each point
for x, y in zip(scenario_x, sec):
    plt.annotate(
        f"{y:.2f}%",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

avg_val = sec.mean()
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
    [98.25, 98.50, 98.75, 99.00, 99.25, 99.50, 99.75, 100.00],
    [f"{v:.2f}" for v in [98.25, 98.50, 98.75, 99.00, 99.25, 99.50, 99.75, 100.00]],
    fontsize=11
)

plt.ylim(98.20, 100.00)

plt.xlabel(
    "Simulation Time",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel(
    "Security (%) ↑",
    fontsize=15,
    fontweight="bold"
)

plt.title(
    "Proposed FADO-CLOUD Security Under Dynamic Scenarios",
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
        "09_FADO_CLOUD_Security.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

if __name__ == "__main__":
    plt.show()
    print("All 9 graphs generated successfully.")
    print("All graph colours are selected without blue.")