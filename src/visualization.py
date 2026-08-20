# =============================================================================
# Scoring Visualizations
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Wedge

from src.config import RISK_COLORS


def plot_risk_gauge(
    pd_value,
    score,
    grade,
    risk_level,
    borrower_id=None,
):
    """
    Plot an individual borrower credit-risk gauge.
    """

    zones = [
        {
            "name": "Very High Risk",
            "pd_range": "PD > 20%",
            "score_range": "Score ≤ 484",
            "color": RISK_COLORS["E"],
        },
        {
            "name": "High Risk",
            "pd_range": "10% < PD ≤ 20%",
            "score_range": "484–542",
            "color": RISK_COLORS["D"],
        },
        {
            "name": "Moderate Risk",
            "pd_range": "6% < PD ≤ 10%",
            "score_range": "542–582",
            "color": RISK_COLORS["C"],
        },
        {
            "name": "Low Risk",
            "pd_range": "3% < PD ≤ 6%",
            "score_range": "582–635",
            "color": RISK_COLORS["B"],
        },
        {
            "name": "Very Low Risk",
            "pd_range": "PD ≤ 3%",
            "score_range": "Score ≥ 635",
            "color": RISK_COLORS["A"],
        },
    ]

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    zone_width = 36

    # -------------------------------------------------------------------------
    # Gauge zones
    # -------------------------------------------------------------------------

    for i, zone in enumerate(zones):

        theta2 = 180 - i * zone_width
        theta1 = theta2 - zone_width

        ax.add_patch(
            Wedge(
                (0, 0),
                1,
                theta1,
                theta2,
                width=0.30,
                facecolor=zone["color"],
                edgecolor="white",
                linewidth=3,
            )
        )

        angle = np.deg2rad(
            (theta1 + theta2) / 2
        )

        x = 0.82 * np.cos(angle)
        y = 0.82 * np.sin(angle)

        ax.text(
            x,
            y + 0.05,
            zone["name"],
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

        ax.text(
            x,
            y - 0.02,
            zone["pd_range"],
            ha="center",
            va="center",
            fontsize=8,
        )

        ax.text(
            x,
            y - 0.09,
            zone["score_range"],
            ha="center",
            va="center",
            fontsize=8,
        )

    # -------------------------------------------------------------------------
    # Exact PD position within assigned risk zone
    # -------------------------------------------------------------------------

    if pd_value <= 0.03:
        zone_index = 4
        position = pd_value / 0.03

    elif pd_value <= 0.06:
        zone_index = 3
        position = (
            (pd_value - 0.03)
            / 0.03
        )

    elif pd_value <= 0.10:
        zone_index = 2
        position = (
            (pd_value - 0.06)
            / 0.04
        )

    elif pd_value <= 0.20:
        zone_index = 1
        position = (
            (pd_value - 0.10)
            / 0.10
        )

    else:
        zone_index = 0
        position = min(
            (pd_value - 0.20)
            / 0.30,
            1.0,
        )

    theta2 = (
        180
        - zone_index * zone_width
    )

    theta1 = (
        theta2
        - zone_width
    )

    needle_angle = np.deg2rad(
        theta1
        + position * zone_width
    )

    # -------------------------------------------------------------------------
    # Needle
    # -------------------------------------------------------------------------

    needle_length = 0.72

    ax.plot(
        [
            0,
            needle_length
            * np.cos(needle_angle),
        ],
        [
            0,
            needle_length
            * np.sin(needle_angle),
        ],
        linewidth=5,
        solid_capstyle="round",
    )

    ax.add_patch(
        Circle(
            (0, 0),
            0.045,
        )
    )

    # -------------------------------------------------------------------------
    # Borrower information
    # -------------------------------------------------------------------------

    title = (
        "Borrower Credit Risk Assessment"
    )

    if borrower_id is not None:
        title += (
            f" — Client {borrower_id}"
        )

    ax.text(
        0,
        1.08,
        title,
        ha="center",
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        0,
        -0.10,
        (
            f"Credit Score: {int(score)}   |   "
            f"Predicted PD: {pd_value:.2%}   |   "
            f"Grade: {grade}   |   "
            f"{risk_level}"
        ),
        ha="center",
        fontsize=11.5,
        fontweight="bold",
    )

    ax.set_xlim(
        -1.12,
        1.12,
    )

    ax.set_ylim(
        -0.18,
        1.15,
    )

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    return fig