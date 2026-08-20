from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def find_ks_threshold(
    y_true,
    probabilities,
) -> tuple[float, float, pd.DataFrame]:
    """
    Find the threshold that maximizes the Kolmogorov-Smirnov statistic.

    KS = TPR - FPR

    Returns
    -------
    best_threshold : float
        Threshold where KS is maximal.

    best_ks : float
        Maximum KS statistic.

    ks_report : pd.DataFrame
        Threshold-by-threshold KS details.
    """

    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities,
    )

    ks_values = tpr - fpr

    ks_report = pd.DataFrame({
        "threshold": thresholds,
        "tpr_bad": tpr,
        "fpr_good": fpr,
        "ks": ks_values,
    })

    best_index = ks_report["ks"].idxmax()

    best_threshold = float(
        ks_report.loc[
            best_index,
            "threshold",
        ]
    )

    best_ks = float(
        ks_report.loc[
            best_index,
            "ks",
        ]
    )

    return (
        best_threshold,
        best_ks,
        ks_report,
    )

def find_best_f1_threshold(
    y_true,
    probabilities,
    start: float = 0.05,
    stop: float = 0.50,
    step: float = 0.01,
) -> tuple[float, pd.DataFrame]:
    """
    Find the classification threshold that maximizes F1-score.
    """

    thresholds = np.arange(start, stop + step, step)

    rows = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "accuracy": accuracy_score(
                y_true,
                predictions,
            ),
        })

    report = pd.DataFrame(rows)

    best_index = report["f1_score"].idxmax()

    best_threshold = float(
        report.loc[
            best_index,
            "threshold",
        ]
    )

    return best_threshold, report


def classification_metrics(
    y_true,
    probabilities,
    threshold: float,
) -> pd.DataFrame:
    """
    Return main classification metrics for a given threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    return pd.DataFrame({
        "metric": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "gini",
        ],
        "value": [
            accuracy_score(
                y_true,
                predictions,
            ),
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            auc,
            2 * auc - 1,
        ],
    })


def print_classification_report(
    y_true,
    probabilities,
    threshold: float,
    ) -> None:
    """
    Print sklearn classification report.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Good",
                "Bad",
            ],
            zero_division=0,
        )
    )


def plot_confusion_matrix(
    y_true,
    probabilities,
    threshold: float,
) -> np.ndarray:
    """
    Plot confusion matrix for a chosen threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
    )

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Good",
            "Bad",
        ],
    ).plot()

    plt.title(
        f"Confusion Matrix — threshold={threshold:.2f}"
    )
    plt.show()

    return matrix


def plot_ks_curve(
    ks_report: pd.DataFrame,
) -> None:
    """Plot TPR, FPR and KS separation across thresholds."""

    best_index = ks_report["ks"].idxmax()

    best_threshold = ks_report.loc[
        best_index,
        "threshold",
    ]

    best_ks = ks_report.loc[
        best_index,
        "ks",
    ]

    plt.figure(figsize=(8, 6))

    plt.plot(
        ks_report["threshold"],
        ks_report["tpr_bad"],
        label="Cumulative Bad / TPR",
    )

    plt.plot(
        ks_report["threshold"],
        ks_report["fpr_good"],
        label="Cumulative Good / FPR",
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        label=f"Max KS = {best_ks:.3f}",
    )

    plt.xlabel("PD threshold")
    plt.ylabel("Cumulative rate")
    plt.title("Kolmogorov-Smirnov Curve")
    plt.legend()
    plt.show()
    
def plot_roc_curve(
    y_true,
    probabilities,
) -> float:
    """
    Plot ROC curve and return ROC-AUC.
    """

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities,
    )

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.3f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )
    plt.ylabel(
        "True Positive Rate"
    )
    plt.title(
        "ROC Curve"
    )
    plt.legend()

    plt.show()

    return auc