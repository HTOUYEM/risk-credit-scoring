import pandas as pd

from scipy.stats import ks_2samp
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def evaluate_pd_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:

    y_proba = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(
        y,
        y_proba,
    )

    ks_stat = ks_2samp(
        y_proba[y == 0],
        y_proba[y == 1],
    ).statistic

    return pd.DataFrame(
        {
            "ROC-AUC": [roc_auc],
            "Gini": [2 * roc_auc - 1],
            "KS": [ks_stat],
            "Mean Predicted PD": [y_proba.mean()],
            "Observed Default Rate": [y.mean()],
            "Brier Score": [
                brier_score_loss(
                    y,
                    y_proba,
                )
            ],
            "Log Loss": [
                log_loss(
                    y,
                    y_proba,
                )
            ],
        }
    )