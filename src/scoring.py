# =============================================================================
# Credit Scoring Utilities
# =============================================================================

import numpy as np
import pandas as pd

from src.config import (
    BASE_SCORE,
    BASE_ODDS,
    PDO,
    PD_BINS,
    GRADE_LABELS,
    RISK_LABELS,
)


FACTOR = (
    PDO
    / np.log(2)
)

OFFSET = (
    BASE_SCORE
    - FACTOR * np.log(BASE_ODDS)
)


def pd_to_credit_score(pd_values):
    """
    Convert predicted probabilities of default into credit scores
    using good-to-bad odds and PDO scaling.
    """

    pd_values = np.asarray(
        pd_values,
        dtype=float,
    )

    pd_clipped = np.clip(
        pd_values,
        1e-6,
        1 - 1e-6,
    )

    good_bad_odds = (
        (1 - pd_clipped)
        / pd_clipped
    )

    scores = (
        OFFSET
        + FACTOR * np.log(good_bad_odds)
    )

    return np.rint(
        scores
    ).astype(int)


def assign_risk_grade(pd_values):
    """
    Assign risk grades from fixed PD thresholds.
    """

    return pd.cut(
        pd_values,
        bins=PD_BINS,
        labels=GRADE_LABELS,
        include_lowest=True,
    )


def build_scoring_output(
    predicted_pd,
    index=None,
):
    """
    Build borrower-level PD, credit score, risk grade
    and risk level output.
    """

    output = pd.DataFrame(
        {
            "Predicted PD": predicted_pd,
        },
        index=index,
    )

    output["Credit Score"] = (
        pd_to_credit_score(
            output["Predicted PD"]
        )
    )

    output["Risk Grade"] = (
        assign_risk_grade(
            output["Predicted PD"]
        )
    )

    output["Risk Level"] = (
        output["Risk Grade"]
        .map(RISK_LABELS)
    )

    return output