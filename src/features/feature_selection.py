from __future__ import annotations

import numpy as np
import pandas as pd


def remove_correlated_features(
    X: pd.DataFrame,
    iv_report: pd.DataFrame,
    threshold: float = 0.80,
    ):
    """
    Remove highly correlated WoE features.

    Among correlated variables, the feature with the highest
    Information Value (IV) is kept.

    Returns
    -------
    selected_features
    removed_features
    decision_report
    """

    corr = X.corr().abs()

    # IV lookup using original variable names
    iv_lookup = (
        iv_report
        .set_index("variable")["information_value"]
        .to_dict()
    )

    # Match WoE feature names with original variable names
    feature_iv = {}

    for feature in X.columns:

        if feature.endswith("_woe"):
            original_name = feature[:-4]
        else:
            original_name = feature

        if original_name not in iv_lookup:
            raise ValueError(
                f"IV not found for feature '{feature}' "
                f"(expected variable '{original_name}' in iv_report)."
            )

        feature_iv[feature] = iv_lookup[original_name]

    # Highest IV first
    ordered_features = sorted(
        X.columns,
        key=lambda feature: feature_iv[feature],
        reverse=True,
    )

    selected = []
    removed = set()
    decisions = []

    for feature in ordered_features:

        if feature in removed:
            continue

        # Keep the current feature because it has the highest
        # remaining IV
        selected.append(feature)

        for other_feature in ordered_features:

            if (
                other_feature == feature
                or other_feature in removed
                or other_feature in selected
            ):
                continue

            corr_value = corr.loc[
                feature,
                other_feature,
            ]

            if corr_value >= threshold:

                removed.add(other_feature)

                decisions.append({
                    "removed_feature": other_feature,
                    "kept_feature": feature,
                    "correlation": round(
                        corr_value,
                        3,
                    ),
                    "removed_iv": round(
                        feature_iv[other_feature],
                        4,
                    ),
                    "kept_iv": round(
                        feature_iv[feature],
                        4,
                    ),
                    "reason": "Lower IV among correlated features",
                })

    decision_report = pd.DataFrame(decisions)

    if not decision_report.empty:
        decision_report = (
            decision_report
            .sort_values(
                "correlation",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    return (
        selected,
        sorted(removed),
        decision_report,
    )