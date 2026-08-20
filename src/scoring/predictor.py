from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import scorecardpy as sc


def load_credit_risk_pipeline(path: str | Path) -> Dict[str, Any]:
    import pickle

    path = Path(path)

    with path.open("rb") as file:
        pipeline = pickle.load(file)

    required_keys = {
        "lr_model",
        "woe_bins",
        "selected_features",
    }

    missing_keys = required_keys - set(pipeline)

    if missing_keys:
        raise KeyError(
            f"Pipeline is missing required keys: {sorted(missing_keys)}"
        )

    return pipeline


def _prepare_for_scorecardpy(data: pd.DataFrame) -> pd.DataFrame:
    prepared = data.copy()

    for column in prepared.columns:

        if pd.api.types.is_numeric_dtype(prepared[column]):
            prepared[column] = (
                pd.to_numeric(
                    prepared[column],
                    errors="coerce",
                )
                .astype("float64")
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

        else:
            prepared[column] = (
                prepared[column]
                .astype("object")
                .where(
                    prepared[column].notna(),
                    np.nan,
                )
            )

    return prepared


def transform_to_woe(
    data: pd.DataFrame,
    bins: Dict,
) -> pd.DataFrame:
    prepared = _prepare_for_scorecardpy(data)

    return sc.woebin_ply(
        prepared,
        bins,
        print_step=0,
    )


def select_model_features(
    woe_data: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    missing_features = [
        feature
        for feature in selected_features
        if feature not in woe_data.columns
    ]

    if missing_features:
        raise KeyError(
            "Missing WoE features required by the model: "
            f"{missing_features}"
        )

    return woe_data[selected_features].copy()


def predict_pd(
    data: pd.DataFrame,
    pipeline: Dict[str, Any],
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data is empty.")

    woe_data = transform_to_woe(
        data=data,
        bins=pipeline["woe_bins"],
    )

    model_data = select_model_features(
        woe_data=woe_data,
        selected_features=pipeline["selected_features"],
    )

    probabilities = pipeline["lr_model"].predict_proba(
        model_data
    )[:, 1]

    result = pd.DataFrame(
        {
            "probability_default": probabilities,
        },
        index=data.index,
    )

    if "ks_threshold" in pipeline:
        threshold = float(
            pipeline["ks_threshold"]
        )

        result["statistical_class"] = np.where(
            probabilities >= threshold,
            "Bad",
            "Good",
        )

    return result
