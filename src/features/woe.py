"""
woe.py
------

Utilities for:
- applying WoE transformations;
- analysing original categorical WoE profiles;
- comparing scorecardpy automatic bins;
- reviewing all categorical variables.
"""

from __future__ import annotations

from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scorecardpy as sc


def apply_woe(
    data: pd.DataFrame,
    bins: Dict,
    target: str | None = None,
) -> pd.DataFrame:
    """
    Apply saved WoE bins to a dataset.

    Converts pandas nullable dtypes and pd.NA values into formats
    safely handled by scorecardpy.
    """

    prepared_data = data.copy()

    for column in prepared_data.columns:

        # Numerical variables
        if pd.api.types.is_numeric_dtype(prepared_data[column]):
            prepared_data[column] = (
                pd.to_numeric(
                    prepared_data[column],
                    errors="coerce",
                )
                .astype("float64")
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

        # Categorical variables
        else:
            prepared_data[column] = (
                prepared_data[column]
                .astype("object")
                .where(
                    prepared_data[column].notna(),
                    np.nan,
                )
            )

    transformed = sc.woebin_ply(
        prepared_data,
        bins,
        print_step=0,
    )

    if (
        target is not None
        and target in transformed.columns
    ):
        transformed = transformed.drop(
            columns=target
        )

    return transformed


def merge_bins(
    categorical_bins: Dict,
    numeric_bins: Dict,
) -> Dict:
    """Merge categorical and numerical bin dictionaries."""

    return {**categorical_bins, **numeric_bins}


def categorical_woe_profile(
    data: pd.DataFrame,
    variable: str,
    target: str,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """
    Calculate WoE for every original category before grouping.

    Target convention:
        1 = Bad
        0 = Good
    """

    missing_columns = {variable, target} - set(data.columns)

    if missing_columns:
        raise KeyError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    working_data = data[[variable, target]].copy()

    working_data[variable] = (
        working_data[variable]
        .astype("object")
        .where(
            working_data[variable].notna(),
            "missing",
        )
        .astype(str)
    )

    table = (
        working_data
        .groupby(variable, dropna=False)[target]
        .agg(
            total="count",
            bad="sum",
        )
        .reset_index()
    )

    table["good"] = table["total"] - table["bad"]

    total_good = table["good"].sum()
    total_bad = table["bad"].sum()
    number_of_categories = len(table)

    table["distribution"] = (
        table["total"] / table["total"].sum()
    )

    table["bad_rate"] = (
        table["bad"] / table["total"]
    )

    table["dist_good"] = (
        table["good"] + smoothing
    ) / (
        total_good + smoothing * number_of_categories
    )

    table["dist_bad"] = (
        table["bad"] + smoothing
    ) / (
        total_bad + smoothing * number_of_categories
    )

    table["woe"] = np.log(
        table["dist_good"] / table["dist_bad"]
    )

    table["iv_component"] = (
        table["dist_good"] - table["dist_bad"]
    ) * table["woe"]

    return (
        table
        .sort_values("woe")
        .reset_index(drop=True)
    )


def plot_categorical_woe_profile(
    profile: pd.DataFrame,
    variable: str,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Plot original-category WoE and population distribution."""

    if figsize is None:
        width = max(
            10,
            min(24, len(profile) * 0.40),
        )
        figsize = (width, 6)

    labels = profile[variable].astype(str)

    fig, ax_woe = plt.subplots(figsize=figsize)

    ax_woe.plot(
        labels,
        profile["woe"],
        marker="o",
        linewidth=1.5,
    )

    ax_woe.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax_woe.set_xlabel(variable)
    ax_woe.set_ylabel("Weight of Evidence")
    ax_woe.tick_params(
        axis="x",
        rotation=90,
    )

    ax_distribution = ax_woe.twinx()

    ax_distribution.bar(
        labels,
        profile["distribution"],
        alpha=0.18,
    )

    ax_distribution.set_ylabel(
        "Population distribution"
    )

    plt.title(
        f"Original-category WoE profile — {variable}"
    )

    plt.tight_layout()
    plt.show()


def _extract_bin_categories(
    bin_value: object,
) -> list[str]:
    """Extract category names from one scorecardpy bin label."""

    if pd.isna(bin_value):
        return ["missing"]

    return [
        category.strip()
        for category in str(bin_value).split("%,%")
    ]


def compare_categorical_groups(
    original_profile: pd.DataFrame,
    binned_table: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:
    """
    Compare original-category WoE dispersion inside each automatic bin.
    """

    if "bin" not in binned_table.columns:
        raise KeyError(
            "The scorecardpy table must contain a 'bin' column."
        )

    profile = original_profile.copy()
    profile[variable] = (
        profile[variable]
        .astype(str)
    )

    report_rows = []

    for _, row in binned_table.iterrows():
        bin_label = row["bin"]

        categories = _extract_bin_categories(
            bin_label
        )

        members = profile[
            profile[variable].isin(categories)
        ].copy()

        if members.empty:
            report_rows.append({
                "bin": str(bin_label),
                "categories_expected": len(categories),
                "categories_matched": 0,
                "minimum_original_woe": np.nan,
                "maximum_original_woe": np.nan,
                "woe_range": np.nan,
                "weighted_bad_rate": np.nan,
                "total_observations": 0,
                "group_quality": "Unmatched",
            })
            continue

        minimum_woe = members["woe"].min()
        maximum_woe = members["woe"].max()
        woe_range = maximum_woe - minimum_woe

        if woe_range <= 0.20:
            group_quality = "Homogeneous"
        elif woe_range <= 0.40:
            group_quality = "Review"
        else:
            group_quality = "Too heterogeneous"

        report_rows.append({
            "bin": str(bin_label),
            "categories_expected": len(categories),
            "categories_matched": len(members),
            "minimum_original_woe": minimum_woe,
            "maximum_original_woe": maximum_woe,
            "woe_range": woe_range,
            "weighted_bad_rate": (
                members["bad"].sum()
                / members["total"].sum()
            ),
            "total_observations": int(
                members["total"].sum()
            ),
            "group_quality": group_quality,
        })

    return pd.DataFrame(report_rows)


def review_categorical_binning(
    data: pd.DataFrame,
    variable: str,
    target: str,
    bins: Dict,
    show_profile_table: bool = True,
    show_plot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Review one categorical variable from original categories
    to automatic scorecardpy groups.
    """

    if variable not in bins:
        raise KeyError(
            f"No successful scorecardpy binning found for '{variable}'."
        )

    profile = categorical_woe_profile(
        data=data,
        variable=variable,
        target=target,
    )

    automatic_table = bins[variable]

    comparison = compare_categorical_groups(
        original_profile=profile,
        binned_table=automatic_table,
        variable=variable,
    )

    original_iv = float(
        profile["iv_component"].sum()
    )

    if "total_iv" in automatic_table.columns:
        automatic_iv = float(
            automatic_table["total_iv"].iloc[0]
        )
    elif "bin_iv" in automatic_table.columns:
        automatic_iv = float(
            automatic_table["bin_iv"].sum()
        )
    else:
        automatic_iv = np.nan

    print("=" * 70)
    print(f"Variable: {variable}")
    print(f"Original categories: {len(profile)}")
    print(f"Automatic bins: {len(automatic_table)}")
    print(f"Original-category IV: {original_iv:.4f}")

    if np.isnan(automatic_iv):
        print("Automatic-binning IV: unavailable")
    else:
        print(
            f"Automatic-binning IV: {automatic_iv:.4f}"
        )

    print("=" * 70)

    if show_profile_table:
        display(
            profile[
                [
                    variable,
                    "total",
                    "distribution",
                    "bad_rate",
                    "woe",
                    "iv_component",
                ]
            ]
        )

    if show_plot:
        plot_categorical_woe_profile(
            profile=profile,
            variable=variable,
        )

    print("Automatic bins proposed by scorecardpy:")
    display(automatic_table)

    print(
        "Original-category dispersion "
        "inside each automatic bin:"
    )
    display(comparison)

    return profile, comparison


def review_all_categorical_binnings(
    data: pd.DataFrame,
    variables: Iterable[str],
    target: str,
    bins: Dict,
    binning_report: pd.DataFrame | None = None,
    show_profile_tables: bool = True,
    show_plots: bool = True,
) -> tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """
    Review all categorical variables.

    Failed automatic binnings are skipped without stopping the loop.
    """

    variables = list(variables)

    profiles: Dict[str, pd.DataFrame] = {}
    comparisons: Dict[str, pd.DataFrame] = {}
    report_rows = []

    failure_messages = {}

    if (
        binning_report is not None
        and not binning_report.empty
        and {"variable", "status"}.issubset(
            binning_report.columns
        )
    ):
        failed_rows = binning_report[
            binning_report["status"].eq("failed")
        ]

        if "error_message" in failed_rows.columns:
            failure_messages = dict(
                zip(
                    failed_rows["variable"],
                    failed_rows["error_message"],
                )
            )

    for position, variable in enumerate(
        variables,
        start=1,
    ):
        print(
            f"\n[{position}/{len(variables)}] "
            f"Reviewing {variable}"
        )

        if variable not in data.columns:
            report_rows.append({
                "variable": variable,
                "review_status": "failed",
                "original_categories": np.nan,
                "automatic_bins": np.nan,
                "original_iv": np.nan,
                "automatic_iv": np.nan,
                "heterogeneous_groups": np.nan,
                "groups_to_review": np.nan,
                "message": (
                    "Variable not found in training data."
                ),
            })
            continue

        if variable not in bins:
            report_rows.append({
                "variable": variable,
                "review_status": "skipped",
                "original_categories": int(
                    data[variable].nunique(
                        dropna=False
                    )
                ),
                "automatic_bins": 0,
                "original_iv": np.nan,
                "automatic_iv": np.nan,
                "heterogeneous_groups": np.nan,
                "groups_to_review": np.nan,
                "message": failure_messages.get(
                    variable,
                    (
                        "No successful automatic "
                        "binning available."
                    ),
                ),
            })

            print(
                f"Skipped: "
                f"{report_rows[-1]['message']}"
            )
            continue

        try:
            profile, comparison = (
                review_categorical_binning(
                    data=data,
                    variable=variable,
                    target=target,
                    bins=bins,
                    show_profile_table=(
                        show_profile_tables
                    ),
                    show_plot=show_plots,
                )
            )

            profiles[variable] = profile
            comparisons[variable] = comparison

            automatic_table = bins[variable]

            if "total_iv" in automatic_table.columns:
                automatic_iv = float(
                    automatic_table[
                        "total_iv"
                    ].iloc[0]
                )
            elif "bin_iv" in automatic_table.columns:
                automatic_iv = float(
                    automatic_table[
                        "bin_iv"
                    ].sum()
                )
            else:
                automatic_iv = np.nan

            heterogeneous_groups = int(
                comparison["group_quality"]
                .eq("Too heterogeneous")
                .sum()
            )

            groups_to_review = int(
                comparison["group_quality"]
                .eq("Review")
                .sum()
            )

            review_status = (
                "review"
                if (
                    heterogeneous_groups > 0
                    or groups_to_review > 0
                )
                else "acceptable"
            )

            report_rows.append({
                "variable": variable,
                "review_status": review_status,
                "original_categories": len(profile),
                "automatic_bins": len(
                    automatic_table
                ),
                "original_iv": float(
                    profile[
                        "iv_component"
                    ].sum()
                ),
                "automatic_iv": automatic_iv,
                "heterogeneous_groups": (
                    heterogeneous_groups
                ),
                "groups_to_review": (
                    groups_to_review
                ),
                "message": "",
            })

        except Exception as error:
            report_rows.append({
                "variable": variable,
                "review_status": "failed",
                "original_categories": int(
                    data[variable].nunique(
                        dropna=False
                    )
                ),
                "automatic_bins": (
                    len(bins[variable])
                    if variable in bins
                    else 0
                ),
                "original_iv": np.nan,
                "automatic_iv": np.nan,
                "heterogeneous_groups": np.nan,
                "groups_to_review": np.nan,
                "message": str(error),
            })

            print(f"Review failed: {error}")

    return (
        profiles,
        comparisons,
        pd.DataFrame(report_rows),
    )
