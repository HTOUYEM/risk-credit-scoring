# =============================================================================
# PD Model Explainability Utilities
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# =============================================================================
# Display Names
# =============================================================================

FEATURE_DISPLAY_NAMES = {

    # -------------------------------------------------------------------------
    # Original borrower / loan variables
    # -------------------------------------------------------------------------

    "loan_amnt": "Loan Amount",
    "funded_amnt": "Funded Amount",
    "funded_amnt_inv": "Investor Funded Amount",
    "term": "Loan Term",
    "int_rate": "Interest Rate",
    "installment": "Monthly Installment",

    "annual_inc": "Annual Income",
    "dti": "Debt-to-Income Ratio",

    "delinq_2yrs": "Recent Delinquencies",
    "inq_last_6mths": "Recent Credit Inquiries",

    "open_acc": "Open Credit Accounts",
    "pub_rec": "Derogatory Public Records",

    "revol_bal": "Revolving Credit Balance",
    "revol_util": "Revolving Credit Utilization",
    "total_acc": "Total Credit Accounts",

    "collections_12_mths_ex_med": (
        "Collections in Last 12 Months"
    ),

    "acc_now_delinq": (
        "Accounts Currently Delinquent"
    ),

    "tot_coll_amt": (
        "Total Collection Amount"
    ),

    "tot_cur_bal": (
        "Total Current Balance"
    ),

    "total_rev_hi_lim": (
        "Total Revolving Credit Limit"
    ),


    # -------------------------------------------------------------------------
    # Engineered features
    # -------------------------------------------------------------------------

    "emp_length_years": (
        "Employment Length"
    ),

    "emp_length_missing": (
        "Employment Length Information"
    ),

    "emp_title_missing": (
        "Employment Information"
    ),

    "credit_history_months": (
        "Credit History Length"
    ),

    "funding_ratio": (
        "Funding Ratio"
    ),

    "investor_loan_ratio": (
        "Investor Funding Ratio"
    ),

    "open_account_ratio": (
        "Open Account Ratio"
    ),

    "delinquency_rate": (
        "Delinquency Rate"
    ),

    "credit_inquiry_rate": (
        "Credit Inquiry Rate"
    ),

    "loan_to_income_ratio": (
        "Loan-to-Income Ratio"
    ),

    "installment_to_income_ratio": (
        "Installment-to-Income Ratio"
    ),

    "loan_burden_interest": (
        "Loan Burden × Interest Rate"
    ),


    # -------------------------------------------------------------------------
    # Missingness indicators
    # -------------------------------------------------------------------------

    "mths_since_last_delinq_is_missing": (
        "Last Delinquency Information"
    ),

    "mths_since_last_record_is_missing": (
        "Last Public Record Information"
    ),

    "mths_since_last_major_derog_is_missing": (
        "Major Derogatory Event Information"
    ),

    "mths_since_last_delinq": (
        "Months Since Last Delinquency"
    ),

    "mths_since_last_record": (
        "Months Since Last Public Record"
    ),

    "mths_since_last_major_derog": (
        "Months Since Last Major Derogatory Event"
    ),
}


# =============================================================================
# Build SHAP Explainer
# =============================================================================

def build_shap_explainer(model):
    """
    Build a SHAP TreeExplainer for the fitted PD model.

    Parameters
    ----------
    model
        Fitted tree-based PD model.

    Returns
    -------
    shap.TreeExplainer
        SHAP explainer associated with the model.
    """

    return shap.TreeExplainer(
        model
    )


# =============================================================================
# Compute Local Borrower Explanation
# =============================================================================

def explain_borrower(
    model,
    encoded_features: pd.DataFrame,
):
    """
    Compute a local SHAP explanation for exactly one borrower.

    Parameters
    ----------
    model
        Fitted PD model.

    encoded_features : pd.DataFrame
        Single borrower already transformed into the encoded
        feature space expected by the final PD model.

    Returns
    -------
    shap.Explanation
        Borrower-specific SHAP explanation.
    """

    if not isinstance(
        encoded_features,
        pd.DataFrame,
    ):
        raise TypeError(
            "encoded_features must be a pandas DataFrame."
        )

    if encoded_features.shape[0] != 1:
        raise ValueError(
            "explain_borrower expects exactly one borrower."
        )

    explainer = build_shap_explainer(
        model
    )

    explanation = explainer(
        encoded_features
    )

    return explanation


# =============================================================================
# Human-Readable Feature Names
# =============================================================================

def humanize_feature_name(
    feature_name: str,
) -> str:
    """
    Convert a technical encoded model feature name into a
    user-friendly business label.
    """

    # -------------------------------------------------------------------------
    # Direct mapping
    # -------------------------------------------------------------------------

    if feature_name in FEATURE_DISPLAY_NAMES:

        return FEATURE_DISPLAY_NAMES[
            feature_name
        ]


    # -------------------------------------------------------------------------
    # Purpose groups
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "purpose_grouped_"
    ):

        return "Loan Purpose"


    # -------------------------------------------------------------------------
    # Home ownership
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "home_ownership_"
    ):

        return "Home Ownership"


    # -------------------------------------------------------------------------
    # Verification status
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "verification_status_"
    ):

        return "Income Verification Status"


    # -------------------------------------------------------------------------
    # Lending Club Sub-Grade
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "sub_grade_"
    ):

        return "Lending Club Loan Sub-Grade"


    # -------------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "addr_state_grouped_"
    ):

        return "Borrower State"


    # -------------------------------------------------------------------------
    # Initial listing status
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "initial_list_status_"
    ):

        return "Initial Listing Status"


    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------

    return (
        feature_name
        .replace("_", " ")
        .title()
    )


# =============================================================================
# Human-Readable Borrower Values
# =============================================================================

def format_feature_value(
    feature_name: str,
    encoded_value,
    raw_input: dict | None = None,
    native_features: pd.DataFrame | None = None,
) -> str:
    """
    Return a business-friendly borrower value for a model feature.
    """

    raw_input = (
        {}
        if raw_input is None
        else raw_input
    )


    # -------------------------------------------------------------------------
    # Currency features
    # -------------------------------------------------------------------------

    currency_features = {
        "loan_amnt",
        "funded_amnt",
        "funded_amnt_inv",
        "installment",
        "annual_inc",
        "revol_bal",
        "tot_coll_amt",
        "tot_cur_bal",
        "total_rev_hi_lim",
    }

    if feature_name in currency_features:

        value = raw_input.get(
            feature_name,
            encoded_value,
        )

        return (
            f"${float(value):,.0f}"
        )


    # -------------------------------------------------------------------------
    # Percentage features
    # -------------------------------------------------------------------------

    percent_features = {
        "int_rate",
        "dti",
        "revol_util",
    }

    if feature_name in percent_features:

        value = raw_input.get(
            feature_name,
            encoded_value,
        )

        return (
            f"{float(value):.1f}%"
        )


    # -------------------------------------------------------------------------
    # Ratio features stored as fractions
    # -------------------------------------------------------------------------

    ratio_percent_features = {
        "funding_ratio",
        "investor_loan_ratio",
        "open_account_ratio",
        "delinquency_rate",
        "installment_to_income_ratio",
    }

    if feature_name in ratio_percent_features:

        return (
            f"{float(encoded_value):.1%}"
        )


    # -------------------------------------------------------------------------
    # Employment missingness
    # -------------------------------------------------------------------------

    if feature_name == "emp_title_missing":

        return (
            "Missing"
            if float(encoded_value) == 1
            else "Provided"
        )


    if feature_name == "emp_length_missing":

        return (
            "Missing"
            if float(encoded_value) == 1
            else "Provided"
        )


    # -------------------------------------------------------------------------
    # Credit-history missingness indicators
    # -------------------------------------------------------------------------

    if feature_name.endswith(
        "_is_missing"
    ):

        return (
            "Missing"
            if float(encoded_value) == 1
            else "Available"
        )


    # -------------------------------------------------------------------------
    # Loan purpose
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "purpose_grouped_"
    ):

        purpose = raw_input.get(
            "purpose"
        )

        if purpose is not None:

            return (
                str(purpose)
                .replace("_", " ")
                .title()
            )

        category = feature_name.replace(
            "purpose_grouped_",
            "",
        )

        return (
            category
            .replace("_", " ")
            .title()
        )


    # -------------------------------------------------------------------------
    # Home ownership
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "home_ownership_"
    ):

        return str(
            raw_input.get(
                "home_ownership",
                feature_name.replace(
                    "home_ownership_",
                    "",
                ),
            )
        ).title()


    # -------------------------------------------------------------------------
    # Verification status
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "verification_status_"
    ):

        return str(
            raw_input.get(
                "verification_status",
                feature_name.replace(
                    "verification_status_",
                    "",
                ),
            )
        )


    # -------------------------------------------------------------------------
    # Sub-grade
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "sub_grade_"
    ):

        return str(
            raw_input.get(
                "sub_grade",
                feature_name.replace(
                    "sub_grade_",
                    "",
                ),
            )
        )


    # -------------------------------------------------------------------------
    # Borrower state
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "addr_state_grouped_"
    ):

        return str(
            raw_input.get(
                "addr_state",
                feature_name.replace(
                    "addr_state_grouped_",
                    "",
                ),
            )
        )


    # -------------------------------------------------------------------------
    # Initial listing status
    # -------------------------------------------------------------------------

    if feature_name.startswith(
        "initial_list_status_"
    ):

        status = raw_input.get(
            "initial_list_status"
        )

        if status == "f":
            return "Fractional (F)"

        if status == "w":
            return "Whole (W)"

        return str(status)


    # -------------------------------------------------------------------------
    # Credit-history length
    # -------------------------------------------------------------------------

    if feature_name == "credit_history_months":

        months = float(
            encoded_value
        )

        years = (
            months / 12
        )

        return (
            f"{months:.0f} months "
            f"({years:.1f} years)"
        )


    # -------------------------------------------------------------------------
    # Loan term
    # -------------------------------------------------------------------------

    if feature_name == "term":

        return (
            f"{int(float(encoded_value))} months"
        )


    # -------------------------------------------------------------------------
    # Loan burden × interest
    # -------------------------------------------------------------------------

    if feature_name == "loan_burden_interest":

        return (
            f"{float(encoded_value):.2f}"
        )


    # -------------------------------------------------------------------------
    # Generic numeric formatting
    # -------------------------------------------------------------------------

    try:

        numeric_value = float(
            encoded_value
        )

        if numeric_value.is_integer():

            return str(
                int(numeric_value)
            )

        return (
            f"{numeric_value:.3f}"
        )

    except (
        TypeError,
        ValueError,
    ):

        return str(
            encoded_value
        )


# =============================================================================
# Build Local Explanation Table
# =============================================================================

def build_local_explanation_table(
    explanation,
    encoded_features: pd.DataFrame,
    raw_input: dict | None = None,
    native_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a borrower-level SHAP contribution table.

    The table contains technical feature names, human-readable
    business labels, borrower values and signed SHAP contributions.
    """

    if not isinstance(
        encoded_features,
        pd.DataFrame,
    ):
        raise TypeError(
            "encoded_features must be a pandas DataFrame."
        )

    if encoded_features.shape[0] != 1:

        raise ValueError(
            "Exactly one borrower is required."
        )


    # -------------------------------------------------------------------------
    # SHAP values
    # -------------------------------------------------------------------------

    shap_values = np.asarray(
        explanation.values[0],
        dtype=float,
    )


    if len(shap_values) != encoded_features.shape[1]:

        raise ValueError(
            "SHAP values and encoded feature columns "
            "do not have matching dimensions."
        )


    # -------------------------------------------------------------------------
    # Base table
    # -------------------------------------------------------------------------

    table = pd.DataFrame(
        {
            "technical_feature": (
                encoded_features
                .columns
                .tolist()
            ),

            "encoded_value": (
                encoded_features
                .iloc[0]
                .to_numpy()
            ),

            "shap_value": (
                shap_values
            ),
        }
    )


    # -------------------------------------------------------------------------
    # Human-readable labels
    # -------------------------------------------------------------------------

    table["factor"] = (
        table["technical_feature"]
        .map(
            humanize_feature_name
        )
    )


    # -------------------------------------------------------------------------
    # Human-readable borrower values
    # -------------------------------------------------------------------------

    table["borrower_value"] = (
        table.apply(
            lambda row: format_feature_value(
                feature_name=(
                    row["technical_feature"]
                ),
                encoded_value=(
                    row["encoded_value"]
                ),
                raw_input=raw_input,
                native_features=native_features,
            ),
            axis=1,
        )
    )


    # -------------------------------------------------------------------------
    # Direction of local contribution
    # -------------------------------------------------------------------------

    table["direction"] = np.where(
        table["shap_value"] > 0,
        "Increasing Risk",
        np.where(
            table["shap_value"] < 0,
            "Reducing Risk",
            "Neutral",
        ),
    )


    # -------------------------------------------------------------------------
    # Absolute local importance
    # -------------------------------------------------------------------------

    table["absolute_shap"] = (
        table["shap_value"]
        .abs()
    )


    # -------------------------------------------------------------------------
    # Sort by local importance
    # -------------------------------------------------------------------------

    table = (
        table
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    return table


# =============================================================================
# Select Main Risk Drivers
# =============================================================================

def get_top_risk_drivers(
    explanation_table: pd.DataFrame,
    top_positive: int = 7,
    top_negative: int = 5,
):
    """
    Select the strongest borrower-specific factors increasing
    and reducing predicted default risk.

    Parameters
    ----------
    explanation_table : pd.DataFrame
        Borrower-level SHAP explanation table.

    top_positive : int
        Maximum number of positive SHAP contributions to retain.

    top_negative : int
        Maximum number of negative SHAP contributions to retain.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Increasing-risk drivers and reducing-risk drivers.
    """

    if top_positive < 0:
        raise ValueError(
            "top_positive must be non-negative."
        )

    if top_negative < 0:
        raise ValueError(
            "top_negative must be non-negative."
        )


    # -------------------------------------------------------------------------
    # Positive contributions
    # -------------------------------------------------------------------------

    increasing = (
        explanation_table[
            explanation_table[
                "shap_value"
            ] > 0
        ]
        .nlargest(
            top_positive,
            "absolute_shap",
        )
        .copy()
    )


    # -------------------------------------------------------------------------
    # Negative contributions
    # -------------------------------------------------------------------------

    reducing = (
        explanation_table[
            explanation_table[
                "shap_value"
            ] < 0
        ]
        .nlargest(
            top_negative,
            "absolute_shap",
        )
        .copy()
    )


    return (
        increasing,
        reducing,
    )


# =============================================================================
# Plot Local SHAP Risk Drivers
# =============================================================================

def plot_local_risk_drivers(
    increasing: pd.DataFrame,
    reducing: pd.DataFrame,
):
    """
    Plot borrower-specific SHAP drivers around a neutral zero line.

    Negative SHAP contributions push the model prediction toward
    lower predicted default risk.

    Positive SHAP contributions push the model prediction toward
    higher predicted default risk.
    """

    # -------------------------------------------------------------------------
    # Combine selected drivers
    # -------------------------------------------------------------------------

    drivers = pd.concat(
        [
            reducing,
            increasing,
        ],
        ignore_index=True,
    )


    if drivers.empty:

        raise ValueError(
            "No SHAP drivers available to plot."
        )


    # -------------------------------------------------------------------------
    # Sort drivers by signed contribution
    # -------------------------------------------------------------------------

    drivers = (
        drivers
        .sort_values(
            "shap_value",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )


    # -------------------------------------------------------------------------
    # Display labels
    # -------------------------------------------------------------------------

    labels = (
        drivers["factor"]
        + " — "
        + drivers["borrower_value"]
    )


    values = (
        drivers["shap_value"]
        .to_numpy(
            dtype=float
        )
    )


    # -------------------------------------------------------------------------
    # Figure size
    # -------------------------------------------------------------------------

    figure_height = max(
        4.5,
        0.55 * len(drivers) + 1.5,
    )


    fig, ax = plt.subplots(
        figsize=(
            11,
            figure_height,
        )
    )


    # -------------------------------------------------------------------------
    # Signed horizontal bars
    # -------------------------------------------------------------------------

    bar_colors = [
        "#2F7EA8"
        if value < 0
        else "#D95F59"
        for value in values
    ]


    ax.barh(
        labels,
        values,
        color=bar_colors,
        alpha=0.90,
    )


    # -------------------------------------------------------------------------
    # Neutral line
    # -------------------------------------------------------------------------

    ax.axvline(
        0,
        linewidth=1.2,
        color="#64748B",
    )


    # -------------------------------------------------------------------------
    # Contribution label spacing
    # -------------------------------------------------------------------------

    max_abs_value = max(
        np.abs(values).max(),
        0.01,
    )


    offset = (
        max_abs_value * 0.025
    )


    # -------------------------------------------------------------------------
    # SHAP values on bars
    # -------------------------------------------------------------------------

    for index, value in enumerate(
        values
    ):

        if value >= 0:

            x_position = (
                value + offset
            )

            horizontal_alignment = (
                "left"
            )

        else:

            x_position = (
                value - offset
            )

            horizontal_alignment = (
                "right"
            )


        ax.text(
            x_position,
            index,
            f"{value:+.3f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=9,
            fontweight="bold",
        )


    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    ax.set_title(
        "Why This Prediction?",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )


    # -------------------------------------------------------------------------
    # X-axis interpretation
    # -------------------------------------------------------------------------

    ax.set_xlabel(
        (
            "← Reduces predicted risk"
            "          SHAP contribution          "
            "Increases predicted risk →"
        ),
        fontsize=10,
    )


    ax.set_ylabel(
        ""
    )


    # -------------------------------------------------------------------------
    # Clean visual style
    # -------------------------------------------------------------------------

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.spines[
        "left"
    ].set_visible(
        False
    )


    # Remove y-axis tick marks / little trailing dashes
    ax.tick_params(
        axis="y",
        length=0,
    )


    # -------------------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------------------

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.20,
    )

    ax.set_axisbelow(
        True
    )


    # -------------------------------------------------------------------------
    # Give contribution labels enough room
    # -------------------------------------------------------------------------

    current_left, current_right = (
        ax.get_xlim()
    )


    extra_margin = (
        max_abs_value * 0.08
    )


    ax.set_xlim(
        current_left - extra_margin,
        current_right + extra_margin,
    )


    # -------------------------------------------------------------------------
    # Final layout
    # -------------------------------------------------------------------------

    plt.tight_layout()


    return fig