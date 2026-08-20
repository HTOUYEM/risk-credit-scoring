# =============================================================================
# PD Inference Utilities
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

from joblib import load
from src.scoring import build_scoring_output


# =============================================================================
# Project Paths
# =============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

PD_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pd"
)

DEPLOYMENT_ARTIFACTS_PATH = (
    PD_DATA_DIR
    / "pd_deployment_artifacts.joblib"
)

PD_PREPROCESSOR_PATH = (
    PD_DATA_DIR
    / "pd_preprocessor.joblib"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "pd"
)

FINAL_PD_MODEL_PATH = (
    MODEL_DIR
    / "final_pd_model.joblib"
)


# =============================================================================
# Load Deployment Artifacts
# =============================================================================

deployment_artifacts = load(
    DEPLOYMENT_ARTIFACTS_PATH
)

pd_preprocessor = load(
    PD_PREPROCESSOR_PATH
)


PURPOSE_MAPPING = (
    deployment_artifacts["purpose_mapping"]
)

KNOWN_STATES = set(
    deployment_artifacts["known_states"]
)

RARE_STATES = set(
    deployment_artifacts["rare_states"]
)

NUMERIC_MEDIANS = (
    deployment_artifacts["numeric_medians"]
)

CATEGORICAL_MODES = (
    deployment_artifacts["categorical_modes"]
)

DATETIME_MEDIANS = (
    deployment_artifacts["datetime_medians"]
)

NATIVE_COLUMNS = (
    deployment_artifacts["native_columns"]
)

final_pd_model = load(
    FINAL_PD_MODEL_PATH
)

# =============================================================================
# Employment Length Mapping
# =============================================================================

EMP_LENGTH_MAPPING = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}


# =============================================================================
# Special Missingness Variables
# =============================================================================

CREDIT_HISTORY_MISSING_COLUMNS = [
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
]


# =============================================================================
# Development-Sample Imputation
# =============================================================================

def apply_development_imputation(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply imputation rules learned from the PD development sample.

    Numerical variables are imputed using development-sample medians,
    categorical variables using development-sample modes, and datetime
    variables using development-sample median dates.

    Variables with dedicated missingness indicators are excluded from
    generic numerical imputation because their missing values are handled
    separately through sentinel values and explicit indicators.

    Parameters
    ----------
    data : pd.DataFrame
        Raw borrower and loan information.

    Returns
    -------
    pd.DataFrame
        Data with development-sample imputation rules applied.
    """

    data = data.copy()

    # -------------------------------------------------------------------------
    # Numerical variables
    # -------------------------------------------------------------------------

    for column, median_value in NUMERIC_MEDIANS.items():

        if (
            column in data.columns
            and column not in CREDIT_HISTORY_MISSING_COLUMNS
        ):
            data[column] = (
                data[column]
                .fillna(median_value)
            )

    # -------------------------------------------------------------------------
    # Categorical variables
    # -------------------------------------------------------------------------

    for column, mode_value in CATEGORICAL_MODES.items():

        if (
            column in data.columns
            and column != "emp_length"
        ):
            data[column] = (
                data[column]
                .fillna(mode_value)
            )

    # -------------------------------------------------------------------------
    # Datetime variables
    # -------------------------------------------------------------------------

    for column, median_date in DATETIME_MEDIANS.items():

        if column in data.columns:

            data[column] = pd.to_datetime(
                data[column],
                errors="coerce",
            )

            data[column] = (
                data[column]
                .fillna(median_date)
            )

    return data


# =============================================================================
# Native Feature Construction
# =============================================================================

def build_native_features(
    raw_input: dict,
) -> pd.DataFrame:
    """
    Build the 40 native PD model features from raw borrower inputs.

    The function reproduces the feature-engineering logic established
    during PD dataset construction.

    Parameters
    ----------
    raw_input : dict
        Raw borrower and loan information collected by the application.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame containing the 40 native features expected
        by the PD preprocessing pipeline.
    """

    data = pd.DataFrame(
        [raw_input]
    )

    # =========================================================================
    # Preserve Original Missingness Information
    # =========================================================================

    # -------------------------------------------------------------------------
    # Employment length missingness
    # -------------------------------------------------------------------------

    data["emp_length_missing"] = (
        data["emp_length"]
        .isna()
        .astype("int8")
    )

    # -------------------------------------------------------------------------
    # Employment title missingness
    # -------------------------------------------------------------------------

    data["emp_title_missing"] = (
        data["emp_title"]
        .isna()
        .astype("int8")
    )

    # -------------------------------------------------------------------------
    # Credit-history missingness indicators
    # -------------------------------------------------------------------------

    for column in CREDIT_HISTORY_MISSING_COLUMNS:

        missing_indicator = (
            f"{column}_is_missing"
        )

        data[missing_indicator] = (
            data[column]
            .isna()
            .astype("int8")
        )


    # =========================================================================
    # Apply Development-Sample Imputation
    # =========================================================================

    data = apply_development_imputation(
        data
    )


    # =========================================================================
    # Employment Length
    # =========================================================================

    data["emp_length_years"] = (
        data["emp_length"]
        .map(EMP_LENGTH_MAPPING)
        .fillna(-1)
        .astype("int8")
    )


    # =========================================================================
    # Credit-History Missing Values
    # =========================================================================

    for column in CREDIT_HISTORY_MISSING_COLUMNS:

        data[column] = (
            data[column]
            .fillna(-1)
            .astype(float)
        )


    # =========================================================================
    # Credit History Length
    # =========================================================================

    data["issue_d"] = pd.to_datetime(
        data["issue_d"],
        errors="coerce",
    )

    data["earliest_cr_line"] = pd.to_datetime(
        data["earliest_cr_line"],
        errors="coerce",
    )

    if data["issue_d"].isna().any():
        raise ValueError(
            "issue_d must contain a valid loan issue date."
        )

    if data["earliest_cr_line"].isna().any():
        raise ValueError(
            "earliest_cr_line could not be resolved "
            "after development-sample imputation."
        )

    data["credit_history_months"] = (
        (
            data["issue_d"].dt.year
            - data["earliest_cr_line"].dt.year
        ) * 12
        + (
            data["issue_d"].dt.month
            - data["earliest_cr_line"].dt.month
        )
    ).astype("int16")

    if (
        data["credit_history_months"] < 0
    ).any():
        raise ValueError(
            "earliest_cr_line cannot be later than issue_d."
        )


    # =========================================================================
    # Purpose Grouping
    # =========================================================================

    data["purpose_grouped"] = (
        data["purpose"]
        .astype("string")
        .map(PURPOSE_MAPPING)
        .fillna("other_or_unknown")
        .astype("category")
    )


    # =========================================================================
    # State Grouping
    # =========================================================================

    states = (
        data["addr_state"]
        .astype("string")
    )

    data["addr_state_grouped"] = (
        states.where(
            states.isin(KNOWN_STATES)
            & ~states.isin(RARE_STATES),
            "OTHER_STATE",
        )
        .fillna("OTHER_STATE")
        .astype("category")
    )


    # =========================================================================
    # Loan and Funding Ratios
    # =========================================================================

    if (
        data["loan_amnt"] <= 0
    ).any():
        raise ValueError(
            "loan_amnt must be greater than zero."
        )

    data["funding_ratio"] = (
        data["funded_amnt"]
        / data["loan_amnt"]
    ).astype("float32")

    data["investor_loan_ratio"] = (
        data["funded_amnt_inv"]
        / data["loan_amnt"]
    ).astype("float32")


    # =========================================================================
    # Credit-History Ratios
    # =========================================================================

    total_acc_safe = (
        data["total_acc"]
        .replace(0, np.nan)
    )

    data["open_account_ratio"] = (
        data["open_acc"]
        / total_acc_safe
    )

    data["delinquency_rate"] = (
        data["delinq_2yrs"]
        / total_acc_safe
    )

    credit_history_safe = (
        data["credit_history_months"]
        .replace(0, np.nan)
    )

    data["credit_inquiry_rate"] = (
        data["inq_last_6mths"]
        / credit_history_safe
    )


    # =========================================================================
    # Income and Loan-Burden Ratios
    # =========================================================================

    income = (
        data["annual_inc"]
        .replace(0, np.nan)
    )

    data["installment_to_income_ratio"] = (
        data["installment"] * 12
        / income
    )

    loan_to_income_ratio = (
        data["loan_amnt"]
        / income
    )

    data["loan_burden_interest"] = (
        loan_to_income_ratio
        * data["int_rate"]
    )


    # =========================================================================
    # Final Native Schema Alignment
    # =========================================================================

    missing_columns = [
        column
        for column in NATIVE_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing native model features: "
            f"{missing_columns}"
        )

    native_data = (
        data[NATIVE_COLUMNS]
        .copy()
    )

    if native_data.isna().any().any():

        missing_native_values = (
            native_data
            .columns[
                native_data.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "Missing values remain in native model features: "
            f"{missing_native_values}"
        )

    return native_data


# =============================================================================
# Encoded Feature Construction
# =============================================================================

def build_encoded_features(
    native_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform native PD model features into the encoded feature space
    used during model training.

    Parameters
    ----------
    native_data : pd.DataFrame
        Native PD model features aligned with the 40-feature schema.

    Returns
    -------
    pd.DataFrame
        Encoded feature matrix expected by the final PD model.
    """

    encoded_data = (
        pd_preprocessor
        .transform(native_data)
    )

    return encoded_data


# =============================================================================
# PD Prediction
# =============================================================================

def predict_pd(
    raw_input: dict,
) -> float:
    """
    Predict the Probability of Default for a new loan application.

    Parameters
    ----------
    raw_input : dict
        Raw borrower and loan information collected by the application.

    Returns
    -------
    float
        Predicted Probability of Default between 0 and 1.
    """

    native_data = build_native_features(
        raw_input
    )

    encoded_data = build_encoded_features(
        native_data
    )

    pd_probability = (
        final_pd_model
        .predict_proba(encoded_data)[:, 1]
        .item()
    )

    return float(pd_probability)

# =============================================================================
# Complete Credit Risk Prediction
# =============================================================================

def predict_credit_risk(
    raw_input: dict,
) -> dict:
    """
    Generate the complete credit-risk output for a new loan application.

    Parameters
    ----------
    raw_input : dict
        Raw borrower and loan information collected by the application.

    Returns
    -------
    dict
        Predicted PD, credit score, risk grade and risk level.
    """

    predicted_pd = predict_pd(
        raw_input
    )

    scoring_output = build_scoring_output(
        predicted_pd=[predicted_pd],
        index=[0],
    )

    result = scoring_output.iloc[0]

    return {
        "predicted_pd": float(
            result["Predicted PD"]
        ),
        "credit_score": int(
            result["Credit Score"]
        ),
        "risk_grade": str(
            result["Risk Grade"]
        ),
        "risk_level": str(
            result["Risk Level"]
        ),
    }