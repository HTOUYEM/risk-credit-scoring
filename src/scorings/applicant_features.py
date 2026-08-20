from __future__ import annotations

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Raw inputs expected from the user / application
# -------------------------------------------------------------------

RAW_INPUT_VARIABLES = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "annual_inc",
    "installment",
    "int_rate",
    "inq_last_6mths",
    "open_acc",
    "total_acc",
    "earliest_cr_line",
    "issue_d",
    "emp_length",
    "term",
    "purpose",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "tot_coll_amt",
    "revol_util",
    "initial_list_status",
    "dti",
    "verification_status",
    "home_ownership",
    "addr_state",
    "revol_bal",
    "mths_since_last_record",
]


# -------------------------------------------------------------------
# Final pre-WoE variables required by the trained model
# -------------------------------------------------------------------

MODEL_INPUT_VARIABLES = [
    "int_rate",
    "loan_burden_interest",
    "installment_to_income_ratio",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "loan_to_income_ratio",
    "credit_inquiry_rate",
    "annual_inc",
    "inq_last_6mths",
    "term",
    "purpose",
    "tot_coll_amt",
    "credit_history_months",
    "revol_util",
    "initial_list_status",
    "dti",
    "verification_status",
    "home_ownership",
    "addr_state",
    "open_account_ratio",
    "investor_funding_ratio",
    "total_acc",
    "emp_length_years",
    "revol_bal",
    "installment",
    "mths_since_last_record",
]


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


NUMERIC_RAW_VARIABLES = [
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "annual_inc",
    "installment",
    "int_rate",
    "inq_last_6mths",
    "open_acc",
    "total_acc",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "tot_coll_amt",
    "revol_util",
    "dti",
    "revol_bal",
    "mths_since_last_record",
]


CATEGORICAL_RAW_VARIABLES = [
    "earliest_cr_line",
    "issue_d",
    "emp_length",
    "term",
    "purpose",
    "initial_list_status",
    "verification_status",
    "home_ownership",
    "addr_state",
]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def required_raw_fields() -> list[str]:
    """Return the raw fields required by the deployment pipeline."""
    return RAW_INPUT_VARIABLES.copy()


def required_model_fields() -> list[str]:
    """Return the final pre-WoE fields expected by the trained model."""
    return MODEL_INPUT_VARIABLES.copy()


def _validate_columns(
    data: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Applicant data is missing required fields: "
            f"{missing}"
        )


def _clean_numeric(
    series: pd.Series,
) -> pd.Series:
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .astype("float64")
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def _clean_categorical(
    series: pd.Series,
) -> pd.Series:
    cleaned = (
        series
        .astype("object")
        .where(
            series.notna(),
            np.nan,
        )
    )

    cleaned = cleaned.apply(
        lambda value: (
            value.strip()
            if isinstance(value, str)
            else value
        )
    )

    cleaned = cleaned.replace(
        {
            "": np.nan,
            "None": np.nan,
            "nan": np.nan,
        }
    )

    return cleaned


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Divide safely and return NaN when denominator is zero or missing.
    """
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = np.where(
        denominator.notna()
        & (denominator != 0),
        numerator / denominator,
        np.nan,
    )

    return pd.Series(
        result,
        index=numerator.index,
        dtype="float64",
    )


def _to_datetime(
    series: pd.Series,
) -> pd.Series:
    """
    Parse LendingClub-style date strings safely.
    """
    return pd.to_datetime(
        series,
        errors="coerce",
    )


# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------

def build_engineered_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recreate the engineered features used during model training.

    Expected raw inputs:
        loan_amnt
        funded_amnt
        funded_amnt_inv
        annual_inc
        installment
        int_rate
        inq_last_6mths
        open_acc
        total_acc
        earliest_cr_line
        issue_d
        emp_length
        ...

    Derived features:
        loan_to_income_ratio
        installment_to_income_ratio
        open_account_ratio
        credit_inquiry_rate
        investor_funding_ratio
        credit_history_months
        emp_length_years
        loan_burden_interest
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Applicant data is empty."
        )

    _validate_columns(
        data,
        RAW_INPUT_VARIABLES,
    )

    df = data.copy()

    # Clean raw numeric variables.
    for column in NUMERIC_RAW_VARIABLES:
        df[column] = _clean_numeric(
            df[column]
        )

    # Clean raw categorical variables.
    for column in CATEGORICAL_RAW_VARIABLES:
        df[column] = _clean_categorical(
            df[column]
        )

    # ---------------------------------------------------------------
    # 1. Loan-to-income ratio
    # ---------------------------------------------------------------
    df["loan_to_income_ratio"] = _safe_divide(
        df["loan_amnt"],
        df["annual_inc"],
    )

    # ---------------------------------------------------------------
    # 2. Installment-to-income ratio
    # Annualized installment burden over annual income
    # ---------------------------------------------------------------
    df[
        "installment_to_income_ratio"
    ] = _safe_divide(
        df["installment"] * 12,
        df["annual_inc"],
    )

    # ---------------------------------------------------------------
    # 3. Open-account ratio
    # ---------------------------------------------------------------
    df["open_account_ratio"] = _safe_divide(
        df["open_acc"],
        df["total_acc"],
    )

    # ---------------------------------------------------------------
    # 4. Credit-history duration in months
    # ---------------------------------------------------------------
    earliest = _to_datetime(
        df["earliest_cr_line"]
    )

    issue = _to_datetime(
        df["issue_d"]
    )

    df["credit_history_months"] = (
        (issue.dt.year - earliest.dt.year) * 12
        + (
            issue.dt.month
            - earliest.dt.month
        )
    ).astype("float64")

    # Negative values indicate invalid date ordering.
    df.loc[
        df["credit_history_months"] < 0,
        "credit_history_months",
    ] = np.nan

    # ---------------------------------------------------------------
    # 5. Credit inquiry rate
    # inquiries per year of credit history
    # ---------------------------------------------------------------
    credit_history_years = (
        df["credit_history_months"] / 12
    )

    df["credit_inquiry_rate"] = _safe_divide(
        df["inq_last_6mths"],
        credit_history_years,
    )

    # ---------------------------------------------------------------
    # 6. Investor funding ratio
    # ---------------------------------------------------------------
    df["investor_funding_ratio"] = _safe_divide(
        df["funded_amnt_inv"],
        df["funded_amnt"],
    )

    # ---------------------------------------------------------------
    # 7. Employment length in years
    # ---------------------------------------------------------------
    df["emp_length_years"] = (
        df["emp_length"]
        .map(EMP_LENGTH_MAPPING)
        .astype("float64")
    )

    # ---------------------------------------------------------------
    # 8. Loan burden adjusted by interest rate
    # ---------------------------------------------------------------
    df["loan_burden_interest"] = (
        df["loan_to_income_ratio"]
        * df["int_rate"]
    )

    # ---------------------------------------------------------------
    # 9. Missing months since last record
    # Training convention: no record -> -1
    # ---------------------------------------------------------------
    df["mths_since_last_record"] = (
        df["mths_since_last_record"]
        .fillna(-1)
        .astype("float64")
    )

    # ---------------------------------------------------------------
    # Final model-ready pre-WoE dataset
    # ---------------------------------------------------------------
    result = df[
        MODEL_INPUT_VARIABLES
    ].copy()

    return result


# -------------------------------------------------------------------
# Deployment entry points
# -------------------------------------------------------------------

def prepare_applicant(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert raw applicant information into the exact pre-WoE feature set
    required by the trained PD model.
    """
    return build_engineered_features(
        data
    )


def applicant_from_dict(
    applicant: dict,
) -> pd.DataFrame:
    """
    Convert one applicant dictionary into a one-row model-ready DataFrame.
    """
    if not isinstance(applicant, dict):
        raise TypeError(
            "applicant must be a dictionary."
        )

    raw_df = pd.DataFrame(
        [applicant]
    )

    return prepare_applicant(
        raw_df
    )


def validate_raw_applicant(
    data: pd.DataFrame,
) -> dict:
    """
    Validate raw applicant input before feature engineering.
    """

    missing_fields = [
        column
        for column in RAW_INPUT_VARIABLES
        if column not in data.columns
    ]

    null_fields = []

    if not missing_fields:
        null_fields = [
            column
            for column in RAW_INPUT_VARIABLES
            if data[column].isna().any()
        ]

    return {
        "valid": len(missing_fields) == 0,
        "missing_fields": missing_fields,
        "null_fields": null_fields,
    }


def validate_model_input(
    data: pd.DataFrame,
) -> dict:
    """
    Validate the engineered pre-WoE dataset.
    """

    missing_fields = [
        column
        for column in MODEL_INPUT_VARIABLES
        if column not in data.columns
    ]

    null_fields = []

    if not missing_fields:
        null_fields = [
            column
            for column in MODEL_INPUT_VARIABLES
            if data[column].isna().any()
        ]

    return {
        "valid": len(missing_fields) == 0,
        "missing_fields": missing_fields,
        "null_fields": null_fields,
    }
