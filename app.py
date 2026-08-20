# =============================================================================
# Credit Risk Scoring Application
# =============================================================================

from datetime import date
from html import escape

import streamlit as st

from src.pd_inference import (
    predict_credit_risk,
    final_pd_model,
    build_native_features,
    build_encoded_features,
)

from src.visualization import (
    plot_risk_gauge,
)

from src.explainability import (
    explain_borrower,
    build_local_explanation_table,
    get_top_risk_drivers,
    plot_local_risk_drivers,
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="🛡️",
    layout="wide",
)


# =============================================================================
# Custom Styling
# =============================================================================

st.html(
    """
    <style>

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* =====================================================================
       HEADER
       ===================================================================== */

    .credit-risk-header {
        background: linear-gradient(
            135deg,
            #0A2F52 0%,
            #0B5270 55%,
            #0F766E 100%
        );

        border-radius: 22px;
        padding: 2.4rem 2.6rem;
        margin-bottom: 1.4rem;

        box-shadow:
            0 14px 32px rgba(15, 53, 88, 0.16);

        position: relative;
        overflow: hidden;
    }

    .credit-risk-header::after {
        content: "";
        position: absolute;

        width: 360px;
        height: 360px;

        right: -90px;
        top: -160px;

        border-radius: 50%;

        background: radial-gradient(
            circle,
            rgba(94, 234, 212, 0.22) 0%,
            rgba(94, 234, 212, 0.04) 65%,
            transparent 72%
        );
    }

    .credit-risk-eyebrow {
        color: #5EEAD4;
        font-size: 0.88rem;
        font-weight: 900;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .credit-risk-title {
        color: #FFFFFF;
        font-size: 3.25rem;
        line-height: 1.05;
        font-weight: 900;
        margin-bottom: 0.85rem;
    }

    .credit-risk-subtitle {
        color: #E6F1F7;
        font-size: 1.08rem;
        line-height: 1.65;
        max-width: 900px;
    }


    /* =====================================================================
       USER GUIDE
       ===================================================================== */

    .application-guide {
        background: linear-gradient(
            90deg,
            #ECFDF8 0%,
            #F7FCFD 100%
        );

        border: 1px solid #BCE8DD;
        border-left: 6px solid #0F766E;
        border-radius: 15px;

        padding: 1.15rem 1.4rem;
        margin-bottom: 1.8rem;

        color: #334E68;
        font-size: 0.97rem;
        line-height: 1.65;
    }

    .application-guide-title {
        color: #0F5F58;
        font-size: 1rem;
        font-weight: 900;
        margin-bottom: 0.35rem;
    }


    /* =====================================================================
       FEATURE LABELS
       ===================================================================== */

    div[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] p {
        font-weight: 850 !important;
        font-size: 0.98rem !important;
        color: #102A43 !important;
    }


    /* =====================================================================
       TITLES
       ===================================================================== */

    h1,
    h2,
    h3 {
        color: #173F5F;
    }

    h2,
    h3 {
        font-weight: 850 !important;
    }


    /* =====================================================================
       SECTION CARDS
       ===================================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 17px;

        box-shadow:
            0 3px 10px rgba(15, 53, 88, 0.035);
    }


    /* =====================================================================
       INPUTS
       ===================================================================== */

    div[data-baseweb="input"] {
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
    }

    [data-testid="stTooltipIcon"] {
        color: #0F766E !important;
    }


    /* =====================================================================
       SUBMIT BUTTON
       ===================================================================== */

    div[data-testid="stFormSubmitButton"] {
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }

    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        min-height: 4.25rem;

        border: none !important;
        border-radius: 15px !important;

        background: linear-gradient(
            90deg,
            #05603A 0%,
            #087F5B 50%,
            #059669 100%
        ) !important;

        color: #FFFFFF !important;

        font-size: 1.35rem !important;
        font-weight: 900 !important;

        letter-spacing: 0.05em;

        box-shadow:
            0 8px 22px rgba(5, 150, 105, 0.28);

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease,
            filter 0.15s ease;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 12px 28px rgba(5, 150, 105, 0.36);

        filter: brightness(1.05);
    }

    div[data-testid="stFormSubmitButton"] > button:active {
        transform: translateY(0);
    }

    div[data-testid="stFormSubmitButton"] > button p {
        color: #FFFFFF !important;
        font-size: 1.35rem !important;
        font-weight: 900 !important;
    }


    /* =====================================================================
       RESULTS HEADER
       ===================================================================== */

    .result-header {
        background: linear-gradient(
            90deg,
            #EFFAF7 0%,
            #F5FAFD 100%
        );

        border: 1px solid #C9E8E0;
        border-left: 6px solid #0F766E;
        border-radius: 14px;

        padding: 1.15rem 1.35rem;
        margin-bottom: 1.25rem;
    }

    .result-header-title {
        color: #173F5F;
        font-size: 1.9rem;
        font-weight: 900;
        margin-bottom: 0.25rem;
    }

    .result-header-text {
        color: #627D98;
        font-size: 0.96rem;
    }


    /* =====================================================================
       RESULT CARDS
       ===================================================================== */

    .risk-results-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);

        gap: 16px;

        margin-top: 1.2rem;
        margin-bottom: 2rem;
    }

    .risk-result-card {
        background: #FFFFFF;

        border: 1px solid #D9E5EC;
        border-top: 5px solid #0F766E;

        border-radius: 16px;

        padding: 1.3rem 1.35rem;

        min-height: 145px;

        box-shadow:
            0 5px 16px rgba(15, 53, 88, 0.07);
    }

    .risk-result-label {
        color: #627D98;

        font-size: 0.80rem;
        font-weight: 850;

        letter-spacing: 0.06em;
        text-transform: uppercase;

        margin-bottom: 0.55rem;
    }

    .risk-result-value {
        color: #173F5F;

        font-size: 2.35rem;
        font-weight: 900;

        line-height: 1.1;

        margin-bottom: 0.5rem;
    }

    .risk-result-text {
        font-size: 1.55rem;
    }

    .risk-result-description {
        color: #829AB1;
        font-size: 0.82rem;
        line-height: 1.4;
    }


    /* =====================================================================
       RESPONSIVE
       ===================================================================== */

    @media (max-width: 950px) {

        .risk-results-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    @media (max-width: 600px) {

        .risk-results-grid {
            grid-template-columns: 1fr;
        }

        .credit-risk-title {
            font-size: 2.4rem;
        }
    }
/* =====================================================================
   SHAP EXPLAINABILITY
   ===================================================================== */

.explainability-header {
    background: linear-gradient(
        90deg,
        #EFFAF7 0%,
        #F5FAFD 100%
    );

    border: 1px solid #C9E8E0;
    border-left: 6px solid #0F766E;

    border-radius: 14px;

    padding: 1.2rem 1.4rem;

    margin-top: 2rem;
    margin-bottom: 1.2rem;
}

.explainability-title {
    color: #173F5F;

    font-size: 1.9rem;
    font-weight: 900;

    margin-bottom: 0.35rem;
}

.explainability-text {
    color: #627D98;

    font-size: 0.96rem;
    line-height: 1.65;
}

.explainability-guide {
    background: linear-gradient(
        90deg,
        #ECFDF8 0%,
        #F7FCFD 100%
    );

    border: 1px solid #BCE8DD;
    border-left: 6px solid #0F766E;

    border-radius: 15px;

    padding: 1.15rem 1.4rem;

    margin-bottom: 1.4rem;

    color: #334E68;

    font-size: 0.95rem;
    line-height: 1.65;
}

.explainability-guide-title {
    color: #0F5F58;

    font-size: 1rem;
    font-weight: 900;

    margin-bottom: 0.35rem;
}


    /* =====================================================================
       SHAP DRIVER TABLES
       ===================================================================== */

    .driver-section {
        margin-top: 1.4rem;
        margin-bottom: 1.8rem;
    }

    .driver-section-title {
        color: #173F5F;
        font-size: 1.35rem;
        font-weight: 900;
        margin-bottom: 0.3rem;
    }

    .driver-section-subtitle {
        color: #627D98;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }

    .driver-table-wrapper {
        background: #FFFFFF;
        border: 1px solid #D9E5EC;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 5px 16px rgba(15, 53, 88, 0.06);
        margin-bottom: 1.25rem;
    }

    .driver-table-heading {
        padding: 0.95rem 1.1rem;
        font-size: 0.95rem;
        font-weight: 900;
        letter-spacing: 0.035em;
        text-transform: uppercase;
    }

    .driver-table-heading.increasing {
        background: #FFF3F2;
        color: #A83D38;
        border-bottom: 1px solid #F3D3D0;
    }

    .driver-table-heading.reducing {
        background: #EEF7FB;
        color: #256A8C;
        border-bottom: 1px solid #D3E8F2;
    }

    .driver-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }

    .driver-table th {
        background: #F8FAFC;
        color: #627D98;
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.055em;
        text-transform: uppercase;
        text-align: left;
        padding: 0.8rem 1rem;
        border-bottom: 1px solid #E6EDF2;
    }

    .driver-table td {
        color: #334E68;
        font-size: 0.92rem;
        padding: 0.85rem 1rem;
        border-bottom: 1px solid #EEF2F5;
        vertical-align: middle;
    }

    .driver-table tr:last-child td {
        border-bottom: none;
    }

    .driver-table td.factor-cell {
        color: #173F5F;
        font-weight: 800;
        width: 48%;
    }

    .driver-table td.value-cell {
        font-weight: 650;
        width: 30%;
    }

    .driver-table td.shap-cell {
        font-weight: 900;
        text-align: right;
        width: 22%;
        font-variant-numeric: tabular-nums;
    }

    .shap-positive {
        color: #C74E48;
    }

    .shap-negative {
        color: #2F7EA8;
    }

    .driver-empty {
        color: #829AB1;
        font-size: 0.9rem;
        padding: 1rem;
        font-style: italic;
    }

    @media (max-width: 700px) {
        .driver-table th,
        .driver-table td {
            padding: 0.7rem 0.65rem;
            font-size: 0.82rem;
        }
    }

    </style>
    """
)


# =============================================================================
# Application Header
# =============================================================================

st.html(
    """
    <div class="credit-risk-header">

        <div class="credit-risk-eyebrow">
            CREDIT RISK ANALYTICS
        </div>

        <div class="credit-risk-title">
            Credit Risk Scoring
        </div>

        <div class="credit-risk-subtitle">
            Assess a borrower's credit risk using loan, financial and
            credit-history information. The application estimates the
            Probability of Default (PD) and translates it into an
            interpretable credit score, model risk grade and overall
            risk level.
        </div>

    </div>
    """
)


# =============================================================================
# User Guide
# =============================================================================

st.html(
    """
    <div class="application-guide">

        <div class="application-guide-title">
            HOW TO USE THIS APPLICATION
        </div>

        <div>
            Complete the borrower and loan information in the sections below,
            then click <strong>Assess Credit Risk</strong>.
            Use the <strong>?</strong> icon next to each feature to view
            its definition and understand the information required.
        </div>

    </div>
    """
)


# =============================================================================
# Borrower Input Form
# =============================================================================

st.header(
    "Loan Application"
)

st.caption(
    "Complete the six sections below to generate the borrower-level "
    "credit-risk assessment."
)


with st.form(
    "credit_risk_form"
):

    # =========================================================================
    # 1. Loan Information
    # =========================================================================

    with st.container(border=True):

        st.subheader(
            "1. Loan Information"
        )

        st.caption(
            "Main characteristics and funding information "
            "for the requested loan."
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            loan_amnt = st.number_input(
                "Loan Amount ($)",
                min_value=0.0,
                value=35000.0,
                step=500.0,
                help=(
                    "Listed amount of the loan applied for by the borrower. "
                    "If the requested amount is reduced before origination, "
                    "the updated loan amount is reflected here."
                ),
            )

            term = st.selectbox(
                "Loan Term",
                options=[
                    36,
                    60,
                ],
                index=1,
                format_func=lambda value: f"{value} months",
                help=(
                    "Number of scheduled monthly payments on the loan. "
                    "Loans in this dataset have terms of either "
                    "36 or 60 months."
                ),
            )


        with col2:

            int_rate = st.number_input(
                "Interest Rate (%)",
                min_value=0.0,
                value=28.0,
                step=0.1,
                help=(
                    "Interest rate assigned to the loan, "
                    "expressed as an annual percentage rate."
                ),
            )

            installment = st.number_input(
                "Monthly Installment ($)",
                min_value=0.0,
                value=1100.0,
                step=10.0,
                help=(
                    "Monthly payment owed by the borrower "
                    "if the loan originates."
                ),
            )


        with col3:

            sub_grade = st.selectbox(
                "Lending Club Loan Sub-Grade",
                options=[
                    f"{grade}{number}"
                    for grade in "ABCDEFG"
                    for number in range(1, 6)
                ],
                index=34,
                help=(
                    "Loan risk sub-grade assigned by Lending Club. "
                    "It provides a more detailed classification within "
                    "the broader Lending Club loan grade. "
                    "For example, B3 belongs to Grade B."
                ),
            )

            funded_amnt = st.number_input(
                "Funded Amount ($)",
                min_value=0.0,
                value=35000.0,
                step=500.0,
                help=(
                    "Total amount committed to the loan "
                    "at the time of funding."
                ),
            )

            funded_amnt_inv = st.number_input(
                "Investor Funded Amount ($)",
                min_value=0.0,
                value=35000.0,
                step=500.0,
                help=(
                    "Total amount committed by investors "
                    "to the loan."
                ),
            )


    # =========================================================================
    # 2. Borrower & Employment Information
    # =========================================================================

    with st.container(border=True):

        st.subheader(
            "2. Borrower & Employment Information"
        )

        st.caption(
            "Income, employment and housing characteristics "
            "reported for the borrower."
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            annual_inc = st.number_input(
                "Annual Income ($)",
                min_value=0.0,
                value=30000.0,
                step=1000.0,
                help=(
                    "Self-reported annual income provided "
                    "by the borrower during registration."
                ),
            )

            emp_length = st.selectbox(
                "Employment Length",
                options=[
                    "< 1 year",
                    "1 year",
                    "2 years",
                    "3 years",
                    "4 years",
                    "5 years",
                    "6 years",
                    "7 years",
                    "8 years",
                    "9 years",
                    "10+ years",
                    None,
                ],
                index=0,
                format_func=lambda value: (
                    "Not provided"
                    if value is None
                    else value
                ),
                help=(
                    "Length of the borrower's employment. "
                    "Select 'Not provided' when employment length "
                    "is unavailable. Missing employment information "
                    "is explicitly captured by the model."
                ),
            )


        with col2:

            emp_title = st.text_input(
                "Employment Title",
                value="",
                placeholder="Not provided",
                help=(
                    "Job title supplied by the borrower when applying "
                    "for the loan. The exact title is not directly used "
                    "by the final PD model, but whether the title was "
                    "provided is retained as information."
                ),
            )

            home_ownership = st.selectbox(
                "Home Ownership",
                options=[
                    "MORTGAGE",
                    "OWN",
                    "RENT",
                    "OTHER",
                ],
                index=2,
                help=(
                    "Home-ownership status provided by the borrower "
                    "or obtained from the credit report. "
                    "Possible values are MORTGAGE, OWN, RENT and OTHER."
                ),
            )


        with col3:

            verification_status = st.selectbox(
                "Income Verification Status",
                options=[
                    "Not Verified",
                    "Source Verified",
                    "Verified",
                ],
                index=0,
                help=(
                    "Indicates how the reported income was validated. "
                    "Income may be Not Verified, Source Verified "
                    "or Verified."
                ),
            )


    # =========================================================================
    # 3. Credit Profile & Revolving Behaviour
    # =========================================================================

    with st.container(border=True):

        st.subheader(
            "3. Credit Profile & Revolving Behaviour"
        )

        st.caption(
            "Current debt burden, credit accounts, inquiries "
            "and revolving-credit usage."
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            dti = st.number_input(
                "Debt-to-Income Ratio (%)",
                value=45.0,
                step=0.1,
                help=(
                    "Ratio calculated from the borrower's total monthly "
                    "payments on existing debt obligations, excluding "
                    "mortgage payments and the requested Lending Club loan, "
                    "divided by self-reported monthly income."
                ),
            )

            delinq_2yrs = st.number_input(
                "Delinquencies in the Last 2 Years",
                min_value=0,
                value=6,
                step=1,
                help=(
                    "Number of delinquency incidents that were "
                    "30 days or more past due during the previous "
                    "two years."
                ),
            )

            inq_last_6mths = st.number_input(
                "Credit Inquiries in the Last 6 Months",
                min_value=0,
                value=8,
                step=1,
                help=(
                    "Number of credit inquiries recorded during "
                    "the previous six months, excluding auto-loan "
                    "and mortgage inquiries."
                ),
            )


        with col2:

            open_acc = st.number_input(
                "Open Credit Accounts",
                min_value=0,
                value=12,
                step=1,
                help=(
                    "Number of open credit lines currently appearing "
                    "in the borrower's credit file."
                ),
            )

            total_acc = st.number_input(
                "Total Credit Accounts",
                min_value=0,
                value=18,
                step=1,
                help=(
                    "Total number of credit lines appearing "
                    "in the borrower's credit file."
                ),
            )

            pub_rec = st.number_input(
                "Derogatory Public Records",
                min_value=0,
                value=3,
                step=1,
                help=(
                    "Number of derogatory public records "
                    "reported in the borrower's credit file."
                ),
            )


        with col3:

            revol_bal = st.number_input(
                "Revolving Credit Balance ($)",
                min_value=0.0,
                value=28000.0,
                step=500.0,
                help=(
                    "Total outstanding balance across the borrower's "
                    "revolving credit accounts."
                ),
            )

            revol_util = st.number_input(
                "Revolving Credit Utilization (%)",
                value=110.0,
                step=1.0,
                help=(
                    "Amount of revolving credit currently being used "
                    "relative to all available revolving credit. "
                    "Values above 100% indicate that reported revolving "
                    "balances exceed the available revolving credit limit."
                ),
            )


    # =========================================================================
    # 4. Credit History & Adverse Events
    # =========================================================================

    with st.container(border=True):

        st.subheader(
            "4. Credit History & Adverse Events"
        )

        st.caption(
            "Age of the borrower's credit history and time elapsed "
            "since previous adverse credit events."
        )

        col1, col2 = st.columns(2)


        with col1:

            earliest_cr_line = st.date_input(
                "Earliest Credit Line",
                value=date(2011, 1, 1),
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                format="DD/MM/YYYY",
                help=(
                    "Date corresponding to the borrower's earliest "
                    "reported credit line. It is used to derive "
                    "the length of the borrower's credit history."
                ),
            )

            mths_since_last_delinq = st.number_input(
                "Months Since Last Delinquency",
                min_value=0.0,
                value=1.0,
                step=1.0,
                help=(
                    "Number of months since the borrower's most recent "
                    "delinquency. A low value indicates that the "
                    "delinquency occurred recently."
                ),
            )


        with col2:

            mths_since_last_record = st.number_input(
                "Months Since Last Public Record",
                min_value=0.0,
                value=2.0,
                step=1.0,
                help=(
                    "Number of months since the borrower's "
                    "most recent public record."
                ),
            )

            mths_since_last_major_derog = st.number_input(
                "Months Since Last Major Derogatory Event",
                min_value=0.0,
                value=1.0,
                step=1.0,
                help=(
                    "Number of months since the borrower's most recent "
                    "major derogatory credit event. A low value indicates "
                    "that the event occurred recently."
                ),
            )


    # =========================================================================
    # 5. Loan Purpose, Geography & Listing Information
    # =========================================================================

    with st.container(border=True):

        st.subheader(
            "5. Loan Purpose, Geography & Listing Information"
        )

        st.caption(
            "Borrower-reported loan purpose, location "
            "and loan-origination information."
        )

        col1, col2 = st.columns(2)


        with col1:

            purpose = st.selectbox(
                "Loan Purpose",
                options=[
                    "credit_card",
                    "car",
                    "major_purchase",
                    "home_improvement",
                    "debt_consolidation",
                    "wedding",
                    "vacation",
                    "other",
                    "medical",
                    "house",
                    "moving",
                    "renewable_energy",
                    "educational",
                    "small_business",
                ],
                index=13,
                format_func=lambda value: (
                    value
                    .replace("_", " ")
                    .title()
                ),
                help=(
                    "Category provided by the borrower describing "
                    "the purpose of the loan request."
                ),
            )

            addr_state = st.selectbox(
                "Borrower State",
                options=[
                    "AL", "AK", "AZ", "AR", "CA",
                    "CO", "CT", "DE", "FL", "GA",
                    "HI", "ID", "IL", "IN", "IA",
                    "KS", "KY", "LA", "ME", "MD",
                    "MA", "MI", "MN", "MS", "MO",
                    "MT", "NE", "NV", "NH", "NJ",
                    "NM", "NY", "NC", "ND", "OH",
                    "OK", "OR", "PA", "RI", "SC",
                    "SD", "TN", "TX", "UT", "VT",
                    "VA", "WA", "WV", "WI", "WY",
                    "DC",
                ],
                index=4,
                help=(
                    "U.S. state provided by the borrower "
                    "in the loan application."
                ),
            )


        with col2:

            initial_list_status = st.selectbox(
                "Initial Listing Status",
                options=[
                    "f",
                    "w",
                ],
                index=0,
                format_func=lambda value: (
                    "Fractional (F)"
                    if value == "f"
                    else "Whole (W)"
                ),
                help=(
                    "Initial listing status assigned to the loan. "
                    "Possible values are Fractional (F) and Whole (W)."
                ),
            )

            issue_d = st.date_input(
                "Loan Issue Date",
                value=date(2014, 12, 1),
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                format="DD/MM/YYYY",
                help=(
                    "Date corresponding to the month "
                    "in which the loan was funded."
                ),
            )


    # =========================================================================
    # 6. Collections, Delinquency & Total Credit Exposure
    # =========================================================================

    with st.container(border=True):

        st.subheader(
            "6. Collections, Delinquency & Total Credit Exposure"
        )

        st.caption(
            "Recent collection activity, current delinquency "
            "and overall credit exposure."
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            collections_12_mths_ex_med = st.number_input(
                "Collections in the Last 12 Months",
                min_value=0,
                value=4,
                step=1,
                help=(
                    "Number of collection events reported during "
                    "the previous 12 months, excluding medical collections."
                ),
            )

            acc_now_delinq = st.number_input(
                "Accounts Currently Delinquent",
                min_value=0,
                value=3,
                step=1,
                help=(
                    "Number of accounts on which the borrower "
                    "is currently delinquent."
                ),
            )


        with col2:

            tot_coll_amt = st.number_input(
                "Total Collection Amount Ever Owed ($)",
                min_value=0.0,
                value=8000.0,
                step=100.0,
                help=(
                    "Total amount ever owed in collections "
                    "according to the borrower's credit information."
                ),
            )

            tot_cur_bal = st.number_input(
                "Total Current Balance ($)",
                min_value=0.0,
                value=120000.0,
                step=1000.0,
                help=(
                    "Total current balance across "
                    "all reported credit accounts."
                ),
            )


        with col3:

            total_rev_hi_lim = st.number_input(
                "Total Revolving Credit Limit ($)",
                min_value=0.0,
                value=25000.0,
                step=500.0,
                help=(
                    "Total revolving high credit or credit limit "
                    "reported across the borrower's revolving accounts."
                ),
            )


    # =========================================================================
    # Submit Button
    # =========================================================================

    st.write("")

    submitted = st.form_submit_button(
        "🛡️  ASSESS CREDIT RISK  →",
        use_container_width=True,
    )


# =============================================================================
# Credit Risk Assessment
# =============================================================================

if submitted:

    # =========================================================================
    # Application Validation
    # =========================================================================

    validation_errors = []


    if loan_amnt <= 0:

        validation_errors.append(
            "Loan Amount must be greater than zero."
        )


    if issue_d is None:

        validation_errors.append(
            "Loan Issue Date is required."
        )


    if (
        issue_d is not None
        and earliest_cr_line is not None
        and earliest_cr_line > issue_d
    ):

        validation_errors.append(
            "Earliest Credit Line cannot be later "
            "than Loan Issue Date."
        )


    if open_acc > total_acc:

        validation_errors.append(
            "Open Credit Accounts cannot exceed "
            "Total Credit Accounts."
        )


    if funded_amnt > loan_amnt:

        validation_errors.append(
            "Funded Amount cannot exceed Loan Amount."
        )


    if funded_amnt_inv > funded_amnt:

        validation_errors.append(
            "Investor Funded Amount cannot exceed "
            "Funded Amount."
        )


    # =========================================================================
    # Validation Errors
    # =========================================================================

    if validation_errors:

        st.error(
            "Please correct the following information:"
        )

        for error in validation_errors:

            st.write(
                f"• {error}"
            )


    else:

        # =====================================================================
        # Build Raw Model Input
        # =====================================================================

        raw_input = {

            "loan_amnt": loan_amnt,

            "funded_amnt": funded_amnt,

            "funded_amnt_inv": funded_amnt_inv,

            "term": term,

            "int_rate": int_rate,

            "installment": installment,

            "sub_grade": sub_grade,

            "emp_length": emp_length,

            "emp_title": (
                emp_title
                if emp_title.strip()
                else None
            ),

            "home_ownership": home_ownership,

            "annual_inc": annual_inc,

            "verification_status": verification_status,

            "issue_d": (
                issue_d.isoformat()
                if issue_d is not None
                else None
            ),

            "earliest_cr_line": (
                earliest_cr_line.isoformat()
                if earliest_cr_line is not None
                else None
            ),

            "purpose": purpose,

            "addr_state": addr_state,

            "dti": dti,

            "delinq_2yrs": delinq_2yrs,

            "inq_last_6mths": inq_last_6mths,

            "mths_since_last_delinq": (
                mths_since_last_delinq
            ),

            "mths_since_last_record": (
                mths_since_last_record
            ),

            "open_acc": open_acc,

            "pub_rec": pub_rec,

            "revol_bal": revol_bal,

            "revol_util": revol_util,

            "total_acc": total_acc,

            "initial_list_status": initial_list_status,

            "collections_12_mths_ex_med": (
                collections_12_mths_ex_med
            ),

            "mths_since_last_major_derog": (
                mths_since_last_major_derog
            ),

            "acc_now_delinq": acc_now_delinq,

            "tot_coll_amt": tot_coll_amt,

            "tot_cur_bal": tot_cur_bal,

            "total_rev_hi_lim": total_rev_hi_lim,
        }


        # =====================================================================
        # Prediction
        # =====================================================================

        try:

            result = predict_credit_risk(
                raw_input
            )


            # =================================================================
            # Results Header
            # =================================================================

            st.divider()

            st.html(
                """
                <div class="result-header">

                    <div class="result-header-title">
                        Credit Risk Assessment
                    </div>

                    <div class="result-header-text">
                        Model-based borrower risk assessment derived
                        from the predicted Probability of Default.
                    </div>

                </div>
                """
            )


            # =================================================================
            # Result Cards
            # =================================================================

            st.html(
                f"""
                <div class="risk-results-grid">

                    <div class="risk-result-card">

                        <div class="risk-result-label">
                            Predicted PD
                        </div>

                        <div class="risk-result-value">
                            {result["predicted_pd"]:.2%}
                        </div>

                        <div class="risk-result-description">
                            Estimated probability of default
                        </div>

                    </div>


                    <div class="risk-result-card">

                        <div class="risk-result-label">
                            Credit Score
                        </div>

                        <div class="risk-result-value">
                            {result["credit_score"]}
                        </div>

                        <div class="risk-result-description">
                            Score derived from predicted PD
                        </div>

                    </div>


                    <div class="risk-result-card">

                        <div class="risk-result-label">
                            Model Risk Grade
                        </div>

                        <div class="risk-result-value">
                            {result["risk_grade"]}
                        </div>

                        <div class="risk-result-description">
                            PD-based risk classification
                        </div>

                    </div>


                    <div class="risk-result-card">

                        <div class="risk-result-label">
                            Risk Level
                        </div>

                        <div class="risk-result-value risk-result-text">
                            {result["risk_level"]}
                        </div>

                        <div class="risk-result-description">
                            Overall borrower risk level
                        </div>

                    </div>

                </div>
                """
            )


            # =================================================================
            # Risk Visualization
            # =================================================================

            st.subheader(
                "Risk Profile"
            )

            st.caption(
                "Visual position of the borrower within the predefined "
                "Probability-of-Default risk bands."
            )


            fig = plot_risk_gauge(
                pd_value=result["predicted_pd"],
                score=result["credit_score"],
                grade=result["risk_grade"],
                risk_level=result["risk_level"],
            )


            st.pyplot(
                fig,
                use_container_width=True,
            )
            # =============================================================================
            # Borrower-Level Explainability
            # =============================================================================

            st.html(
                """
                <div class="explainability-header">
                    <div class="explainability-title">
                        Why This Prediction?
                    </div>
                    <div class="explainability-text">
                        Understand which borrower characteristics had the strongest
                        influence on the Probability of Default predicted by the model.
                    </div>
                </div>
                """
            )

            # -----------------------------------------------------------------------------
            # Simple explanation for non-technical users
            # -----------------------------------------------------------------------------

            st.html(
                """
                <div class="explainability-guide">
                    <div class="explainability-guide-title">
                        HOW TO READ THIS EXPLANATION
                    </div>
                    <div>
                        This chart explains why the model produced this
                        <strong>Probability of Default</strong> for the current borrower.
                        Each bar represents a borrower characteristic that influenced
                        the prediction.
                        Factors extending to the <strong>right</strong> pushed the
                        predicted default risk higher, while factors extending to the
                        <strong>left</strong> pushed it lower.
                        The longer the bar, the stronger that characteristic influenced
                        this borrower's prediction.
                    </div>
                </div>
                """
            )

            # -----------------------------------------------------------------------------
            # Rebuild borrower features for SHAP
            # -----------------------------------------------------------------------------

            native_explanation = build_native_features(raw_input)
            encoded_explanation = build_encoded_features(native_explanation)

            # -----------------------------------------------------------------------------
            # Compute borrower-specific SHAP explanation
            # -----------------------------------------------------------------------------

            shap_explanation = explain_borrower(
                model=final_pd_model,
                encoded_features=encoded_explanation,
            )

            # -----------------------------------------------------------------------------
            # Build human-readable local explanation table
            # -----------------------------------------------------------------------------

            explanation_table = build_local_explanation_table(
                explanation=shap_explanation,
                encoded_features=encoded_explanation,
                raw_input=raw_input,
                native_features=native_explanation,
            )

            # -----------------------------------------------------------------------------
            # Select strongest increasing and reducing factors
            # -----------------------------------------------------------------------------

            increasing_drivers, reducing_drivers = get_top_risk_drivers(
                explanation_table=explanation_table,
                top_positive=7,
                top_negative=5,
            )

            # -----------------------------------------------------------------------------
            # Plot borrower-specific explanation
            # -----------------------------------------------------------------------------

            fig_explanation = plot_local_risk_drivers(
                increasing=increasing_drivers,
                reducing=reducing_drivers,
            )

            st.pyplot(
                fig_explanation,
                use_container_width=True,
            )

            # -----------------------------------------------------------------------------
            # Key Risk Drivers — dynamic borrower-specific tables
            # -----------------------------------------------------------------------------

            st.html(
                """
                <div class="driver-section">
                    <div class="driver-section-title">
                        Key Risk Drivers
                    </div>
                    <div class="driver-section-subtitle">
                        The tables below summarize the strongest local factors
                        identified for this borrower. They are recalculated
                        automatically each time a new application is assessed.
                    </div>
                </div>
                """
            )

            def build_driver_rows(drivers, contribution_class):
                """Build safe HTML rows for a borrower-specific SHAP table."""

                if drivers.empty:
                    return (
                        '<tr><td colspan="3" class="driver-empty">'
                        'No material factors were identified in this direction.'
                        '</td></tr>'
                    )

                rows = []

                for _, row in drivers.iterrows():
                    factor = escape(str(row["factor"]))
                    borrower_value = escape(str(row["borrower_value"]))
                    shap_value = float(row["shap_value"])

                    rows.append(
                        f"""
                        <tr>
                            <td class="factor-cell">{factor}</td>
                            <td class="value-cell">{borrower_value}</td>
                            <td class="shap-cell {contribution_class}">
                                {shap_value:+.3f}
                            </td>
                        </tr>
                        """
                    )

                return "".join(rows)

            increasing_rows = build_driver_rows(
                increasing_drivers,
                "shap-positive",
            )

            reducing_rows = build_driver_rows(
                reducing_drivers,
                "shap-negative",
            )

            st.html(
                f"""
                <div class="driver-table-wrapper">
                    <div class="driver-table-heading increasing">
                        ↑ Factors Increasing Predicted Risk
                    </div>
                    <table class="driver-table">
                        <thead>
                            <tr>
                                <th>Factor</th>
                                <th>Borrower Value</th>
                                <th style="text-align:right;">SHAP Contribution</th>
                            </tr>
                        </thead>
                        <tbody>
                            {increasing_rows}
                        </tbody>
                    </table>
                </div>
                """
            )

            st.html(
                f"""
                <div class="driver-table-wrapper">
                    <div class="driver-table-heading reducing">
                        ↓ Factors Reducing Predicted Risk
                    </div>
                    <table class="driver-table">
                        <thead>
                            <tr>
                                <th>Factor</th>
                                <th>Borrower Value</th>
                                <th style="text-align:right;">SHAP Contribution</th>
                            </tr>
                        </thead>
                        <tbody>
                            {reducing_rows}
                        </tbody>
                    </table>
                </div>
                """
            )

            # -----------------------------------------------------------------------------
            # Interpretation note
            # -----------------------------------------------------------------------------

            st.caption(
            "This is a borrower-specific SHAP explanation. "
            "Positive contributions push the model toward higher predicted "
            "default risk, while negative contributions push it toward lower "
            "predicted default risk. These effects explain the model prediction "
            "for this borrower and should not be interpreted as causal effects."
            )
        


        # =====================================================================
        # Expected Errors
        # =====================================================================

        except ValueError as error:

            st.error(
                str(error)
            )


        # =====================================================================
        # Unexpected Errors
        # =====================================================================

        except Exception as error:

            st.error(
                "An unexpected error occurred while processing "
                "the credit-risk assessment."
            )

            st.exception(
                error
            )
