# =============================================================================
# Global Configuration
# =============================================================================

# Reproducibility
RANDOM_STATE = 42

# Internal validation split
VALIDATION_SIZE = 0.20

# Maximum iterations for iterative estimators
MAX_ITER = 1000


# =============================================================================
# Credit Scoring Configuration
# =============================================================================

# Reference score assigned to the reference odds
BASE_SCORE = 600

# Good-to-bad odds associated with the reference score
BASE_ODDS = 20

# Points to Double the Odds
PDO = 50


# =============================================================================
# Risk Grade Configuration
# =============================================================================

PD_BINS = [
    0.00,
    0.03,
    0.06,
    0.10,
    0.20,
    1.00,
]

GRADE_LABELS = [
    "A",
    "B",
    "C",
    "D",
    "E",
]


# =============================================================================
# Risk Level Labels
# =============================================================================

RISK_LABELS = {
    "A": "Very Low Risk",
    "B": "Low Risk",
    "C": "Moderate Risk",
    "D": "High Risk",
    "E": "Very High Risk",
}


# =============================================================================
# Risk Grade Colors
# =============================================================================

RISK_COLORS = {
    "A": "#A8D5A2",
    "B": "#087F8C",
    "C": "#A6DCEF",
    "D": "#FFB81C",
    "E": "#F4512A",
}