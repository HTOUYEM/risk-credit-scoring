from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def build_logistic_preprocessor(
    X,
    ordinal_features=None,
):

    ordinal_features = (
        []
        if ordinal_features is None
        else ordinal_features
    )

    binary_features = [
        col
        for col in X.columns
        if set(
            X[col]
            .dropna()
            .unique()
        ).issubset({0, 1})
    ]

    numeric_features = [
        col
        for col in X.columns
        if col not in binary_features
        and col not in ordinal_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return (
        preprocessor,
        numeric_features,
        ordinal_features,
        binary_features,
    )