import json

from joblib import dump, load


def save_model_artifacts(
    best_model,
    best_params,
    cv_results,
    model_name,
    model_dir,
):

    model_path = (
        model_dir
        / f"{model_name}_tuned.joblib"
    )

    params_path = (
        model_dir
        / f"{model_name}_best_params.json"
    )

    results_path = (
        model_dir
        / f"{model_name}_cv_results.csv"
    )

    dump(
        best_model,
        model_path,
    )

    with open(
        params_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_params,
            file,
            indent=4,
        )

    cv_results.to_csv(
        results_path,
        index=False,
    )


def load_model_artifacts(
    model_name,
    model_dir,
):

    model_path = (
        model_dir
        / f"{model_name}_tuned.joblib"
    )

    params_path = (
        model_dir
        / f"{model_name}_best_params.json"
    )

    best_model = load(
        model_path
    )

    with open(
        params_path,
        "r",
        encoding="utf-8",
    ) as file:
        best_params = json.load(
            file
        )

    return (
        best_model,
        best_params,
    )