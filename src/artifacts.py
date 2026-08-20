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


# =============================================================================
# Final PD Model Artifacts
# =============================================================================

from joblib import dump


def save_final_pd_artifacts(
    model,
    performance_summary,
    model_dir,
):
    """
    Save the final PD model and its performance summary.
    """

    model_path = (
        model_dir
        / "final_pd_model.joblib"
    )

    performance_path = (
        model_dir
        / "final_pd_performance.csv"
    )

    dump(
        model,
        model_path,
    )

    performance_summary.to_csv(
        performance_path,
        index=False,
    )

    return {
        "model_path": model_path,
        "performance_path": performance_path,
    }