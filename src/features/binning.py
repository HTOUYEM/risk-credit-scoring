from __future__ import annotations

from time import perf_counter
from typing import Dict, Iterable, Tuple
import numpy as np
from numpy import unique_values
import pandas as pd
import scorecardpy as sc


def build_bins_for_variable(
    data: pd.DataFrame,
    variable: str,
    target: str,
    method: str = "tree",
    stop_limit: float = 0.10,
    count_distr_limit: float = 0.05,
    bin_num_limit: int = 8,
    positive: str = "bad|1",
    breaks_list: Dict | None = None,
) -> Dict:
    """Build bins for one variable."""

    variable_data = data[
        [variable, target]
    ].copy()

    if breaks_list:
        return sc.woebin(
            variable_data,
            y=target,
            breaks_list=breaks_list,
            positive=positive,
            check_cate_num=False,
            print_step=0,
        )

    return sc.woebin(
        variable_data,
        y=target,
        method=method,
        stop_limit=stop_limit,
        count_distr_limit=count_distr_limit,
        bin_num_limit=bin_num_limit,
        positive=positive,
        check_cate_num=False,
        print_step=0,
    )


def build_bins(
    data: pd.DataFrame,
    variables: Iterable[str],
    target: str,
    method: str = "tree",
    stop_limit: float = 0.10,
    count_distr_limit: float = 0.05,
    bin_num_limit: int = 8,
    positive: str = "bad|1",
    breaks_list: Dict | None = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Build bins variable by variable and return bins plus a report.

    If a variable is present in breaks_list, its manual breaks are used.
    Otherwise, scorecardpy performs automatic binning.
    """

    variables = list(variables)
    breaks_list = breaks_list or {}

    all_bins: Dict = {}
    report_rows = []

    for position, variable in enumerate(variables, start=1):
        start_time = perf_counter()

        unique_values = (
            int(data[variable].nunique(dropna=False))
            if variable in data.columns
            else None
        )

        manual_breaks_used = variable in breaks_list

        print(
            f"[{position}/{len(variables)}] {variable} "
            f"({unique_values if unique_values is not None else 'N/A'} values)"
            f"{' — manual breaks' if manual_breaks_used else ''}",
            flush=True,
        )

        # Scorecardpy expects only the breaks of the current variable.
        variable_breaks = (
            {variable: breaks_list[variable]}
            if manual_breaks_used
            else None
        )

        retry_used = False
        information_value = None

        try:
            variable_bins = build_bins_for_variable(
                data=data,
                variable=variable,
                target=target,
                method=method,
                stop_limit=stop_limit,
                count_distr_limit=count_distr_limit,
                bin_num_limit=bin_num_limit,
                positive=positive,
                breaks_list=variable_breaks,
            )

            all_bins.update(variable_bins)

            table = variable_bins.get(variable)
            number_of_bins = len(table) if table is not None else 0

            if table is not None and "total_iv" in table.columns:
                information_value = float(
                    table["total_iv"].iloc[0]
                )
            elif table is not None and "bin_iv" in table.columns:
                information_value = float(
                    table["bin_iv"].sum()
                )

            status = "success"
            error_message = None

        except Exception as first_error:
            retry_used = True

            retry_data = data[[variable, target]].copy()

            # Retry after converting nullable numeric types and infinities.
            if pd.api.types.is_numeric_dtype(
                retry_data[variable]
            ):
                retry_data[variable] = (
                    pd.to_numeric(
                        retry_data[variable],
                        errors="coerce",
                    )
                    .astype("float64")
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                )

            try:
                variable_bins = build_bins_for_variable(
                    data=retry_data,
                    variable=variable,
                    target=target,
                    method=method,
                    stop_limit=stop_limit,
                    count_distr_limit=count_distr_limit,
                    bin_num_limit=bin_num_limit,
                    positive=positive,
                    breaks_list=variable_breaks,
                )

                all_bins.update(variable_bins)

                table = variable_bins.get(variable)
                number_of_bins = (
                    len(table)
                    if table is not None
                    else 0
                )

                if (
                    table is not None
                    and "total_iv" in table.columns
                ):
                    information_value = float(
                        table["total_iv"].iloc[0]
                    )
                elif (
                    table is not None
                    and "bin_iv" in table.columns
                ):
                    information_value = float(
                        table["bin_iv"].sum()
                    )

                status = "success_after_retry"
                error_message = None

            except Exception as second_error:
                number_of_bins = 0
                information_value = None
                status = "failed"

                error_message = (
                    f"First attempt: {first_error} | "
                    f"Retry: {second_error}"
                )

        duration_seconds = (
            perf_counter() - start_time
        )

        report_rows.append({
            "variable": variable,
            "status": status,
            "unique_values": unique_values,
            "number_of_bins": number_of_bins,
            "information_value": information_value,
            "manual_breaks_used": manual_breaks_used,
            "retry_used": retry_used,
            "duration_seconds": round(
                duration_seconds,
                2,
            ),
            "error_message": error_message,
        })

        print(
            f"  {status.upper()} — "
            f"{duration_seconds:.2f} seconds",
            flush=True,
        )

    report = pd.DataFrame(report_rows)

    return all_bins, report


'''def summarize_binning_report(report: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row summary of a binning report."""

    success_mask = report["status"].eq("success")
    failed_mask = report["status"].eq("failed")

    return pd.DataFrame([{
        "total_variables": len(report),
        "successful_variables": int(success_mask.sum()),
        "failed_variables": int(failed_mask.sum()),
        "success_rate_percentage": round(success_mask.mean() * 100, 2),
        "total_duration_seconds": round(report["duration_seconds"].sum(), 2),
        "average_duration_seconds": round(report["duration_seconds"].mean(), 2),
    }])'''


'''def rebuild_bins(
    data: pd.DataFrame,
    target: str,
    manual_breaks: Dict,
    positive: str = "bad|1",
) -> Dict:
    """Rebuild selected bins using manually approved breaks."""

    if not manual_breaks:
        return {}

    variables = list(manual_breaks)

    return sc.woebin(
        data[variables + [target]],
        y=target,
        breaks_list=manual_breaks,
        positive=positive,
        print_step=0,
    )'''
