from __future__ import annotations

import numpy as np
import pandas as pd

SOFT_OBJECTIVE_COLS = [
    "objective_travel_time",
    "objective_time_window",
    "objective_priority",
    "objective_client_experience",
    "objective_school_experience",
    "objective_short_term_client_experience",
    "objective_availability_gap",
]

# Distanzen in den Logs sind Meter; 1000 m entsprechen 1 Minute Reisezeit.
METERS_PER_MINUTE = 1000


def meters_to_minutes(
    value: pd.Series | np.ndarray | float,
) -> pd.Series | np.ndarray | float:
    return value / METERS_PER_MINUTE


def gini(x: pd.Series | np.ndarray) -> float:
    values = np.asarray(x, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    if np.all(values == 0):
        return 0.0
    values = np.sort(values)
    n = len(values)
    return float(
        (2 * np.sum((np.arange(1, n + 1) * values)) / (n * np.sum(values)))
        - (n + 1) / n
    )


def max_unserved_streak(group: pd.DataFrame) -> int:
    group = group.sort_values("date")
    streak = 0
    max_streak = 0
    for served in group["assigned"]:
        if not served:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def validate_panel(df: pd.DataFrame) -> dict:
    panel_check = (
        df.groupby(["date", "client_id"])["weight_variant"]
        .nunique()
        .value_counts()
        .sort_index()
    )
    return {
        "weight_variant_counts": df["weight_variant"].value_counts().to_dict(),
        "unique_dates_per_variant": df.groupby("weight_variant")["date"]
        .nunique()
        .to_dict(),
        "unique_clients_per_variant": df.groupby("weight_variant")["client_id"]
        .nunique()
        .to_dict(),
        "panel_variant_counts_per_date_client": panel_check.to_dict(),
        "total_rows": len(df),
        **(
            {
                "priority_value_counts": df["priority"]
                .value_counts()
                .sort_index()
                .to_dict()
            }
            if "priority" in df.columns
            else {}
        ),
    }


def build_client_variant(df: pd.DataFrame) -> pd.DataFrame:
    client_variant = (
        df.groupby(["client_id", "weight_variant"])
        .agg(
            needed_days=("date", "nunique"),
            served_days=("assigned", "sum"),
            locally_eligible_days=("locally_eligible", "sum"),
            mean_eligible_ma=("eligible_ma_count", "mean"),
            mean_min_distance=("min_distance", "mean"),
            mean_chosen_distance=("chosen_distance", "mean"),
            mean_priority=("priority", "mean"),
            mean_client_experience=("client_experience", "mean"),
            mean_school_experience=("school_experience", "mean"),
            mean_short_term_client_experience=(
                "short_term_client_experience",
                "mean",
            ),
        )
        .reset_index()
    )

    client_variant["raw_coverage"] = (
        client_variant["served_days"] / client_variant["needed_days"]
    )
    client_variant["eligible_adjusted_coverage"] = np.where(
        client_variant["locally_eligible_days"] > 0,
        client_variant["served_days"] / client_variant["locally_eligible_days"],
        np.nan,
    )

    streaks = (
        df.groupby(["client_id", "weight_variant"])
        .apply(max_unserved_streak)
        .reset_index(name="max_unserved_streak")
    )
    return client_variant.merge(
        streaks, on=["client_id", "weight_variant"], how="left"
    )


def build_variant_comparison(client_variant: pd.DataFrame) -> pd.DataFrame:
    wide = client_variant.pivot(
        index="client_id",
        columns="weight_variant",
        values="eligible_adjusted_coverage",
    )

    if "default" in wide.columns and "baseline" in wide.columns:
        wide["delta_default_vs_baseline"] = wide["default"] - wide["baseline"]
    if "no-dist" in wide.columns and "default" in wide.columns:
        wide["delta_no_distance_vs_default"] = wide["no-dist"] - wide["default"]
    if "no-exp" in wide.columns and "default" in wide.columns:
        wide["delta_no_experience_vs_default"] = wide["no-exp"] - wide["default"]
    if "default-low-exp" in wide.columns and "default" in wide.columns:
        wide["delta_default_low_exp_vs_default"] = (
            wide["default-low-exp"] - wide["default"]
        )

    return wide.reset_index()


def build_client_features(client_variant: pd.DataFrame) -> pd.DataFrame:
    default_rows = client_variant[client_variant["weight_variant"] == "default"]
    return default_rows[
        [
            "client_id",
            "needed_days",
            "served_days",
            "locally_eligible_days",
            "raw_coverage",
            "max_unserved_streak",
            "mean_min_distance",
            "mean_eligible_ma",
            "mean_priority",
            "mean_client_experience",
            "mean_school_experience",
            "mean_short_term_client_experience",
        ]
    ].rename(
        columns={
            "raw_coverage": "default_raw_coverage",
            "max_unserved_streak": "default_max_unserved_streak",
        }
    )


def build_analysis_table(
    wide: pd.DataFrame, client_features: pd.DataFrame
) -> pd.DataFrame:
    return wide.merge(client_features, on="client_id", how="left")


def build_variant_summary(df: pd.DataFrame) -> pd.DataFrame:
    variant_summary = (
        df.groupby("weight_variant")
        .agg(
            total_requests=("client_id", "count"),
            total_served=("assigned", "sum"),
            locally_eligible_requests=("locally_eligible", "sum"),
            mean_eligible_ma=("eligible_ma_count", "mean"),
            mean_min_distance=("min_distance", "mean"),
            mean_chosen_distance=("chosen_distance", "mean"),
            mean_priority=("priority", "mean"),
            mean_objective_total=("objective_total", "mean"),
        )
        .reset_index()
    )
    variant_summary["raw_coverage"] = (
        variant_summary["total_served"] / variant_summary["total_requests"]
    )
    variant_summary["eligible_adjusted_coverage"] = (
        variant_summary["total_served"]
        / variant_summary["locally_eligible_requests"]
    )
    return variant_summary


def build_objective_summary(df: pd.DataFrame) -> pd.DataFrame:
    present_cols = [c for c in SOFT_OBJECTIVE_COLS if c in df.columns]
    return (
        df.groupby("weight_variant")[present_cols]
        .mean()
        .reset_index()
    )


def build_unassigned_share_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("weight_variant")
        .agg(
            unassigned_rate=("versorgt", lambda x: 1 - x.mean()),
            raw_coverage=("versorgt", "mean"),
            mean_unassigned_penalty=("objective_unassigned", "mean"),
        )
        .reset_index()
    )


def build_fairness_lower_tail_summary(
    client_variant: pd.DataFrame,
) -> pd.DataFrame:
    eligible = client_variant[client_variant["locally_eligible_days"] > 0].copy()
    return (
        eligible.groupby("weight_variant")
        .agg(
            n_clients=("client_id", "count"),
            mean_coverage=("eligible_adjusted_coverage", "mean"),
            median_coverage=("eligible_adjusted_coverage", "median"),
            p10_coverage=(
                "eligible_adjusted_coverage",
                lambda x: x.quantile(0.10),
            ),
            p25_coverage=(
                "eligible_adjusted_coverage",
                lambda x: x.quantile(0.25),
            ),
            min_coverage=("eligible_adjusted_coverage", "min"),
            mean_max_unserved_streak=("max_unserved_streak", "mean"),
            max_max_unserved_streak=("max_unserved_streak", "max"),
        )
        .reset_index()
    )


def _gini_for_subset(frame: pd.DataFrame) -> float:
    values = frame["eligible_adjusted_coverage"].dropna()
    return gini(values)


def build_fairness_inequality_summary(
    client_variant: pd.DataFrame,
) -> pd.DataFrame:
    eligible = client_variant[client_variant["locally_eligible_days"] > 0].copy()
    rows = []
    for variant, group in eligible.groupby("weight_variant"):
        coverage = group["eligible_adjusted_coverage"]
        rows.append(
            {
                "weight_variant": variant,
                "n_clients": len(group),
                "share_zero_coverage": (coverage == 0).mean(),
                "share_below_50": (coverage < 0.5).mean(),
                "gini_all_clients": _gini_for_subset(group),
                "gini_eligible_ge_5": _gini_for_subset(
                    group[group["locally_eligible_days"] >= 5]
                ),
                "gini_eligible_ge_10": _gini_for_subset(
                    group[group["locally_eligible_days"] >= 10]
                ),
                "n_clients_ge_5": int((group["locally_eligible_days"] >= 5).sum()),
                "n_clients_ge_10": int((group["locally_eligible_days"] >= 10).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_fairness_summary(client_variant: pd.DataFrame) -> pd.DataFrame:
    lower_tail = build_fairness_lower_tail_summary(client_variant)
    inequality = build_fairness_inequality_summary(client_variant)
    return lower_tail.merge(inequality, on=["weight_variant", "n_clients"], how="left")


def build_correlation_matrix(analysis: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "delta_default_vs_baseline",
        "delta_no_distance_vs_default",
        "delta_no_experience_vs_default",
        "mean_min_distance",
        "mean_client_experience",
        "mean_school_experience",
        "mean_eligible_ma",
        "needed_days",
        "default",
    ]
    present = [c for c in cols if c in analysis.columns]
    return analysis[present].corr()


def build_problem_cases(
    analysis: pd.DataFrame,
    *,
    min_locally_eligible_days: int = 10,
    max_default_coverage: float = 0.5,
) -> pd.DataFrame:
    if "default" not in analysis.columns:
        return analysis.head(0)
    mask = (analysis["locally_eligible_days"] >= min_locally_eligible_days) & (
        analysis["default"] < max_default_coverage
    )
    sort_cols = ["default"]
    if "delta_default_vs_baseline" in analysis.columns:
        sort_cols.append("delta_default_vs_baseline")
    return analysis[mask].sort_values(sort_cols).reset_index(drop=True)


def build_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    assigned_only = df[df["assigned"]].copy()
    quality_summary = (
        assigned_only.groupby("weight_variant")
        .agg(
            mean_chosen_distance=("chosen_distance", "mean"),
            mean_min_distance=("min_distance", "mean"),
            mean_priority=("priority", "mean"),
            mean_client_experience=("client_experience", "mean"),
            mean_school_experience=("school_experience", "mean"),
            n_assigned=("client_id", "count"),
        )
        .reset_index()
    )
    quality_summary["distance_over_min"] = (
        quality_summary["mean_chosen_distance"]
        / quality_summary["mean_min_distance"]
    )
    quality_summary["mean_chosen_distance_min"] = meters_to_minutes(
        quality_summary["mean_chosen_distance"]
    )
    quality_summary["mean_min_distance_min"] = meters_to_minutes(
        quality_summary["mean_min_distance"]
    )
    return quality_summary


def validate_experience_data(df: pd.DataFrame) -> dict:
    experience_cols = [
        "client_experience",
        "school_experience",
        "short_term_client_experience",
    ]
    objective_cols = [
        "objective_client_experience",
        "objective_school_experience",
        "objective_short_term_client_experience",
    ]
    present_exp = [c for c in experience_cols if c in df.columns]
    present_obj = [c for c in objective_cols if c in df.columns]

    has_variance = False
    for col in present_exp:
        if df[col].nunique(dropna=True) > 1 or df[col].sum() > 0:
            has_variance = True
            break

    has_objective_signal = False
    for col in present_obj:
        if df[col].abs().sum() > 0:
            has_objective_signal = True
            break

    no_exp_differs = False
    if "no-exp" in df["weight_variant"].unique():
        default_served = df.loc[
            df["weight_variant"] == "default", "assigned"
        ].sum()
        no_exp_served = df.loc[
            df["weight_variant"] == "no-exp", "assigned"
        ].sum()
        no_exp_differs = default_served != no_exp_served

    return {
        "has_experience_variance": has_variance,
        "has_objective_experience_signal": has_objective_signal,
        "no_exp_differs_from_default": no_exp_differs,
        "interpretable": has_variance
        and (has_objective_signal or no_exp_differs),
        "max_client_experience": float(df["client_experience"].max())
        if "client_experience" in df.columns
        else 0.0,
        "max_school_experience": float(df["school_experience"].max())
        if "school_experience" in df.columns
        else 0.0,
        "objective_client_experience_sum": float(
            df["objective_client_experience"].sum()
        )
        if "objective_client_experience" in df.columns
        else 0.0,
    }


def enrich_analysis_for_plots(analysis: pd.DataFrame) -> pd.DataFrame:
    enriched = analysis.copy()
    if "mean_min_distance" in enriched.columns:
        enriched["mean_min_distance_min"] = meters_to_minutes(
            enriched["mean_min_distance"]
        )
        enriched["distance_bin"] = pd.cut(
            enriched["mean_min_distance_min"],
            bins=[0, 15, 30, 45, 60, 90, 120, np.inf],
            labels=["0-15", "15-30", "30-45", "45-60", "60-90", "90-120", ">120"],
        )
    if "mean_eligible_ma" in enriched.columns:
        enriched["eligible_pool_bin"] = pd.cut(
            enriched["mean_eligible_ma"],
            bins=[0, 1, 3, 5, 10, 20, 50, np.inf],
            labels=["<=1", "2-3", "4-5", "6-10", "11-20", "21-50", ">50"],
        )
    return enriched


def build_distance_bin_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    if "distance_bin" not in analysis.columns:
        return pd.DataFrame()
    return (
        analysis.groupby("distance_bin", observed=True)
        .agg(
            mean_delta=("delta_no_distance_vs_default", "mean"),
            median_delta=("delta_no_distance_vs_default", "median"),
            n_clients=("client_id", "count"),
            total_eligible_days=("locally_eligible_days", "sum"),
        )
        .reset_index()
    )


def build_pool_bin_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    if "eligible_pool_bin" not in analysis.columns:
        return pd.DataFrame()
    return (
        analysis.groupby("eligible_pool_bin", observed=True)
        .agg(
            mean_default_coverage=("default", "mean"),
            mean_delta_default_vs_baseline=(
                "delta_default_vs_baseline",
                "mean",
            ),
            n_clients=("client_id", "count"),
            total_eligible_days=("locally_eligible_days", "sum"),
        )
        .reset_index()
    )


def build_streak_summary(client_variant: pd.DataFrame) -> pd.DataFrame:
    return (
        client_variant.groupby("weight_variant")
        .agg(
            mean_max_streak=("max_unserved_streak", "mean"),
            median_max_streak=("max_unserved_streak", "median"),
            p90_max_streak=(
                "max_unserved_streak",
                lambda x: x.quantile(0.9),
            ),
            max_streak=("max_unserved_streak", "max"),
        )
        .reset_index()
    )


def build_daily_coverage(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["date", "weight_variant"])
        .agg(
            served=("assigned", "sum"),
            locally_eligible=("locally_eligible", "sum"),
            requests=("client_id", "count"),
        )
        .reset_index()
    )
    daily["raw_coverage"] = daily["served"] / daily["requests"]
    daily["eligible_adjusted_coverage"] = np.where(
        daily["locally_eligible"] > 0,
        daily["served"] / daily["locally_eligible"],
        np.nan,
    )
    return daily
