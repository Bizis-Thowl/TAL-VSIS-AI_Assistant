from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis.client_day_log_analysis.io import safe_to_csv, safe_write_json
from analysis.client_day_log_analysis.load import load_run, filter_by_priority
from analysis.client_day_log_analysis.metrics import (
    build_analysis_table,
    build_client_features,
    build_client_variant,
    build_correlation_matrix,
    build_daily_coverage,
    build_distance_bin_summary,
    build_fairness_inequality_summary,
    build_fairness_lower_tail_summary,
    build_fairness_summary,
    build_objective_summary,
    build_pool_bin_summary,
    build_problem_cases,
    build_quality_summary,
    build_streak_summary,
    build_unassigned_share_summary,
    build_variant_comparison,
    build_variant_summary,
    enrich_analysis_for_plots,
    validate_experience_data,
    validate_panel,
)
from analysis.client_day_log_analysis.visualize import generate_all_plots


@dataclass
class AnalysisResult:
    run_id: str
    validation: dict
    client_variant: pd.DataFrame
    wide: pd.DataFrame
    analysis: pd.DataFrame
    variant_summary: pd.DataFrame
    objective_summary: pd.DataFrame
    unassigned_share_summary: pd.DataFrame
    fairness_summary: pd.DataFrame
    fairness_lower_tail: pd.DataFrame
    fairness_inequality: pd.DataFrame
    quality_summary: pd.DataFrame
    streak_summary: pd.DataFrame
    distance_bin_summary: pd.DataFrame
    pool_bin_summary: pd.DataFrame
    daily_coverage: pd.DataFrame
    experience_validation: dict
    correlation_matrix: pd.DataFrame
    problem_cases: pd.DataFrame
    plot_paths: list[Path]


def run_analysis(
    log_dir: Path,
    run_id: str,
    output_dir: Path,
    *,
    priorities: list[int] | None = None,
    min_locally_eligible_days: int = 10,
    max_default_coverage: float = 0.5,
    save_plots: bool = True,
) -> AnalysisResult:
    df = load_run(log_dir, run_id)
    rows_before_filter = len(df)
    df = filter_by_priority(df, priorities)
    validation = validate_panel(df)
    if priorities:
        validation["priority_filter"] = priorities
        validation["rows_before_priority_filter"] = rows_before_filter
        validation["rows_after_priority_filter"] = len(df)

    client_variant = build_client_variant(df)
    wide = build_variant_comparison(client_variant)
    client_features = build_client_features(client_variant)
    analysis = enrich_analysis_for_plots(
        build_analysis_table(wide, client_features)
    )

    variant_summary = build_variant_summary(df)
    objective_summary = build_objective_summary(df)
    unassigned_share_summary = build_unassigned_share_summary(df)
    fairness_lower_tail = build_fairness_lower_tail_summary(client_variant)
    fairness_inequality = build_fairness_inequality_summary(client_variant)
    fairness_summary = build_fairness_summary(client_variant)
    quality_summary = build_quality_summary(df)
    streak_summary = build_streak_summary(client_variant)
    distance_bin_summary = build_distance_bin_summary(analysis)
    pool_bin_summary = build_pool_bin_summary(analysis)
    daily_coverage = build_daily_coverage(df)
    experience_validation = validate_experience_data(df)
    correlation_matrix = build_correlation_matrix(analysis)
    problem_cases = build_problem_cases(
        analysis,
        min_locally_eligible_days=min_locally_eligible_days,
        max_default_coverage=max_default_coverage,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_to_csv(client_variant, output_dir / "client_variant.csv", index=False)
    safe_to_csv(wide, output_dir / "wide_with_deltas.csv", index=False)
    safe_to_csv(analysis, output_dir / "analysis.csv", index=False)
    safe_to_csv(variant_summary, output_dir / "variant_summary.csv", index=False)
    safe_to_csv(objective_summary, output_dir / "objective_summary.csv", index=False)
    safe_to_csv(
        unassigned_share_summary, output_dir / "unassigned_share.csv", index=False
    )
    safe_to_csv(fairness_summary, output_dir / "fairness_summary.csv", index=False)
    safe_to_csv(
        fairness_lower_tail, output_dir / "fairness_lower_tail.csv", index=False
    )
    safe_to_csv(
        fairness_inequality, output_dir / "fairness_inequality.csv", index=False
    )
    safe_to_csv(quality_summary, output_dir / "quality_summary.csv", index=False)
    safe_to_csv(streak_summary, output_dir / "streak_summary.csv", index=False)
    safe_to_csv(
        distance_bin_summary, output_dir / "distance_bin_summary.csv", index=False
    )
    safe_to_csv(pool_bin_summary, output_dir / "pool_bin_summary.csv", index=False)
    safe_to_csv(daily_coverage, output_dir / "daily_coverage.csv", index=False)
    safe_to_csv(correlation_matrix, output_dir / "correlation_matrix.csv")
    safe_to_csv(
        problem_cases.head(20),
        output_dir / "problem_cases_top20.csv",
        index=False,
    )

    safe_write_json(
        output_dir / "validation.json",
        validation,
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    safe_write_json(
        output_dir / "experience_validation.json",
        experience_validation,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    plot_paths: list[Path] = []
    if save_plots:
        plot_paths = generate_all_plots(
            df=df,
            variant_summary=variant_summary,
            client_variant=client_variant,
            analysis=analysis,
            fairness_lower_tail=fairness_lower_tail,
            fairness_inequality=fairness_inequality,
            objective_summary=objective_summary,
            unassigned_summary=unassigned_share_summary,
            quality_summary=quality_summary,
            experience_validation=experience_validation,
            distance_bin_summary=distance_bin_summary,
            pool_bin_summary=pool_bin_summary,
            streak_summary=streak_summary,
            daily_coverage=daily_coverage,
            output_dir=output_dir / "plots",
        )

    return AnalysisResult(
        run_id=run_id,
        validation=validation,
        client_variant=client_variant,
        wide=wide,
        analysis=analysis,
        variant_summary=variant_summary,
        objective_summary=objective_summary,
        unassigned_share_summary=unassigned_share_summary,
        fairness_summary=fairness_summary,
        fairness_lower_tail=fairness_lower_tail,
        fairness_inequality=fairness_inequality,
        quality_summary=quality_summary,
        streak_summary=streak_summary,
        distance_bin_summary=distance_bin_summary,
        pool_bin_summary=pool_bin_summary,
        daily_coverage=daily_coverage,
        experience_validation=experience_validation,
        correlation_matrix=correlation_matrix,
        problem_cases=problem_cases,
        plot_paths=plot_paths,
    )
