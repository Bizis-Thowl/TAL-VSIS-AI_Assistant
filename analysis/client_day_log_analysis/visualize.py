from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.client_day_log_analysis.metrics import SOFT_OBJECTIVE_COLS

from analysis.client_day_log_analysis.load import PREFERRED_VARIANT_ORDER, sort_variants

VARIANT_ORDER = list(PREFERRED_VARIANT_ORDER)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _point_sizes(
    frame: pd.DataFrame, column: str = "locally_eligible_days", scale: float = 3.0
) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), 30.0)
    return 10 + frame[column].fillna(0) * scale


def _ordered_variants(variants: pd.Series | list) -> list:
    return sort_variants(variants)


def plot_variant_coverage(
    variant_summary: pd.DataFrame, output_path: Path
) -> None:
    summary = variant_summary.copy()
    summary["weight_variant"] = pd.Categorical(
        summary["weight_variant"], categories=_ordered_variants(summary["weight_variant"])
    )
    summary = summary.sort_values("weight_variant")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    width = 0.35
    raw_bars = ax.bar(
        x - width / 2,
        summary["raw_coverage"],
        width,
        label="raw_coverage",
    )
    eligible_bars = ax.bar(
        x + width / 2,
        summary["eligible_adjusted_coverage"],
        width,
        label="eligible_adjusted_coverage",
    )
    for bars in (raw_bars, eligible_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["weight_variant"], rotation=15)
    ax.set_ylabel("Coverage")
    ax.set_title("Gesamtversorgung je Gewichtungsvariante")
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, output_path)


def plot_coverage_ecdf(
    client_variant: pd.DataFrame, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for variant in _ordered_variants(client_variant["weight_variant"].unique()):
        values = (
            client_variant.loc[
                client_variant["weight_variant"] == variant,
                "eligible_adjusted_coverage",
            ]
            .dropna()
            .sort_values()
        )
        if len(values) == 0:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, label=variant, linewidth=2)
    ax.set_xlabel("Eligible-adjusted Coverage")
    ax.set_ylabel("Anteil der Klienten <= Coverage")
    ax.set_title("Kumulative Verteilung der Coverage pro Variante")
    ax.set_xlim(-0.03, 1.03)
    ax.legend()
    ax.grid(alpha=0.2)
    _save_figure(fig, output_path)


def plot_default_vs_baseline(
    analysis: pd.DataFrame, output_path: Path
) -> None:
    if "baseline" not in analysis.columns or "default" not in analysis.columns:
        return
    plot_df = analysis.dropna(subset=["baseline", "default"])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        plot_df["baseline"],
        plot_df["default"],
        s=_point_sizes(plot_df),
        alpha=0.45,
    )
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.text(0.08, 0.92, "Default besser", fontsize=11)
    ax.text(0.62, 0.08, "Default schlechter", fontsize=11)
    ax.text(
        0.72,
        0.25,
        "kritische Faelle:\nBaseline hoch,\nDefault niedrig",
        fontsize=10,
    )
    ax.set_xlabel("Coverage in Baseline")
    ax.set_ylabel("Coverage in Default")
    ax.set_title(
        "Coverage-Vergleich pro Klient: Default vs. Baseline\n"
        "Punktgroesse = Anzahl lokal geeigneter Faelle"
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.2)
    _save_figure(fig, output_path)


def plot_top_delta_clients(
    analysis: pd.DataFrame,
    output_path: Path,
    *,
    ascending: bool,
    title: str,
) -> None:
    if "delta_default_vs_baseline" not in analysis.columns:
        return
    top = analysis.dropna(subset=["delta_default_vs_baseline"]).sort_values(
        "delta_default_vs_baseline", ascending=ascending
    ).head(20)
    if top.empty:
        return
    labels = top["client_id"].astype(str).str[:8]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, top["delta_default_vs_baseline"])
    ax.axvline(0, linestyle="--", color="gray")
    ax.set_xlabel("Default - Baseline Coverage")
    ax.set_ylabel("Klient (ID-Prefix)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    _save_figure(fig, output_path)


def plot_top_losses(analysis: pd.DataFrame, output_path: Path) -> None:
    plot_top_delta_clients(
        analysis,
        output_path,
        ascending=True,
        title="Staerkste Coverage-Verluste im Default-Modell",
    )


def plot_top_gains(analysis: pd.DataFrame, output_path: Path) -> None:
    plot_top_delta_clients(
        analysis,
        output_path,
        ascending=False,
        title="Staerkste Coverage-Gewinne im Default-Modell",
    )


def plot_distance_bias_scatter(
    analysis: pd.DataFrame, output_path: Path
) -> None:
    if "delta_no_distance_vs_default" not in analysis.columns:
        return
    plot_df = analysis.dropna(
        subset=["mean_min_distance_min", "delta_no_distance_vs_default"]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        plot_df["mean_min_distance_min"],
        plot_df["delta_no_distance_vs_default"],
        s=_point_sizes(plot_df),
        alpha=0.45,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mittlere minimale Distanz (Minuten, 1000 m = 1 min)")
    ax.set_ylabel("Coverage-Gewinn ohne Distanzgewicht")
    ax.set_title(
        "Distanznachteil: no-dist vs. default\n"
        "Punktgroesse = Anzahl lokal geeigneter Faelle"
    )
    ax.grid(alpha=0.2)
    _save_figure(fig, output_path)


def plot_distance_bias_bins(
    distance_bins: pd.DataFrame, output_path: Path
) -> None:
    if distance_bins.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = distance_bins["distance_bin"].astype(str)
    ax.bar(labels, distance_bins["mean_delta"])
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlabel("Mittlere minimale Distanz (Minuten)")
    ax.set_ylabel("O Coverage-Aenderung ohne Distanzgewicht")
    ax.set_title("Distanznachteil nach Distanzgruppen")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, output_path)


def plot_experience_validation(
    experience_validation: dict, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    if experience_validation.get("interpretable"):
        message = (
            "Erfahrungs-Lock-in: Varianz in den Erfahrungsdaten vorhanden.\n"
            "Siehe delta_no_experience_vs_default in analysis.csv."
        )
    else:
        message = (
            "Erfahrungs-Lock-in: aktuell nicht messbar\n\n"
            "In der Simulation lagen client_experience, school_experience und\n"
            "short_term_client_experience ohne Varianz vor bzw. die Objective-\n"
            "Beitraege waren konstant 0. no-exp unterscheidet sich nicht von default.\n\n"
            "Hinweis: Experience-Log aktivieren und Run wiederholen."
        )
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=11,
        wrap=True,
        bbox={"boxstyle": "round", "facecolor": "#f5f5f5"},
    )
    ax.set_title("Erfahrungsanalyse: Validierungsbefund")
    _save_figure(fig, output_path)


def plot_fairness_lower_tail(
    fairness_lower_tail: pd.DataFrame, output_path: Path
) -> None:
    summary = fairness_lower_tail.copy()
    summary["weight_variant"] = pd.Categorical(
        summary["weight_variant"], categories=_ordered_variants(summary["weight_variant"])
    )
    summary = summary.sort_values("weight_variant")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    width = 0.2
    metrics = [
        ("min_coverage", "Minimum"),
        ("p10_coverage", "P10"),
        ("p25_coverage", "P25"),
        ("median_coverage", "Median"),
    ]
    for i, (col, label) in enumerate(metrics):
        offset = (i - 1.5) * width
        ax.bar(x + offset, summary[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["weight_variant"], rotation=15)
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Lower-tail Coverage je Variante (nur Klienten mit lokaler Eignung)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, output_path)


def plot_fairness_inequality(
    fairness_inequality: pd.DataFrame, output_path: Path
) -> None:
    summary = fairness_inequality.copy()
    summary["weight_variant"] = pd.Categorical(
        summary["weight_variant"], categories=_ordered_variants(summary["weight_variant"])
    )
    summary = summary.sort_values("weight_variant")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    width = 0.25
    metrics = [
        ("gini_all_clients", "Gini"),
        ("share_zero_coverage", "Anteil Coverage = 0"),
        ("share_below_50", "Anteil Coverage < 0.5"),
    ]
    for i, (col, label) in enumerate(metrics):
        offset = (i - 1) * width
        ax.bar(x + offset, summary[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["weight_variant"], rotation=15)
    ax.set_ylabel("Wert")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ungleichverteilung je Variante")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, output_path)


def plot_gini_by_threshold(
    fairness_inequality: pd.DataFrame, output_path: Path
) -> None:
    summary = fairness_inequality.copy()
    summary["weight_variant"] = pd.Categorical(
        summary["weight_variant"], categories=_ordered_variants(summary["weight_variant"])
    )
    summary = summary.sort_values("weight_variant")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    width = 0.25
    gini_cols = [
        ("gini_all_clients", "alle Klienten"),
        ("gini_eligible_ge_5", "eligible >= 5"),
        ("gini_eligible_ge_10", "eligible >= 10"),
    ]
    for i, (col, label) in enumerate(gini_cols):
        offset = (i - 1) * width
        ax.bar(x + offset, summary[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["weight_variant"], rotation=15)
    ax.set_ylabel("Gini")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Gini-Robustheit: nur Klienten mit ausreichend lokal geeigneten Faellen"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, output_path)


def plot_coverage_by_pool(
    pool_summary: pd.DataFrame, output_path: Path
) -> None:
    if pool_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = pool_summary["eligible_pool_bin"].astype(str)
    ax.bar(labels, pool_summary["mean_default_coverage"])
    ax.set_xlabel("Mittlere Anzahl geeigneter Mitarbeitender")
    ax.set_ylabel("O Default Coverage")
    ax.set_title("Default Coverage nach Groesse des Kandidatenpools")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, output_path)


def plot_streak_summary(
    streak_summary: pd.DataFrame, output_path: Path
) -> None:
    summary = streak_summary.copy()
    summary["weight_variant"] = pd.Categorical(
        summary["weight_variant"], categories=_ordered_variants(summary["weight_variant"])
    )
    summary = summary.sort_values("weight_variant")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["weight_variant"], summary["p90_max_streak"])
    ax.set_ylabel("90%-Perzentil der laengsten Nichtversorgungsserie")
    ax.set_title("Worst-Case-Zeitverlaeufe: lange Nichtversorgung je Variante")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, output_path)


def plot_coverage_and_soft_objective(
    df: pd.DataFrame,
    unassigned_summary: pd.DataFrame,
    soft_objective_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    unassigned = unassigned_summary.copy()
    unassigned["weight_variant"] = pd.Categorical(
        unassigned["weight_variant"], categories=_ordered_variants(unassigned["weight_variant"])
    )
    unassigned = unassigned.sort_values("weight_variant")

    soft = soft_objective_summary.set_index("weight_variant")
    soft = soft.reindex(_ordered_variants(soft.index))
    present_cols = [c for c in SOFT_OBJECTIVE_COLS if c in soft.columns]
    soft = soft[present_cols]

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10, 10))

    bars = ax_a.bar(unassigned["weight_variant"], unassigned["unassigned_rate"])
    for bar in bars:
        height = bar.get_height()
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_a.set_ylabel("Anteil unversorgter Vertretungsfälle")
    ax_a.set_title("Panel A: Anteil unversorgter Vertretungsfälle")
    ax_a.set_ylim(0, 1.12)
    ax_a.tick_params(axis="x", rotation=15)
    ax_a.grid(axis="y", alpha=0.2)

    if not soft.empty:
        soft.plot(kind="bar", stacked=True, ax=ax_b)
        ax_b.axhline(0, linestyle="--", color="gray", linewidth=0.8)
        ax_b.set_ylabel("Mittlerer Objective-Beitrag pro Vertretungsfall")
        ax_b.set_title(
            "Panel B: Mittlerer Objective-Beitrag pro Vertretungsfall\n"
            "(ohne Unversorgungsstrafe; alle Vertretungsfälle)"
        )
        ax_b.tick_params(axis="x", rotation=15)
        ax_b.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax_b.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Versorgung und Objective-Mechanismen je Gewichtungsvariante", y=1.01
    )
    _save_figure(fig, output_path)


def plot_unassigned_share(
    unassigned_summary: pd.DataFrame, output_path: Path
) -> None:
    summary = unassigned_summary.copy()
    summary["weight_variant"] = pd.Categorical(
        summary["weight_variant"], categories=_ordered_variants(summary["weight_variant"])
    )
    summary = summary.sort_values("weight_variant")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary["weight_variant"], summary["unassigned_rate"])
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Anteil unversorgter Klient-Tage")
    ax.set_title("Unversorgungsrate je Gewichtungsvariante")
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, output_path)


def plot_objective_components(
    soft_objective_summary: pd.DataFrame, output_path: Path
) -> None:
    soft = soft_objective_summary.set_index("weight_variant")
    soft = soft.reindex(_ordered_variants(soft.index))
    present_cols = [c for c in SOFT_OBJECTIVE_COLS if c in soft.columns]
    if not present_cols:
        return
    soft = soft[present_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    soft.plot(kind="bar", stacked=True, ax=ax)
    ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
    ax.set_ylabel("Mittlerer Objective-Beitrag pro Klient-Tag")
    ax.set_title(
        "Soft-Objective-Komponenten je Variante\n"
        "(ohne Unversorgungsstrafe; alle Klient-Tage)"
    )
    ax.tick_params(axis="x", rotation=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, output_path)


def plot_tradeoff_coverage_distance(
    variant_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    merged = variant_summary[
        ["weight_variant", "eligible_adjusted_coverage"]
    ].merge(
        quality_summary[["weight_variant", "mean_chosen_distance_min"]],
        on="weight_variant",
    )
    merged["weight_variant"] = pd.Categorical(
        merged["weight_variant"], categories=_ordered_variants(merged["weight_variant"])
    )
    merged = merged.sort_values("weight_variant")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(merged))
    bars = ax1.bar(
        x,
        merged["eligible_adjusted_coverage"],
        color="#4C78A8",
        alpha=0.85,
        label="eligible_adjusted_coverage",
    )
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.set_ylabel("Eligible-adjusted Coverage", color="#4C78A8")
    ax1.set_ylim(0, 1.12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged["weight_variant"], rotation=15)
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        merged["mean_chosen_distance_min"],
        color="#E45756",
        marker="o",
        linewidth=2,
        label="mean_chosen_distance (min)",
    )
    ax2.set_ylabel("Mittlere gewaehlte Distanz (Minuten)", color="#E45756")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Trade-off: Coverage vs. gewaehlte Reisezeit")
    _save_figure(fig, output_path)


def plot_daily_coverage(
    daily_coverage: pd.DataFrame, output_path: Path
) -> None:
    if daily_coverage.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for variant in _ordered_variants(daily_coverage["weight_variant"].unique()):
        subset = daily_coverage[daily_coverage["weight_variant"] == variant].sort_values(
            "date"
        )
        ax.plot(
            subset["date"],
            subset["eligible_adjusted_coverage"],
            label=variant,
            linewidth=1.5,
            alpha=0.9,
        )
    ax.set_xlabel("Datum")
    ax.set_ylabel("Eligible-adjusted Coverage")
    ax.set_title("Zeitverlauf der taeglichen Coverage je Variante")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    _save_figure(fig, output_path)


def _sort_df_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weight_variant"] = pd.Categorical(
        out["weight_variant"], categories=_ordered_variants(out["weight_variant"])
    )
    return out.sort_values("weight_variant")


def build_plot_tables(
    *,
    fairness_lower_tail: pd.DataFrame,
    fairness_inequality: pd.DataFrame,
    unassigned_summary: pd.DataFrame,
    objective_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Tabellen mit denselben Kennzahlen wie in den Plots 09, 10, 11 und 14."""
    plot_09 = _sort_df_by_variant(fairness_lower_tail)[
        [
            "weight_variant",
            "min_coverage",
            "p10_coverage",
            "p25_coverage",
            "median_coverage",
        ]
    ]

    plot_10 = _sort_df_by_variant(fairness_inequality)[
        [
            "weight_variant",
            "gini_all_clients",
            "share_zero_coverage",
            "share_below_50",
        ]
    ]

    plot_11 = _sort_df_by_variant(fairness_inequality)[
        [
            "weight_variant",
            "gini_all_clients",
            "gini_eligible_ge_5",
            "gini_eligible_ge_10",
        ]
    ]

    unassigned = _sort_df_by_variant(unassigned_summary)[
        ["weight_variant", "unassigned_rate"]
    ]
    soft = objective_summary.set_index("weight_variant")
    soft = soft.reindex(_ordered_variants(soft.index))
    present_cols = [c for c in SOFT_OBJECTIVE_COLS if c in soft.columns]
    soft = soft[present_cols].reset_index()
    plot_14 = unassigned.merge(soft, on="weight_variant", how="left")

    return {
        "09_fairness_lower_tail": plot_09,
        "10_fairness_inequality": plot_10,
        "11_gini_by_threshold": plot_11,
        "14_coverage_and_soft_objective": plot_14,
    }


def generate_all_plots(
    *,
    df: pd.DataFrame,
    variant_summary: pd.DataFrame,
    client_variant: pd.DataFrame,
    analysis: pd.DataFrame,
    fairness_lower_tail: pd.DataFrame,
    fairness_inequality: pd.DataFrame,
    objective_summary: pd.DataFrame,
    unassigned_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
    experience_validation: dict,
    distance_bin_summary: pd.DataFrame,
    pool_bin_summary: pd.DataFrame,
    streak_summary: pd.DataFrame,
    daily_coverage: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    plot_specs: list[tuple[str, object, tuple]] = [
        ("01_variant_coverage.png", plot_variant_coverage, (variant_summary,)),
        ("02_coverage_ecdf.png", plot_coverage_ecdf, (client_variant,)),
        ("03_default_vs_baseline.png", plot_default_vs_baseline, (analysis,)),
        (
            "04_top_losses_default_vs_baseline.png",
            plot_top_losses,
            (analysis,),
        ),
        (
            "05_top_gains_default_vs_baseline.png",
            plot_top_gains,
            (analysis,),
        ),
        ("06_distance_bias_scatter.png", plot_distance_bias_scatter, (analysis,)),
        ("07_distance_bias_bins.png", plot_distance_bias_bins, (distance_bin_summary,)),
        (
            "08_experience_validation.png",
            plot_experience_validation,
            (experience_validation,),
        ),
        (
            "09_fairness_lower_tail.png",
            plot_fairness_lower_tail,
            (fairness_lower_tail,),
        ),
        (
            "10_fairness_inequality.png",
            plot_fairness_inequality,
            (fairness_inequality,),
        ),
        (
            "11_gini_by_threshold.png",
            plot_gini_by_threshold,
            (fairness_inequality,),
        ),
        ("12_coverage_by_pool.png", plot_coverage_by_pool, (pool_bin_summary,)),
        ("13_max_unserved_streak.png", plot_streak_summary, (streak_summary,)),
        (
            "14_coverage_and_soft_objective.png",
            plot_coverage_and_soft_objective,
            (df, unassigned_summary, objective_summary),
        ),
        (
            "16_tradeoff_coverage_distance.png",
            plot_tradeoff_coverage_distance,
            (variant_summary, quality_summary),
        ),
        ("17_daily_coverage.png", plot_daily_coverage, (daily_coverage,)),
    ]

    saved: list[Path] = []
    for filename, plot_fn, args in plot_specs:
        path = output_dir / filename
        plot_fn(*args, path)
        saved.append(path)
    return saved
