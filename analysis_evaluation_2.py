from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


GROUP_CHOICES = ["priority_10", "other_priorities", "all_priorities"]

TABLE_COLUMNS = [
    "date",
    "entries_count",
    "cl_experience_avg",
    "school_experience_avg",
    "short_term_cl_experience_avg",
    "time_to_school_avg",
    "ma_availability_rate",
    "car_availability_ratio",
    "qualifications_met_rate",
    "availability_gap_non_positive_rate",
    "availability_gap_positive_rate",
    "availability_gap_positive_average",
    "priority_avg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an analysis run JSON (e.g. analysis_run_0001.json), aggregate "
            "per-day label vs recommendation metrics for a priority group, and "
            "save matplotlib dashboards."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/analysis_runs/analysis_run_0001.json"),
        help="Path to analysis_run_XXXX.json.",
    )
    parser.add_argument(
        "--group",
        choices=GROUP_CHOICES,
        default="all_priorities",
        help="Priority group to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluations"),
        help="Directory where PNG output is saved.",
    )
    return parser.parse_args()


def load_run(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator in (0, None):
        return 0.0
    return float(numerator) / float(denominator)


def _to_float(value) -> float:
    if value is None:
        return 0.0
    return float(value)


def _row_matches_group(row: dict, group: str) -> bool:
    if group == "all_priorities":
        return True
    priority = row.get("priority")
    if group == "priority_10":
        return priority == 10
    if group == "other_priorities":
        return priority != 10
    return True


def _filter_rows(rows: list, group: str) -> list:
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and _row_matches_group(r, group)]


def aggregate_rows_for_day(rows: list) -> dict:
    """Map raw pair rows to the same metrics as analysis_evaluation.build_rows_for_setting."""
    entries = len(rows)
    if entries == 0:
        return {
            "entries_count": 0,
            "cl_experience_avg": 0.0,
            "school_experience_avg": 0.0,
            "short_term_cl_experience_avg": 0.0,
            "time_to_school_avg": 0.0,
            "ma_availability_rate": 0.0,
            "car_availability_ratio": 0.0,
            "qualifications_met_rate": 0.0,
            "availability_gap_non_positive_rate": 0.0,
            "availability_gap_positive_rate": 0.0,
            "availability_gap_positive_average": 0.0,
            "priority_avg": 0.0,
        }

    cl_sum = sum(_to_float(r.get("cl_experience")) for r in rows)
    school_sum = sum(_to_float(r.get("school_experience")) for r in rows)
    st_sum = sum(_to_float(r.get("short_term_cl_experience")) for r in rows)

    tts_values = [
        _to_float(r.get("timeToSchool"))
        for r in rows
        if r.get("timeToSchool") is not None
    ]
    tts_avg = sum(tts_values) / len(tts_values) if tts_values else 0.0

    ma_true = sum(1 for r in rows if r.get("ma_availability") is True)
    mob_true = sum(1 for r in rows if r.get("mobility") is True)
    qual_true = sum(1 for r in rows if r.get("qualifications_met") is True)

    gaps = [_to_float(r.get("availability_gap")) for r in rows]
    pos_gaps = [g for g in gaps if g > 0]
    nonpos_gaps = sum(1 for g in gaps if g <= 0)

    pri_vals = [r.get("priority") for r in rows if r.get("priority") is not None]
    priority_avg = sum(float(p) for p in pri_vals) / len(pri_vals) if pri_vals else 0.0

    return {
        "entries_count": entries,
        "cl_experience_avg": cl_sum / entries,
        "school_experience_avg": school_sum / entries,
        "short_term_cl_experience_avg": st_sum / entries,
        "time_to_school_avg": tts_avg,
        "ma_availability_rate": _safe_div(ma_true, entries),
        "car_availability_ratio": _safe_div(mob_true, entries),
        "qualifications_met_rate": _safe_div(qual_true, entries),
        "availability_gap_non_positive_rate": _safe_div(nonpos_gaps, entries),
        "availability_gap_positive_rate": _safe_div(len(pos_gaps), entries),
        "availability_gap_positive_average": (
            sum(pos_gaps) / len(pos_gaps) if pos_gaps else 0.0
        ),
        "priority_avg": priority_avg,
    }


def build_daily_rows(run_data: dict, group: str, setting: str) -> list:
    rows_out = []
    for date_str in sorted(run_data.keys(), key=_parse_date):
        day = run_data.get(date_str)
        if not isinstance(day, dict):
            continue
        raw = day.get(setting, [])
        filtered = _filter_rows(raw, group)
        stats = aggregate_rows_for_day(filtered)
        rows_out.append({"date": date_str, **stats})
    return rows_out


def _mean(rows: list, key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r.get(key, 0.0)) for r in rows) / len(rows)


def print_compact_table(rows: list, setting: str, group: str) -> None:
    print(f"\nCompact daily table for '{setting}' ({group}):")
    widths = {col: len(col) for col in TABLE_COLUMNS}
    formatted_rows = []
    for row in rows:
        formatted = {}
        for col in TABLE_COLUMNS:
            value = row[col]
            if isinstance(value, float):
                cell = f"{value:.3f}"
            else:
                cell = str(value)
            formatted[col] = cell
            widths[col] = max(widths[col], len(cell))
        formatted_rows.append(formatted)
    header = " | ".join(col.ljust(widths[col]) for col in TABLE_COLUMNS)
    separator = "-+-".join("-" * widths[col] for col in TABLE_COLUMNS)
    print(header)
    print(separator)
    for row in formatted_rows:
        print(" | ".join(row[col].ljust(widths[col]) for col in TABLE_COLUMNS))


def _nonempty_daily_values(daily_rows: list, key: str) -> list:
    """One value per calendar day with at least one assignment; excludes empty days."""
    out = []
    for r in daily_rows:
        if int(r.get("entries_count") or 0) <= 0:
            continue
        out.append(float(r[key]))
    return out


def _pair_boxplot(
    ax,
    labels_values: list,
    rec_values: list,
    title: str,
    ylabel: Optional[str] = None,
    ylim: Optional[tuple] = None,
) -> None:
    datasets = []
    tick_labels = []
    positions = []

    if labels_values:
        datasets.append(labels_values)
        tick_labels.append("labels")
        positions.append(1)
    if rec_values:
        datasets.append(rec_values)
        tick_labels.append("recommendations")
        positions.append(2)

    if not datasets:
        ax.text(
            0.5,
            0.5,
            "no non-empty days",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_title(title)
        return

    bp = ax.boxplot(
        datasets,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
    )
    colors = ("#a8dadc", "#f4a261")
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)


def print_insights(labels_rows: list, rec_rows: list, group: str) -> None:
    print(f"\nMost relevant insights for '{group}':")
    print(
        f"- Days analysed: {len(labels_rows)} | Mean entries: "
        f"labels={_mean(labels_rows, 'entries_count'):.2f}, "
        f"recommendations={_mean(rec_rows, 'entries_count'):.2f}"
    )
    print(
        f"- Mean cl experience: labels={_mean(labels_rows, 'cl_experience_avg'):.3f}, "
        f"recommendations={_mean(rec_rows, 'cl_experience_avg'):.3f}"
    )
    print(
        f"- Mean MA availability rate: labels={_mean(labels_rows, 'ma_availability_rate'):.3f}, "
        f"recommendations={_mean(rec_rows, 'ma_availability_rate'):.3f}"
    )
    print(
        f"- Mean car availability ratio: labels={_mean(labels_rows, 'car_availability_ratio'):.3f}, "
        f"recommendations={_mean(rec_rows, 'car_availability_ratio'):.3f}"
    )
    print(
        f"- Mean qualifications met rate: labels={_mean(labels_rows, 'qualifications_met_rate'):.3f}, "
        f"recommendations={_mean(rec_rows, 'qualifications_met_rate'):.3f}"
    )
    print(
        f"- Mean time to school: labels={_mean(labels_rows, 'time_to_school_avg'):.3f}, "
        f"recommendations={_mean(rec_rows, 'time_to_school_avg'):.3f}"
    )


def create_dashboard_matplotlib(
    labels_rows: list, rec_rows: list, group: str, output_file: Path
) -> None:
    """
    Boxplots compare distributions of **daily** aggregates (one sample per
    non-empty day). Spread across days reflects day-to-day variability.
    """
    l_qual = _nonempty_daily_values(labels_rows, "qualifications_met_rate")
    r_qual = _nonempty_daily_values(rec_rows, "qualifications_met_rate")
    l_gap0 = _nonempty_daily_values(labels_rows, "availability_gap_non_positive_rate")
    r_gap0 = _nonempty_daily_values(rec_rows, "availability_gap_non_positive_rate")
    l_cl = _nonempty_daily_values(labels_rows, "cl_experience_avg")
    r_cl = _nonempty_daily_values(rec_rows, "cl_experience_avg")
    l_sch = _nonempty_daily_values(labels_rows, "school_experience_avg")
    r_sch = _nonempty_daily_values(rec_rows, "school_experience_avg")
    l_st = _nonempty_daily_values(labels_rows, "short_term_cl_experience_avg")
    r_st = _nonempty_daily_values(rec_rows, "short_term_cl_experience_avg")
    l_n = _nonempty_daily_values(labels_rows, "entries_count")
    r_n = _nonempty_daily_values(rec_rows, "entries_count")
    l_ma = _nonempty_daily_values(labels_rows, "ma_availability_rate")
    r_ma = _nonempty_daily_values(rec_rows, "ma_availability_rate")
    l_tts = _nonempty_daily_values(labels_rows, "time_to_school_avg")
    r_tts = _nonempty_daily_values(rec_rows, "time_to_school_avg")
    l_pri = _nonempty_daily_values(labels_rows, "priority_avg")
    r_pri = _nonempty_daily_values(rec_rows, "priority_avg")

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(
        f"Daily distributions: labels vs recommendations — {output_file.stem} ({group})\n"
        "(each point is one non-empty day)",
        fontsize=13,
        fontweight="bold",
    )

    ratio_ylim = (-0.02, 1.02)

    _pair_boxplot(
        axes[0, 0],
        l_qual,
        r_qual,
        "Qualifications met (daily ratio)",
        "ratio",
        ratio_ylim,
    )
    _pair_boxplot(
        axes[0, 1],
        l_gap0,
        r_gap0,
        "Availability gap ≤ 0 (daily ratio)",
        "ratio",
        ratio_ylim,
    )
    _pair_boxplot(axes[0, 2], l_cl, r_cl, "Client experience (daily mean)", "mean")
    _pair_boxplot(axes[1, 0], l_sch, r_sch, "School experience (daily mean)", "mean")
    _pair_boxplot(
        axes[1, 1],
        l_st,
        r_st,
        "Short-term client experience (daily mean)",
        "mean",
    )
    _pair_boxplot(
        axes[1, 2],
        l_n,
        r_n,
        "Clients assigned (count per day)",
        "count",
    )
    _pair_boxplot(
        axes[2, 0],
        l_ma,
        r_ma,
        "MA availability true (daily ratio)",
        "ratio",
        ratio_ylim,
    )
    _pair_boxplot(
        axes[2, 1],
        l_tts,
        r_tts,
        "Time to school (daily mean)",
        "seconds (mean)",
    )
    _pair_boxplot(axes[2, 2], l_pri, r_pri, "Priority (daily mean)", "mean")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_data = load_run(args.input)

    labels_rows = build_daily_rows(run_data, args.group, "labels")
    rec_rows = build_daily_rows(run_data, args.group, "recommendations")

    if not labels_rows and not rec_rows:
        print("No dated entries found in the analysis run file.")
        return

    print_compact_table(labels_rows, "labels", args.group)
    print_compact_table(rec_rows, "recommendations", args.group)
    print_insights(labels_rows, rec_rows, args.group)

    stem = args.input.stem
    dashboard_png = args.output_dir / f"{stem}_dashboard_{args.group}.png"
    create_dashboard_matplotlib(labels_rows, rec_rows, args.group, dashboard_png)
    print(f"\nSaved dashboard to: {dashboard_png}")


if __name__ == "__main__":
    main()
