import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


GROUP_CHOICES = ["priority_10", "other_priorities", "all_priorities"]
SETTING_CHOICES = ["labels", "recommendations"]

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
    "availability_gap_positive_rate",
    "availability_gap_positive_average",
]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator in (0, None):
        return 0.0
    return float(numerator) / float(denominator)


def _to_float(value) -> float:
    if value is None:
        return 0.0
    return float(value)


def _to_ratio_from_percent(value) -> float:
    if value is None:
        return 0.0
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def _extract_stats(day_entry: dict, setting: str, group: str) -> dict:
    return day_entry.get(setting, {}).get("stats", {}).get(group, {})


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def load_comparison_data(json_path: Path) -> list:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_rows_for_setting(comparison_data: list, group: str, setting: str) -> list:
    rows = []

    for day_entry in comparison_data:
        day = day_entry.get("date")
        stats = _extract_stats(day_entry, setting, group)
        entries = int(stats.get("entries_count", 0) or 0)

        row = {
            "date": day,
            "entries_count": entries,
            "cl_experience_avg": _to_float(
                stats.get("experience_average", {}).get("cl_experience")
            ),
            "school_experience_avg": _to_float(
                stats.get("experience_average", {}).get("school_experience")
            ),
            "short_term_cl_experience_avg": _to_float(
                stats.get("experience_average", {}).get("short_term_cl_experience")
            ),
            "time_to_school_avg": _to_float(stats.get("average_time_to_school")),
            "ma_availability_rate": _safe_div(
                stats.get("ma_availability_true_count", 0), entries
            ),
            "car_availability_ratio": _to_ratio_from_percent(
                stats.get("mobility_percentage")
            ),
            "qualifications_met_rate": _safe_div(
                stats.get("qualifications_met_true_count", 0), entries
            ),
            "availability_gap_positive_rate": _safe_div(
                stats.get("availability_gap_positive_count", 0), entries
            ),
            "availability_gap_positive_average": _to_float(
                stats.get("availability_gap_positive_average")
            ),
        }
        rows.append(row)

    rows.sort(key=lambda x: _parse_date(x["date"]))
    return rows


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


def write_csv(rows: list, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def create_dashboard_matplotlib(
    labels_rows: list, rec_rows: list, group: str, output_file: Path
) -> None:
    dates = [_parse_date(row["date"]) for row in labels_rows]

    labels_entries = [row["entries_count"] for row in labels_rows]
    rec_entries = [row["entries_count"] for row in rec_rows]

    labels_cl_exp = [row["cl_experience_avg"] for row in labels_rows]
    rec_cl_exp = [row["cl_experience_avg"] for row in rec_rows]

    labels_ma_rate = [row["ma_availability_rate"] for row in labels_rows]
    rec_ma_rate = [row["ma_availability_rate"] for row in rec_rows]

    labels_car_ratio = [row["car_availability_ratio"] for row in labels_rows]
    rec_car_ratio = [row["car_availability_ratio"] for row in rec_rows]

    labels_qual_rate = [row["qualifications_met_rate"] for row in labels_rows]
    rec_qual_rate = [row["qualifications_met_rate"] for row in rec_rows]

    labels_time_to_school = [row["time_to_school_avg"] for row in labels_rows]
    rec_time_to_school = [row["time_to_school_avg"] for row in rec_rows]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(
        f"Labels vs Recommendations Dashboard ({group})",
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].plot(dates, labels_entries, marker="o", label="labels")
    axes[0, 0].plot(dates, rec_entries, marker="o", label="recommendations")
    axes[0, 0].set_title("Entries count per day")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(dates, labels_cl_exp, marker="o", label="labels")
    axes[0, 1].plot(dates, rec_cl_exp, marker="o", label="recommendations")
    axes[0, 1].set_title("Average cl_experience per day")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(dates, labels_ma_rate, marker="o", label="labels")
    axes[1, 0].plot(dates, rec_ma_rate, marker="o", label="recommendations")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("MA availability rate per day")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(dates, labels_car_ratio, marker="o", label="labels")
    axes[1, 1].plot(dates, rec_car_ratio, marker="o", label="recommendations")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Car availability ratio per day")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    axes[2, 0].plot(dates, labels_qual_rate, marker="o", label="labels")
    axes[2, 0].plot(dates, rec_qual_rate, marker="o", label="recommendations")
    axes[2, 0].set_ylim(0, 1.05)
    axes[2, 0].set_title("Qualifications met rate per day")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)

    axes[2, 1].plot(dates, labels_time_to_school, marker="o", label="labels")
    axes[2, 1].plot(dates, rec_time_to_school, marker="o", label="recommendations")
    axes[2, 1].set_title("Average time to school per day")
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.3)

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def create_dashboard_html(labels_rows: list, rec_rows: list, group: str, output_file: Path) -> None:
    def _row_map(rows):
        return {r["date"]: r for r in rows}

    labels_map = _row_map(labels_rows)
    rec_map = _row_map(rec_rows)
    dates = sorted(labels_map.keys(), key=_parse_date)

    html_rows = []
    for day in dates:
        l = labels_map[day]
        r = rec_map.get(day, {})
        html_rows.append(
            "<tr>"
            f"<td>{day}</td>"
            f"<td>{l.get('entries_count', 0)}</td>"
            f"<td>{r.get('entries_count', 0)}</td>"
            f"<td>{l.get('cl_experience_avg', 0):.3f}</td>"
            f"<td>{r.get('cl_experience_avg', 0):.3f}</td>"
            f"<td>{l.get('ma_availability_rate', 0):.3f}</td>"
            f"<td>{r.get('ma_availability_rate', 0):.3f}</td>"
            f"<td>{l.get('car_availability_ratio', 0):.3f}</td>"
            f"<td>{r.get('car_availability_ratio', 0):.3f}</td>"
            f"<td>{l.get('qualifications_met_rate', 0):.3f}</td>"
            f"<td>{r.get('qualifications_met_rate', 0):.3f}</td>"
            f"<td>{l.get('time_to_school_avg', 0):.3f}</td>"
            f"<td>{r.get('time_to_school_avg', 0):.3f}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Labels vs Recommendations Dashboard - {group}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: center; font-size: 12px; }}
    th {{ background: #f0f0f0; }}
  </style>
</head>
<body>
  <h1>Labels vs Recommendations Dashboard ({group})</h1>
  <p>Matplotlib is not available in this Python environment, so this HTML comparison table was generated instead.</p>
  <table>
    <thead>
      <tr>
        <th>Date</th>
        <th>Label entries</th><th>Rec entries</th>
        <th>Label cl_exp avg</th><th>Rec cl_exp avg</th>
        <th>Label MA avail rate</th><th>Rec MA avail rate</th>
        <th>Label car ratio</th><th>Rec car ratio</th>
        <th>Label qual rate</th><th>Rec qual rate</th>
        <th>Label time to school</th><th>Rec time to school</th>
      </tr>
    </thead>
    <tbody>
      {"".join(html_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate label and recommendation metrics from comparison_analysis.json "
            "for a selected priority group."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/comparison_analysis.json"),
        help="Path to comparison analysis JSON file.",
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
        default=Path("data"),
        help="Directory where tables and dashboard are saved.",
    )
    parser.add_argument(
        "--only",
        choices=SETTING_CHOICES,
        default=None,
        help="If set, only prints/saves one setting table (labels or recommendations).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_data = load_comparison_data(args.input)

    labels_rows = build_rows_for_setting(comparison_data, args.group, "labels")
    rec_rows = build_rows_for_setting(comparison_data, args.group, "recommendations")

    if not labels_rows and not rec_rows:
        print("No data found for evaluation.")
        return

    labels_csv = args.output_dir / f"comparison_table_labels_{args.group}.csv"
    rec_csv = args.output_dir / f"comparison_table_recommendations_{args.group}.csv"
    dashboard_png = args.output_dir / f"comparison_dashboard_{args.group}.png"
    dashboard_html = args.output_dir / f"comparison_dashboard_{args.group}.html"

    if args.only in (None, "labels"):
        write_csv(labels_rows, labels_csv)
        print_compact_table(labels_rows, "labels", args.group)
    if args.only in (None, "recommendations"):
        write_csv(rec_rows, rec_csv)
        print_compact_table(rec_rows, "recommendations", args.group)

    print_insights(labels_rows, rec_rows, args.group)
    if HAS_MATPLOTLIB:
        create_dashboard_matplotlib(labels_rows, rec_rows, args.group, dashboard_png)
    else:
        create_dashboard_html(labels_rows, rec_rows, args.group, dashboard_html)

    if args.only in (None, "labels"):
        print(f"\nSaved labels table to: {labels_csv}")
    if args.only in (None, "recommendations"):
        print(f"Saved recommendations table to: {rec_csv}")
    if HAS_MATPLOTLIB:
        print(f"Saved dashboard to: {dashboard_png}")
    else:
        print(
            "Saved dashboard to: "
            f"{dashboard_html} (matplotlib not available in this Python environment)"
        )


if __name__ == "__main__":
    main()
