import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False


GROUP_CHOICES = ["priority_10", "other_priorities", "all_priorities"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate client assignment/non-assignment counts from "
            "comparison_client_assignment_tracking.json for a selected priority group."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/comparison_client_assignment_tracking.json"),
        help="Path to client assignment tracking JSON file.",
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
        help="Directory where dashboard output is saved.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many clients to include in the ratio comparison chart.",
    )
    return parser.parse_args()


def load_tracking_data(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_div(numerator: float, denominator: float):
    if denominator in (0, None):
        return None
    return float(numerator) / float(denominator)


def index_by_client(rows: list) -> dict:
    return {str(row.get("client_id")): row for row in rows}


def build_recommendations_rows(tracking_data: dict, group: str) -> list:
    label_rows = tracking_data.get("labels", {}).get(group, [])
    recommendation_rows = tracking_data.get("recommendations", {}).get(group, [])
    label_map = index_by_client(label_rows)
    recommendation_map = index_by_client(recommendation_rows)

    client_ids = sorted(set(label_map.keys()) | set(recommendation_map.keys()))
    rows = []
    for client_id in client_ids:
        label_row = label_map.get(client_id, {})
        recommendation_row = recommendation_map.get(client_id, {})

        labels_percentage = float(label_row.get("assignment_percentage", 0) or 0.0)
        recommendations_percentage = float(
            recommendation_row.get("assignment_percentage", 0) or 0.0
        )

        rows.append(
            {
                "client_id": str(client_id),
                "labels_assignment_percentage": labels_percentage,
                "recommendations_assignment_percentage": recommendations_percentage,
            }
        )

    rows.sort(
        key=lambda item: (
            item["recommendations_assignment_percentage"],
            item["labels_assignment_percentage"],
        ),
        reverse=True,
    )
    return rows


def print_insights(rows: list, group: str) -> None:
    if not rows:
        print(f"No client rows found for group '{group}'.")
        return

    n = len(rows)
    labels_percentage_avg = sum(row["labels_assignment_percentage"] for row in rows) / n
    recommendations_percentage_avg = (
        sum(row["recommendations_assignment_percentage"] for row in rows) / n
    )
    max_row = rows[0]

    print(f"\nAssignment percentage insights for '{group}':")
    print(f"- Compared clients: {n}")
    print(
        f"- Average labels assignment percentage: {labels_percentage_avg:.3f}%"
    )
    print(
        f"- Average recommendations assignment percentage: "
        f"{recommendations_percentage_avg:.3f}%"
    )
    print(
        "- Highest recommendation assignment percentage client: "
        f"{max_row['client_id']} ({max_row['recommendations_assignment_percentage']:.3f}%)"
    )


def create_dashboard_matplotlib(rows: list, group: str, top_n: int, output_file: Path):
    top_rows = rows[: max(top_n, 1)]

    client_labels = [row["client_id"][:8] for row in top_rows]
    labels_percentages = [row["labels_assignment_percentage"] for row in top_rows]
    recommendations_percentages = [
        row["recommendations_assignment_percentage"] for row in top_rows
    ]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = list(range(len(top_rows)))
    width = 0.4
    ax.bar(
        [i - width / 2 for i in x],
        labels_percentages,
        width=width,
        label="labels",
    )
    ax.bar(
        [i + width / 2 for i in x],
        recommendations_percentages,
        width=width,
        label="recommendations",
    )
    ax.set_title(
        f"Assignment Percentage per Client ({group})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Client")
    ax.set_ylabel("Assignment percentage")
    ax.set_ylim(0, 105)
    ax.set_xticks(list(x))
    ax.set_xticklabels(client_labels, rotation=60, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def create_dashboard_html(rows: list, group: str, top_n: int, output_file: Path):
    top_rows = rows[: max(top_n, 1)]

    html_rows = []
    for row in top_rows:
        html_rows.append(
            "<tr>"
            f"<td>{row['client_id']}</td>"
            f"<td>{row['labels_assignment_percentage']:.3f}%</td>"
            f"<td>{row['recommendations_assignment_percentage']:.3f}%</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Assignment Percentage Dashboard - {group}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: center; font-size: 12px; }}
    th {{ background: #f0f0f0; }}
  </style>
</head>
<body>
  <h1>Assignment Percentage per Client ({group})</h1>
  <p>Matplotlib is not available in this Python environment, so this HTML table was generated instead.</p>
  <p>Showing top {len(top_rows)} clients by recommendation assignment percentage (descending).</p>
  <table>
    <thead>
      <tr>
        <th>Client ID</th>
        <th>Labels Assignment Percentage</th>
        <th>Recommendations Assignment Percentage</th>
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


def main() -> None:
    args = parse_args()
    tracking_data = load_tracking_data(args.input)
    rows = build_recommendations_rows(tracking_data, args.group)

    if not rows:
        print(f"No data found for group '{args.group}'.")
        return

    print_insights(rows, args.group)

    dashboard_png = (
        args.output_dir / f"comparison_assigned_ratio_labels_vs_recommendations_{args.group}.png"
    )
    dashboard_html = (
        args.output_dir / f"comparison_assigned_ratio_labels_vs_recommendations_{args.group}.html"
    )

    if HAS_MATPLOTLIB:
        create_dashboard_matplotlib(rows, args.group, args.top_n, dashboard_png)
        print(f"Saved dashboard to: {dashboard_png}")
    else:
        create_dashboard_html(rows, args.group, args.top_n, dashboard_html)
        print(
            "Saved dashboard to: "
            f"{dashboard_html} (matplotlib not available in this Python environment)"
        )


if __name__ == "__main__":
    main()
