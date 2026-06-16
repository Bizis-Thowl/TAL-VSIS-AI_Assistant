from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


GROUP_CHOICES = ["priority_10", "other_priorities", "all_priorities"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an analysis run (analysis_run_XXXX.json), count per-client "
            "assignment days for labels vs recommendations, and plot assignment "
            "ratio distributions. Ratio is assignment days divided by days the "
            "client appears in clients_available (when present in the file); "
            "otherwise run calendar days are used."
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
        help="Priority filter applied to each day's rows before counting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluations"),
        help="Directory for the saved figure.",
    )
    return parser.parse_args()


def load_run(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_date(value: str) -> datetime:
    return datetime.strptime(str(value), "%Y-%m-%d")


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


def sorted_run_dates(run_data: dict) -> list[str]:
    return sorted(run_data.keys(), key=_parse_date)


def run_has_clients_available_schema(run_data: dict) -> bool:
    """True if at least one day stores the clients_available key (new schema)."""
    for date_str in sorted_run_dates(run_data):
        day = run_data.get(date_str)
        if isinstance(day, dict) and "clients_available" in day:
            return True
    return False


def _client_id_from_row(row: dict) -> str | None:
    cid = row.get("client_id")
    if cid is None:
        cid = row.get("id")
    if cid is None:
        return None
    return str(cid)


def count_availability_days_per_client(run_data: dict, group: str) -> dict[str, int]:
    """
    For each client_id, number of distinct days the client appears in
    clients_available (after priority group filter). Matches the pool of
    open clients for that day, including unassigned clients.
    """
    counts = defaultdict(int)
    for date_str in sorted_run_dates(run_data):
        day = run_data.get(date_str)
        if not isinstance(day, dict):
            continue
        avail = day.get("clients_available")
        if not isinstance(avail, list):
            continue
        seen: set[str] = set()
        for r in avail:
            if not isinstance(r, dict) or not _row_matches_group(r, group):
                continue
            cid = _client_id_from_row(r)
            if cid is None:
                continue
            seen.add(cid)
        for cid in seen:
            counts[cid] += 1
    return dict(counts)


def count_assignment_days_per_client(
    run_data: dict, group: str, setting: str
) -> dict[str, int]:
    """
    For each client_id, number of distinct calendar days with at least one
    filtered assignment row in the given setting.
    """
    counts = defaultdict(int)
    for date_str in sorted_run_dates(run_data):
        day = run_data.get(date_str)
        if not isinstance(day, dict):
            continue
        rows = _filter_rows(day.get(setting, []), group)
        clients_this_day = set()
        for r in rows:
            cid = _client_id_from_row(r)
            if cid is None:
                continue
            clients_this_day.add(cid)
        for cid in clients_this_day:
            counts[cid] += 1
    return dict(counts)


def _ratio_maps(
    assign_days: dict[str, int],
    avail_days: dict[str, int],
    total_days: int,
    use_availability_denominator: bool,
) -> dict[str, float]:
    """Per-client ratio: assignment days / availability days (or run days fallback)."""
    out: dict[str, float] = {}
    for cid, a_count in assign_days.items():
        if a_count <= 0:
            continue
        d = avail_days.get(cid, 0) if use_availability_denominator else 0
        if use_availability_denominator and d > 0:
            out[cid] = min(1.0, float(a_count) / float(d))
        elif total_days > 0:
            out[cid] = min(1.0, float(a_count) / float(total_days))
    return out


def build_client_assignment_stats(
    run_data: dict, group: str
) -> tuple[
    int,
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, float],
    dict[str, float],
    bool,
]:
    """
    Returns:
        total_days, avail_days, label_days, rec_days, label_ratio, rec_ratio,
        use_availability_denominator (True when clients_available yields any counts).
    """
    total_days = len(sorted_run_dates(run_data))
    avail_days = count_availability_days_per_client(run_data, group)
    label_days = count_assignment_days_per_client(run_data, group, "labels")
    rec_days = count_assignment_days_per_client(run_data, group, "recommendations")

    if total_days <= 0:
        return 0, {}, {}, {}, {}, {}, False

    schema = run_has_clients_available_schema(run_data)
    use_avail = schema and bool(avail_days)

    label_ratio = _ratio_maps(label_days, avail_days, total_days, use_avail)
    rec_ratio = _ratio_maps(rec_days, avail_days, total_days, use_avail)

    return (
        total_days,
        avail_days,
        label_days,
        rec_days,
        label_ratio,
        rec_ratio,
        use_avail,
    )


def _clients_union(
    label_days: dict[str, int], rec_days: dict[str, int]
) -> list[str]:
    return sorted(set(label_days) | set(rec_days))


def print_summary(
    group: str,
    total_days: int,
    avail_days: dict[str, int],
    label_days: dict[str, int],
    rec_days: dict[str, int],
    label_ratio: dict[str, float],
    rec_ratio: dict[str, float],
    use_avail: bool,
    schema_present: bool,
) -> None:
    union = _clients_union(label_days, rec_days)
    denom_desc = (
        "assignment days / days listed in clients_available (same priority group)"
        if use_avail
        else f"assignment days / {total_days} run calendar days"
    )
    print(f"\nPer-client assignment summary ({group}):")
    print(f"- Denominator: {denom_desc}")
    print(f"- Run calendar days: {total_days}")
    if use_avail:
        print(
            f"- Client-days in clients_available (filtered): "
            f"{sum(avail_days.values())} across {len(avail_days)} clients"
        )
    elif schema_present:
        print(
            "- Note: clients_available is present but no rows matched the "
            "priority filter (or lists were empty); ratios use run calendar days."
        )
    print(f"- Clients with ≥1 label assignment day: {len(label_days)}")
    print(f"- Clients with ≥1 recommendation assignment day: {len(rec_days)}")
    print(f"- Union (any assignment in either setting): {len(union)}")
    if union:
        lr = [label_ratio.get(c, 0.0) for c in union]
        rr = [rec_ratio.get(c, 0.0) for c in union]
        print(
            f"- Mean assignment ratio (over union, zeros where no assignments): "
            f"labels={sum(lr)/len(lr):.4f}, recommendations={sum(rr)/len(rr):.4f}"
        )


def _short_id(client_id: str, n: int = 8) -> str:
    s = str(client_id)
    return s[:n] if len(s) > n else s


def create_assignment_figure(
    total_days: int,
    group: str,
    label_days: dict[str, int],
    rec_days: dict[str, int],
    label_ratio: dict[str, float],
    rec_ratio: dict[str, float],
    use_avail: bool,
    output_file: Path,
) -> None:
    union = _clients_union(label_days, rec_days)
    if total_days <= 0 or not union:
        raise ValueError("No calendar days or no client assignments to plot.")

    box_l = [label_ratio.get(c, 0.0) for c in union]
    box_r = [rec_ratio.get(c, 0.0) for c in union]

    clients_l = sorted(label_ratio.keys(), key=lambda c: (-label_ratio[c], c))
    clients_r = sorted(rec_ratio.keys(), key=lambda c: (-rec_ratio[c], c))

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.35, 1.35], hspace=0.42)
    ax_box = fig.add_subplot(gs[0, 0])
    ax_bar_l = fig.add_subplot(gs[1, 0])
    ax_bar_r = fig.add_subplot(gs[2, 0])

    stem = output_file.stem
    ratio_line = (
        "Assignment ratio = assignment days / days client appears in clients_available "
        "(same priority group)"
        if use_avail
        else f"Assignment ratio = assignment days / {total_days} run calendar days"
    )
    fig.suptitle(
        f"Client assignment over run — {stem} ({group})\n{ratio_line}",
        fontsize=13,
        fontweight="bold",
    )

    bp = ax_box.boxplot(
        [box_l, box_r],
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.65},
    )
    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels(["labels", "recommendations"])
    for patch, color in zip(bp["boxes"], ("#a8dadc", "#f4a261")):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)
    ax_box.set_ylabel("assignment ratio")
    ax_box.set_ylim(-0.02, 1.02)
    ax_box.set_title(
        "Distribution of per-client assignment ratio\n"
        f"(clients with ≥1 assignment in labels or recommendations, n={len(union)})"
    )
    ax_box.grid(axis="y", alpha=0.3)

    def _bar_ratios(
        ax,
        clients: list[str],
        ratios: dict[str, float],
        title: str,
        bar_color: str,
    ) -> None:
        if not clients:
            ax.text(0.5, 0.5, "no clients", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        xs = list(range(len(clients)))
        ys = [ratios[c] for c in clients]
        ax.bar(xs, ys, color=bar_color, alpha=0.85, edgecolor="#1d3557", linewidth=0.3)
        ax.set_xticks(xs)
        ax.set_xticklabels([_short_id(c) for c in clients], rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("assignment ratio")
        ax.set_xlabel("client id (prefix)")
        ax.set_ylim(0, max(0.05, max(ys) * 1.08) if ys else 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    _bar_ratios(
        ax_bar_l,
        clients_l,
        label_ratio,
        f"Labels — assignment ratio per client (n={len(clients_l)})",
        "#457b9d",
    )
    _bar_ratios(
        ax_bar_r,
        clients_r,
        rec_ratio,
        f"Recommendations — assignment ratio per client (n={len(clients_r)})",
        "#e76f51",
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_data = load_run(args.input)

    (
        total_days,
        avail_days,
        label_days,
        rec_days,
        label_ratio,
        rec_ratio,
        use_avail,
    ) = build_client_assignment_stats(run_data, args.group)
    schema_present = run_has_clients_available_schema(run_data)

    if total_days <= 0:
        print("No dated entries in the analysis run file.")
        return

    print_summary(
        args.group,
        total_days,
        avail_days,
        label_days,
        rec_days,
        label_ratio,
        rec_ratio,
        use_avail,
        schema_present,
    )

    union = _clients_union(label_days, rec_days)
    if not union:
        print("No client assignments found after filtering; skipping figure.")
        return

    stem = args.input.stem
    out_png = args.output_dir / f"{stem}_assignment_client_{args.group}.png"
    create_assignment_figure(
        total_days,
        args.group,
        label_days,
        rec_days,
        label_ratio,
        rec_ratio,
        use_avail,
        out_png,
    )
    print(f"\nSaved figure to: {out_png}")


if __name__ == "__main__":
    main()
