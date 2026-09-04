from __future__ import annotations

import argparse
from pathlib import Path

from analysis.client_day_log_analysis.load import discover_run_ids, priority_output_suffix
from analysis.client_day_log_analysis.pipeline import run_analysis
from analysis.client_day_log_analysis.visualize import build_plot_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analysiert client_day_log CSV-Dateien: Panel-Check, "
            "Client×Variante-Metriken, Bias-Deltas, Fairness und Plots."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("data/analysis_client_day_logs"),
        help="Verzeichnis mit client_day_log_XXXX_<variante>.csv",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Lauf-ID (z. B. 0002). Standard: höchste gefundene ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/client_day_log_evaluations"),
        help="Basisverzeichnis für CSV- und Plot-Ausgaben.",
    )
    parser.add_argument(
        "--min-locally-eligible-days",
        type=int,
        default=10,
        help="Schwellwert für Worst-Case-Filter (problem_cases).",
    )
    parser.add_argument(
        "--max-default-coverage",
        type=float,
        default=0.5,
        help="Max. Default-Coverage für problem_cases.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Keine PNG-Plots erzeugen.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Nur Klient-Tage mit dieser Prioritaet auswerten "
            "(z. B. --priority 10 oder --priority 10 20)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ids = discover_run_ids(args.log_dir)
    if not run_ids:
        raise SystemExit(f"Keine Logs in {args.log_dir}")

    run_id = args.run_id or run_ids[-1]
    if run_id not in run_ids:
        raise SystemExit(
            f"Run-ID {run_id} nicht gefunden. Verfügbar: {', '.join(run_ids)}"
        )

    output_dir = args.output_dir / f"run_{run_id}"
    priority_suffix = priority_output_suffix(args.priority)
    if priority_suffix:
        output_dir = output_dir / priority_suffix

    result = run_analysis(
        args.log_dir,
        run_id,
        output_dir,
        priorities=args.priority,
        min_locally_eligible_days=args.min_locally_eligible_days,
        max_default_coverage=args.max_default_coverage,
        save_plots=not args.no_plots,
    )

    filter_note = (
        f" (Prioritaet {', '.join(str(p) for p in args.priority)})"
        if args.priority
        else ""
    )
    print(f"Analyse fuer Run {result.run_id}{filter_note} -> {output_dir}")
    print("\n--- Panel-Check ---")
    for key, value in result.validation.items():
        print(f"{key}: {value}")

    print("\n--- variant_summary ---")
    print(result.variant_summary.to_string(index=False))

    plot_tables = build_plot_tables(
        fairness_lower_tail=result.fairness_lower_tail,
        fairness_inequality=result.fairness_inequality,
        unassigned_summary=result.unassigned_share_summary,
        objective_summary=result.objective_summary,
    )
    plot_titles = {
        "09_fairness_lower_tail": "Plot 09: Lower-tail Coverage je Variante",
        "10_fairness_inequality": "Plot 10: Ungleichverteilung je Variante",
        "11_gini_by_threshold": "Plot 11: Gini-Robustheit nach Eignungsschwelle",
        "14_coverage_and_soft_objective": (
            "Plot 14: Unversorgungsrate und Soft-Objective je Variante"
        ),
    }
    for key, table in plot_tables.items():
        print(f"\n--- {plot_titles[key]} ---")
        print(table.to_string(index=False))

    print("\n--- experience_validation ---")
    print(
        f"interpretable={result.experience_validation.get('interpretable')} "
        f"(variance={result.experience_validation.get('has_experience_variance')}, "
        f"no-exp differs={result.experience_validation.get('no_exp_differs_from_default')})"
    )

    if len(result.problem_cases) > 0:
        print("\n--- problem_cases (Top 5) ---")
        print(result.problem_cases.head(5).to_string(index=False))
    else:
        print("\n--- problem_cases: keine Treffer ---")

    if result.plot_paths:
        print("\n--- Plots ---")
        for path in result.plot_paths:
            print(path)


if __name__ == "__main__":
    main()
