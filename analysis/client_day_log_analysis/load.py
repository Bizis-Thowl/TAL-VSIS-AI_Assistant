from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Bevorzugte Reihenfolge fuer Plots; weitere Varianten werden dynamisch ergaenzt.
PREFERRED_VARIANT_ORDER = (
    "human-baseline",
    "baseline",
    "default",
    "default-low-exp",
    "default-high-dist",
    "no-dist",
    "no-exp",
)

_RUN_FILE_PATTERN = re.compile(
    r"^client_day_log_(?P<run_id>\d+)_(?P<variant>.+)\.csv$"
)


def sort_variants(variants: list[str] | set[str]) -> list[str]:
    present = list(variants)
    return [v for v in PREFERRED_VARIANT_ORDER if v in present] + sorted(
        v for v in present if v not in PREFERRED_VARIANT_ORDER
    )


def _normalize_experience_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ältere Logs nutzen has_*-Booleans statt Zähler."""
    rename_map = {
        "has_client_experience": "client_experience",
        "has_short_term_client_experience": "short_term_client_experience",
        "has_school_experience": "school_experience",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old].astype(int)

    for col in (
        "client_experience",
        "short_term_client_experience",
        "school_experience",
    ):
        if col not in df.columns:
            df[col] = 0

    return df


def discover_run_ids(log_dir: Path) -> list[str]:
    run_ids: set[str] = set()
    for path in log_dir.glob("client_day_log_*.csv"):
        match = _RUN_FILE_PATTERN.match(path.name)
        if match:
            run_ids.add(match.group("run_id"))
    return sorted(run_ids)


def discover_variants_for_run(log_dir: Path, run_id: str) -> list[str]:
    variants: list[str] = []
    for path in log_dir.glob(f"client_day_log_{run_id}_*.csv"):
        match = _RUN_FILE_PATTERN.match(path.name)
        if match and match.group("run_id") == run_id:
            variants.append(match.group("variant"))
    discovered = sort_variants(variants)
    if not discovered:
        raise FileNotFoundError(
            f"Keine client_day_log_{run_id}_<variante>.csv in {log_dir}"
        )
    return discovered


def load_run(log_dir: Path, run_id: str) -> pd.DataFrame:
    frames = []
    for variant in discover_variants_for_run(log_dir, run_id):
        path = log_dir / f"client_day_log_{run_id}_{variant}.csv"
        part = pd.read_csv(path)
        part["weight_variant"] = variant
        frames.append(_normalize_experience_columns(part))

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["versorgt"] = df["versorgt"].astype(bool)
    df["assigned"] = df["versorgt"]
    df["locally_eligible"] = df["eligible_ma_count"] > 0
    return df


def filter_by_priority(
    df: pd.DataFrame, priorities: list[int] | None
) -> pd.DataFrame:
    if not priorities:
        return df
    if "priority" not in df.columns:
        raise ValueError("Spalte 'priority' fehlt in den Log-Daten.")
    filtered = df[df["priority"].isin(priorities)].copy()
    if filtered.empty:
        available = sorted(df["priority"].dropna().unique().tolist())
        raise ValueError(
            f"Keine Zeilen fuer Prioritaet(en) {priorities}. "
            f"Verfuegbar: {available}"
        )
    return filtered


def priority_output_suffix(priorities: list[int] | None) -> str | None:
    if not priorities:
        return None
    return "priority_" + "_".join(str(p) for p in sorted(set(priorities)))


def load_all_runs(log_dir: Path) -> pd.DataFrame:
    run_ids = discover_run_ids(log_dir)
    if not run_ids:
        raise FileNotFoundError(f"Keine client_day_log_*.csv in {log_dir}")
    frames = []
    for run_id in run_ids:
        part = load_run(log_dir, run_id)
        part["run_id"] = run_id
        frames.append(part)
    return pd.concat(frames, ignore_index=True)
