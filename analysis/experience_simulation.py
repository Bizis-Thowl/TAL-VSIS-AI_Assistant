from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import pandas as pd


def empty_experience_log() -> List[Dict]:
    return []


def _experience_maps_from_log(
    experience_log: List[Dict],
) -> tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    ma_experience_map: Dict[str, Dict[str, List[str]]] = {}
    ma_school_experience_map: Dict[str, Dict[str, List[str]]] = {}

    for entry in experience_log:
        ma_id = entry["ma"]
        ma_experience_map[ma_id] = {
            client_id: list(dates)
            for client_id, dates in entry.get("client_experience", {}).items()
        }
        ma_school_experience_map[ma_id] = {
            school_id: list(dates)
            for school_id, dates in entry.get("school_experience", {}).items()
        }

    return ma_experience_map, ma_school_experience_map


def _log_from_maps(
    ma_experience_map: Dict[str, Dict[str, List[str]]],
    ma_school_experience_map: Dict[str, Dict[str, List[str]]],
) -> List[Dict]:
    all_ma_ids = set(ma_experience_map) | set(ma_school_experience_map)
    return [
        {
            "ma": ma_id,
            "client_experience": ma_experience_map.get(ma_id, {}),
            "school_experience": ma_school_experience_map.get(ma_id, {}),
        }
        for ma_id in sorted(all_ma_ids)
    ]


def record_assignment(
    experience_log: List[Dict],
    ma_id: str,
    client_id: str,
    school_id: str,
    date_str: str,
) -> List[Dict]:
    """Append one MA-client-school assignment for a day (experience_log.json format)."""
    ma_experience_map, ma_school_experience_map = _experience_maps_from_log(experience_log)

    if ma_id not in ma_experience_map:
        ma_experience_map[ma_id] = {}
    if ma_id not in ma_school_experience_map:
        ma_school_experience_map[ma_id] = {}

    client_dates = ma_experience_map[ma_id].setdefault(client_id, [])
    if date_str not in client_dates:
        client_dates.append(date_str)

    school_dates = ma_school_experience_map[ma_id].setdefault(school_id, [])
    if date_str not in school_dates:
        school_dates.append(date_str)

    return _log_from_maps(ma_experience_map, ma_school_experience_map)


def record_optimizer_assignments(
    experience_log: List[Dict],
    assigned_pairs: List[Dict],
    clients_by_id: Dict[str, Dict],
    date_str: str,
) -> List[Dict]:
    updated_log = experience_log
    for pair in assigned_pairs:
        client = clients_by_id.get(pair["klient"])
        if client is None:
            continue
        school = client.get("schule") or {}
        school_id = school.get("id")
        if school_id is None:
            continue
        updated_log = record_assignment(
            updated_log,
            pair["ma"],
            pair["klient"],
            school_id,
            date_str,
        )
    return updated_log


def experience_logs_path(output_dir: str, run_id: str) -> str:
    return os.path.join(output_dir, f"run_{run_id}_experience_logs.json")


def save_variant_experience_logs(
    output_dir: str, run_id: str, variant_logs: Dict[str, List[Dict]]
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = experience_logs_path(output_dir, run_id)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(variant_logs, f, indent=2)
    os.replace(tmp_path, path)


def load_variant_experience_logs(
    output_dir: str, run_id: str, weight_names: List[str]
) -> Dict[str, List[Dict]] | None:
    path = experience_logs_path(output_dir, run_id)
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        stored = json.load(f)

    return {
        weight_name: deepcopy(stored.get(weight_name, empty_experience_log()))
        for weight_name in weight_names
    }


def rebuild_variant_experience_logs_from_csv(
    output_dir: str,
    run_id: str,
    weight_names: List[str],
    clients_by_id: Dict[str, Dict],
) -> Dict[str, List[Dict]]:
    variant_logs = {weight_name: empty_experience_log() for weight_name in weight_names}

    for weight_name in weight_names:
        csv_path = Path(output_dir) / f"client_day_log_{run_id}_{weight_name}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        served = df[df["versorgt"].astype(bool) & df["assigned_ma_id"].notna()]

        for date_str in sorted(served["date"].unique()):
            day_rows = served[served["date"] == date_str]
            assigned_pairs = [
                {"ma": row["assigned_ma_id"], "klient": row["client_id"]}
                for _, row in day_rows.iterrows()
            ]
            variant_logs[weight_name] = record_optimizer_assignments(
                variant_logs[weight_name],
                assigned_pairs,
                clients_by_id,
                date_str,
            )

    return variant_logs


def initialize_variant_experience_logs(
    output_dir: str,
    run_id: str,
    weight_names: List[str],
    clients_by_id: Dict[str, Dict],
) -> Dict[str, List[Dict]]:
    loaded = load_variant_experience_logs(output_dir, run_id, weight_names)
    if loaded is not None:
        return loaded

    rebuilt = rebuild_variant_experience_logs_from_csv(
        output_dir, run_id, weight_names, clients_by_id
    )
    if any(rebuilt[weight_name] for weight_name in weight_names):
        save_variant_experience_logs(output_dir, run_id, rebuilt)
        return rebuilt

    return {weight_name: empty_experience_log() for weight_name in weight_names}
