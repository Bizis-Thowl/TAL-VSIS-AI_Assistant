"""Shared helpers for client-day log generation scripts."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

HUMAN_BASELINE_VARIANT = "human-baseline"


def merge_consecutive_free_ma_records(records):
    grouped_records = {}

    for record in records:
        ma_id = record.get("mafrei", {}).get("id")
        if ma_id is None:
            continue

        grouped_records.setdefault(ma_id, []).append(record)

    merged_records = []

    for ma_id_records in grouped_records.values():
        sorted_records = sorted(
            ma_id_records,
            key=lambda item: (
                datetime.strptime(item["startdatum"], "%Y-%m-%d"),
                datetime.strptime(item["enddatum"], "%Y-%m-%d"),
            ),
        )

        current_record = dict(sorted_records[0])
        current_start = datetime.strptime(current_record["startdatum"], "%Y-%m-%d")
        current_end = datetime.strptime(current_record["enddatum"], "%Y-%m-%d")

        for next_record in sorted_records[1:]:
            next_start = datetime.strptime(next_record["startdatum"], "%Y-%m-%d")
            next_end = datetime.strptime(next_record["enddatum"], "%Y-%m-%d")

            if next_start <= current_end + timedelta(days=1):
                current_end = max(current_end, next_end)
                current_record["enddatum"] = current_end.strftime("%Y-%m-%d")
                continue

            merged_records.append(current_record)
            current_record = dict(next_record)
            current_start = next_start
            current_end = next_end

        current_record["startdatum"] = current_start.strftime("%Y-%m-%d")
        current_record["enddatum"] = current_end.strftime("%Y-%m-%d")
        merged_records.append(current_record)

    return merged_records


def run_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, "run_state.json")


def variant_date_key(date_str: str, weight_name: str) -> str:
    return f"{date_str}:{weight_name}"


def load_existing_run_state(output_dir: str) -> dict:
    state_path = run_state_path(output_dir)
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"No run_state.json in {output_dir}. Run analysis.py first."
        )
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)


def save_run_state(output_dir: str, state: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    state_path = run_state_path(output_dir)
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


def is_variant_date_done(state: dict, date_str: str, weight_name: str) -> bool:
    return variant_date_key(date_str, weight_name) in state.get("completed", [])


def append_log_rows(
    output_dir: str, run_id: str, weight_name: str, rows: list
) -> None:
    from analysis.client_day_logging import rows_to_dataframe

    log_path = os.path.join(output_dir, f"client_day_log_{run_id}_{weight_name}.csv")
    log_df = rows_to_dataframe(rows)
    log_df.to_csv(
        log_path,
        mode="a",
        header=not os.path.exists(log_path),
        index=False,
        encoding="utf-8",
    )


def human_assignments_from_vertretungen(vertretungen) -> list[dict]:
    """Actual Missy assignments (labels): mavertretend -> klientzubegleiten."""
    assigned_records = [
        record
        for record in vertretungen
        if record.get("klientzubegleiten") is not None
    ]
    return [
        {
            "ma": record["mavertretend"]["id"],
            "klient": record["klientzubegleiten"]["id"],
        }
        for record in assigned_records
        if record.get("mavertretend") is not None
        and record.get("klientzubegleiten") is not None
    ]


def prepare_day_context(vertretungen, mas):
    """Build open-client and free-MA context shared by optimizer and human logs."""
    assigned_records = [
        record
        for record in vertretungen
        if record.get("klientzubegleiten") is not None
    ]
    assignments = human_assignments_from_vertretungen(vertretungen)

    absent_ma_records = [
        record
        for record in vertretungen
        if record.get("maabwesend") is not None
        and record.get("klientzubegleiten") is None
    ]
    free_and_assigned_ma_records = [
        {
            **record,
            "mafrei": {"id": record.get("mavertretend").get("id")},
        }
        for record in assigned_records
        if record.get("mavertretend") is not None
    ]

    free_ma_records = merge_consecutive_free_ma_records(
        [record for record in vertretungen if record.get("mafrei") is not None]
        + free_and_assigned_ma_records
    )

    absent_ma_ids = [record["maabwesend"]["id"] for record in absent_ma_records]
    absent_mas = [ma for ma in mas if ma.get("id") in absent_ma_ids]
    all_open_clients = {
        ma.get("id"): client_id.get("id")
        for ma in absent_mas
        for client_id in ma["aktiveklientinnen"]
    }
    all_open_clients = {
        **all_open_clients,
        **{pair["ma"]: pair["klient"] for pair in assignments},
    }

    free_mas = [
        {
            "id": record["mafrei"]["id"],
            "until": datetime.strptime(record["enddatum"], "%Y-%m-%d"),
        }
        for record in free_ma_records
    ]
    free_ma_ids = [item["id"] for item in free_mas]

    open_clients = [
        {
            "id": all_open_clients.get(record["maabwesend"]["id"]),
            "until": datetime.strptime(record.get("enddatum", None), "%Y-%m-%d"),
            "ma_blacklist": record.get("mavorschlagblacklist", []),
        }
        for record in absent_ma_records
        if all_open_clients.get(record["maabwesend"]["id"]) is not None
    ]

    open_client_ids = list({item["id"] for item in open_clients})
    free_ma_ids = list(set(free_ma_ids + [pair["ma"] for pair in assignments]))

    return {
        "assignments": assignments,
        "open_client_ids": open_client_ids,
        "free_ma_ids": free_ma_ids,
        "open_clients": open_clients,
        "free_mas": free_mas,
    }
