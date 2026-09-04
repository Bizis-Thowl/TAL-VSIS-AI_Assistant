"""Generate client-day logs for the human baseline (Missy labels / assignments)."""

from __future__ import annotations

import json
import os

import pandas as pd
from dotenv import load_dotenv

from analysis.client_day_logging import build_human_baseline_client_day_log_rows
from analysis.experience_simulation import (
    empty_experience_log,
    experience_logs_path,
    record_optimizer_assignments,
    save_variant_experience_logs,
)
from analysis.log_run_shared import (
    HUMAN_BASELINE_VARIANT,
    append_log_rows,
    is_variant_date_done,
    load_existing_run_state,
    prepare_day_context,
    save_run_state,
    variant_date_key,
)
from config import base_url_ai, base_url_missy
from data_processing.data_processor import DataProcessor
from fetching.missy_fetching import (
    get_clients,
    get_distances,
    get_mas,
    get_prio_assignments,
    get_schools,
    get_vertretungen,
)

load_dotenv(override=True)

LOG_OUTPUT_DIR = "data/analysis_client_day_logs"


def _load_request_info():
    request_specs = json.loads(os.getenv("REQUEST_INFO"))
    return [
        {
            "user": spec["user"],
            "pw": spec["pw"],
            "url": base_url_missy.format(domain=spec["domain"]),
            "url_ai": base_url_ai.format(system_spec=spec["domain"]),
        }
        for spec in request_specs
    ]


def _load_human_experience_log(output_dir: str, run_id: str) -> list:
    path = experience_logs_path(output_dir, run_id)
    if not os.path.exists(path):
        return empty_experience_log()
    with open(path, encoding="utf-8") as f:
        stored = json.load(f)
    return stored.get(HUMAN_BASELINE_VARIANT, empty_experience_log())


def _save_human_experience_log(
    output_dir: str, run_id: str, human_experience_log: list
) -> None:
    path = experience_logs_path(output_dir, run_id)
    stored = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            stored = json.load(f)
    stored[HUMAN_BASELINE_VARIANT] = human_experience_log
    save_variant_experience_logs(output_dir, run_id, stored)


def main() -> None:
    request_info = _load_request_info()
    distances = get_distances(request_info)
    clients = get_clients(request_info)
    mas = get_mas(request_info)
    schools = get_schools(request_info)
    prio_assignments = get_prio_assignments(request_info)

    global_schools_mapping = {
        school.get("id", None): school.get("systemuebergreifendeid", None)
        for school in schools
    }
    clients_by_id = {client["id"]: client for client in clients}

    run_state = load_existing_run_state(LOG_OUTPUT_DIR)
    run_id = run_state["run_id"]
    start_date = run_state["start_date"]
    end_date = run_state["end_date"]
    human_experience_log = _load_human_experience_log(LOG_OUTPUT_DIR, run_id)

    print(
        f"Generating human-baseline logs for run {run_id} "
        f"({start_date} to {end_date})"
    )

    data_processor = DataProcessor(
        mas,
        clients,
        prio_assignments,
        distances,
        [],
        global_schools_mapping,
    )

    for relevant_date in pd.date_range(start=start_date, end=end_date):
        date_str = relevant_date.strftime("%Y-%m-%d")

        if is_variant_date_done(run_state, date_str, HUMAN_BASELINE_VARIANT):
            continue

        vertretungen = get_vertretungen(request_info, date_str)
        if len(vertretungen) == 0:
            continue

        day_context = prepare_day_context(vertretungen, mas)
        assigned_pairs = day_context["assignments"]
        relevant_date_dt = pd.to_datetime(date_str).to_pydatetime()

        data_processor.experience_log = human_experience_log
        clients_df, mas_df = data_processor.create_day_dataset(
            day_context["open_client_ids"],
            day_context["free_ma_ids"],
            relevant_date_dt,
        )

        clients_df["available_until"] = clients_df["id"].map(
            lambda client_id: next(
                (
                    item["until"]
                    for item in day_context["open_clients"]
                    if item["id"] == client_id
                ),
                None,
            )
        )
        clients_df["ma_blacklist"] = clients_df["id"].map(
            lambda client_id: next(
                (
                    item["ma_blacklist"]
                    for item in day_context["open_clients"]
                    if item["id"] == client_id
                ),
                [],
            )
        )
        mas_df["available_until"] = mas_df["id"].map(
            lambda ma_id: next(
                (item["until"] for item in day_context["free_mas"] if item["id"] == ma_id),
                None,
            )
        )

        day_log_rows = build_human_baseline_client_day_log_rows(
            clients_df,
            mas_df,
            date_str,
            HUMAN_BASELINE_VARIANT,
            assigned_pairs,
        )

        append_log_rows(LOG_OUTPUT_DIR, run_id, HUMAN_BASELINE_VARIANT, day_log_rows)
        human_experience_log = record_optimizer_assignments(
            human_experience_log,
            assigned_pairs,
            clients_by_id,
            date_str,
        )
        _save_human_experience_log(LOG_OUTPUT_DIR, run_id, human_experience_log)

        run_state["completed"].append(
            variant_date_key(date_str, HUMAN_BASELINE_VARIANT)
        )
        save_run_state(LOG_OUTPUT_DIR, run_state)

        print(
            f"Saved {len(day_log_rows)} human-baseline rows for {date_str} "
            f"({len(assigned_pairs)} human assignments)"
        )

    print(
        f"Human-baseline logs appended to "
        f"{LOG_OUTPUT_DIR}/client_day_log_{run_id}_{HUMAN_BASELINE_VARIANT}.csv"
    )


if __name__ == "__main__":
    main()
