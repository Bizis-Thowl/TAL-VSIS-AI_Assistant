from fetching.missy_fetching import get_vertretungen
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import json

from config import (
    base_url_missy,
    base_url_ai,
    solver_time_limit_seconds,
)
from fetching.missy_fetching import (
    get_distances,
    get_clients,
    get_mas,
    get_prio_assignments,
    get_schools,
)

from data_processing.data_processor import DataProcessor
from optimize.optimize import Optimizer

from analysis.client_day_logging import build_client_day_log_rows, rows_to_dataframe
from analysis.experience_simulation import (
    initialize_variant_experience_logs,
    record_optimizer_assignments,
    save_variant_experience_logs,
)

load_dotenv(override=True)

request_specs = os.getenv("REQUEST_INFO")
print(request_specs)
request_specs = json.loads(request_specs)

request_info = [
    {
        "user": spec["user"],
        "pw": spec["pw"],
        "url": base_url_missy.format(domain=spec["domain"]),
        "url_ai": base_url_ai.format(system_spec=spec["domain"]),
    }
    for spec in request_specs
]

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


def _get_next_log_run_id(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    existing = []
    for filename in os.listdir(output_dir):
        if not filename.startswith("client_day_log_") or not filename.endswith(".csv"):
            continue
        parts = filename[len("client_day_log_") : -len(".csv")].split("_", 1)
        if parts and parts[0].isdigit():
            existing.append(int(parts[0]))

    return f"{max(existing, default=0) + 1:04d}"


def _run_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, "run_state.json")


def _variant_date_key(date_str: str, weight_name: str) -> str:
    return f"{date_str}:{weight_name}"


def _load_or_create_run_state(output_dir: str, start_date: str, end_date: str) -> dict:
    state_path = _run_state_path(output_dir)
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        if state.get("start_date") == start_date and state.get("end_date") == end_date:
            completed = len(state.get("completed", []))
            print(
                f"Resuming run {state['run_id']} "
                f"({completed} variant-date combinations already saved)"
            )
            return state

    state = {
        "run_id": _get_next_log_run_id(output_dir),
        "start_date": start_date,
        "end_date": end_date,
        "completed": [],
    }
    _save_run_state(output_dir, state)
    print(f"Starting new run {state['run_id']}")
    return state


def _save_run_state(output_dir: str, state: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    state_path = _run_state_path(output_dir)
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


def _is_variant_date_done(state: dict, date_str: str, weight_name: str) -> bool:
    return _variant_date_key(date_str, weight_name) in state["completed"]


def _append_log_rows(
    output_dir: str, run_id: str, weight_name: str, rows: list
) -> None:
    log_path = os.path.join(output_dir, f"client_day_log_{run_id}_{weight_name}.csv")
    log_df = rows_to_dataframe(rows)
    log_df.to_csv(
        log_path,
        mode="a",
        header=not os.path.exists(log_path),
        index=False,
        encoding="utf-8",
    )


weights_list = {
    "baseline": {
        "unassigned": 100000,
        "travel_time": 0,
        "time_window": 0,
        "priority": 0,
        "abnormality": 0,
        "client_experience": 0,
        "school_experience": 0,
        "short_term_client_experience": 0,
        "availability_gap": 0,
    },
    "default": {
        "unassigned": 100000,
        "travel_time": 30,
        "time_window": 10,
        "priority": 1000,
        "abnormality": 200,
        "client_experience": 1000,
        "school_experience": 333,
        "short_term_client_experience": 1000,
        "availability_gap": 100,
    },
    "default-low-exp": {
        "unassigned": 100000,
        "travel_time": 30,
        "time_window": 10,
        "priority": 1000,
        "abnormality": 200,
        "client_experience": 100,
        "school_experience": 33,
        "short_term_client_experience": 100,
        "availability_gap": 100,
    },
    "default-high-dist": {
        "unassigned": 100000,
        "travel_time": 120,
        "time_window": 10,
        "priority": 1000,
        "abnormality": 200,
        "client_experience": 100,
        "school_experience": 33,
        "short_term_client_experience": 100,
        "availability_gap": 100,
    },
    "no-dist": {
        "unassigned": 100000,
        "travel_time": 0,
        "time_window": 10,
        "priority": 1000,
        "abnormality": 200,
        "client_experience": 1000,
        "school_experience": 333,
        "short_term_client_experience": 1000,
        "availability_gap": 100,
    },
    "no-exp": {
        "unassigned": 100000,
        "travel_time": 30,
        "time_window": 10,
        "priority": 1000,
        "abnormality": 200,
        "client_experience": 0,
        "school_experience": 0,
        "short_term_client_experience": 0,
        "availability_gap": 100,
    },
}


def _run_weight_variant(weight_name, weights, mas_df, clients_df, date_str):
    print(f"Running weight variant: {weight_name} | date: {date_str}")
    optimizer = Optimizer(mas_df, clients_df)
    optimizer.create_model(weights=weights)
    if optimizer.solve_model() is None:
        print(
            f"No solution for '{weight_name}' on {date_str} "
            f"(infeasible or timed out after {solver_time_limit_seconds}s)"
        )
        return weight_name, [], []
    assigned_pairs, _ = optimizer.get_solution_assignments()
    day_log_rows = build_client_day_log_rows(
        optimizer,
        clients_df,
        mas_df,
        date_str,
        weight_name,
    )
    return weight_name, day_log_rows, assigned_pairs


def main():
    log_output_dir = "data/analysis_client_day_logs"
    start_date = "2026-03-03"
    end_date = "2026-05-29"
    run_state = _load_or_create_run_state(log_output_dir, start_date, end_date)
    run_id = run_state["run_id"]
    weight_names = list(weights_list.keys())
    variant_experience_logs = initialize_variant_experience_logs(
        log_output_dir,
        run_id,
        weight_names,
        clients_by_id,
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
        relevant_date = relevant_date.strftime("%Y-%m-%d")
        date_str = relevant_date

        if all(
            _is_variant_date_done(run_state, date_str, weight_name)
            for weight_name in weights_list
        ):
            continue

        vertretungen = get_vertretungen(request_info, relevant_date)

        if len(vertretungen) == 0:
            continue

        relevant_date = datetime.strptime(relevant_date, "%Y-%m-%d")

        assigned_records = list(
            filter(lambda x: x.get("klientzubegleiten") != None, vertretungen)
        )
        assignments = [
            {
                "ma": elem["mavertretend"]["id"],
                "klient": elem["klientzubegleiten"]["id"],
            }
            for elem in assigned_records
            if elem.get("mavertretend") != None
            and elem.get("klientzubegleiten") != None
        ]
        absent_ma_records = list(
            filter(
                lambda x: x.get("maabwesend") != None
                and x.get("klientzubegleiten") == None,
                vertretungen,
            )
        )
        free_and_assigned_ma_records = [
            {
                **elem,
                "mafrei": {"id": elem.get("mavertretend").get("id")},
            }
            for elem in assigned_records
            if elem.get("mavertretend") != None
        ]

        free_ma_records = merge_consecutive_free_ma_records(
            list(filter(lambda x: x.get("mafrei") != None, vertretungen))
            + free_and_assigned_ma_records
        )

        absent_ma_ids = [elem["maabwesend"]["id"] for elem in absent_ma_records]

        absent_mas = [ma for ma in mas if ma.get("id") in absent_ma_ids]
        all_open_clients = {
            ma.get("id"): client_id.get("id")
            for ma in absent_mas
            for client_id in ma["aktiveklientinnen"]
        }
        all_open_clients = {
            **all_open_clients,
            **{elem["ma"]: elem["klient"] for elem in assignments},
        }

        free_mas = [
            {
                "id": elem["mafrei"]["id"],
                "until": datetime.strptime(elem["enddatum"], "%Y-%m-%d"),
            }
            for elem in free_ma_records
        ]
        free_ma_ids = [elem["id"] for elem in free_mas]

        open_clients = [
            {
                "id": all_open_clients.get(elem["maabwesend"]["id"]),
                "until": datetime.strptime(elem.get("enddatum", None), "%Y-%m-%d"),
                "ma_blacklist": elem.get("mavorschlagblacklist", []),
            }
            for elem in absent_ma_records
            if all_open_clients.get(elem["maabwesend"]["id"]) != None
        ]
        open_client_ids = [elem["id"] for elem in open_clients]

        open_client_ids = list(set(open_client_ids))
        free_ma_ids = list(set(free_ma_ids + [elem["ma"] for elem in assignments]))

        data_processor.experience_log = []
        clients_df, _ = data_processor.create_day_dataset(
            open_client_ids, free_ma_ids, relevant_date
        )

        clients_df["available_until"] = clients_df["id"].map(
            lambda x: next(
                (item["until"] for item in open_clients if item["id"] == x), None
            )
        )

        clients_df["ma_blacklist"] = clients_df["id"].map(
            lambda x: next(
                (item["ma_blacklist"] for item in open_clients if item["id"] == x),
                [],
            )
        )

        pending_variants = [
            weight_name
            for weight_name in weight_names
            if not _is_variant_date_done(run_state, date_str, weight_name)
        ]
        if not pending_variants:
            continue

        variant_mas_dfs = {}
        for weight_name in pending_variants:
            data_processor.experience_log = variant_experience_logs[weight_name]
            _, mas_df = data_processor.create_day_dataset(
                open_client_ids, free_ma_ids, relevant_date
            )
            mas_df["available_until"] = mas_df["id"].map(
                lambda x: next(
                    (item["until"] for item in free_mas if item["id"] == x), None
                )
            )
            variant_mas_dfs[weight_name] = mas_df

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    _run_weight_variant,
                    weight_name,
                    weights_list[weight_name],
                    variant_mas_dfs[weight_name],
                    clients_df,
                    date_str,
                )
                for weight_name in pending_variants
            ]
            for future in futures:
                weight_name, day_log_rows, assigned_pairs = future.result()
                _append_log_rows(log_output_dir, run_id, weight_name, day_log_rows)
                variant_experience_logs[weight_name] = record_optimizer_assignments(
                    variant_experience_logs[weight_name],
                    assigned_pairs,
                    clients_by_id,
                    date_str,
                )
                run_state["completed"].append(
                    _variant_date_key(date_str, weight_name)
                )
                _save_run_state(log_output_dir, run_state)
                save_variant_experience_logs(
                    log_output_dir, run_id, variant_experience_logs
                )
                print(
                    f"Saved {len(day_log_rows)} rows for "
                    f"'{weight_name}' on {date_str} "
                    f"({len(assigned_pairs)} assignments added to experience log)"
                )

    print(f"Run {run_id} complete. Logs in {log_output_dir}")


if __name__ == "__main__":
    main()
