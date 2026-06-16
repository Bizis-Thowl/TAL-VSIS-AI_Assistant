from fetching.missy_fetching import get_vertretungen
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

from config import (
    base_url_missy,
    base_url_ai,
)
from fetching.missy_fetching import (
    get_distances,
    get_clients,
    get_mas,
    get_prio_assignments,
    get_schools,
)
from fetching.experience_logging import get_experience_log

from data_processing.data_processor import DataProcessor
from optimize.optimize import Optimizer
from learning.model import AbnormalityModel
from data_processing.features_retrieval.create_single_df import create_single_df
from data_processing.features_retrieval.create_replacements import create_replacements

from analysis.client_day_logging import build_client_day_log_rows, rows_to_dataframe

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

# Retrieve mostly static data
distances = get_distances(request_info)
clients = get_clients(request_info)
mas = get_mas(request_info)
schools = get_schools(request_info)
prio_assignments = get_prio_assignments(request_info)
# experience_log = get_experience_log()
experience_log = []

global_schools_mapping = {
    school.get("id", None): school.get("systemuebergreifendeid", None)
    for school in schools
}


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


def _safe_bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def _safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _to_json_serializable(value):
    if isinstance(value, dict):
        return {key: _to_json_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_json_serializable(val) for val in value]
    if isinstance(value, tuple):
        return [_to_json_serializable(val) for val in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _serialize_client_row(client_row: pd.Series) -> dict:
    return {
        column: _to_json_serializable(value) for column, value in client_row.items()
    }


def _serialize_dataframe(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return [
        {column: _to_json_serializable(value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _get_next_run_file(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    existing = []
    for filename in os.listdir(output_dir):
        if not filename.startswith("analysis_run_") or not filename.endswith(".json"):
            continue
        numeric_part = filename[len("analysis_run_") : -len(".json")]
        if numeric_part.isdigit():
            existing.append(int(numeric_part))

    next_number = max(existing, default=0) + 1
    return os.path.join(output_dir, f"analysis_run_{next_number:04d}.json")


def _init_experience_maps(experience_log: list[dict]) -> tuple[dict, dict]:
    ma_client_map = {}
    ma_school_map = {}

    for entry in experience_log:
        ma_id = entry.get("ma")
        if ma_id is None:
            continue
        ma_client_map[ma_id] = entry.get("client_experience", {})
        ma_school_map[ma_id] = entry.get("school_experience", {})

    return ma_client_map, ma_school_map


def _append_experience(
    ma_client_map: dict,
    ma_school_map: dict,
    ma_id: str,
    client_id: str,
    school_id: str,
    date_str: str,
) -> None:
    if ma_id not in ma_client_map:
        ma_client_map[ma_id] = {}
    if ma_id not in ma_school_map:
        ma_school_map[ma_id] = {}

    if client_id not in ma_client_map[ma_id]:
        ma_client_map[ma_id][client_id] = []
    if date_str not in ma_client_map[ma_id][client_id]:
        ma_client_map[ma_id][client_id].append(date_str)

    if school_id not in ma_school_map[ma_id]:
        ma_school_map[ma_id][school_id] = []
    if date_str not in ma_school_map[ma_id][school_id]:
        ma_school_map[ma_id][school_id].append(date_str)


def _materialize_experience_log(ma_client_map: dict, ma_school_map: dict) -> list[dict]:
    ma_ids = set(ma_client_map.keys()) | set(ma_school_map.keys())
    return [
        {
            "ma": ma_id,
            "client_experience": ma_client_map.get(ma_id, {}),
            "school_experience": ma_school_map.get(ma_id, {}),
        }
        for ma_id in ma_ids
    ]


weights_list = {
    "baseline": {
        "unassigned": 1000,
        "travel_time": 0,
        "time_window": 0,
        "priority": 0,
        "abnormality": 0,
        "client_experience": 0,
        "school_experience": 0,
        "short_term_client_experience": 0,
        "availability_gap": 0,
    },
    "default":{
        "unassigned": 1000,
        "travel_time": 30,
        "time_window": 10,
        "priority": 1000,
        "abnormality": 200,
        "client_experience": 1000,
        "school_experience": 333,
        "short_term_client_experience": 1000,
        "availability_gap": 100,
    },
    "no-dist":{
        "unassigned": 1000,
        "travel_time": 0,
        "time_window": 10,
        "priority": 1000,
        "abnormality": 200,
        "client_experience": 1000,
        "school_experience": 333,
        "short_term_client_experience": 1000,
        "availability_gap": 100,
    },
    "no-exp":{
        "unassigned": 1000,
        "travel_time": 30,
        "time_window": 10,
        "priority": 1000,
        "abnormality": 200,
        "client_experience": 1000,
        "school_experience": 333,
        "short_term_client_experience": 1000,
        "availability_gap": 100,
    },
}


def main():

    data_processor = DataProcessor(
        mas,
        clients,
        prio_assignments,
        distances,
        experience_log,
        global_schools_mapping,
    )
    output_by_date = {}
    client_day_logs = {weight_name: [] for weight_name in weights_list}
    ma_client_map, ma_school_map = _init_experience_maps(experience_log)
    start_date = "2026-03-02"
    end_date = "2026-05-29"
    for relevant_date in pd.date_range(start=start_date, end=end_date):
        relevant_date = relevant_date.strftime("%Y-%m-%d")

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

        data_processor.experience_log = _materialize_experience_log(
            ma_client_map, ma_school_map
        )
        clients_df, mas_df = data_processor.create_day_dataset(
            open_client_ids, free_ma_ids, relevant_date
        )

        mas_df["available_until"] = mas_df["id"].map(
            lambda x: next(
                (item["until"] for item in free_mas if item["id"] == x), None
            )
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

        abnormality_model = AbnormalityModel()
        date_str = relevant_date.strftime("%Y-%m-%d")
        default_optimizer = None

        for weight_name, weights in weights_list.items():
            log_optimizer = Optimizer(mas_df, clients_df, abnormality_model)
            log_optimizer.create_model(weights=weights)
            if log_optimizer.solve_model() is None:
                continue
            day_log_rows = build_client_day_log_rows(
                log_optimizer,
                clients_df,
                mas_df,
                date_str,
                ma_client_map,
                ma_school_map,
                weight_name,
            )
            client_day_logs[weight_name].extend(day_log_rows)
            if weight_name == "default":
                default_optimizer = log_optimizer

        if default_optimizer is None:
            default_optimizer = Optimizer(mas_df, clients_df, abnormality_model)
            default_optimizer.create_model(weights=weights_list["default"])
            if default_optimizer.solve_model() is None:
                continue

        assigned_pairs, _ = default_optimizer.process_results()

        assigned_pairs = {elem["ma"]: elem["klient"] for elem in assigned_pairs}
        assignments = {elem["ma"]: elem["klient"] for elem in assignments}

        replacements = create_replacements(assignments)
        replacement_recommendations = create_replacements(assigned_pairs)

        print(replacements)
        print(replacement_recommendations)

        single_df_labels = create_single_df(
            clients_df, mas_df, replacements, relevant_date
        )
        single_df_recommendations = create_single_df(
            clients_df, mas_df, replacement_recommendations, relevant_date
        )
        client_school_map = (
            clients_df.set_index("id")["school"].to_dict()
            if "school" in clients_df.columns
            else {}
        )
        if isinstance(replacement_recommendations, pd.DataFrame):
            for _, rec_row in replacement_recommendations.iterrows():
                ma_id = rec_row.get("mas")
                client_id = rec_row.get("clients")
                if ma_id is None or client_id is None:
                    continue
                school_id = client_school_map.get(client_id)
                if school_id is None:
                    continue
                _append_experience(
                    ma_client_map,
                    ma_school_map,
                    ma_id,
                    client_id,
                    school_id,
                    date_str,
                )

        output_by_date[date_str] = {
            "labels": _serialize_dataframe(single_df_labels),
            "recommendations": _serialize_dataframe(single_df_recommendations),
            "clients_available": _serialize_dataframe(clients_df),
            "mas_available": _serialize_dataframe(mas_df),
        }

    output_dir = "data/analysis_runs"
    output_file = _get_next_run_file(output_dir)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_by_date, f, ensure_ascii=False, indent=2)

    print(f"Saved analysis run output to {output_file}")

    log_output_dir = "data/analysis_client_day_logs"
    os.makedirs(log_output_dir, exist_ok=True)
    run_id = os.path.basename(output_file).replace("analysis_run_", "").replace(
        ".json", ""
    )
    for weight_name, rows in client_day_logs.items():
        log_df = rows_to_dataframe(rows)
        log_path = os.path.join(
            log_output_dir, f"client_day_log_{run_id}_{weight_name}.csv"
        )
        log_df.to_csv(log_path, index=False, encoding="utf-8")
        print(f"Saved client-day log for '{weight_name}' to {log_path}")


if __name__ == "__main__":
    main()
