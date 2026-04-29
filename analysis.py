from fetching.missy_fetching import get_vertretungen
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
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
experience_log = get_experience_log()

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
    return {column: _to_json_serializable(value) for column, value in client_row.items()}


def _init_client_assignment_tracking() -> dict:
    return {
        "labels": {
            "priority_10": {},
            "other_priorities": {},
            "all_priorities": {},
        },
        "recommendations": {
            "priority_10": {},
            "other_priorities": {},
            "all_priorities": {},
        },
    }


def _update_client_assignment_tracking(
    tracking: dict,
    clients_df: pd.DataFrame,
    assigned_client_ids: set,
    analysis_type: str,
):
    for _, client_row in clients_df.iterrows():
        client_id = client_row.get("id")
        if client_id is None:
            continue

        group_names = ["all_priorities"]
        if client_row.get("priority") == 10:
            group_names.append("priority_10")
        else:
            group_names.append("other_priorities")

        is_assigned = client_id in assigned_client_ids
        serialized_row = _serialize_client_row(client_row)

        for group_name in group_names:
            group_tracking = tracking[analysis_type][group_name]
            if client_id not in group_tracking:
                group_tracking[client_id] = {
                    "client": serialized_row,
                    "assigned_count": 0,
                    "not_assigned_count": 0,
                }
            if is_assigned:
                group_tracking[client_id]["assigned_count"] += 1
            else:
                group_tracking[client_id]["not_assigned_count"] += 1


def _finalize_client_assignment_tracking(tracking: dict) -> dict:
    output = {}
    for analysis_type, groups in tracking.items():
        output[analysis_type] = {}
        for group_name, clients in groups.items():
            rows = []
            for client_id, entry in clients.items():
                assigned_count = int(entry["assigned_count"])
                not_assigned_count = int(entry["not_assigned_count"])
                total_count = assigned_count + not_assigned_count
                rows.append(
                    {
                        "client_id": client_id,
                        "group": group_name,
                        "client": entry["client"],
                        "assigned_count": assigned_count,
                        "not_assigned_count": not_assigned_count,
                        "total_count": total_count,
                        "assignment_percentage": (
                            float((assigned_count / total_count) * 100)
                            if total_count > 0
                            else None
                        ),
                    }
                )
            rows.sort(
                key=lambda row: (
                    row["not_assigned_count"],
                    -row["assigned_count"],
                    str(row["client_id"]),
                ),
                reverse=True,
            )
            output[analysis_type][group_name] = rows
    return output


def compute_priority_stats(
    df: pd.DataFrame, group_name: str, total_clients_count: int = 0
) -> dict:
    assigned_count = int(len(df))
    assigned_percentage = (
        float((assigned_count / total_clients_count) * 100)
        if total_clients_count > 0
        else None
    )

    if df.empty:
        return {
            "group": group_name,
            "entries_count": assigned_count,
            "total_clients_count": int(total_clients_count),
            "assigned_percentage": assigned_percentage,
            "experience_gt_1_count": {
                "cl_experience": 0,
                "school_experience": 0,
                "short_term_cl_experience": 0,
            },
            "experience_average": {
                "cl_experience": None,
                "school_experience": None,
                "short_term_cl_experience": None,
            },
            "ma_availability_true_count": 0,
            "qualifications_met_true_count": 0,
            "availability_gap_positive_count": 0,
            "availability_gap_positive_average": None,
            "average_time_to_school": None,
            "mobility_percentage": None,
        }

    cl_experience = _safe_numeric_series(df, "cl_experience")
    school_experience = _safe_numeric_series(df, "school_experience")
    short_term_cl_experience = _safe_numeric_series(df, "short_term_cl_experience")
    ma_availability = _safe_bool_series(df, "ma_availability")
    qualifications_met = _safe_bool_series(df, "qualifications_met")
    availability_gap = _safe_numeric_series(df, "availability_gap")
    time_to_school = _safe_numeric_series(df, "timeToSchool")
    mobility = _safe_bool_series(df, "mobility")
    availability_gap_positive = availability_gap[availability_gap > 0]

    return {
        "group": group_name,
        "entries_count": assigned_count,
        "total_clients_count": int(total_clients_count),
        "assigned_percentage": assigned_percentage,
        "experience_gt_1_count": {
            "cl_experience": int((cl_experience > 1).sum()),
            "school_experience": int((school_experience > 1).sum()),
            "short_term_cl_experience": int((short_term_cl_experience > 1).sum()),
        },
        "experience_average": {
            "cl_experience": float(cl_experience.mean()) if not cl_experience.dropna().empty else None,
            "school_experience": float(school_experience.mean()) if not school_experience.dropna().empty else None,
            "short_term_cl_experience": float(short_term_cl_experience.mean()) if not short_term_cl_experience.dropna().empty else None,
        },
        "ma_availability_true_count": int(ma_availability.sum()),
        "qualifications_met_true_count": int(qualifications_met.sum()),
        "availability_gap_positive_count": int((availability_gap > 0).sum()),
        "availability_gap_positive_average": (
            float(availability_gap_positive.mean())
            if not availability_gap_positive.dropna().empty
            else None
        ),
        "average_time_to_school": (
            float(time_to_school.mean()) if not time_to_school.dropna().empty else None
        ),
        "mobility_percentage": float(mobility.mean() * 100),
    }


def build_df_analysis(
    df: pd.DataFrame, clients_df: pd.DataFrame, df_type: str, date_value
) -> dict:
    date_str = date_value.strftime("%Y-%m-%d") if isinstance(date_value, datetime) else str(date_value)

    priority_values = _safe_numeric_series(df, "priority")
    priority_10_df = df[priority_values == 10]
    priority_other_df = df[priority_values != 10]

    client_priority_values = _safe_numeric_series(clients_df, "priority")
    total_priority_10_count = int((client_priority_values == 10).sum())
    total_other_priorities_count = int((client_priority_values != 10).sum())
    total_all_priorities_count = int(len(clients_df))

    return {
        "type": df_type,
        "date": date_str,
        "stats": {
            "priority_10": compute_priority_stats(
                priority_10_df, "priority_10", total_priority_10_count
            ),
            "other_priorities": compute_priority_stats(
                priority_other_df, "other_priorities", total_other_priorities_count
            ),
            "all_priorities": compute_priority_stats(
                df, "all_priorities", total_all_priorities_count
            ),
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
    comparison = []
    client_assignment_tracking = _init_client_assignment_tracking()
    start_date = "2026-03-23"
    end_date = "2026-04-27"
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
            for elem in assigned_records if elem.get("mavertretend") != None
        ]

        free_ma_records = merge_consecutive_free_ma_records(
            list(filter(lambda x: x.get("mafrei") != None, vertretungen)) +
            free_and_assigned_ma_records
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
            **{elem["ma"]: elem["klient"] for elem in assignments}
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
            for elem in absent_ma_records if all_open_clients.get(elem["maabwesend"]["id"]) != None
        ]
        open_client_ids = [elem["id"] for elem in open_clients]

        
        open_client_ids = list(set(open_client_ids))
        free_ma_ids = list(set(free_ma_ids + [elem["ma"] for elem in assignments]))

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

        optimizer = Optimizer(mas_df, clients_df, abnormality_model)
        optimizer.create_model()

        optimizer.solve_model()
        assigned_pairs, _ = optimizer.process_results()

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

        _update_client_assignment_tracking(
            client_assignment_tracking,
            clients_df,
            set(assignments.values()),
            "labels",
        )
        _update_client_assignment_tracking(
            client_assignment_tracking,
            clients_df,
            set(assigned_pairs.values()),
            "recommendations",
        )

        comparison.append(
            {
                "date": relevant_date.strftime("%Y-%m-%d"),
                "labels": build_df_analysis(
                    single_df_labels, clients_df, "labels", relevant_date
                ),
                "recommendations": build_df_analysis(
                    single_df_recommendations, clients_df, "recommendations", relevant_date
                ),
            }
        )

    output_file = "data/comparison_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    client_tracking_output = _finalize_client_assignment_tracking(client_assignment_tracking)
    tracking_output_file = "data/comparison_client_assignment_tracking.json"
    with open(tracking_output_file, "w", encoding="utf-8") as f:
        json.dump(client_tracking_output, f, ensure_ascii=False, indent=2)

    print(f"Saved comparison analysis to {output_file}")
    print(f"Saved client assignment tracking to {tracking_output_file}")


if __name__ == "__main__":
    main()
