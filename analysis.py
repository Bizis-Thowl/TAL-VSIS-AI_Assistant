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

def create_comparison_plots(my_dict):
    # Get all keys that end with '_labels' or '_recommendations'
    keys = [
        k.replace("_labels", "")
        for k in my_dict.keys()
        if k.endswith("_labels") and isinstance(my_dict[k], dict)
    ]

    print(keys)

    # Create a figure with subplots
    n_keys = int(len(keys) / 2) + 1
    fig, axes = plt.subplots(n_keys, 2, figsize=(10, 5 * n_keys))
    if n_keys == 1:
        axes = [axes]

    print(my_dict["count_labels"])
    print(my_dict["timeToSchool_labels"])
    print(my_dict["timeToSchool_labels"]["min"])

    # Create box plots for each key
    for idx, key in enumerate(keys):
        # Prepare data for box plot
        labels_data = [
            my_dict[f"{key}_labels"]["min"],
            my_dict[f"{key}_labels"]["mean"],
            my_dict[f"{key}_labels"]["median"],
            my_dict[f"{key}_labels"]["max"],
        ]

        recommendations_data = [
            my_dict[f"{key}_recommendations"]["min"],
            my_dict[f"{key}_recommendations"]["mean"],
            my_dict[f"{key}_recommendations"]["median"],
            my_dict[f"{key}_recommendations"]["max"],
        ]

        # Create box plot
        data = [labels_data, recommendations_data]
        d_index = int(idx / 2), idx % 2
        sns.boxplot(data=data, ax=axes[d_index])
        axes[d_index].set_title(f"Comparison of {key}")
        axes[d_index].set_ylabel("Value")

    plt.tight_layout()
    plt.show()


def create_time_series_plots(comparison):
    # Get all keys that end with '_labels' or '_recommendations'
    keys = [
        k.replace("_labels", "")
        for k in comparison[0].keys()
        if k.endswith("_labels") and isinstance(comparison[0][k], dict)
    ]

    # Create a figure with subplots
    n_keys = len(keys)
    fig, axes = plt.subplots(n_keys, 1, figsize=(15, 5 * n_keys))
    if n_keys == 1:
        axes = [axes]

    # Sort comparison by date
    comparison.sort(key=lambda x: x["date"])
    dates = [entry["date"] for entry in comparison]

    # Create line plots for each key
    for idx, key in enumerate(keys):
        # Extract all statistics for labels and recommendations
        labels_data = {
            "mean": [entry[f"{key}_labels"]["mean"] for entry in comparison],
            "median": [entry[f"{key}_labels"]["median"] for entry in comparison],
            "min": [entry[f"{key}_labels"]["min"] for entry in comparison],
            "max": [entry[f"{key}_labels"]["max"] for entry in comparison],
            "std": [entry[f"{key}_labels"]["std"] for entry in comparison],
        }

        recommendations_data = {
            "mean": [entry[f"{key}_recommendations"]["mean"] for entry in comparison],
            "median": [
                entry[f"{key}_recommendations"]["median"] for entry in comparison
            ],
            "min": [entry[f"{key}_recommendations"]["min"] for entry in comparison],
            "max": [entry[f"{key}_recommendations"]["max"] for entry in comparison],
            "std": [entry[f"{key}_recommendations"]["std"] for entry in comparison],
        }

        # Create box plot data
        labels_box_data = []
        recommendations_box_data = []
        for i in range(len(dates)):
            # Create box plot data using min, max, median, and quartiles
            labels_box_data.append(
                [
                    labels_data["min"][i],
                    labels_data["median"][i] - labels_data["std"][i],
                    labels_data["median"][i],
                    labels_data["median"][i] + labels_data["std"][i],
                    labels_data["max"][i],
                ]
            )
            recommendations_box_data.append(
                [
                    recommendations_data["min"][i],
                    recommendations_data["median"][i] - recommendations_data["std"][i],
                    recommendations_data["median"][i],
                    recommendations_data["median"][i] + recommendations_data["std"][i],
                    recommendations_data["max"][i],
                ]
            )

        # Create line plot with box plots
        axes[idx].plot(
            range(len(dates)),
            labels_data["mean"],
            "b-",
            label="Labels Mean",
            marker="o",
            alpha=0.7,
        )
        axes[idx].plot(
            range(len(dates)),
            recommendations_data["mean"],
            "r-",
            label="Recommendations Mean",
            marker="s",
            alpha=0.7,
        )

        # Add box plots
        for i in range(len(dates)):
            # Labels box plot
            axes[idx].boxplot(
                [labels_box_data[i]],
                positions=[i - 0.2],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="blue", alpha=0.1),
            )
            # Recommendations box plot
            axes[idx].boxplot(
                [recommendations_box_data[i]],
                positions=[i + 0.2],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="red", alpha=0.1),
            )

        # Customize plot
        axes[idx].set_title(f"Time Series of {key} with Distribution")
        axes[idx].set_xlabel("Date")
        axes[idx].set_ylabel("Value")
        axes[idx].legend()
        axes[idx].grid(True)

        # Set x-axis ticks to dates
        axes[idx].set_xticks(range(len(dates)))
        axes[idx].set_xticklabels(dates, rotation=45, ha="right")

        # Add some padding to prevent box plots from being cut off
        axes[idx].margins(x=0.1)

    plt.tight_layout()
    plt.show()


def main():

    data_processor = DataProcessor(
        mas, clients, prio_assignments, distances, experience_log, global_schools_mapping
    )
    comparison = []
    start_date = "2026-04-14"
    end_date = "2026-04-18"
    for relevant_date in pd.date_range(start=start_date, end=end_date):
        relevant_date = relevant_date.strftime("%Y-%m-%d")

        vertretungen = get_vertretungen(request_info, relevant_date)

        if len(vertretungen) == 0:
            continue

        relevant_date = datetime.strptime(relevant_date, "%Y-%m-%d")

        free_ma_records = merge_consecutive_free_ma_records(
            list(filter(lambda x: x.get("mafrei") != None, vertretungen))
        )
        assigned_records = list(
            filter(lambda x: x.get("klientzubegleiten") != None, vertretungen)
        )
        absent_ma_records = list(	
            filter(lambda x: x.get("maabwesend") != None and x.get("klientabwesend") == None, vertretungen)
        )
        
        absent_ma_ids = [elem["maabwesend"]["id"] for elem in absent_ma_records]
        
        all_open_clients = [elem["id"] for sublist in [sublist["aktiveklientinnen"] for sublist in mas if sublist.get("id") in absent_ma_ids] for elem in sublist]
        all_open_clients = list(set(all_open_clients))
        assigned_clients = list(set([elem["klientzubegleiten"]["id"] for elem in assigned_records]))
        
        free_mas = [
            {
                "id": elem["mafrei"]["id"],
                "until": datetime.strptime(elem["enddatum"], "%Y-%m-%d"),
            }
            for elem in free_ma_records
        ]
        free_ma_ids = [elem["id"] for elem in free_mas]
        
        print(len(all_open_clients))
        print(len(assigned_clients))
        print(len(free_mas))
        
        assignments = [
            {
                "ma": elem["mavertretend"]["id"],
                "klient": elem["klientzubegleiten"]["id"],
            }
            for elem in assigned_records
        ]

        clients_df, mas_df = data_processor.create_day_dataset(
            all_open_clients, free_ma_ids, relevant_date
        )
        
        mas_df["available_until"] = mas_df["id"].map(
            lambda x: next(
                (item["until"] for item in free_mas if item["id"] == x), None
            )
        )
        clients_df["available_until"] = clients_df["id"].map(
            lambda x: next(
                (item["until"] for item in all_open_clients if item["id"] == x), None
            )
        )

        clients_df["ma_blacklist"] = clients_df["id"].map(
            lambda x: next(
                (item["ma_blacklist"] for item in all_open_clients if item["id"] == x),
                None,
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

        single_df_labels = create_single_df(
            clients_df, mas_df, replacements, relevant_date
        )
        single_df_recommendations = create_single_df(
            clients_df, mas_df, replacement_recommendations, relevant_date
        )

        print(f"Labels: {single_df_labels['qualifications_met']}")
        # print(f"Recommendations: {single_df_recommendations.describe()}")

        # Create a dictionary that retrieves entries from the describe function
        description_labels = single_df_labels.describe().to_dict()
        description_recommendations = single_df_recommendations.describe().to_dict()

        my_dict = {
            "date": relevant_date,
            "count_labels": len(single_df_labels),
            "count_recommendations": len(single_df_recommendations),
        }

        # Loop through all keys in the description dictionaries
        for key in description_labels.keys():
            if key == "date":
                continue
            my_dict[f"{key}_labels"] = {
                "mean": description_labels[key]["mean"],
                "median": description_labels[key]["50%"],
                "std": description_labels[key]["std"],
                "min": description_labels[key]["min"],
                "max": description_labels[key]["max"],
            }
            my_dict[f"{key}_recommendations"] = {
                "mean": description_recommendations[key]["mean"],
                "median": description_recommendations[key]["50%"],
                "std": description_recommendations[key]["std"],
                "min": description_recommendations[key]["min"],
                "max": description_recommendations[key]["max"],
            }

        comparison.append(my_dict)

    print(my_dict)
    create_comparison_plots(my_dict)
    create_time_series_plots(comparison)


if __name__ == "__main__":
    main()
