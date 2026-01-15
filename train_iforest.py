import logging
import pandas as pd
import shap

from datetime import datetime, timedelta

from fetching.missy_fetching import (
    get_distances,
    get_clients,
    get_mas,
    get_prio_assignments,
    get_schools,
)
from fetching.experience_logging import get_experience_log

from data_processing.data_processor import DataProcessor
from data_processing.features_retrieval.create_replacements import create_replacements
from data_processing.features_retrieval.create_single_df import create_single_df

from learning.model import AbnormalityModel

from utils.daterange import daterange
from utils.min_max_date import min_max_date
from utils.read_file import read_file

from config import training_features, base_url_missy
import os
import json
from dotenv import load_dotenv

from tqdm import tqdm

# load .env file to environment
load_dotenv(override=True)

logging.basicConfig(
    filename="ai-assistant.log",
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

request_specs = os.getenv("REQUEST_INFO")
request_specs = json.loads(request_specs)

request_info = [{'user': spec['user'], 'pw': spec['pw'], 'url': base_url_missy.format(domain=spec['domain'])} for spec in request_specs]

# Retrieve mostly static data
distances = get_distances(request_info, use_cache=True)
clients = get_clients(request_info, use_cache=True)
mas = get_mas(request_info, use_cache=True)
prio_assignments = get_prio_assignments(request_info)
experience_log = get_experience_log()
schools = get_schools(request_info)
global_schools_mapping = {school.get("id", None): school.get("systemuebergreifendeid", None) for school in schools}

def main():
    
    use_cache = False
    
    full_dataset_path = "data/final_dataset.csv"
    
    if use_cache and os.path.exists(full_dataset_path):
        full_dataset = pd.read_csv(full_dataset_path)
    else:
        vertretungen = read_file("vertretungsfall_all")
        data_processor = DataProcessor(
            mas, clients, prio_assignments, distances, experience_log, global_schools_mapping
        )

        min_date, max_date = min_max_date(vertretungen)

        full_dataset = None

        for current_date in daterange(min_date, max_date + timedelta(days=1)):
            # Filter vertretungen for current date
            filtered_vertretungen = [
                v
                for v in vertretungen
                if datetime.strptime(v["startdatum"], "%Y-%m-%d").date()
                <= current_date
                <= datetime.strptime(v["enddatum"], "%Y-%m-%d").date()
            ]

            if filtered_vertretungen:
                print(
                    f"Processing {len(filtered_vertretungen)} vertretungen for {current_date}"
                )
                
                mabw_records = data_processor.get_mabw_records(filtered_vertretungen)
                rescheduled_ma_records = mabw_records["rescheduled_mas"]
                open_client_records = mabw_records["open_clients"]
        
                ma_assignments = data_processor.get_ma_assignments(rescheduled_ma_records)
                assigned_mas = list(ma_assignments.keys())
                assigned_clients = list(ma_assignments.values())
                
                client_record_assignments = data_processor.get_client_record_assignments(mabw_records["open_clients"])

                clients_df, mas_df = (
                    data_processor.create_day_dataset(assigned_clients, assigned_mas, current_date)
                )
                
                # iterate over the mas_df and add a column "available_until" based on the free_ma_ids in the form {"id": "123", "until": "2025-01-01"}
                # First, generate the column with the correct values and then add it to the dataframe
                mas_df["available_until"] = mas_df["id"].map(lambda x: next(datetime.strptime(item["enddatum"], "%Y-%m-%d") for item in rescheduled_ma_records if item["mavertretend"]["id"] == x), None)
                clients_df["available_until"] = clients_df["id"].map(lambda x: next((datetime.strptime(item["enddatum"], "%Y-%m-%d") for item in rescheduled_ma_records if item["klientzubegleiten"]["id"] == x), None))
                
                replacements = create_replacements(ma_assignments)
                single_df = create_single_df(clients_df, mas_df, replacements, current_date)
                print("single_df: ", single_df)
                

                if full_dataset is None:
                    full_dataset = single_df
                else:
                    full_dataset = pd.concat([full_dataset, single_df])
                    
                

        full_dataset.to_csv(full_dataset_path, index=False)  
        
        

    non_nan_dataset = full_dataset[~full_dataset.isna().any(axis=1)]
    X_train = non_nan_dataset[training_features]
    
    model = AbnormalityModel(use_cache=False)
    model.train(X_train)
    
    test_row = X_train.iloc[[-1]]
    
    model.visualize(X_train)
    
    print(f"Last row ofX_train: {test_row}")
    print(f"model.predict(X_train[-1]): {model.predict(test_row)}")
    print(f"model.score_samples(X_train[-1]): {model.score_samples(test_row)}")
    print(f"model.decision_function(X_train[-1]): {model.get_decision_function(test_row)}")
    print(f"model.evaluate(X_train): {model.evaluate(X_train)}")
    
    # Use SHAP's TreeExplainer for IsolationForest
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(test_row)
    
    print(f"shap_values: {shap_values}")
    
    # shap.initjs()
    # shap.force_plot(explainer.expected_value, shap_values, test_row)

if __name__ == "__main__":
    main()

