import logging
import pandas as pd
import shap

from datetime import datetime, timedelta

from fetching.missy_fetching import (
    get_distances,
    get_clients,
    get_mas,
    get_prio_assignments,
)
from fetching.experience_logging import get_experience_log

from core.data.data_processor import DataProcessor
from core.data.features_retrieval.create_replacements import create_replacements
from core.data.features_retrieval.create_single_df import create_single_df

from learning.model import AbnormalityModel

from utils.daterange import daterange
from utils.min_max_date import min_max_date
from utils.read_file import read_file

from config import update_cache
import os
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

user = os.getenv("USER")
pw = os.getenv("PASSWORD")


# Retrieve mostly static data
distances = get_distances(user, pw, update_cache=update_cache)
clients = get_clients(user, pw, update_cache=update_cache)
mas = get_mas(user, pw, update_cache=update_cache)
prio_assignments = get_prio_assignments(user, pw, update_cache=update_cache)
experience_log = get_experience_log()


def main():
    
    use_cache = True
    
    full_dataset_path = "data/final_dataset.csv"
    
    if use_cache and os.path.exists(full_dataset_path):
        full_dataset = pd.read_csv(full_dataset_path)
    else:
        vertretungen = read_file("vertretungsfall_all")
        data_processor = DataProcessor(
            mas, clients, prio_assignments, distances, experience_log
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
        
                ma_assignments = data_processor.get_ma_assignments(rescheduled_ma_records)
                assigned_mas = list(ma_assignments.keys())
                assigned_clients = list(ma_assignments.values())
                
                client_record_assignments = data_processor.get_client_record_assignments(mabw_records["open_clients"])

                clients_df, mas_df = (
                    data_processor.create_day_dataset(assigned_clients, assigned_mas, current_date)
                )
                
                replacements = create_replacements(ma_assignments)
                single_df = create_single_df(clients_df, mas_df, replacements, current_date)

                if full_dataset is None:
                    full_dataset = single_df
                else:
                    full_dataset = pd.concat([full_dataset, single_df])
                

        full_dataset.to_csv(full_dataset_path, index=False)  
    
    training_features = ["timeToSchool", "cl_experience", "school_experience", "priority", "ma_availability", "mobility", "geschlecht_relevant", "qualifications_met"]

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
    
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values, test_row)

if __name__ == "__main__":
    main()

