import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import time
from fetching.missy_fetching import get_distances, get_clients, get_mas, get_prio_assignments
from fetching.experience_logging import get_experience_log
from utils.add_comment import add_abnormality_comment
from utils.append_to_json_file import append_to_json_file

from config import update_cache, relevant_date_test

from data_processing.data_processor import DataProcessor
from fetching.missy_fetching import get_vertretungen
from data_processing.features_retrieval.retrieve_ids import (
    get_open_client_ids,
    get_free_ma_ids,
)
from optimize.optimize import Optimizer
from learning.model import AbnormalityModel
from learning.LearningHandler import LearningHandler
from utils.assignment_alternatives import collect_alternatives
from utils.send_update import send_update, send_empty_update

from config import training_features_de, include_abnormality

# load .env file to environment
load_dotenv(override=True)

logging.basicConfig(
    filename="ai-assistant.log",
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

user = os.getenv("USER")
pw = os.getenv("PASSWORD")


# Retrieve mostly static data
distances = get_distances(user, pw, update_cache=update_cache)
clients = get_clients(user, pw, update_cache=update_cache)
mas = get_mas(user, pw, update_cache=update_cache)
prio_assignments = get_prio_assignments(user, pw, update_cache=update_cache)
experience_log = get_experience_log()

def main():
    
    data_processor = DataProcessor(mas, clients, prio_assignments, distances, experience_log)
    old_vertretungen = None
    
    while True:
        
        if relevant_date_test:
            relevant_date = relevant_date_test
        else:
            today = datetime.today()
            if today.hour >= 10 and today.minute >= 30:
                tomorrow = today + timedelta(days=1)
                relevant_date = tomorrow.strftime('%Y-%m-%d')
            else:
                relevant_date = today.strftime('%Y-%m-%d')
        
        vertretungen = get_vertretungen(relevant_date, user, pw, update_cache=True)
        
        relevant_date = datetime.strptime(relevant_date, '%Y-%m-%d')
        
        print("vertretungen: ", vertretungen)
        
        if vertretungen != old_vertretungen:
            old_vertretungen = vertretungen
        else:
            print("No new updates")
            time.sleep(10)
            continue
        
        mabw_records = data_processor.get_mabw_records(vertretungen)
        open_client_records = mabw_records["open_clients"]
        rescheduled_ma_records = mabw_records["rescheduled_mas"]

        ma_assignments = data_processor.get_ma_assignments(rescheduled_ma_records)
        assigned_mas = list(ma_assignments.keys())
        print("assigned_mas: ", assigned_mas)
        kabw_records = data_processor.get_kabw_records(vertretungen, assigned_mas)
        
        print("kabw_records: ", kabw_records)
        
        client_record_assignments = data_processor.get_client_record_assignments(mabw_records["open_clients"])

        print("client_record_assignments: ", client_record_assignments)

        absent_client_records = kabw_records["absent_clients"]
        free_ma_records = kabw_records["free_mas"]
        
        

        print("Records extrahiert")

        open_client_ids = get_open_client_ids(open_client_records)
        free_ma_ids = get_free_ma_ids(free_ma_records, absent_client_records, mas)
        free_ma_ids_only = list(map(lambda x: x["id"], free_ma_ids))
        open_client_ids_only = list(map(lambda x: x["id"], open_client_ids))
        
        print("ids gesammelt")
        
        print("freie MA's: ", free_ma_ids_only)
        print("offene Klienten: ", open_client_ids_only)
        
        clients_df, mas_df = data_processor.create_day_dataset(open_client_ids_only, free_ma_ids_only, relevant_date)
        
        print("clients_df: ", clients_df)
        print("mas_df: ", mas_df)
        
        # iterate over the mas_df and add a column "available_until" based on the free_ma_ids in the form {"id": "123", "until": "2025-01-01"}
        # First, generate the column with the correct values and then add it to the dataframe
        mas_df["available_until"] = mas_df["id"].map(lambda x: next((item["until"] for item in free_ma_ids if item["id"] == x), None))
        clients_df["available_until"] = clients_df["id"].map(lambda x: next((item["until"] for item in open_client_ids if item["id"] == x), None))
        
        abnormality_model = AbnormalityModel()
        
        optimizer = Optimizer(mas_df, clients_df, abnormality_model)
        optimizer.create_model()
        
        learner = LearningHandler(abnormality_model)
        
        objective_value = None
        assigned_pairs_list = []
        recommendation_ids = []
        for _ in range(3):
            objective_value = optimizer.solve_model(objective_value)
            if objective_value is None:
                break
            objective_value = int(objective_value * 1.10) # increase objective value by 10%
            assigned_pairs, recommendation_id = optimizer.process_results()
            assigned_pairs_list.append(assigned_pairs)
            recommendation_ids.append(recommendation_id)
        
        transposed_pair_list = collect_alternatives(assigned_pairs_list)
        recommendations = []
        for assigned_pairs in transposed_pair_list:
            if include_abnormality:
                learner_infos = []
                for i in range(len(assigned_pairs)):
                    learner_data = learner.prepare_data(assigned_pairs[i], mas_df, clients_df)
                    print(f"learner_data: {learner_data}")
                    learner_info = learner.predict_and_score(learner_data)
                    shap_values = learner.get_explanation(learner_data)
                    if learner_info[0] == 1: # assignment is abnormal
                        add_abnormality_comment(recommendation_ids[i], shap_values, learner_data[0], training_features_de)
                    learner_infos.append(learner_info)

            recommendation = send_update(user, pw, assigned_pairs, recommendation_ids, client_record_assignments)
            
            recommendations.append(recommendation)
        
        # Clear all old recommendations that are not in the new recommendations
        unassigned_incidents = filter_unassigned_incidents(client_record_assignments, assigned_pairs_list)
        print("unassigned_incidents: ", unassigned_incidents)
        send_empty_update(user, pw, unassigned_incidents)
        
        append_to_json_file(recommendations, "user_recommendations.json")
        
        time.sleep(10)
    
# assigned_pairs is a list of dictionaries with the following structure:
# [{"ma": "123", "klient": "456"}, {"ma": "231", "klient": "322"}]
# client_record_assignments is a dictionary with the following structure: {"456": "256", ...}
# where 256 is the incident_id and 456 is the client_id
# the assigned_pairs_list is a list of assigned_pairs.
# I want to filter all incident_ids from the client_record_assignments that are not included in the assigned_pairs_list:
def filter_unassigned_incidents(client_record_assignments, assigned_pairs_list):
    """
    Filter out incident IDs from client_record_assignments that are not included in assigned_pairs_list.
    
    Args:
        client_record_assignments (dict): Dictionary mapping client_ids to incident_ids {"456": "256", ...}
        assigned_pairs_list (list): List of assigned_pairs dictionaries [{"ma": "123", "klient": "456"}, ...]
    
    Returns:
        dict: Filtered client_record_assignments containing only entries where client_id is in assigned_pairs_list
    """
    # Create a set of all client_ids from assigned_pairs_list
    assigned_client_ids = {
        pair["klient"] for pairs in assigned_pairs_list 
        for pair in pairs
    }
    
    # Filter client_record_assignments to keep only entries where client_id is NOT in assigned_client_ids
    filtered_assignments = {
        client_id: incident_id 
        for client_id, incident_id in client_record_assignments.items()
        if client_id not in assigned_client_ids
    }
    
    return list(filtered_assignments.values())



    
if __name__ == '__main__':
    main()
    # pass