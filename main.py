import os
from dotenv import load_dotenv
import logging

import time
from datetime import datetime, timedelta
from fetching.missy_fetching import get_distances, get_clients, get_mas, get_prio_assignments
from fetching.experience_logging import get_experience_log

from utils.append_to_json_file import append_to_json_file

from config import update_cache

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
from utils.send_update import send_update
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
        
        # set a date
        relevant_date = "2025-04-03"
        
        # today = datetime.today()
        # if today.hour >= 10 and today.minute >= 30:
        #     tomorrow = today + timedelta(days=1)
        #     relevant_date = tomorrow.strftime('%Y-%m-%d')
        # else:
        #     relevant_date = today.strftime('%Y-%m-%d')
        
        vertretungen = get_vertretungen(relevant_date, user, pw, update_cache=True)
        
        print("vertretungen: ", vertretungen)
        
        if vertretungen != old_vertretungen:
            old_vertretungen = vertretungen
        else:
            print("No new updates")
            time.sleep(10)
            continue
        
        mabw_records = data_processor.get_mabw_records(vertretungen)
        open_client_records = mabw_records["open_clients"]

        ma_assignments = data_processor.get_ma_assignments(mabw_records)
        assigned_mas = list(ma_assignments.keys())
        print("assigned_mas: ", assigned_mas)
        kabw_records = data_processor.get_kabw_records(vertretungen, assigned_mas)
        
        client_record_assignments = data_processor.get_client_record_assignments(mabw_records["open_clients"])

        print("client_record_assignments: ", client_record_assignments)

        absent_client_records = kabw_records["absent_clients"]
        free_ma_records = mabw_records["rescheduled_mas"]

        print("Records extrahiert")

        open_client_ids = get_open_client_ids(open_client_records)
        free_ma_ids = get_free_ma_ids(free_ma_records, absent_client_records, mas)
        
        print("ids gesammelt")
        
        print("freie MA's: ", free_ma_ids)
        print("offene Klienten: ", open_client_ids)
        
        clients_df, mas_df = data_processor.create_day_dataset(open_client_ids, free_ma_ids, relevant_date)
        
        abnormality_model = AbnormalityModel()
        
        optimizer = Optimizer(mas_df, clients_df, abnormality_model)
        optimizer.create_model()
        
        learner = LearningHandler(abnormality_model)
        
        objective_value = None
        assigned_pairs_list = []
        recommendation_ids = []
        for _ in range(3):
            objective_value = optimizer.solve_model(objective_value)
            assigned_pairs, recommendation_id = optimizer.process_results()
            assigned_pairs_list.append(assigned_pairs)
            recommendation_ids.append(recommendation_id)
        
        transposed_pair_list = collect_alternatives(assigned_pairs_list)
        recommendations = []
        for assigned_pairs in transposed_pair_list:
            learner_infos = []
            for i in range(len(assigned_pairs)):
                learner_data = learner.prepare_data(assigned_pairs[i], mas_df, clients_df)
                learner_info = learner.predict_and_score(learner_data)
                learner_infos.append(learner_info)

            recommendation = send_update(user, pw, assigned_pairs, recommendation_ids, learner_infos, client_record_assignments)
            
            recommendations.append(recommendation)
        
        append_to_json_file(recommendations, "user_recommendations.json")
        
        time.sleep(10)
    
    
if __name__ == '__main__':
    main()
    # pass