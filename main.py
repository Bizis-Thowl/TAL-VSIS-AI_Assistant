import os
from dotenv import load_dotenv
import logging

from AIAssistant import AIAssistant
import time

from fetching.missy_fetching import get_distances, get_clients, get_mas, get_prio_assignments

from utils.append_to_json_file import append_to_json_file

from config import update_cache

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

def main():
    
    assistant = AIAssistant(user, pw, mas, clients, prio_assignments, distances)
    
    while True:
        did_update = assistant.update_dataset()
        
        if not did_update:
            print("No new updates")
            time.sleep(10)
            continue
        
        assistant.update_solver_model()
        objective_value = None
        assigned_pairs_list = []
        recommendation_ids = []
        for _ in range(3):
            objective_value = assistant.solve_model(objective_value)
            assigned_pairs, recommendation_id = assistant.process_results()
            assigned_pairs_list.append(assigned_pairs)
            recommendation_ids.append(recommendation_id)
        
        transposed_pair_list = _collect_alternatives(assigned_pairs_list)
        recommendations = []
        for assigned_pairs in transposed_pair_list:
            print(assigned_pairs)
            learner_infos = []
            for i in range(len(assigned_pairs)):
                learner_data = assistant.prepare_learner_data(assigned_pairs[i])
                learner_info = assistant.retrieve_learner_scores(learner_data)
                learner_infos.append(learner_info)

            recommendation = assistant.send_update(assigned_pairs, recommendation_ids, learner_infos)
            
            recommendations.append(recommendation)
        
        append_to_json_file(recommendations, "user_recommendations.json")
        time.sleep(10)
    
    
def _collect_alternatives(assigned_pairs_list):
    alternatives_list = []
    
    for assigned_pair in assigned_pairs_list[0]:
        alternatives_element = [assigned_pair]
        client = assigned_pair["klient"]
        for i in range(1,3):
            for j, alternative_pair in enumerate(assigned_pairs_list[i]):
                alternative_client = alternative_pair["klient"]
                if alternative_client == client:
                    alternatives_element.append(alternative_pair)
                    break
        alternatives_list.append(alternatives_element)
    
    return alternatives_list
    
if __name__ == '__main__':
    main()
    # pass