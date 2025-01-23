import os
from dotenv import load_dotenv
import logging

from AIAssistant import AIAssistant

from fetching.missy_fetching import get_distances, get_clients, get_mas, get_prio_assignments

from optimize.optimize import Optimizer

# load .env file to environment
load_dotenv()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

user = os.getenv("USER")
pw = os.getenv("PASSWORD")

update_cache = False

# Retrieve mostly static data
clients = get_clients(user, pw, update_cache=update_cache)
distances = get_distances(user, pw, update_cache=update_cache)
mas = get_mas(user, pw, update_cache=update_cache)
prio_assignments = get_prio_assignments(user, pw, update_cache=update_cache)

def main():
    
    assistant = AIAssistant(user, pw, mas, clients, prio_assignments, distances)
    
    assistant.update_dataset()
    
    print("Datensätze erstellt")
    
    optimizer = Optimizer(assistant.mas_df, assistant.clients_df)
    optimizer.create_model()
    optimizer.solve_model()
    optimizer.display_results()
    
if __name__ == '__main__':
    main()