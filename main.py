import os
from dotenv import load_dotenv
import logging

from AIAssistant import AIAssistant
import time

from fetching.missy_fetching import get_distances, get_clients, get_mas, get_prio_assignments

# load .env file to environment
load_dotenv(override=True)

logging.basicConfig(
    filename="ai-assistant.log",
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

user = os.getenv("USER")
pw = os.getenv("PASSWORD")

print(user)
print(pw)

update_cache = True

# Retrieve mostly static data
clients = get_clients(user, pw, update_cache=update_cache)
distances = get_distances(user, pw, update_cache=update_cache)
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
        assistant.solve_model()
        assistant.display_results()

        assistant.send_update()
        
        time.sleep(10)
    
if __name__ == '__main__':
    main()