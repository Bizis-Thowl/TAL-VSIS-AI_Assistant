import os
from dotenv import load_dotenv
from fetching.missy_fetching import fetch_date_objects_in_range
from update_ma_client_history import process_data_for_date
from utils.daterange import daterange

from datetime import date

# load .env file to environment
load_dotenv(override=True)

user = os.getenv("USER")
pw = os.getenv("PASSWORD")

OUTPUT_FILE = "data/experience_log.json"  # Where the adapted list is saved

if __name__ == "__main__":
    
    start_date = date(2024, 10, 1)
    end_date = date.today()
    endpoint_key = 'vertretungsfall'
    
    vertretungen = fetch_date_objects_in_range(user, pw, endpoint_key, start_date, end_date)
    
    # for date in daterange(start_date, end_date):
    #     today = date.strftime("%Y-%m-%d")
    #     process_data_for_date(vertretungen, today, OUTPUT_FILE)