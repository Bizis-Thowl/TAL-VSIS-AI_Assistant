import os
from dotenv import load_dotenv
from fetching.missy_fetching import fetch_date_objects_in_range
from update_ma_client_history import process_data_for_date
from utils.daterange import daterange
from config import base_url_missy
from datetime import date, timedelta
import json
# load .env file to environment
load_dotenv(override=True)

request_specs = os.getenv("REQUEST_INFO")
request_specs = json.loads(request_specs)

OUTPUT_FILE = "data/experience_log.json"  # Where the adapted list is saved

if __name__ == "__main__":
    
    start_date = date(2025, 11, 23)
    end_date = date.today() + timedelta(days=1)
    endpoint_key = 'vertretungsfall'
    
    request_info = [{'user': spec['user'], 'pw': spec['pw'], 'url': base_url_missy.format(domain=spec['domain'])} for spec in request_specs]
    
    vertretungen = fetch_date_objects_in_range(request_info, endpoint_key, start_date, end_date)
    
    for date in daterange(start_date, end_date):
        process_data_for_date(request_info, vertretungen, date, OUTPUT_FILE)