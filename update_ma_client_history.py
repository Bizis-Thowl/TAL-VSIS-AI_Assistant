import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

from fetching.missy_fetching import (
    get_clients
)
from utils.daterange import daterange
from utils.min_max_date import min_max_date

import os
from dotenv import load_dotenv
import schedule
import time

# load .env file to environment
load_dotenv(override=True)

user = os.getenv("USER")
pw = os.getenv("PASSWORD")

INPUT_FILE = "data/vertretungsfall_all.json"      # Contains the list of raw elements
OUTPUT_FILE = "data/experience_log.json"  # Where the adapted list is saved

def load_json_file(path: str) -> List[Dict]:
    """Load JSON data from file or return empty list if file doesn't exist."""
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_json_file(path: str, data: List[Dict]) -> None:
    """Save JSON data to file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def _initialize_experience_maps(experience_list: List[Dict]) -> tuple[Dict, Dict]:
    """Initialize experience maps from existing experience list."""
    ma_experience_map = {}
    ma_school_experience_map = {}
    
    for entry in experience_list:
        ma_id = entry["ma"]
        ma_experience_map[ma_id] = entry.get("client_experience", {})
        ma_school_experience_map[ma_id] = entry.get("school_experience", {})
    
    return ma_experience_map, ma_school_experience_map

def _process_entry_dates(entry: Dict) -> tuple[Optional[datetime.date], Optional[datetime.date]]:
    """Extract and validate start and end dates from an entry."""
    try:
        start = datetime.strptime(entry["startdatum"], "%Y-%m-%d").date()
        end = datetime.strptime(entry["enddatum"], "%Y-%m-%d").date()
        return start, end
    except (ValueError, KeyError) as e:
        print(f"Skipping entry with invalid dates: {entry}. Error: {e}")
        return None, None

def _update_experience_maps(
    ma_experience_map: Dict,
    ma_school_experience_map: Dict,
    ma_id: str,
    client_id: str,
    school_id: str,
    target_date_str: str
) -> None:
    """Update both client and school experience maps for a given MA."""
    # Initialize maps for new MA
    if ma_id not in ma_experience_map:
        ma_experience_map[ma_id] = {}
    if ma_id not in ma_school_experience_map:
        ma_school_experience_map[ma_id] = {}

    # Update client experience
    if client_id not in ma_experience_map[ma_id]:
        ma_experience_map[ma_id][client_id] = []
    if target_date_str not in ma_experience_map[ma_id][client_id]:
        ma_experience_map[ma_id][client_id].append(target_date_str)
        print(f"Added client assignment: MA {ma_id} with client {client_id} on {target_date_str}")

    # Update school experience
    if school_id not in ma_school_experience_map[ma_id]:
        ma_school_experience_map[ma_id][school_id] = []
    if target_date_str not in ma_school_experience_map[ma_id][school_id]:
        ma_school_experience_map[ma_id][school_id].append(target_date_str)
        print(f"Added school assignment: MA {ma_id} at school {school_id} on {target_date_str}")

def process_data_for_date(request_info: List[Dict], data: List[Dict], target_date: datetime, output_path: str) -> None:
    """Process assignments for a specific date and update experience logs."""
    
    clients = get_clients(request_info, use_cache=True)
    
    date_str = target_date.strftime("%Y-%m-%d")
    print(f"Processing assignments for date: {date_str}")
    experience_list = load_json_file(output_path)
    ma_experience_map, ma_school_experience_map = _initialize_experience_maps(experience_list)

    for entry in data:
        start, end = _process_entry_dates(entry)
        if not start or not end or not (start <= target_date <= end):
            continue

        ma_vertretend = entry.get("mavertretend")
        if not ma_vertretend:
            continue
            
        ma_id = ma_vertretend["id"]
        client_id = entry["klientzubegleiten"]["id"]
        client_object = next((client for client in clients if client["id"] == client_id), None)
        
        if not client_object:
            print(f"Warning: Client {client_id} not found in clients list")
            continue
            
        school_id = client_object["schule"]["id"]

        _update_experience_maps(
            ma_experience_map,
            ma_school_experience_map,
            ma_id,
            client_id,
            school_id,
            date_str
        )

    # Rebuild and save the experience list
    updated_experience_list = [
        {
            "ma": ma_id,
            "client_experience": ma_experience_map[ma_id],
            "school_experience": ma_school_experience_map[ma_id]
        }
        for ma_id in set(ma_experience_map.keys()) | set(ma_school_experience_map.keys())
    ]
    save_json_file(output_path, updated_experience_list)

if __name__ == "__main__":
    input_data = load_json_file(INPUT_FILE)
    min_date, max_date = min_max_date(input_data)
    for date in daterange(min_date, max_date + timedelta(days=1)):
        process_data_for_date(input_data, date, OUTPUT_FILE)
        
    # Schedule the function to run daily at 10 PM
    # schedule.every().day.at("22:00").do(process_data_for_date, input_data, date, OUTPUT_FILE)
    # # Keep the script running
    # while True:
    #     schedule.run_pending()
    #     time.sleep(6000)
    