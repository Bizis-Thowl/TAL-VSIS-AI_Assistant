import json
from datetime import datetime
from pathlib import Path

INPUT_FILE = "data/vertretungsfall.json"      # Contains the list of raw elements
OUTPUT_FILE = "data/experience_log.json"  # Where the adapted list is saved

def load_json_file(path):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def process_data_for_date(data, target_date_str, output_path):
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    experience_list = load_json_file(output_path)

    # Convert to dict for easier updating
    ma_experience_map = {entry["ma"]: entry["client_experience"] for entry in experience_list}

    for entry in data:
        start = datetime.strptime(entry["startdatum"], "%Y-%m-%d").date()
        end = datetime.strptime(entry["enddatum"], "%Y-%m-%d").date()

        if start <= target_date <= end:
            has_mavertretend = entry.get("mavertretend") != None
            if not has_mavertretend: continue
            ma_id = entry["mavertretend"]["id"]
            client_id = entry["klientzubegleiten"]["id"]

            if ma_id not in ma_experience_map:
                ma_experience_map[ma_id] = {}

            if client_id in ma_experience_map[ma_id]:
                ma_experience_map[ma_id][client_id] += 1
            else:
                ma_experience_map[ma_id][client_id] = 1

    # Rebuild the list for saving
    updated_experience_list = [
        {"ma": ma_id, "client_experience": client_exp}
        for ma_id, client_exp in ma_experience_map.items()
    ]

    save_json_file(output_path, updated_experience_list)

# Example usage
if __name__ == "__main__":
    input_data = load_json_file(INPUT_FILE)["VMBegleitungList"]
    process_data_for_date(input_data, "2025-01-13", OUTPUT_FILE)