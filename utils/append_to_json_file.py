import json
import os
from datetime import datetime

from config import log_store

def append_to_json_file(obj, file_name):
    """
    Appends a given Python object to a JSON file, using the current timestamp as the key.
    
    Parameters:
        obj (any): The Python object to store (must be JSON serializable).
        file_name (str): Path to the JSON file where the data should be stored.
    """
    timestamp = datetime.now().isoformat()  # Generate timestamp key
    file_path = f"{log_store}/{file_name}"

    # Load existing data if the file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}  # Handle empty or corrupted files
    else:
        data = {}

    # Append new data with timestamp key
    data[timestamp] = obj

    # Save updated data back to the file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    return timestamp  # Return the timestamp for reference