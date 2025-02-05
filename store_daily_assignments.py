import json
import schedule
import time

from config import log_store

def store_daily_assignments():
    # Load the JSON data
    with open(f"{log_store}/ma_assignments.json", "r") as f:
        data = json.load(f)
    
    # Sort timestamps and group by date
    sorted_entries = sorted(data.items(), key=lambda x: x[0])
    daily_snapshots = {}
    
    for timestamp, value in sorted_entries:
        date_str = timestamp.split("T")[0]  # Extract YYYY-MM-DD
        daily_snapshots[date_str] = (timestamp, value)  # Keep last entry of the day
    
    # Prepare the final snapshot dictionary
    snapshot = {date: daily_snapshots[date][1] for date in daily_snapshots}
    
    # Save the snapshot file
    with open(f"{log_store}/daily_assignments.json", "w") as f:
        json.dump(snapshot, f, indent=4)

# Schedule the function to run daily at 10 PM
schedule.every().day.at("22:00").do(store_daily_assignments)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(60)