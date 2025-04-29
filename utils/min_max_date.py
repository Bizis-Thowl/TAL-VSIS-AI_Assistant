from datetime import datetime
from typing import List, Tuple

def min_max_date(vertretungen: List) -> Tuple[datetime, datetime]:
    
    # Get min and max dates from vertretungen
    start_dates = [
        datetime.strptime(v["startdatum"], "%Y-%m-%d").date() for v in vertretungen
    ]
    end_dates = [
        datetime.strptime(v["enddatum"], "%Y-%m-%d").date() for v in vertretungen
    ]

    min_date = min(start_dates)
    max_date = max(end_dates)
    
    return min_date, max_date
