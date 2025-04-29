from datetime import date, timedelta
from typing import Generator
def daterange(start_date: date, end_date: date) -> Generator[date, None, None]:
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)