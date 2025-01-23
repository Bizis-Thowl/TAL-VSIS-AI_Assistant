from datetime import datetime

weekDaysMapping = ("montag", "dienstag", 
                   "mittwoch", "donnerstag",
                   "freitag", "samstag", "sonntag")

def get_weekday(target_date: str) -> str:
    weekday_num = datetime.strptime(target_date, '%Y-%m-%d').weekday()
    weekday = weekDaysMapping[weekday_num]
    return weekday