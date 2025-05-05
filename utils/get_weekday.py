from datetime import datetime

weekDaysMapping = ("montag", "dienstag", 
                   "mittwoch", "donnerstag",
                   "freitag", "samstag", "sonntag")

def get_weekday(target_date: str) -> str:
    target_date = datetime.strptime(target_date, '%Y-%m-%d')
    weekday_num = target_date.weekday()
    weekday = weekDaysMapping[weekday_num]
    return weekday