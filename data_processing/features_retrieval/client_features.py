from typing import List, Dict
from datetime import datetime
import pandas as pd
from utils.get_weekday import get_weekday
from utils.add_comment import add_customer_comment

def aggregate_client_features(open_client_objects: List, date: datetime, prio_assignments: List, global_schools_mapping: Dict):
    client_dict = {
        "id": [],
        "neededQualifications": [],
        "requiredSex": [],
        "timeWindow": [],
        "priority": [],
        "school": []
    }
    weekday = get_weekday(date)
    for client in open_client_objects:
        if get_timewindow(client, weekday) is None:
            # add_customer_comment(client["id"], "Für diesen Klienten fehlt der Stundenplan")
            pass
        client_dict["id"].append(client["id"])
        client_dict["neededQualifications"].append(get_qualifications(client))
        client_dict["requiredSex"].append(client.get("begleitergeschlecht"))
        client_dict["timeWindow"].append(get_timewindow(client, weekday)) 
        priority_id = client.get("vertretungab")["id"] if client.get("vertretungab") != None else 100       
        client_dict["priority"].append(convert_priority(prio_assignments, priority_id))
        global_school_id = global_schools_mapping.get(client.get("schule", {}).get("id", None), None)
        client_dict["school"].append(global_school_id)
        
    client_df = pd.DataFrame.from_dict(client_dict)
    
    
    return client_df, client_dict

def get_qualifications(client):
    attributes = []
    if client.get("hatdiabetes", 0) == 1:
        attributes.append("diabetes")
    if client.get("brauchtpflege", 0) == 1:
        attributes.append("pflege")

    return attributes

def get_timewindow(client, weekday):
    timetable = client.get("aktuellerstundenplan")
    if (timetable == None) or timetable.get(f"{weekday}von") == None:
        return None
    start = timetable.get(f"{weekday}von")
    start_formatted = datetime.strptime(start, '%H:%M:%S').time()
    end = timetable.get(f"{weekday}bis")
    end_formatted = datetime.strptime(end, '%H:%M:%S').time()
    start_as_float = start_formatted.hour + start_formatted.minute / 60
    end_as_float = end_formatted.hour + end_formatted.minute / 60
    
    return (start_as_float, end_as_float)

def convert_priority(prio_assignments, priority_id):
    if priority_id == None:
        return 100
    else:
        for item in prio_assignments:
            if item.get('id') == priority_id:
                return item.get('reihenfolge', 100)
    return 100