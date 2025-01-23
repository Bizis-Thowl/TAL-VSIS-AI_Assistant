from datetime import datetime
from typing import List, Dict, Tuple
import json
import pandas as pd

def aggregate_ma_features(ma_objects: List, distances: List, clients_dict: Dict) -> Tuple[pd.DataFrame, Dict]:
    ma_dict = {
        "id": [],
        "qualifications": [],
        "sex": [],
        "hasCar": [],
        "timeToSchool": [],
        "availability": [],
    }
    for ma in ma_objects:
        ma_dict["id"].append(ma["id"][0:6])
        ma_dict["qualifications"].append(get_ma_qualifications(ma))
        # TODO Implement
        ma_dict["sex"].append(None)
        commute_time = create_commute_info(ma["id"], clients_dict, distances)
        ma_dict["timeToSchool"].append(json.dumps(commute_time))        
        ma_dict["hasCar"].append(get_mobility(ma))        
        ma_dict["availability"].append(get_ma_availability(ma))
        
    ma_df = pd.DataFrame.from_dict(ma_dict)
    
    return ma_df, ma_dict


def get_ma_qualifications(ma):
    attributes = []
    if ma.get("kanndiabetes", 0) == 1:
        attributes.append("diabetes")
    if ma.get("kannpflege", 0) == 1:
        attributes.append("pflege")

    return attributes

def get_ma_availability(ma):
    start = datetime.strptime("00:00:00", '%H:%M:%S').time()
    default_end = datetime.strptime("23:59:59", '%H:%M:%S').time()
    start_as_float = start.hour + start.minute / 60
    if (ma.get("zeitlicheeinschraenkung-uhrzeit") == None):
        end_as_float = default_end.hour + default_end.minute / 60
        return (start_as_float, end_as_float)
    else:
        end = datetime.strptime(ma["zeitlicheeinschraenkung-uhrzeit"], '%H:%M:%S').time()
        end_as_float = end.hour + end.minute / 60
        return (start_as_float, end_as_float)
    
def create_commute_info(ma_id: str, clients: dict, distances: list):
    result = {}
    
    print(f"ma id: {ma_id}")

    for i, school_id in enumerate(clients["school"]):
        distance = list(filter(lambda x: x["mitarbeiterin"]["id"] == ma_id and x["schule"]["id"][0:6] == school_id, distances))
        if (type(distance) is list and len(distance) > 0):
            result[school_id[0:6]] = distance[0].get("einfachdistanzluft")
            
    return result

def get_mobility(ma):
    if ma.get("mobilitaet") != None and ma["mobilitaet"] != []:
        return True
    else:
        return False