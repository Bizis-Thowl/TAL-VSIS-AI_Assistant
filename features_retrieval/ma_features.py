from datetime import datetime
from typing import List, Dict, Tuple
import json
import pandas as pd
from utils.add_comment import add_employee_comment

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

def prepare_distances(distances, ma_id):
    # Preprocess the distances into a dictionary for faster lookups
    distance_dict = {}
    for distance in distances:
        if distance["mitarbeiterin"]["id"] == ma_id:
            school_id = distance["schule"]["id"][0:6]  # Extract the first 6 characters of the school ID
            if school_id not in distance_dict:
                distance_dict[school_id] = distance
    
    return distance_dict
    
def create_commute_info(ma_id: str, clients: dict, distances: list):
    # Preprocess the distances into a dictionary for faster lookups
    distance_dict = prepare_distances(distances, ma_id)

    # Build the result using the preprocessed dictionary
    result = {}
    for school_id in clients["school"]:
        if school_id is None: continue
        school_prefix = school_id[:6]  # Extract the first 6 characters of the school ID
        if school_prefix in distance_dict:
            dist_data = distance_dict[school_prefix]
            dist = dist_data.get("einfachdistanzluft")
            if dist is not None and dist < 80000:
                result[school_prefix] = dist

    if len(result) == 0:
        add_employee_comment(ma_id, "Es gibt keine Klienten im Einzugsgebiet des Mitarbeiters (< 80 km)")
        
    if len(result) == 1:
        add_employee_comment(ma_id, "Es gibt nur einen Klienten im Einzugsgebiet des Mitarbeiters (< 80 km)")

    return result

def get_mobility(ma):
    if ma.get("mobilitaet") != None and ma["mobilitaet"] != []:
        return True
    else:
        return False