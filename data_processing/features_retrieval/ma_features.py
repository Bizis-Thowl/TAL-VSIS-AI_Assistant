from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
import pandas as pd
from utils.add_comment import add_employee_comment

def aggregate_ma_features(ma_objects: List, distances: List, clients_dict: Dict, experience_log: List, date: str) -> Tuple[pd.DataFrame, Dict]:
    ma_dict = {
        "id": [],
        "qualifications": [],
        "sex": [],
        "cl_experience": [],
        "school_experience": [],
        "short_term_cl_experience": [],
        "hasCar": [],
        "timeToSchool": [],
        "availability": [],
    }
    for ma in ma_objects:
        ma_dict["id"].append(ma["id"])
        ma_dict["qualifications"].append(get_ma_qualifications(ma))
        # TODO Implement
        ma_dict["sex"].append(None)
        experiences = get_experiences(ma["id"], clients_dict, experience_log, date)
        ma_dict["cl_experience"].append(json.dumps(experiences["client_experience"]))
        ma_dict["school_experience"].append(json.dumps(experiences["school_experience"]))
        ma_dict["short_term_cl_experience"].append(json.dumps(experiences["short_term_client_experience"]))
        commute_time = create_commute_info(ma["id"], clients_dict, distances)
        ma_dict["timeToSchool"].append(json.dumps(commute_time))
        ma_dict["hasCar"].append(get_mobility(ma))        
        ma_dict["availability"].append(get_ma_availability(ma))
        
    ma_df = pd.DataFrame.from_dict(ma_dict)
    
    return ma_df, ma_dict

def get_short_term_client_experience_dict(ma_experience: Dict, clients_dict: Dict, date: str) -> Dict[str, int]:
    
    experience_dict = {}	
    # Get the client experience data
    experience_data = ma_experience.get("client_experience", {})
    
    # Calculate the date two weeks ago
    two_weeks_ago = datetime.strptime(date, "%Y-%m-%d") - timedelta(weeks=2)
    
    # Count experience for each client within the last two weeks
    for client_id in clients_dict["id"]:
        client_experience = experience_data.get(client_id, [])
        if client_experience:
            # Filter dates within last two weeks and count them
            recent_experiences = [
                date for date in client_experience 
                if datetime.fromisoformat(date) >= two_weeks_ago
            ]
            experience_dict[client_id] = len(recent_experiences)
    
    return experience_dict
    
def get_experiences(ma_id: str, clients_dict: Dict, experience_log: List[Dict], date: str) -> Dict[str, int]:
    
    experience_dict = {
        "client_experience": {},
        "short_term_client_experience": {},
        "school_experience": {}
    }
    
    # Find MA's experience entry
    ma_experience = next(
        (entry for entry in experience_log if entry["ma"] == ma_id),
        None
    )
    
    if not ma_experience:
        return experience_dict
    
    experience_dict["client_experience"] = get_client_experience_dict(ma_experience, clients_dict)
    experience_dict["short_term_client_experience"] = get_short_term_client_experience_dict(ma_experience, clients_dict, date)
    experience_dict["school_experience"] = get_school_experience_dict(ma_experience, clients_dict)
    
    return experience_dict

def get_client_experience_dict(ma_experience: Dict, clients_dict: Dict) -> Dict[str, int]:
    
    experience_dict = {}	
    # Get the client experience data
    experience_data = ma_experience.get("client_experience", {})
    
    # Count experience for each client
    for client_id in clients_dict["id"]:
        client_experience = experience_data.get(client_id, [])
        if client_experience:
            experience_dict[client_id] = len(client_experience)
    
    return experience_dict

def get_school_experience_dict(ma_experience: Dict, clients_dict: Dict) -> Dict[str, int]:
    
    experience_dict = {}
        # Get the school experience data
    experience_data = ma_experience.get("school_experience", {})
    
    # Get unique school IDs from clients
    school_ids = set(clients_dict.get("school"))
    
    # Count experience for each school
    for school_id in school_ids:
        school_experience = experience_data.get(school_id, [])
        if school_experience:
            experience_dict[school_id] = len(school_experience)
    
    return experience_dict

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
            school_id = distance["schule"]["id"]
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
        school_prefix = school_id
        if school_prefix in distance_dict:
            dist_data = distance_dict[school_prefix]
            dist = dist_data.get("einfachdistanzluft")
            if dist is not None and dist < 60000:
                result[school_prefix] = dist

    if len(result) == 0:
        add_employee_comment(ma_id, "Es gibt keine Klienten im Einzugsgebiet des Mitarbeiters (< 60 km)")
        
    if len(result) == 1:
        add_employee_comment(ma_id, "Es gibt nur einen Klienten im Einzugsgebiet des Mitarbeiters (< 60 km)")

    return result

def get_mobility(ma):
    if ma.get("mobilitaet") != None and ma["mobilitaet"] != []:
        return True
    else:
        return False