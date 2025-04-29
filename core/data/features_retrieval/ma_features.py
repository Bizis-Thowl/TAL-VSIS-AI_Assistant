from datetime import datetime
from typing import List, Dict, Tuple
import json
import pandas as pd
from utils.add_comment import add_employee_comment

def aggregate_ma_features(ma_objects: List, distances: List, clients_dict: Dict, experience_log: List) -> Tuple[pd.DataFrame, Dict]:
    ma_dict = {
        "id": [],
        "qualifications": [],
        "sex": [],
        "cl_experience": [],
        "school_experience": [],
        "hasCar": [],
        "timeToSchool": [],
        "availability": [],
    }
    for ma in ma_objects:
        ma_dict["id"].append(ma["id"])
        ma_dict["qualifications"].append(get_ma_qualifications(ma))
        # TODO Implement
        ma_dict["sex"].append(None)
        clients_experience = get_client_experience(ma["id"], clients_dict, experience_log)
        schools_experience = get_school_experience(ma["id"], clients_dict, experience_log)
        ma_dict["cl_experience"].append(json.dumps(clients_experience))
        ma_dict["school_experience"].append(json.dumps(schools_experience))
        commute_time = create_commute_info(ma["id"], clients_dict, distances)
        ma_dict["timeToSchool"].append(json.dumps(commute_time))
        ma_dict["hasCar"].append(get_mobility(ma))        
        ma_dict["availability"].append(get_ma_availability(ma))
        
    ma_df = pd.DataFrame.from_dict(ma_dict)
    
    return ma_df, ma_dict

def _get_experience(ma_id: str, experience_log: List[Dict], experience_type: str, entity_ids: List[str]) -> Dict[str, int]:
    """
    Helper function to get experience counts for a given MA.
    
    Args:
        ma_id: The ID of the MA
        experience_log: List of experience entries
        experience_type: Type of experience ('client_experience' or 'school_experience')
        entity_ids: List of entity IDs to check experience for
        
    Returns:
        Dictionary mapping entity IDs to their experience count
    """
    experience_dict = {}
    
    # Find MA's experience entry
    ma_experience = next(
        (entry for entry in experience_log if entry["ma"] == ma_id),
        None
    )
    
    if not ma_experience:
        return experience_dict
    
    # Get the specific experience type (client or school)
    experience_data = ma_experience.get(experience_type, {})
    
    # Count experience for each entity
    for entity_id in entity_ids:
        entity_experience = experience_data.get(entity_id, [])
        if entity_experience:
            experience_dict[entity_id] = len(entity_experience)
    
    return experience_dict

def get_client_experience(ma_id: str, clients_dict: Dict, experience_log: List[Dict]) -> Dict[str, int]:
    """
    Get the number of times an MA has worked with each client.
    
    Args:
        ma_id: The ID of the MA
        clients_dict: Dictionary containing client information
        experience_log: List of experience entries
        
    Returns:
        Dictionary mapping client IDs to their experience count
    """
    return _get_experience(
        ma_id=ma_id,
        experience_log=experience_log,
        experience_type="client_experience",
        entity_ids=clients_dict["id"]
    )

def get_school_experience(ma_id: str, clients_dict: Dict, experience_log: List[Dict]) -> Dict[str, int]:
    """
    Get the number of times an MA has worked at each school.
    
    Args:
        ma_id: The ID of the MA
        clients_dict: Dictionary containing client information
        experience_log: List of experience entries
        
    Returns:
        Dictionary mapping school IDs to their experience count
    """
    return _get_experience(
        ma_id=ma_id,
        experience_log=experience_log,
        experience_type="school_experience",
        entity_ids=clients_dict["school"]
    )
    
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