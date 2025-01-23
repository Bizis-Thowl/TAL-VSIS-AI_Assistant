from typing import Tuple, List, Dict

def filter_records(records: List) -> Tuple[List]:
    
    # Initialize filtered list
    entities = {
        "open_clients": [],
        "rescheduled_mas": [],
        "free_mas": [],
        "absent_clients": [],
    }
    
    for record in records:
        if record["typ"] == "mabw":
            entities = assign_mabw_record(record, entities)
        elif record["typ"] == "kabw":
            entities = assign_kabw_record(record, entities)
                
    return entities

def assign_mabw_record(record: Dict, entities: Dict) -> Dict:
    
    if 'klientzubegleiten' in record and 'maabwesend' in record:
        entities["open_clients"].append(record)
        if 'mavertretend' in record:
            entities["rescheduled_mas"].append(record)
    
    return entities

def assign_kabw_record(record: Dict, entities: Dict) -> Dict:
    
    if 'klientabwesend' in record:
        entities["absent_clients"].append(record)
    elif 'mafrei' in record:
        entities["free_mas"].append(record)
        
    return entities