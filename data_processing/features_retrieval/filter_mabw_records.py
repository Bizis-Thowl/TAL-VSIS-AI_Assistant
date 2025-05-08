from typing import Tuple, List, Dict

def filter_mabw_records(records: List) -> Tuple[List]:
    
    # Initialize filtered list
    entities = {
        "open_clients": [],
        "rescheduled_mas": []
    }
    
    for record in records:
        if record["typ"] == "mabw":
            entities = assign_mabw_record(record, entities)
                
    return entities

def assign_mabw_record(record: Dict, entities: Dict) -> Dict:
    
    if 'klientzubegleiten' in record and 'maabwesend' in record and 'mavertretend' not in record:
        entities["open_clients"].append(record)
    elif 'klientzubegleiten' in record and 'mavertretend' not in record:
        entities["open_clients"].append(record)
    elif 'mavertretend' in record:
        entities["rescheduled_mas"].append(record)
    
    return entities