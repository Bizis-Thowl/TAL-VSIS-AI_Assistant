from typing import Tuple, List, Dict

def filter_kabw_records(records: List, assigned_mas: List) -> Tuple[List]:
    
    # Initialize filtered list
    entities = {
        "free_mas": [],
        "absent_clients": [],
    }
    
    for record in records:
        if record["typ"] == "kabw":
            entities = assign_kabw_record(record, entities, assigned_mas)
                
    return entities

def assign_kabw_record(record: Dict, entities: Dict, assigned_mas: List) -> Dict:
    
    if 'mavertretend' not in record:
        if 'klientabwesend' in record:
            entities["absent_clients"].append(record)
        if 'mafrei' in record and record["mafrei"]["id"]: # not in assigned_mas:
            entities["free_mas"].append(record)
            
        
    return entities