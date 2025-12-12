from typing import List, Tuple, Dict
from utils.create_explanations import create_explanation, create_short_explanation
from fetching.ai_communication import update_recommendation

def send_update(
    request_info: List[Dict[str, str]],
    assigned_pairs: List[List],
    recommendation_ids: List[str],
    client_record_assignments: Dict[str, Dict],
) -> Dict:

    '''
    Send the update to the AI assistant
    '''
    client_id = None

    ma_ids = []
    expl_shorts = []
    expls = []
    
    incident_ids = []
    orgs = []
    recommended_ma_ids = []
    recommended_expl_shorts = []
    recommended_expls = []
    recommended_client_ids = []

    for i, assigned_pair in enumerate(assigned_pairs):
        client_id = assigned_pair["klient"]
        ma_id = assigned_pair["ma"]
        ma_ids.append(ma_id)
        print(f"Client ID: {client_id}")
        incident_id = client_record_assignments[client_id]["id"]
        org = client_record_assignments[client_id]["org"]
        expl_short = create_short_explanation(recommendation_ids[i])
        expl_shorts.append(expl_short)
        expl = create_explanation(
            ma_id, client_id, recommendation_ids[i]
        )
        expls.append(expl)
        incident_ids.append(incident_id)
        orgs.append(org)
        recommended_ma_ids.append(ma_id)
        recommended_expl_shorts.append(expl_short)
        recommended_expls.append(expl)
        recommended_client_ids.append(client_id)
    out = update_recommendation(
        request_info, incident_id, list(zip(ma_ids, expl_shorts, expls)), org
    )
    print(out)
    return incident_ids, orgs, recommended_ma_ids, recommended_expl_shorts, recommended_expls, recommended_client_ids


def send_empty_update(request_info: List[Dict[str, str]], incident_id: str, org: str):
    update_recommendation(request_info, incident_id, [], org)
