from typing import List, Tuple, Dict
from utils.create_explanations import create_explanation, create_short_explanation
from fetching.ai_communication import update_recommendation

def send_update(
    user: str,
    pw: str,
    assigned_pairs: List[List],
    recommendation_ids: List[str],
    learner_infos: List[Tuple[int, float]],
    client_record_assignments: Dict[str, str],
) -> Dict:

    '''
    Send the update to the AI assistant
    '''
    client_id = None

    ma_ids = []
    expl_shorts = []
    expls = []

    for i, assigned_pair in enumerate(assigned_pairs):
        client_id = assigned_pair["klient"]
        ma_id = assigned_pair["ma"]
        ma_ids.append(ma_id)
        print(f"Client ID: {client_id}")
        incident_id = client_record_assignments[client_id]
        expl_short = create_short_explanation(recommendation_ids[i], learner_infos[i])
        expl_shorts.append(expl_short)
        expl = create_explanation(
            ma_id, client_id, recommendation_ids[i], learner_infos[i]
        )
        expls.append(expl)
    out = update_recommendation(
        user, pw, incident_id, list(zip(ma_ids, expl_shorts, expls))
    )
    user_recommendation = {
        "incident_id": incident_id,
        "ma_id": ma_id,
        "client_id": client_id,
        "short_explanation": expl_short,
        "explanation": expl,
    }
    print(out)
    return user_recommendation
