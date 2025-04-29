from typing import Dict, List, Tuple
import html as html_escape

from fetching.ai_communication import update_recommendation
from utils.add_comment import (
    get_employee_customer_comment,
    get_employee_comments,
    get_customer_comments,
    get_ai_comments
)
from utils.flatten_list import flatten


class RecommendationManager:
    """Handles recommendation generation and updates."""
    
    def __init__(self, user: str, pw: str):
        self.user = user
        self.pw = pw

    def create_short_explanation(self, recommendation_id: str, learner_info: Tuple[int, float]) -> str:
        """Create a short explanation for a recommendation."""
        expl = []
        expl.append(get_ai_comments(recommendation_id))
        expl = flatten(expl)
        expl = ", ".join(expl)
        return expl

    def create_explanation(self, ma_id: str, client_id: str, recommendation_id: str, learner_info: Tuple[int, float]) -> str:
        """Create a detailed explanation for a recommendation."""
        expl_specific = []
        expl_general = []
        
        expl_specific.append(get_employee_customer_comment(ma_id, client_id))
        expl_specific.append(get_employee_comments(ma_id))
        expl_specific.append(get_customer_comments(client_id))
        expl_general.append(get_ai_comments(recommendation_id))
        
        expl_specific = flatten(expl_specific)
        expl_general = flatten(expl_general)
        
        return self._generate_html(expl_specific, expl_general)

    def send_update(
        self,
        assigned_pairs: List[Dict],
        recommendation_ids: List[str],
        learner_infos: List[Tuple[int, float]],
        client_record_assignments: Dict
    ) -> Dict:
        """Send recommendation updates to the system."""
        client_id = None
        ma_ids = []
        expl_shorts = []
        expls = []

        for i, assigned_pair in enumerate(assigned_pairs):
            client_id = assigned_pair["klient"]
            ma_id = assigned_pair["ma"]
            ma_ids.append(ma_id)
            
            incident_id = client_record_assignments[client_id]
            expl_short = self.create_short_explanation(recommendation_ids[i], learner_infos[i])
            expl_shorts.append(expl_short)
            
            expl = self.create_explanation(
                ma_id, client_id, recommendation_ids[i], learner_infos[i]
            )
            expls.append(expl)

        out = update_recommendation(
            self.user,
            self.pw,
            incident_id,
            list(zip(ma_ids, expl_shorts, expls))
        )

        user_recommendation = {
            "incident_id": incident_id,
            "ma_id": ma_id,
            "client_id": client_id,
            "short_explanation": expl_short,
            "explanation": expl,
        }
        
        return user_recommendation

    def _generate_html(self, input_list1: List[str], input_list2: List[str]) -> str:
        """Generate HTML for explanations."""
        escaped_items1 = [html_escape.escape(item) for item in input_list1]
        escaped_items2 = [html_escape.escape(item) for item in input_list2]

        list_items1 = "\n".join([f"    <li>{item}</li>" for item in escaped_items1])
        list_items2 = "\n".join([f"    <li>{item}</li>" for item in escaped_items2])

        html_template = f"""<!DOCTYPE html>
        <html>
        <head>
        <style>
        .list-container {{
            max-width: 600px;
            margin: 20px auto;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .custom-list {{
            font-family: Arial, sans-serif;
            font-size: 16px;
            line-height: 1.6;
            color: #333;
            padding-left: 30px;
        }}

        .custom-list li {{
            margin-bottom: 8px;
            padding: 6px 0;
            border-bottom: 1px solid #eee;
        }}

        .custom-list li:last-child {{
            border-bottom: none;
        }}

        .custom-list li:hover {{
            background-color: #f1f1f1;
            transition: background-color 0.2s;
        }}
        </style>
        </head>
        <body>
        <div class="list-container">
        <ul class="custom-list">
        {list_items1}
        </ul>
        <br />
        <br />
        <ul class="custom-list">
        {list_items2}
        </ul>
        </div>
        </body>
        </html>"""

        return html_template 