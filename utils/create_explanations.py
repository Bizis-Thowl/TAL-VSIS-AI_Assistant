from utils.add_comment import (
    get_employee_customer_comment,
    get_employee_comments,
    get_customer_comments,
    get_ai_comments,
)
from utils.generate_html import generate_html
from typing import Tuple
from utils.flatten_list import flatten

def create_explanation(ma_id, client_id, recommendation_id):
    expl_specific = []
    expl_general = []
    expl_specific.append(get_employee_customer_comment(ma_id, client_id))
    expl_specific.append(get_employee_comments(ma_id))
    expl_specific.append(get_customer_comments(client_id))
    expl_general.append(get_ai_comments(recommendation_id))
    expl_specific = flatten(expl_specific)
    expl_general = flatten(expl_general)
    expl = generate_html(expl_specific, expl_general)

    return expl

def create_short_explanation(
    recommendation_id: str
) -> str:
    expl = []
    expl.append(get_ai_comments(recommendation_id))
    expl = flatten(expl)
    expl = ", ".join(expl)

    return expl