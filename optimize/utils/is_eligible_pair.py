import json
from typing import Any

from optimize.utils.has_required_qualifications import has_required_qualifications


def is_eligible_pair(employee: Any, client: Any) -> bool:
    blacklist_ids = {elem["id"] for elem in client["ma_blacklist"]}
    if employee["id"] in blacklist_ids:
        return False

    time_to_school = json.loads(employee["timeToSchool"])
    if client["school"] not in time_to_school:
        return False

    return has_required_qualifications(
        employee["qualifications"], client["neededQualifications"]
    )


def get_travel_distance(employee: Any, client: Any) -> int | None:
    if not is_eligible_pair(employee, client):
        return None
    return json.loads(employee["timeToSchool"]).get(client["school"])
