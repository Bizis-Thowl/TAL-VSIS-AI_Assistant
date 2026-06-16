from datetime import datetime, timedelta

import pandas as pd

from optimize.utils.is_eligible_pair import get_travel_distance, is_eligible_pair


def _has_client_experience_in_log(ma_id, client_id, ma_client_map):
    return bool(ma_client_map.get(ma_id, {}).get(client_id))


def _has_school_experience_in_log(ma_id, school_id, ma_school_map):
    return bool(ma_school_map.get(ma_id, {}).get(school_id))


def _has_short_term_client_experience_in_log(
    ma_id, client_id, date_str, ma_client_map
):
    experience_dates = ma_client_map.get(ma_id, {}).get(client_id, [])
    if not experience_dates:
        return False
    reference_date = datetime.strptime(date_str, "%Y-%m-%d")
    two_weeks_ago = reference_date - timedelta(weeks=2)
    return any(
        datetime.fromisoformat(experience_date) >= two_weeks_ago
        for experience_date in experience_dates
    )


def _eligible_employee_indices(clients_df, mas_df, client_pos):
    client = clients_df.iloc[client_pos]
    eligible_indices = []
    distances = []
    for emp_pos in range(len(mas_df)):
        employee = mas_df.iloc[emp_pos]
        if not is_eligible_pair(employee, client):
            continue
        distance = get_travel_distance(employee, client)
        if distance is None:
            continue
        eligible_indices.append(emp_pos)
        distances.append(distance)
    return eligible_indices, distances


def _employee_pos_for_id(mas_df, ma_id):
    if ma_id is None:
        return None
    matches = mas_df.index[mas_df["id"] == ma_id]
    if len(matches) == 0:
        return None
    return mas_df.index.get_loc(matches[0])


def build_client_day_log_rows(
    optimizer,
    clients_df,
    mas_df,
    date_str,
    ma_client_map,
    ma_school_map,
    weight_variant,
):
    if optimizer.soft_constrained_handler is None:
        return []

    objective_contributions = (
        optimizer.soft_constrained_handler.compute_per_client_objective_contributions()
    )
    assigned_pairs, unassigned_client_ids = optimizer.get_solution_assignments()
    assignment_by_client = {pair["klient"]: pair["ma"] for pair in assigned_pairs}
    unassigned_client_ids = set(unassigned_client_ids)

    rows = []
    for client_pos in range(len(clients_df)):
        client = clients_df.iloc[client_pos]
        client_id = client["id"]
        school_id = client.get("school")
        eligible_indices, eligible_distances = _eligible_employee_indices(
            clients_df, mas_df, client_pos
        )
        assigned_ma_id = assignment_by_client.get(client_id)
        assigned_ma_pos = _employee_pos_for_id(mas_df, assigned_ma_id)

        chosen_distance = None
        if assigned_ma_pos is not None:
            chosen_distance = get_travel_distance(
                mas_df.iloc[assigned_ma_pos], client
            )

        objective = objective_contributions[client_pos]
        rows.append(
            {
                "date": date_str,
                "client_id": client_id,
                "weight_variant": weight_variant,
                "versorgt": client_id not in unassigned_client_ids,
                "eligible_ma_count": len(eligible_indices),
                "min_distance": min(eligible_distances) if eligible_distances else None,
                "chosen_distance": chosen_distance,
                "assigned_ma_id": assigned_ma_id,
                "has_client_experience": (
                    _has_client_experience_in_log(
                        assigned_ma_id, client_id, ma_client_map
                    )
                    if assigned_ma_id is not None
                    else False
                ),
                "has_short_term_client_experience": (
                    _has_short_term_client_experience_in_log(
                        assigned_ma_id, client_id, date_str, ma_client_map
                    )
                    if assigned_ma_id is not None
                    else False
                ),
                "has_school_experience": (
                    _has_school_experience_in_log(
                        assigned_ma_id, school_id, ma_school_map
                    )
                    if assigned_ma_id is not None and school_id is not None
                    else False
                ),
                "priority": client.get("priority"),
                "objective_unassigned": objective["unassigned"],
                "objective_travel_time": objective["travel_time"],
                "objective_time_window": objective["time_window"],
                "objective_priority": objective["priority"],
                "objective_client_experience": objective["client_experience"],
                "objective_school_experience": objective["school_experience"],
                "objective_short_term_client_experience": objective[
                    "short_term_client_experience"
                ],
                "objective_availability_gap": objective["availability_gap"],
                "objective_abnormality": objective.get("abnormality", 0),
                "objective_total": sum(objective.values()),
            }
        )
    return rows


def rows_to_dataframe(rows):
    return pd.DataFrame(rows)
