import pandas as pd

from optimize.utils.is_eligible_pair import get_travel_distance, is_eligible_pair


def _experience_counts_for_assignment(employee, client_id, school_id):
    return {
        "client_experience": employee["cl_experience"].get(client_id, 0),
        "short_term_client_experience": employee["short_term_cl_experience"].get(
            client_id, 0
        ),
        "school_experience": (
            employee["school_experience"].get(school_id, 0)
            if school_id is not None
            else 0
        ),
    }

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
        experience_counts = {
            "client_experience": 0,
            "short_term_client_experience": 0,
            "school_experience": 0,
        }
        if assigned_ma_pos is not None:
            chosen_distance = get_travel_distance(
                mas_df.iloc[assigned_ma_pos], client
            )
            experience_counts = _experience_counts_for_assignment(
                mas_df.iloc[assigned_ma_pos], client_id, school_id
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
                "client_experience": experience_counts["client_experience"],
                "short_term_client_experience": experience_counts[
                    "short_term_client_experience"
                ],
                "school_experience": experience_counts["school_experience"],
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


def build_human_baseline_client_day_log_rows(
    clients_df,
    mas_df,
    date_str,
    weight_variant,
    assigned_pairs,
):
    """Log rows from actual human assignments (Missy labels). No optimizer involved."""
    assignment_by_client = {pair["klient"]: pair["ma"] for pair in assigned_pairs}
    assigned_client_ids = set(assignment_by_client.keys())

    zero_objective = {
        "unassigned": 0,
        "travel_time": 0,
        "time_window": 0,
        "priority": 0,
        "client_experience": 0,
        "school_experience": 0,
        "short_term_client_experience": 0,
        "availability_gap": 0,
        "abnormality": 0,
    }

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
        experience_counts = {
            "client_experience": 0,
            "short_term_client_experience": 0,
            "school_experience": 0,
        }
        if assigned_ma_pos is not None:
            chosen_distance = get_travel_distance(
                mas_df.iloc[assigned_ma_pos], client
            )
            experience_counts = _experience_counts_for_assignment(
                mas_df.iloc[assigned_ma_pos], client_id, school_id
            )

        rows.append(
            {
                "date": date_str,
                "client_id": client_id,
                "weight_variant": weight_variant,
                "versorgt": client_id in assigned_client_ids,
                "eligible_ma_count": len(eligible_indices),
                "min_distance": min(eligible_distances) if eligible_distances else None,
                "chosen_distance": chosen_distance,
                "assigned_ma_id": assigned_ma_id,
                "client_experience": experience_counts["client_experience"],
                "short_term_client_experience": experience_counts[
                    "short_term_client_experience"
                ],
                "school_experience": experience_counts["school_experience"],
                "priority": client.get("priority"),
                "objective_unassigned": zero_objective["unassigned"],
                "objective_travel_time": zero_objective["travel_time"],
                "objective_time_window": zero_objective["time_window"],
                "objective_priority": zero_objective["priority"],
                "objective_client_experience": zero_objective["client_experience"],
                "objective_school_experience": zero_objective["school_experience"],
                "objective_short_term_client_experience": zero_objective[
                    "short_term_client_experience"
                ],
                "objective_availability_gap": zero_objective["availability_gap"],
                "objective_abnormality": zero_objective["abnormality"],
                "objective_total": 0,
            }
        )
    return rows


def rows_to_dataframe(rows):
    return pd.DataFrame(rows)
