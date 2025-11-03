import numpy as np
from datetime import datetime
import json

def compute_availability_gap_stats(employees, clients):
    """Compute mean and standard deviation of availability gaps."""
    availability_gaps = []
    for i, employee in employees.iterrows():
        for j, client in clients.iterrows():
            availability_gap = (
                datetime.strptime(employee["available_until"], "%Y-%m-%d") - datetime.strptime(client["available_until"], "%Y-%m-%d")
            )
            availability_gaps.append(availability_gap)
    return np.mean(availability_gaps), (np.std(availability_gaps) if availability_gaps else (0, 1))

def compute_travel_time_stats(employees, clients):
    """Compute mean and standard deviation of travel times for standardization."""
    travel_times = []
    for i, employee in employees.iterrows():
        for j, client in clients.iterrows():
            client_school = client["school"]
            time_to_school = json.loads(employee["timeToSchool"]).get(
                client_school, 0
            )
            travel_times.append(time_to_school)
    return np.mean(travel_times), np.std(travel_times) if travel_times else (0, 1)


def compute_short_term_client_experience_stats(employees):
    """Compute mean and standard deviation of short term client experience scores."""
    short_term_client_experience_scores = []
    for i, ma in employees.iterrows():
        short_term_client_experience_scores.append(ma["short_term_cl_experience"])
    return np.mean(short_term_client_experience_scores), (
        np.std(short_term_client_experience_scores)
        if short_term_client_experience_scores
        else (0, 1)
    )
    
def compute_client_experience_stats(employees):
    """Compute mean and standard deviation of client experience scores."""
    client_experience_scores = []
    for i, ma in employees.iterrows():
        client_experience_scores.append(ma["cl_experience"])
    return np.mean(client_experience_scores), (
        np.std(client_experience_scores) if client_experience_scores else (0, 1)
    )
    
def compute_school_experience_stats(employees):
    """Compute mean and standard deviation of school experience scores."""
    school_experience_scores = []
    for i, ma in employees.iterrows():
        school_experience_scores.append(ma["school_experience"])
    return np.mean(school_experience_scores), (np.std(school_experience_scores) if school_experience_scores else (0, 1))

def compute_time_window_stats(employees, clients):
    """Compute mean and standard deviation of time window differences."""
    time_diffs = []
    for i, employee in employees.iterrows():
        for j, client in clients.iterrows():
            client_time_window = client["timeWindow"]
            if client_time_window:
                client_time_end = client_time_window[1]
                time_diff = employee["availability"][1] - client_time_end
                time_diffs.append(time_diff)
    return np.mean(time_diffs), np.std(time_diffs) if time_diffs else (0, 1)

def compute_priority_stats(clients):
    """Compute mean and standard deviation of client priority values."""
    priorities = [client["priority"] for _, client in clients.iterrows()]
    return np.mean(priorities), np.std(priorities) if priorities else (0, 1)