import numpy as np
import json

class SoftConstrainedHandler:
    def __init__(self, employees, clients, assignments, unassigned_clients, model, weights=None):
        self.employees = employees
        self.clients = clients
        self.assignments = assignments
        self.unassigned_clients = unassigned_clients
        self.model = model

        # Compute feature statistics for standardization
        self.travel_time_mean, self.travel_time_std = self._compute_travel_time_stats()
        self.time_window_mean, self.time_window_std = self._compute_time_window_stats()
        self.priority_mean, self.priority_std = self._compute_priority_stats()

        # Weights for each objective (default values if not provided)
        self.weights = weights or {
            "unassigned": 5.0,
            "travel_time": 2.0,
            "time_window": 1.0,
            "priority": 3.0,
        }

    def _compute_travel_time_stats(self):
        """Compute mean and standard deviation of travel times for standardization."""
        travel_times = []
        for i, employee in self.employees.iterrows():
            for j, client in self.clients.iterrows():
                client_school = client["school"]
                time_to_school = json.loads(employee["timeToSchool"]).get(client_school, 0)
                travel_times.append(time_to_school)
        return np.mean(travel_times), np.std(travel_times) if travel_times else (0, 1)

    def _compute_time_window_stats(self):
        """Compute mean and standard deviation of time window differences."""
        time_diffs = []
        for i, employee in self.employees.iterrows():
            for j, client in self.clients.iterrows():
                client_time_window = client["timeWindow"]
                if client_time_window:
                    client_time_end = client_time_window[1]
                    time_diff = employee["availability"][1] - client_time_end
                    time_diffs.append(time_diff)
        return np.mean(time_diffs), np.std(time_diffs) if time_diffs else (0, 1)

    def _compute_priority_stats(self):
        """Compute mean and standard deviation of client priority values."""
        priorities = [client["priority"] for _, client in self.clients.iterrows()]
        return np.mean(priorities), np.std(priorities) if priorities else (0, 1)

    def _normalize(self, value, mean, std):
        """Normalize value using z-score normalization."""
        return (value - mean) / std if std > 0 else 0

    def _compute_unassigned_objective(self):
        """Objective 1: Minimize unassigned clients."""
        return self.weights["unassigned"] * sum(self.unassigned_clients)

    def _get_travel_time_term(self, i, j):
        """Normalized travel time term for assignment (i,j)."""
        employee = self.employees.iloc[i]
        client_school = self.clients.iloc[j]["school"]
        time_to_school = json.loads(employee["timeToSchool"]).get(client_school, 0)
        normalized_time = self._normalize(time_to_school, self.travel_time_mean, self.travel_time_std)
        return self.assignments[(i, j)] * normalized_time

    def _compute_travel_time_objective(self):
        """Objective 2: Minimize total normalized travel time."""
        return self.weights["travel_time"] * sum(self._get_travel_time_term(i, j) for (i, j) in self.assignments)

    def _get_time_window_diff_term(self, i, j):
        """Normalized time window difference term for assignment (i,j)."""
        employee_avail_end = self.employees.iloc[i]["availability"][1]
        client_time_window = self.clients.iloc[j]["timeWindow"]

        if client_time_window is None:
            return 0  # No penalty if client has no time window

        client_time_end = client_time_window[1]
        time_diff = employee_avail_end - client_time_end
        normalized_diff = self._normalize(time_diff, self.time_window_mean, self.time_window_std)
        return self.assignments[(i, j)] * normalized_diff

    def _compute_time_window_objective(self):
        """Objective 3: Minimize total time window differences."""
        return self.weights["time_window"] * sum(self._get_time_window_diff_term(i, j) for (i, j) in self.assignments)

    def _get_priority_term(self, i, j):
        """Normalized priority term for assignment (i,j)."""
        client_priority = self.clients.iloc[j]["priority"]
        normalized_priority = self._normalize(client_priority, self.priority_mean, self.priority_std)
        return self.assignments[(i, j)] * normalized_priority

    def _compute_priority_objective(self):
        """Objective 4: Minimize total priority scores of assigned clients."""
        return self.weights["priority"] * sum(self._get_priority_term(i, j) for (i, j) in self.assignments)

    def set_up_objectives(self):
        """Combine and set all optimization objectives in the model."""
        total_objective = (
            self._compute_unassigned_objective()
            + self._compute_travel_time_objective()
            + self._compute_time_window_objective()
            + self._compute_priority_objective()
        )
        self.model.minimize(total_objective)
        return self.model
