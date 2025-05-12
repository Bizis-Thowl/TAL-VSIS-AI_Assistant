import numpy as np
import json
from learning.model import AbnormalityModel 
# To ensure that the minimized value is high and can be converted to ints for using it to set constraints
scaling_factor = 1000000

class SoftConstrainedHandler:
    def __init__(self, employees, clients, assignments, unassigned_clients, model, learner_dataset, abnormality_model: AbnormalityModel, weights=None):
        self.employees = employees
        self.clients = clients
        self.assignments = assignments
        self.unassigned_clients = unassigned_clients
        self.model = model
        self.learner_dataset = learner_dataset
        self.abnormality_model = abnormality_model
        # Compute feature statistics for standardization
        self.travel_time_mean, self.travel_time_std = self._compute_travel_time_stats()
        self.time_window_mean, self.time_window_std = self._compute_time_window_stats()
        self.priority_mean, self.priority_std = self._compute_priority_stats()

        # Weights for each objective (default values if not provided)
        self.weights = weights or {
            "unassigned": 1000,
            "travel_time": 30,
            "time_window": 10,
            "priority": 160,
            "abnormality": 200,
            "client_experience": 100,
            "school_experience": 100,
            "short_term_client_experience": 100
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

    def _compute_abnormality(self, i, j):
        """Compute abnormality of employee-client pair."""
        pair_features = self.learner_dataset[(i, j)]
        
        print("pair_features: ", pair_features)
        datapoint = list(pair_features.values())
        print("datapoint: ", datapoint)
        
        score = self.abnormality_model.score_samples([datapoint])[0]
        
        int_score = int(round(score * scaling_factor))
        print("int_score: ", int_score)
        # return negative score to minimize
        return -int_score
    
    def _compute_short_term_client_experience_stats(self):
        """Compute mean and standard deviation of short term client experience scores."""
        short_term_client_experience_scores = []
        for i, ma in self.employees.iterrows():
            short_term_client_experience_scores.append(ma["short_term_cl_experience"])
        return np.mean(short_term_client_experience_scores), np.std(short_term_client_experience_scores) if short_term_client_experience_scores else (0, 1)
    
    def _compute_short_term_client_experience_objective(self):
        """Objective 8: Minimize total short term client experience scores."""
        return self.weights["short_term_client_experience"] * sum(self._compute_short_term_client_experience(i, j) for (i, j) in self.assignments)
    
    def _compute_short_term_client_experience(self, i, j):
        """Compute short term client experience score for assignment (i,j)."""
        employee = self.employees.iloc[i]
        client_id = self.clients.iloc[j]["id"]
        short_term_experience = json.loads(employee["short_term_cl_experience"]).get(client_id, 0)
        return short_term_experience
    
    def _compute_client_experience_stats(self):
        """Compute mean and standard deviation of client experience scores."""
        client_experience_scores = []
        for i, ma in self.employees.iterrows():
            client_experience_scores.append(ma["cl_experience"])
        return np.mean(client_experience_scores), np.std(client_experience_scores) if client_experience_scores else (0, 1)
    
    def _compute_school_experience_stats(self):
        """Compute mean and standard deviation of school experience scores."""
        school_experience_scores = []
        for i, ma in self.employees.iterrows():
            school_experience_scores.append(ma["school_experience"])
        return np.mean(school_experience_scores), np.std(school_experience_scores) if school_experience_scores else (0, 1)

    def _compute_client_experience_objective(self):
        """Objective 6: Minimize total client experience scores."""
        return self.weights["client_experience"] * sum(self._compute_client_experience(i, j) for (i, j) in self.assignments)

    def _compute_client_experience(self, i, j):
        """Compute client experience score for assignment (i,j)."""
        employee = self.employees.iloc[i]
        client_id = self.clients.iloc[j]["id"]
        client_experience = json.loads(employee["cl_experience"]).get(client_id, 0)
        return client_experience


    def _compute_school_experience_objective(self):
        """Objective 7: Minimize total school experience scores."""
        return self.weights["school_experience"] * sum(self._compute_school_experience(i, j) for (i, j) in self.assignments)
    
    def _compute_school_experience(self, i, j):
        """Compute school experience score for assignment (i,j)."""
        employee = self.employees.iloc[i]
        client_school = self.clients.iloc[j]["school"]
        school_experience = json.loads(employee["school_experience"]).get(client_school, 0)
        return school_experience

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
        return self.weights["unassigned"] * sum(self.unassigned_clients) * scaling_factor

    def _get_travel_time_term(self, i, j):
        """Normalized travel time term for assignment (i,j)."""
        employee = self.employees.iloc[i]
        client_school = self.clients.iloc[j]["school"]
        time_to_school = json.loads(employee["timeToSchool"]).get(client_school, 0)
        normalized_time = self._normalize(time_to_school, self.travel_time_mean, self.travel_time_std)
        
        # Scale and round to integer
        scaled_time = int(round(normalized_time * scaling_factor))    
        
        return self.assignments[(i, j)] * scaled_time

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
        
        # Scale and round to integer
        scaled_diff = int(round(normalized_diff * scaling_factor))
        
        return self.assignments[(i, j)] * scaled_diff

    def _compute_time_window_objective(self):
        """Objective 3: Minimize total time window differences."""
        return self.weights["time_window"] * sum(self._get_time_window_diff_term(i, j) for (i, j) in self.assignments)

    def _get_priority_term(self, i, j):
        """Normalized priority term for assignment (i,j)."""
        client_priority = self.clients.iloc[j]["priority"]
        normalized_priority = self._normalize(client_priority, self.priority_mean, self.priority_std)
        
        # Scale and round to integer
        scaled_priority = int(round(normalized_priority * scaling_factor))
        
        return self.assignments[(i, j)] * scaled_priority

    def _compute_priority_objective(self):
        """Objective 4: Minimize total priority scores of assigned clients."""
        return self.weights["priority"] * sum(self._get_priority_term(i, j) for (i, j) in self.assignments)
    
    def _compute_abnormality_objective(self):
        """Objective 5: Minimize total abnormality scores of assigned clients."""
        return self.weights["abnormality"] * sum(self._compute_abnormality(i, j) for (i, j) in self.assignments)

    def set_up_objectives(self):
        """Combine and set all optimization objectives in the model."""
        total_objective = (
            self._compute_unassigned_objective()
            + self._compute_travel_time_objective()
            + self._compute_time_window_objective()
            + self._compute_priority_objective()
            + self._compute_abnormality_objective()
            + self._compute_client_experience_objective()
            + self._compute_school_experience_objective()
            + self._compute_short_term_client_experience_objective()
        )
        self.model.minimize(total_objective)
        return self.model
