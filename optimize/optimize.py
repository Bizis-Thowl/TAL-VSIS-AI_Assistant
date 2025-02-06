import cpmpy as cp
import pandas as pd
import json
from optimize.utils.has_required_qualifications import has_required_qualifications
from optimize.SoftConstraintHandler import SoftConstrainedHandler
import logging
from utils.append_to_json_file import append_to_json_file
from utils.add_comment import add_ai_comment, get_ai_comments
import uuid

logger = logging.getLogger(__name__)

class Optimizer:
    
    def __init__(self, employees: pd.DataFrame, clients: pd.DataFrame):
        # Define variables for employee self.assignments and client unassignment indicators
        self.assignments = {}
        self.unassigned_clients = []
        # Model instance
        self.model = cp.Model()
        
        self.employees = employees
        self.clients = clients

    def create_model(self):

        # Create decision variables and filter based on eligibility
        for i, emp in self.employees.iterrows():
            for j, client in self.clients.iterrows():
                if client["school"] in json.loads(emp["timeToSchool"]) and has_required_qualifications(emp["qualifications"], client["neededQualifications"]):
                    # Define a binary variable for this assignment
                    self.assignments[(i, j)] = cp.boolvar(name=f"assign_E{i}_C{j}")
                    self.assignments[(i, j)].set_description(f"E{i} is assigned to C{j}")

        # Create binary variables to represent unassigned clients
        for j in range(len(self.clients)):
            unassigned_var = cp.boolvar(name=f"unassigned_C{j}")
            unassigned_var.set_description(f"C{j} is not assigned")
            self.unassigned_clients.append(unassigned_var)

        # Primary Objective: Minimize the number of unassigned clients
        for j in range(len(self.clients)):
            self.model += [self.unassigned_clients[j] == 1 - sum(self.assignments[(i, j)] for i in range(len(self.employees)) if (i, j) in self.assignments)]

        soft_constrained_handler = SoftConstrainedHandler(self.employees, self.clients, self.assignments, self.unassigned_clients, self.model)
        self.model = soft_constrained_handler.set_up_objectives()

        # Constraints: Each employee and client can only be assigned once
        # Each employee can only be assigned to one client
        for i in range(len(self.employees)):
            self.model += [sum(self.assignments[(i, j)] for j in range(len(self.clients)) if (i, j) in self.assignments) <= 1]

        # Each client can only be assigned to one employee
        for j in range(len(self.clients)):
            self.model += [sum(self.assignments[(i, j)] for i in range(len(self.employees)) if (i, j) in self.assignments) <= 1]

        # All self.unassigned_clients shall be assigned
        # for j in range(len(self.clients)):
        #     self.model += sum(self.unassigned_clients) == 0

        # for elem in time_window_diffs:
        #     self.model += elem >= 0

    def solve_model(self):
        if self.model.solve(solver="ortools"):
            logger.info("Optimal solution found!")
            print("Optimal solution found!")
            # Extract solution
            solution_assignments = {(i, j): self.assignments[(i, j)].value() for (i, j) in self.assignments}
            solution_unassigned_clients = [var.value() for var in self.unassigned_clients]
            
            return solution_assignments, solution_unassigned_clients
        else:
            logger.info("No feasible solution found.")
            print("No feasible solution found.")
            return None
        
    def process_results(self):
        store_dict = {
            "assigned_pairs": None,
            "unassigned_clients": None,
            "avg_travel_time": None,
            "avg_priority": None
        }
        assigned_pairs = []
        for (i, j), var in self.assignments.items():
            if var.value() == 1:
                assigned_pairs.append({"ma": self.employees.iloc[i]["id"], "klient": self.clients.iloc[j]["id"]})
                print(f"Employee {self.employees.iloc[i]['id']} assigned to Client {self.clients.iloc[j]['id']}")
        
        # Output the unassigned clients
        unassigned_clients_list = [self.clients.iloc[j]["id"] for j in range(len(self.clients))
                                if self.unassigned_clients[j].value() == 1]

        print("\nUnassigned Clients:")
        print(unassigned_clients_list)
        store_dict["assigned_pairs"] = assigned_pairs
        store_dict["unassigned_clients"] = unassigned_clients_list

        # Display total travel time and time window difference for the optimal solution
        total_travel_time = [var.value() * json.loads(self.employees.iloc[i]["timeToSchool"])[self.clients.iloc[j]["school"]] 
                                for (i, j), var in self.assignments.items() if var.value() == 1]
        # total_priority = [var.value() * self.clients.iloc[j]["priority"] for (i, j), var in self.assignments.items() if var.value() == 1]
        total_priority = [self.clients.to_dict()["priority"][j] for (i, j), var in self.assignments.items() if var.value() == 1]
        total_time_window_diff = []

        for (i, j), var in self.assignments.items():
            if var.value() == 1:
                availability_end = self.employees.iloc[i]["availability"][1]
                kl_time_window = self.clients.iloc[j]["timeWindow"]
                if kl_time_window is None:
                    time_window_end = availability_end
                else:
                    time_window_end = kl_time_window[1]
                diff = var.value() * availability_end - time_window_end
                total_time_window_diff.append(diff)

        print(f"travel times: {total_travel_time}")
        # print(f"window diff times: {total_time_window_diff}")
        print("\nTotal Travel Time:", sum(total_travel_time))
        # print("Total Time Window Difference:", sum(total_time_window_diff))
        
        store_dict["avg_travel_time"] = sum(total_travel_time) / len(assigned_pairs)
        print(total_priority)
        store_dict["avg_priority"] = sum(total_priority) / len(assigned_pairs)
        
        recommendation_id = self._calculate_unique_recommendation_id()
        add_ai_comment(recommendation_id, f"Ø Luftlinie: {(store_dict['avg_travel_time'] / 1000):.2f} km")
        add_ai_comment(recommendation_id, f"Ø Prio: {store_dict['avg_priority']:.2f}")
        
        append_to_json_file(store_dict, "recommendations.json")
        
        return assigned_pairs, recommendation_id
    
    def _calculate_unique_recommendation_id(self):
        
        return uuid.uuid4()