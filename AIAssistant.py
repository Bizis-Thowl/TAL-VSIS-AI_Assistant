import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

from fetching.missy_fetching import get_vertretungen
from fetching.ai_communication import update_recommendation

from basic_retrieval.retrieve_ids import get_free_ma_ids, get_ma_assignments, get_open_client_ids, get_client_record_assignments
from basic_retrieval.retrieve_objects import get_objects_by_id
from basic_retrieval.filter_records import filter_records

from features_retrieval.client_features import aggregate_client_features
from features_retrieval.ma_features import aggregate_ma_features

from optimize.optimize import Optimizer
from utils.add_comment import reset_comments, get_customer_comments, get_employee_comments, get_ai_comments
from utils.flatten_list import flatten
from utils.append_to_json_file import append_to_json_file

from learning.LearningHandler import LearningHandler

class AIAssistant:
    
    def __init__(self, user, pw, mas, clients, prio_assignments, distances):
        
        self.mas = mas
        self.clients = clients
        self.prio_assignments = prio_assignments
        self.distances = distances
        
        self.vertretungen = []
        self.user = user
        self.pw = pw
        
        # Dataset
        self.clients_df = pd.DataFrame()
        self.mas_df = pd.DataFrame()
        self.client_record_assignments = {}
        
        # Solver
        self.optimizer = None
        
        # Learner
        self.learner = LearningHandler()
    
    def update_dataset(self) -> bool:
        
        # today = datetime.today().strftime('%Y-%m-%d')
        today = '2025-02-06'
        
        reset_comments()
        vertretungen = get_vertretungen(today, self.user, self.pw, update_cache=True)
        
        if len(vertretungen) != len(self.vertretungen):
            self.vertretungen = vertretungen
        else: return False
    
        print(f"Vertretungen für {today} gefetcht")
        
        filtered_records = filter_records(vertretungen)
        open_client_records = filtered_records["open_clients"]
        rescheduled_ma_records = filtered_records["rescheduled_mas"]
        absent_client_records = filtered_records["absent_clients"]
        free_ma_records = filtered_records["free_mas"]
        
        print("Records extrahiert")
        
        open_client_ids = get_open_client_ids(open_client_records)
        free_ma_ids = get_free_ma_ids(free_ma_records, absent_client_records, self.mas)
        ma_assignments = get_ma_assignments(rescheduled_ma_records)
        self.client_record_assignments = get_client_record_assignments(open_client_records)
        
        append_to_json_file(open_client_ids, "open_clients.json")
        append_to_json_file(free_ma_ids, "free_mas.json")
        append_to_json_file(ma_assignments, "ma_assignments.json")
        
        print("ids gesammelt")
        
        open_client_objects = get_objects_by_id(self.clients, open_client_ids)
        free_ma_objects = get_objects_by_id(self.mas, free_ma_ids)
        
        print(open_client_objects)
        print(free_ma_objects)
        
        print("Objekte erzeugt")
        
        self.clients_df, clients_dict = aggregate_client_features(open_client_objects, today, self.prio_assignments)
        self.mas_df, mas_dict = aggregate_ma_features(free_ma_objects, self.distances, clients_dict)
        
        print(self.clients_df)
        print(self.mas_df)
        
        print("Datensätze erstellt")
        
        return True
        
    def update_solver_model(self):
        is_not_initialized = self._handle_data_not_initialized() and not self._handle_optimizer_not_initialized()
        if not is_not_initialized:
            self.optimizer = Optimizer(self.mas_df, self.clients_df)
            self.optimizer.create_model()
        else:
            self.optimizer.create_model()
        
    def solve_model(self):
        is_not_initialized = self._handle_data_not_initialized() and self._handle_optimizer_not_initialized()
        if not is_not_initialized:
            self.optimizer.solve_model()
        
    def process_results(self):
        is_not_initialized = self._handle_data_not_initialized() and self._handle_optimizer_not_initialized()
        if not is_not_initialized:
            assigned_pairs, recommendation_id = self.optimizer.process_results()

        return assigned_pairs, recommendation_id
        
    def prepare_learner_data(self, assignments: Dict) -> List[List]:
        
        return self.learner.prepare_data(assignments, self.mas_df, self.clients_df)
    
    def retrieve_learner_scores(self, datapoint: List[List]) -> Tuple[int, float]:
        
        return self.learner.predict_and_score(datapoint)
    
    def send_update(self, assigned_pair: Dict, recommendation_id: str, learner_info: Tuple[int, float]):
        
        
        client_id = assigned_pair["klient"]
        ma_id = assigned_pair["ma"]
        incident_id = self.client_record_assignments[client_id]
        expl_short = self._create_short_explanation(recommendation_id, learner_info)
        expl = self._create_explanation(ma_id, client_id, recommendation_id)
        out = update_recommendation(self.user, self.pw, incident_id, ma_id, expl_short, expl)
        user_recommendation = {
            "incident_id": incident_id,
            "ma_id": ma_id,
            "client_id": client_id,
            "short_explanation": expl_short,
            "explanation": expl
        }
        print(out)
        return user_recommendation
    
    # Some helper functions are here...
    
    def _create_short_explanation(self, recommendation_id: str, learner_info: Tuple[int, float]) -> str:
        expl = []
        learner_pred = learner_info[0]
        learner_score = learner_info[1]
        expl.append([f"Normal ({learner_score})" if learner_pred == 1 else f"Unnormal ({learner_score})"])
        # expl.append(get_ai_comments(recommendation_id))
        expl = str(expl)
        expl = ', '.join(expl)
        
        print(expl)
        
        return expl
    
    def _create_explanation(self, ma_id, client_id, recommendation_id):
        expl = []
        expl.append(get_employee_comments(ma_id))
        expl.append(get_customer_comments(client_id))
        expl.append(get_ai_comments(recommendation_id))
        expl = str(expl)
        expl = ', '.join(expl)
        
        return expl
    
    def _data_not_initialized(self):
        return self.clients_df.empty or self.mas_df.empty
    
    def _optimizer_not_initialized(self):
        return self.optimizer is None
    
    def _handle_data_not_initialized(self):
        if self._data_not_initialized(): 
            print("Please update the dataset, for it is empty")
            return True
        else:
            return False
        
    def _handle_optimizer_not_initialized(self):
        if self._optimizer_not_initialized(): 
            print("Please update the optimizer, for it is not initialized")
            return True
        else:
            return False