import pandas as pd

from fetching.missy_fetching import get_vertretungen, get_distances, get_clients, get_mas, get_prio_assignments

from basic_retrieval.retrieve_ids import get_free_ma_ids, get_ma_assignments, get_open_client_ids, get_client_record_assignments
from basic_retrieval.retrieve_objects import get_objects_by_id
from basic_retrieval.filter_records import filter_records

from features_retrieval.client_features import aggregate_client_features
from features_retrieval.ma_features import aggregate_ma_features

from optimize.optimize import Optimizer
from utils.add_comment import add_customer_comments, add_employee_comment, reset_comments

today = "2024-08-22"

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
        self.clients_df = pd.DataFrame
        self.mas_df = pd.DataFrame
        self.client_record_assignments = {}
    
    def update_dataset(self):
        reset_comments()
        vertretungen = get_vertretungen(today, self.user, self.pw)
        
        if len(vertretungen) != len(self.vertretungen):
            self.vertretungen = vertretungen
        else: return
    
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
        
        print("ids gesammelt")
        
        open_client_objects = get_objects_by_id(self.clients, open_client_ids)
        free_ma_objects = get_objects_by_id(self.mas, free_ma_ids)
        
        print("Objekte erzeugt")
        
        self.clients_df, clients_dict = aggregate_client_features(open_client_objects, today, self.prio_assignments)
        self.mas_df, mas_dict = aggregate_ma_features(free_ma_objects, self.distances, clients_dict)
        
        print("Datensätze erstellt")