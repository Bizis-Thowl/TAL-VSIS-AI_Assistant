from learning.retrieve_model import retrieve_model
from typing import Tuple, List, Dict
import pandas as pd
from datetime import datetime

import json

features_names = ["timeToSchool", "priority", "ma_availability", "mobility", "geschlecht_relevant", "qualifications_met"]

class LearningHandler:
    
    def __init__(self):
        
        self.model = retrieve_model("isolation_forest.pkl")
    
    
    def predict_and_score(self, datapoint: List[List]) -> Tuple[int, float]:
        print(datapoint)
        pred = self.model.predict(datapoint)
        sample = self.model.score_samples(datapoint)
        
        return pred, float("{:.2f}".format(sample[0]))

    def prepare_data(self, assignment: Dict, employees: pd.DataFrame, clients: pd.DataFrame) -> List[List] | None:
        
        # Find the corresponding rows
        m_id = assignment["ma"]
        c_id = assignment["klient"]
        replacements = self._create_replacements_df(m_id, c_id)
        # Ensure the IDs in the mapping are comparable with `mas` and `clients`
        replacements = replacements.merge(employees, left_on="mas", right_on="id", how="inner")
        replacements = replacements.merge(clients, left_on="clients", right_on="id", how="inner", suffixes=("_mas", "_client"))

        print(replacements)
        
        date = datetime.today().strftime('%Y-%m-%d')
        base_availability = self._get_base_availability()
        
        # Extract the relevant information
        result = replacements.apply(lambda row: {
            "ma_id": row["id_mas"],
            "client_id": row["id_client"],
            "date": date,
            "timeToSchool": json.loads(row["timeToSchool"]).get(row["school"]),
            "priority": row["priority"],
            "ma_availability": row["availability"] == base_availability,
            "mobility": row["hasCar"],
            "geschlecht_relevant": row["requiredSex"] != None,
            "qualifications_met": all(e in row["qualifications"] for e in row["neededQualifications"])
        }, axis=1)
        
        result_df = pd.DataFrame(result.tolist())
        
        result_df = result_df.filter(items=features_names)
        print(result_df)
        
        # Format the single row contained in the dataframe
        formatted_row = [list(result_df.iloc[0])]
        
        return formatted_row
    
    def _get_base_availability(self):
        # Base availability
        start = datetime.strptime("00:00:00", '%H:%M:%S').time()
        default_end = datetime.strptime("23:59:59", '%H:%M:%S').time()
        start_as_float = start.hour + start.minute / 60
        default_end_as_float = default_end.hour + default_end.minute / 60
        base_availability = (start_as_float, default_end_as_float)
        
        return base_availability
    

    def _convert_list_to_df(self, data: List[Dict], id_cols: List) -> pd.DataFrame:
        # Convert to DataFrame
        df = pd.json_normalize(data)

        # Flatten list columns by extracting 'id' values
        for col in id_cols:
            try:
                df[col] = df[col].apply(lambda x: ', '.join(d['id'] for d in x) if isinstance(x, list) else '')
            except KeyError:
                print(f"Column {col} does not exist")
            
        return df
    
    def _create_replacements_df(self, ma: str, klient: str) -> pd.DataFrame:
        ma_client_mapping = {"mas": [ma], "clients": [klient]}
        for i, elem in enumerate(ma_client_mapping["mas"]):
            ma_client_mapping["mas"][i] = elem
            ma_client_mapping["clients"][i] = ma_client_mapping["clients"][i]
            
        df = pd.DataFrame(ma_client_mapping)
        return df