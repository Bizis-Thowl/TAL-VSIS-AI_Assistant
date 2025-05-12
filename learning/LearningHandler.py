from typing import Tuple, List, Dict
import pandas as pd
from utils.add_comment import add_employee_customer_comment
from learning.model import AbnormalityModel
from utils.base_availability import base_availability

import json

class LearningHandler:
    
    def __init__(self, abnormality_model: AbnormalityModel):
        
        self.abnormality_model = abnormality_model
    
    def predict_and_score(self, datapoint: List[List]) -> Tuple[int, float]:
        pred = self.abnormality_model.predict(datapoint)
        sample = self.abnormality_model.score_samples(datapoint)
        
        return pred, float("{:.2f}".format(sample[0]))
    
    def get_explanation(self, datapoint: List[List]) -> List[List]:
        shap_values = self.abnormality_model.get_explanation(datapoint[0])
        return shap_values[0]

    def prepare_data(self, assignment: Dict, employees: pd.DataFrame, clients: pd.DataFrame) -> List[List] | None:
        
        # Find the corresponding rows
        m_id = assignment["ma"]
        c_id = assignment["klient"]
        
        emp = employees.loc[employees['id'] == m_id].iloc[0]
        client = clients.loc[clients['id'] == c_id].iloc[0]
        
        time_to_school = json.loads(emp["timeToSchool"]).get(client["school"])
        priority = client["priority"]
        qualifications_met = all(e in emp["qualifications"] for e in client["neededQualifications"])
        
        add_employee_customer_comment(m_id, c_id, f"Luftlinie: {time_to_school / 1000} km")
        # add_employee_customer_comment(ma_id, kl_id, f"Priorität: {priority}")
        if not qualifications_met:
            add_employee_customer_comment(m_id, c_id, "Qualifikationen sind laut Datensatz nicht ausreichend")
        
        cl_experience = json.loads(emp["cl_experience"]).get(client["id"])
        school_experience = json.loads(emp["school_experience"]).get(client["school"])
        short_term_cl_experience = json.loads(emp["short_term_cl_experience"]).get(client["id"])

        add_employee_customer_comment(m_id, c_id, "Mit Auto" if emp["hasCar"] else "Ohne Auto")
        add_employee_customer_comment(m_id, c_id, f"Erfahrung mit Klient: {cl_experience}")
        add_employee_customer_comment(m_id, c_id, f"Erfahrung mit Schule: {school_experience}")
        add_employee_customer_comment(m_id, c_id, f"Erfahrung mit Klient in letzter Woche: {short_term_cl_experience}")

        combined_data = {
            "timeToSchool": time_to_school,
            "cl_experience": cl_experience,
            "school_experience": school_experience,
            "short_term_cl_experience": short_term_cl_experience,
            "priority": priority,
            "ma_availability": emp["availability"] == base_availability,
            "mobility": emp["hasCar"],
            "geschlecht_relevant": client["requiredSex"] != None,
            "qualifications_met": qualifications_met
        }
        
        print("combined_data: ", combined_data)
        
        return [list(combined_data.values())]