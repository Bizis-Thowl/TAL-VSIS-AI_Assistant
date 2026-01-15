import pandas as pd
import json
from utils.base_availability import base_availability

feature_names = ["timeToSchool", "cl_experience", "short_term_cl_experience", "school_experience", "priority", "ma_availability", "mobility", "geschlecht_relevant", "qualifications_met", "available_until_mas", "available_until_client"]

def create_single_df(clients_df: pd.DataFrame, mas_df: pd.DataFrame, replacements: pd.DataFrame, date: str = None) -> pd.DataFrame:
    
    # Ensure the IDs in the mapping are comparable with `mas` and `clients`
    replacements = replacements.merge(mas_df, left_on="mas", right_on="id", how="inner")
    replacements = replacements.merge(clients_df, left_on="clients", right_on="id", how="inner", suffixes=("_mas", "_client"))

    if len(replacements) == 0:
        return pd.DataFrame(columns=feature_names)

    # Extract the relevant information
    result = replacements.apply(lambda row: create_single_row(row, date), axis=1)
    
    # Convert the Series of dictionaries to a DataFrame with the correct columns
    result_df = pd.DataFrame(list(result))

    return result_df

def create_single_row(row: pd.Series, date: str) -> pd.Series:
    row_dict = {
        "ma_id": row["id_mas"],
        "client_id": row["id_client"],
        "date": date,
        "timeToSchool": json.loads(row["timeToSchool"]).get(row["school"], None),
        "cl_experience": row["cl_experience"].get(row["id_client"], 0),
        "school_experience": row["school_experience"].get(row["school"], 0),
        "short_term_cl_experience": row["short_term_cl_experience"].get(row["id_client"], 0),
        "priority": row["priority"],
        "ma_availability": row["availability"] == base_availability,
        "mobility": row["hasCar"],
        "geschlecht_relevant": row["requiredSex"] != None,
        "qualifications_met": all(e in row["qualifications"] for e in row["neededQualifications"]),
        "availability_gap": (row["available_until_mas"] - row["available_until_client"]).days
    }
    return row_dict
