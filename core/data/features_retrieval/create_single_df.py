import pandas as pd
import json

from utils.base_availability import base_availability

def create_single_df(clients_df: pd.DataFrame, mas_df: pd.DataFrame, replacements: pd.DataFrame, date: str) -> pd.DataFrame:
    
    # Ensure the IDs in the mapping are comparable with `mas` and `clients`
    replacements = replacements.merge(mas_df, left_on="mas", right_on="id", how="inner")
    replacements = replacements.merge(clients_df, left_on="clients", right_on="id", how="inner", suffixes=("_mas", "_client"))

    # Extract the relevant information
    result = replacements.apply(lambda row: {
        "ma_id": row["id_mas"],
        "client_id": row["id_client"],
        "date": date,
        "timeToSchool": json.loads(row["timeToSchool"]).get(row["school"]),
        "cl_experience": json.loads(row["cl_experience"]).get(row["id_client"]),
        "school_experience": json.loads(row["school_experience"]).get(row["school"]),
        "priority": row["priority"],
        "ma_availability": row["availability"] == base_availability,
        "mobility": row["hasCar"],
        "geschlecht_relevant": row["requiredSex"] != None,
        "qualifications_met": all(e in row["qualifications"] for e in row["neededQualifications"])
    }, axis=1)

    # Explicitly convert to DataFrame
    return pd.DataFrame(result.tolist())
