import pandas as pd
from typing import Dict

def create_replacements(replacing_mas_to_clients: Dict):
    ma_client_mapping = {"mas": list(replacing_mas_to_clients.keys()), "clients": list(replacing_mas_to_clients.values())}
    for i, elem in enumerate(ma_client_mapping["mas"]):
        ma_client_mapping["mas"][i] = elem
        
    df = pd.DataFrame(ma_client_mapping)
    return df