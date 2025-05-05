from typing import List, Dict

def collect_alternatives(assigned_pairs_list: List[List[Dict]]) -> List[List[Dict]]:
    '''
    Transpose the list of assigned pairs
    This is done to get the best alternative for each client
    The first list contains the best alternative for each client
    The second list contains the second best alternative for each client
    The third list contains the third best alternative for each client
    '''
    alternatives_list = []
    
    for assigned_pair in assigned_pairs_list[0]:
        alternatives_element = [assigned_pair]
        client = assigned_pair["klient"]
        for i in range(1,3):
            for j, alternative_pair in enumerate(assigned_pairs_list[i]):
                alternative_client = alternative_pair["klient"]
                if alternative_client == client:
                    alternatives_element.append(alternative_pair)
                    break
        alternatives_list.append(alternatives_element)  
    
    return alternatives_list