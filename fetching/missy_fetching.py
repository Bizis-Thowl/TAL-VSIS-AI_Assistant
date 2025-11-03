import requests
import json
from datetime import datetime
from endpoints import endpoints_missy
from utils.read_file import read_file
from base64 import b64encode
from utils.daterange import daterange
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

def get_vertretungen(request_info: List[Dict[str, str]], date: str, use_cache = False) -> List[Any]:
    
    endpoint_key = 'vertretungsfall'
    
    vertretungen = fetch_many(request_info, use_cache=use_cache, endpoint_key=endpoint_key, date=date)
    
    return vertretungen

def get_clients(request_info: List[Dict[str, str]], use_cache = True) -> List[Any]:
    
    endpoint_key = 'klient'
    
    clients = fetch_many(request_info, use_cache=use_cache, endpoint_key=endpoint_key)
    
    return clients

def get_mas(request_info: List[Dict[str, str]], use_cache = True) -> List[Any]:
    
    endpoint_key = 'ma'
    
    mas = fetch_many(request_info, use_cache=use_cache, endpoint_key=endpoint_key)
    
    return mas

def get_prio_assignments(request_info: List[Dict[str, str]]) -> List[Any]:
    
    endpoint_key = 'prio_assignment'
    
    assignments = fetch_object(request_info[0], endpoint_key)
    
    return assignments

def get_distances(request_info: List[Dict[str, str]], use_cache = True) -> List[Any]:
    
    endpoint_key = 'dist_ma_sch'
    
    distances = fetch_many(request_info, use_cache=use_cache, endpoint_key=endpoint_key)
    
    return distances

def parallel_fetch_object(request_info: Dict[str, str], endpoint_key: str, parallel: bool = False, max_workers: int = 5, date: str = None) -> List[Any]:
    responses = []
    if parallel and len(request_info) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_info = {executor.submit(fetch_object, info, endpoint_key, date): info for info in request_info}
            for future in as_completed(future_to_info):
                responses.append(future.result())
    else:
        for info in request_info:
            responses.append(fetch_object(info, endpoint_key, date))
            
    return responses

def fetch_object(request_info: Dict[str, str], endpoint_key: str, date: str = None) -> List[Any]:
    
    user = request_info['user']
    pw = request_info['pw']
    base_url = request_info['url']
    url = f"{base_url}{endpoints_missy[endpoint_key]}"
    if date:
        url += f"?datum={date}"
    
    token = b64encode(f"{user}:{pw}".encode('utf-8')).decode("ascii")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Basic {token}'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response_object = response.json()
    
    # Always only one element in the response
    return list(response_object.values())[0]

def fetch_many(
    request_info: List[Dict[str, str]],
    parallel: bool = False,
    max_workers: int = 5,
    use_cache: bool = True,
    endpoint_key: str = None,
    date: str = None
) -> List[Any]:

	# requests_info: [{ 'user': str, 'pw': str, 'endpoint_key': str }, ...]

    if not use_cache:
        responses = parallel_fetch_object(request_info, endpoint_key, parallel, max_workers, date)
        handle_cache_update(responses, endpoint_key)
    else:
        responses = read_file(endpoint_key)
        if responses is None:
            responses = parallel_fetch_object(request_info, endpoint_key, parallel, max_workers, date)
            handle_cache_update(responses, endpoint_key)
            
    combined = []
    for resp in responses:
        combined.extend(resp)
    
    return combined

def fetch_date_objects_in_range(request_info: List[Dict[str, str]], endpoint_key: str, start_date: date, end_date: date):
    
    response_objects = []
    
    for date in daterange(start_date, end_date):
        print(f"Fetching {endpoint_key} for {date}")
        response_object = fetch_many(request_info, endpoint_key=endpoint_key, date=date)
        for elem in response_object:
            response_objects.append(elem)
        
    handle_cache_update(response_objects, f"{endpoint_key}_all")
        
    return response_objects


def handle_cache_update(response_object: Any, endpoint_key: str) -> None:
    json_object = json.dumps(response_object, indent=4)
    with open(f"data/{endpoint_key}.json", "w") as outfile:
        outfile.write(json_object)

def filter_records_w_date(records, date):
    
    filtered_records = []
    
    # Convert target_date to datetime for comparison
    target_date = datetime.strptime(date, '%Y-%m-%d')
    
    for record in records:
        # Convert startdatum and enddatum to datetime for comparison
        start_date = datetime.strptime(record['startdatum'], '%Y-%m-%d')
        end_date = datetime.strptime(record['enddatum'], '%Y-%m-%d')
        
        if start_date <= target_date <= end_date:
            filtered_records.append(record)
            
    return filtered_records