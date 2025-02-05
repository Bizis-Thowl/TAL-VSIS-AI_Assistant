import requests
from requests.auth import HTTPBasicAuth
import json
from datetime import datetime
from endpoints import endpoints_missy
from config import base_url_missy
from utils.read_file import read_file
from base64 import b64encode


def get_vertretungen(date, user, pw, use_cache = True, update_cache = False):
    
    endpoint_key = 'vertretungsfall'
    
    vertretungen = abstract_fetch_w_date(user, pw, endpoint_key, date, use_cache, update_cache)
    
    vertretungen = vertretungen["VMBegleitungList"]
    
    return vertretungen

def get_clients(user, pw, use_cache = True, update_cache = False):
    
    endpoint_key = 'klient'
    
    clients = abstract_fetch(user, pw, endpoint_key, use_cache, update_cache)
    
    clients = clients["KlientInList"]
    
    return clients

def get_mas(user, pw, use_cache = True, update_cache = False):
    
    endpoint_key = 'ma'
    
    mas = abstract_fetch(user, pw, endpoint_key, use_cache, update_cache)
    
    mas = mas["MitarbeiterInList"]
    
    return mas

def get_prio_assignments(user, pw, use_cache = True, update_cache = False):
    
    endpoint_key = 'prio_assignment'
    
    assignments = abstract_fetch(user, pw, endpoint_key, use_cache, update_cache)
    
    assignments = assignments["VertretungAbList"]
    
    return assignments

def get_distances(user, pw, use_cache = True, update_cache = False):
    
    endpoint_key = 'dist_ma_sch'
    
    distances = abstract_fetch(user, pw, endpoint_key, use_cache, update_cache)
    
    distances = distances["MASchuleDistanzGMapList"]
    
    return distances

def fetch_object(user, pw, endpoint_key):
    
    token = b64encode(f"{user}:{pw}".encode('utf-8')).decode("ascii")
    url = f"{base_url_missy}{endpoints_missy[endpoint_key]}"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Basic {token}'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response_object = response.json()
    
    return response_object

def fetch_object_w_date(user, pw, endpoint_key, date):
    
    
    token = b64encode(f"{user}:{pw}".encode('utf-8')).decode("ascii")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Basic {token}'
    }
    response = requests.get(f"{base_url_missy}{endpoints_missy[endpoint_key]}?datum={date}", headers=headers)
    response_object = response.json()
    
    return response_object

def handle_cache_update(response_object, endpoint_key):
    json_object = json.dumps(response_object, indent=4)
    with open(f"data/{endpoint_key}.json", "w") as outfile:
        outfile.write(json_object)
        
    return response_object

def abstract_fetch(user, pw, endpoint_key, use_cache = True, update_cache = False):
    if update_cache:
        response_object = fetch_object(user, pw, endpoint_key)
        handle_cache_update(response_object, endpoint_key)
    elif use_cache:
        response_object = read_file(endpoint_key)
        if response_object is None:
            response_object = fetch_object(user, pw, endpoint_key)
            handle_cache_update(response_object, endpoint_key)
    else:
        response_object = fetch_object(user, pw, endpoint_key)
        
    return response_object

def abstract_fetch_w_date(user, pw, endpoint_key, date, use_cache = True, update_cache = False):
    if update_cache:
        response_object = fetch_object_w_date(user, pw, endpoint_key, date)
        handle_cache_update(response_object, endpoint_key)
    elif use_cache:
        response_object = read_file(endpoint_key)
        response_object = response_object["VMBegleitungList"]
        response_object = filter_records_w_date(response_object, date)
        if response_object is None:
            response_object = fetch_object_w_date(user, pw, endpoint_key, date)
            handle_cache_update(response_object, endpoint_key)
    else:
        response_object = fetch_object_w_date(user, pw, endpoint_key, date)
        
    return response_object

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