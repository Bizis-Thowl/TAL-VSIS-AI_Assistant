import requests
from requests.auth import HTTPBasicAuth
import json
from base64 import b64encode

from config import base_url_ai

endpoint = "VMBegleitung"

def update_recommendation(
    user: str,
    pw: str,
    id: str,
    mavertretendvorschlag1: str,
    erklaerungvorschlagkurz1: str,
    erklaerungvorschlag1: str
):
    """
    Updates a AI recommendation record.

    Args:
        id (str): ID of the VMBegleitung record to update.
        mavertretendvorschlag1 (str): ID of the proposed substitute employee.
        erklaerungvorschlagkurz1 (str): Short description for a tabular listing (<= 200 chars).
        erklaerungvorschlag1 (str): Detailed information in HTML format (<= 8000 chars).
    
    Returns:
        dict: Response from the web service.
    """
    if len(erklaerungvorschlagkurz1) > 200:
        raise ValueError("erklaerungvorschlagkurz1 must be 200 characters or fewer.")
    if len(erklaerungvorschlag1) > 8000:
        raise ValueError("erklaerungvorschlag1 must be 8000 characters or fewer.")
    
    url = f"{base_url_ai}{endpoint}"
    
    print(url)
    
    payload = {
        "id": id,
        "mavertretendvorschlag1": mavertretendvorschlag1,
        "erklaerungvorschlagkurz1": erklaerungvorschlagkurz1,
        "erklaerungvorschlag1": erklaerungvorschlag1,
    }
    
    token = b64encode(f"{user}:{pw}".encode('utf-8')).decode("ascii")
    
    print(token)
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Basic {token}'
    }
    
    try:
        response = requests.put(url, data=payload, headers=headers)
        print(response.request.method)
        print(response.request.url)
        print(response.request.headers)
        print(response.request.body)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None