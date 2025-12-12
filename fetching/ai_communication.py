import requests
from typing import List, Tuple, Dict
import json
from base64 import b64encode

endpoint = "VMBegleitung"

def update_recommendation(
    request_info: List[Dict[str, str]],
    id: str,
    recommendations: List[Tuple[str, str, str]] = [],
    org: str = None
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
    
    # if recommendations:
    #     for i, (mavertretend, erklaerungkurz, erklaerung) in enumerate(recommendations, start=1):
    #         if len(erklaerungkurz) > 200:
    #             raise ValueError(f"erklaerungvorschlagkurz{i} must be 200 characters or fewer.")
    #         if len(erklaerung) > 8000:
    #             raise ValueError(f"erklaerungvorschlag{i} must be 8000 characters or fewer.")

    # Construct the URL
    for request_data in request_info:
        if org == request_data["url"]:
            url = f"{request_data['url_ai']}{endpoint}"
            print(url)

            # Dynamically build the payload
            payload = {"id": id}

            for i, (mavertretend, erklaerungkurz, erklaerung) in enumerate(recommendations, start=1):
                payload[f"mavertretendvorschlag{i}"] = mavertretend
                payload[f"erklaerungvorschlagkurz{i}"] = erklaerungkurz[:199]
                payload[f"erklaerungvorschlag{i}"] = erklaerung[:7999]

            
            token = b64encode(f"{request_data['user']}:{request_data['pw']}".encode('utf-8')).decode("ascii")
            
            print(payload)
            
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