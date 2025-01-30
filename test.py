from fetching.ai_communication import update_recommendation
import os
from dotenv import load_dotenv

# load .env file to environment
load_dotenv(override=True)

user = os.getenv("USER")
pw = os.getenv("PASSWORD")

print(user)
print(pw)

def endpoint_test():
    
    incident_id = "b9ee8ed2-3196-45d6-b79e-5ef2da6c8ab6"
    ma_id = "d1f8e88a-02cf-4317-9512-a2a66f931a7a"
    expl_short = "Test Nummer 2"
    expl = "<p><b>Hallo</b> <span style=\"color:green\">Test</span>! :-)<br/>2. Zeile</p>"
    
    out = update_recommendation(user, pw, incident_id, ma_id, expl_short, expl)
    
    print(out)
    
endpoint_test()