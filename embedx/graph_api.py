import os
import msal
import requests
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID") or "common"
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["Files.Read.All", "Files.ReadWrite.All"]
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

def start_device_flow():
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
    flow = app.initiate_device_flow(scopes=SCOPE)
    if "user_code" not in flow:
        raise Exception("Could not create device flow")
    return app, flow, flow["message"]

def acquire_token(app, flow):
    return app.acquire_token_by_device_flow(flow)

def list_excel_files(token):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{GRAPH_API_ENDPOINT}/me/drive/root/children"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to list files: {response.status_code} {response.text}")

def download_file(token, file_id):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/content"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception(f"Failed to download file: {response.status_code} {response.text}")
    
def upload_to_onedrive(filename, file_bytes, token):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"}
    upload_url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{filename}:/content"
    response = requests.put(upload_url, headers=headers, data=file_bytes)
    response.raise_for_status()
    return response.json()
