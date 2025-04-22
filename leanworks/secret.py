from google.cloud import secretmanager
from google.oauth2 import service_account
import json

class GCPSecretLoader:
    def __init__(self, key_path, client_name):
        self.key_path = key_path
        self.client_name = client_name
        credentials = service_account.Credentials.from_service_account_file(self.key_path)
        self.client = secretmanager.SecretManagerServiceClient(credentials=credentials)

    def get(self, name):
        if self.client_name != "leanworks":
            name = f"{self.client_name.upper()}_{name}"
        with open(self.key_path, "r") as f_in:
            credential = json.load(f_in)
        project_id = credential["project_id"]
        name = f"projects/{project_id}/secrets/{name}/versions/latest"
        response = self.client.access_secret_version(name=name)
        secret_string = response.payload.data.decode("UTF-8")
        return secret_string