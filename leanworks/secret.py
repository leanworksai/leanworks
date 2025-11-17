from google.cloud import secretmanager
from google.oauth2 import service_account
import json

class GCPSecretLoader:
    def __init__(self, key_path):
        self.key_path = key_path
        credentials = service_account.Credentials.from_service_account_file(self.key_path)
        self.client = secretmanager.SecretManagerServiceClient(credentials=credentials)

    def get(self, name):
        """
        Retrieve a secret value using the secret name.
        
        Args:
            name: Secret name (e.g., 'my-secret', 'API_KEY')
                  The project_id is automatically extracted from the credential file.
        
        Returns:
            The secret value as a string
        """
        with open(self.key_path, "r") as f_in:
            credential = json.load(f_in)
        project_id = credential["project_id"]
        full_name = f"projects/{project_id}/secrets/{name}/versions/latest"
        response = self.client.access_secret_version(name=full_name)
        secret_string = response.payload.data.decode("UTF-8")
        return secret_string