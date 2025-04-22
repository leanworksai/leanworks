import yaml
import requests
from typing import Dict, Any


class GitHub:
    """GitHub wrapper for interacting with the GitHub API."""
    
    def __init__(self, secret_client):
        """
        Initialize GitHub instance.
        
        Args:
            secret_client: Secret client for GitHub token
        """
        self.token = secret_client.get("GITHUB_KEY")
        if not self.token:
            raise ValueError("GitHub token is required. Provide it as an argument or set GITHUB_KEY environment variable.")
        
        self.owner = "LeanWorks-ai"
        self.repo = "client-onboarding"
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.base_url = "https://api.github.com"
    
    def get_client_config(self, client_name: str, branch: str = "main") -> Dict[str, Any]:
        """
        Download client configuration YAML file from GitHub repository.
        
        Args:
            client_name: Name of the client
            branch: Branch name to use
            
        Returns:
            Dict containing the client configuration
            
        Raises:
            ValueError: If client configuration file cannot be found
            requests.RequestException: If there's an issue with the GitHub API request
        """
        file_path = f"{client_name}.yaml"
        
        # Construct API URL for repository content
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/contents/{file_path}"
        params = {"ref": branch}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code != 200:
            raise ValueError(f"Could not find client configuration for {client_name}. "
                            f"Status code: {response.status_code}, "
                            f"Response: {response.text}")
        
        # Get the download URL from the response
        content_data = response.json()
        download_url = content_data.get("download_url")
        
        if not download_url:
            raise ValueError(f"Download URL not found for {file_path}")
        
        # Download the actual file content
        file_response = requests.get(download_url)
        
        if file_response.status_code != 200:
            raise ValueError(f"Failed to download client configuration. "
                           f"Status code: {file_response.status_code}")
        
        # Parse YAML content
        try:
            config = yaml.safe_load(file_response.text)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing client configuration YAML: {str(e)}")
