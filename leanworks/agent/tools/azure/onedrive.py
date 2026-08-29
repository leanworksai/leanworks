import logging
import datetime
import pytz
from typing import List, Dict, Optional, Any
import requests
from msal import ConfidentialClientApplication
import json
import base64

logger = logging.getLogger(__name__)


class OneDriveTool:
    """
    OneDrive tool for accessing and managing files using Microsoft Graph API.

    Provides methods for listing, searching, downloading, uploading, and creating folders in OneDrive.
    Uses Microsoft Graph API with OAuth 2.0 client credentials flow.
    """

    def __init__(self, client_id: str = None, client_secret: str = None, tenant_id: str = None, authority: str = None):
        """
        Initialize OneDriveTool with Microsoft Graph API credentials.

        Args:
            client_id: Azure AD application client ID
            client_secret: Azure AD application client secret
            tenant_id: Azure AD tenant ID
            authority: Azure AD authority URL
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.authority = authority or f"https://login.microsoftonline.com/{tenant_id}"
        self.scopes = ['https://graph.microsoft.com/.default']
        self.access_token = None
        self.base_url = "https://graph.microsoft.com/v1.0"

    def _authenticate(self):
        """Authenticate with Microsoft Graph API using client credentials flow."""
        try:
            if not all([self.client_id, self.client_secret, self.tenant_id]):
                logger.error("Missing required credentials: client_id, client_secret, and tenant_id")
                return False

            # Create confidential client application
            app = ConfidentialClientApplication(
                client_id=self.client_id,
                client_credential=self.client_secret,
                authority=self.authority
            )

            # Get token using client credentials flow
            result = app.acquire_token_for_client(scopes=self.scopes)

            if "access_token" in result:
                self.access_token = result["access_token"]
                logger.info("OneDrive authentication successful")
                return True
            else:
                logger.error(
                    "Failed to acquire OneDrive token (error_present=%s)",
                    bool(result.get('error')),
                )
                return False

        except Exception as e:
            logger.error(f"OneDrive authentication failed: {str(e)}")
            return False

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Any:
        """
        Make authenticated request to Microsoft Graph API.

        Args:
            method: HTTP method (GET, POST, PATCH, PUT)
            endpoint: API endpoint path (e.g., '/me/drive/root/children')
            **kwargs: Additional arguments for requests

        Returns:
            Response JSON data or None
        """
        url = f"{self.base_url}{endpoint}"

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

        if 'headers' in kwargs:
            headers.update(kwargs.pop('headers'))

        kwargs['headers'] = headers

        response = requests.request(method, url, **kwargs)
        response.raise_for_status()

        if response.content:
            return response.json()
        return None

    @property
    def list_files_property(self):
        description = """
        List files and folders in a OneDrive folder.

        Lists all files and folders within a specified folder ID. If no folder_id is provided,
        lists files from the root directory. Returns file metadata including ID, name, type,
        size, and modification date.

        Uses Microsoft Graph API to access OneDrive files.
        """
        return {
            "type": "custom",
            "name": "onedrive_list_files",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "folder_id": {
                        "type": "string",
                        "description": "ID of the folder to list files from. If not provided, lists from root directory."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of files to return (default: 50, max: 200)",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 50
                    },
                    "include_folders": {
                        "type": "boolean",
                        "description": "Whether to include folders in results (default: true)",
                        "default": True
                    }
                }
            }
        }

    def list_files(self, folder_id: str = None, max_results: int = 50, include_folders: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """
        List files and folders in a OneDrive folder.

        Args:
            folder_id: ID of the folder to list files from (optional, defaults to root)
            max_results: Maximum number of files to return
            include_folders: Whether to include folders in results

        Returns:
            List of file/folder dictionaries, or error dictionary
        """
        logger.info(f"Listing files in folder: {folder_id}, max_results: {max_results}, include_folders: {include_folders}")

        try:
            if not self.access_token and not self._authenticate():
                return {"error": "Failed to authenticate with Microsoft Graph API"}

            # Validate max_results
            max_results = min(max(1, max_results), 200)

            # Build endpoint
            if folder_id:
                endpoint = f"/me/drive/items/{folder_id}/children"
            else:
                endpoint = "/me/drive/root/children"

            # Make request with pagination
            params = {
                "$top": max_results,
                "$orderby": "lastModifiedDateTime desc"
            }

            response = self._make_request("GET", endpoint, params=params)
            items = response.get('value', [])

            # Filter and format results
            formatted_items = []
            for item in items:
                # Skip folders if not requested
                if not include_folders and item.get('folder'):
                    continue

                formatted_items.append({
                    "id": item.get('id'),
                    "name": item.get('name'),
                    "type": "folder" if item.get('folder') else "file",
                    "size": item.get('size'),
                    "modified_time": item.get('lastModifiedDateTime'),
                    "created_time": item.get('createdDateTime'),
                    "web_url": item.get('webUrl'),
                    "download_url": item.get('@microsoft.graph.downloadUrl')
                })

            logger.info(f"Listed {len(formatted_items)} files/folders")
            return formatted_items

        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def search_files_property(self):
        description = """
        Search for files in OneDrive by name or content.

        Searches for files and folders using Microsoft Graph API search capabilities.
        Supports searching by filename or content keywords.

        Uses Microsoft Graph API search functionality.
        """
        return {
            "type": "custom",
            "name": "onedrive_search_files",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for filename or content"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 50, max: 200)",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 50
                    }
                },
                "required": ["query"]
            }
        }

    def search_files(self, query: str, max_results: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for files in OneDrive.

        Args:
            query: Search query for filename or content
            max_results: Maximum number of results to return

        Returns:
            List of file/folder dictionaries, or error dictionary
        """
        logger.info(
            "Searching OneDrive files (query_chars=%d, max_results=%d)",
            len(query), max_results,
        )

        try:
            if not self.access_token and not self._authenticate():
                return {"error": "Failed to authenticate with Microsoft Graph API"}

            # Validate max_results
            max_results = min(max(1, max_results), 200)

            # Use Microsoft Graph search API
            endpoint = "/me/drive/root/search(q='{}')".format(query)

            params = {
                "$top": max_results,
                "$orderby": "lastModifiedDateTime desc"
            }

            response = self._make_request("GET", endpoint, params=params)
            items = response.get('value', [])

            # Format results
            formatted_items = []
            for item in items:
                formatted_items.append({
                    "id": item.get('id'),
                    "name": item.get('name'),
                    "type": "folder" if item.get('folder') else "file",
                    "size": item.get('size'),
                    "modified_time": item.get('lastModifiedDateTime'),
                    "created_time": item.get('createdDateTime'),
                    "web_url": item.get('webUrl'),
                    "download_url": item.get('@microsoft.graph.downloadUrl')
                })

            logger.info(f"Search returned {len(formatted_items)} results")
            return formatted_items

        except Exception as e:
            logger.error(f"Error searching files: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def get_file_property(self):
        description = """
        Get metadata for a specific file or folder in OneDrive.

        Retrieves detailed information about a file or folder including name, size,
        modification date, and download URLs.

        Uses Microsoft Graph API to get item metadata.
        """
        return {
            "type": "custom",
            "name": "onedrive_get_file",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "ID of the file or folder to get information for"
                    }
                },
                "required": ["file_id"]
            }
        }

    def get_file(self, file_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get metadata for a specific file or folder.

        Args:
            file_id: ID of the file or folder

        Returns:
            File/folder metadata dictionary, or error dictionary
        """
        logger.info(f"Getting file metadata for: {file_id}")

        try:
            if not self.access_token and not self._authenticate():
                return {"error": "Failed to authenticate with Microsoft Graph API"}

            if not file_id:
                return {"error": "file_id is required"}

            # Get file metadata
            endpoint = f"/me/drive/items/{file_id}"
            item = self._make_request("GET", endpoint)

            # Format the response
            result = {
                "id": item.get('id'),
                "name": item.get('name'),
                "type": "folder" if item.get('folder') else "file",
                "size": item.get('size'),
                "modified_time": item.get('lastModifiedDateTime'),
                "created_time": item.get('createdDateTime'),
                "web_url": item.get('webUrl'),
                "download_url": item.get('@microsoft.graph.downloadUrl')
            }

            logger.info(f"Retrieved metadata for file: {item.get('name')}")
            return result

        except Exception as e:
            logger.error(f"Error getting file metadata: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def download_file_property(self):
        description = """
        Download the content of a file from OneDrive.

        Downloads the binary content of a file. Returns the content as base64-encoded data
        along with file metadata.

        Uses Microsoft Graph API download URL.
        """
        return {
            "type": "custom",
            "name": "onedrive_download_file",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "ID of the file to download"
                    }
                },
                "required": ["file_id"]
            }
        }

    def download_file(self, file_id: str, **kwargs) -> Dict[str, Any]:
        """
        Download the content of a file from OneDrive.

        Args:
            file_id: ID of the file to download

        Returns:
            Dictionary with content (base64 encoded) and metadata, or error dictionary
        """
        logger.info(f"Downloading file: {file_id}")

        try:
            if not self.access_token and not self._authenticate():
                return {"error": "Failed to authenticate with Microsoft Graph API"}

            if not file_id:
                return {"error": "file_id is required"}

            # Get file metadata first to get download URL
            endpoint = f"/me/drive/items/{file_id}"
            item = self._make_request("GET", endpoint)

            download_url = item.get('@microsoft.graph.downloadUrl')
            if not download_url:
                return {"error": "File download URL not available"}

            # Download the file content
            response = requests.get(download_url)
            response.raise_for_status()

            content = response.content

            # Base64 encode the content for JSON response
            encoded_content = base64.b64encode(content).decode('utf-8')

            result = {
                "file_id": file_id,
                "file_name": item.get('name'),
                "content": encoded_content,
                "size_bytes": len(content),
                "mime_type": response.headers.get('content-type')
            }

            logger.info(f"Downloaded file: {item.get('name')} ({len(content)} bytes)")
            return result

        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def upload_file_property(self):
        description = """
        Upload a file to OneDrive.

        Uploads a new file to OneDrive. The file content should be provided as base64-encoded data.
        Can optionally specify a parent folder. Returns the uploaded file's metadata.

        Uses Microsoft Graph API simple upload for files up to 4MB.
        """
        return {
            "type": "custom",
            "name": "onedrive_upload_file",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Name of the file to upload"
                    },
                    "content": {
                        "type": "string",
                        "description": "File content as base64-encoded string"
                    },
                    "parent_folder_id": {
                        "type": "string",
                        "description": "Optional ID of the parent folder. If not provided, uploads to root."
                    }
                },
                "required": ["file_name", "content"]
            }
        }

    def upload_file(self, file_name: str, content: str, parent_folder_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Upload a file to OneDrive.

        Args:
            file_name: Name of the file to upload
            content: File content as base64-encoded string
            parent_folder_id: Optional parent folder ID

        Returns:
            Uploaded file metadata dictionary, or error dictionary
        """
        logger.info(f"Uploading file: {file_name}, parent_folder: {parent_folder_id}")

        try:
            if not self.access_token and not self._authenticate():
                return {"error": "Failed to authenticate with Microsoft Graph API"}

            if not file_name or not content:
                return {"error": "file_name and content are required"}

            # Decode base64 content
            file_content = base64.b64decode(content)

            # Build upload endpoint
            if parent_folder_id:
                endpoint = f"/me/drive/items/{parent_folder_id}:/{file_name}:/content"
            else:
                endpoint = f"/me/drive/root:/{file_name}:/content"

            # Upload the file using simple upload (up to 4MB)
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/octet-stream'
            }

            url = f"{self.base_url}{endpoint}"
            response = requests.put(url, data=file_content, headers=headers)
            response.raise_for_status()

            item = response.json()

            # Format the response
            result = {
                "id": item.get('id'),
                "name": item.get('name'),
                "type": "file",
                "size": item.get('size'),
                "modified_time": item.get('lastModifiedDateTime'),
                "created_time": item.get('createdDateTime'),
                "web_url": item.get('webUrl')
            }

            logger.info(f"Uploaded file: {file_name} with ID: {item.get('id')}")
            return result

        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def create_folder_property(self):
        description = """
        Create a new folder in OneDrive.

        Creates a new folder with the specified name. Can optionally specify a parent folder.
        Returns the created folder's metadata.

        Uses Microsoft Graph API to create folder items.
        """
        return {
            "type": "custom",
            "name": "onedrive_create_folder",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "folder_name": {
                        "type": "string",
                        "description": "Name of the folder to create"
                    },
                    "parent_folder_id": {
                        "type": "string",
                        "description": "Optional ID of the parent folder. If not provided, creates in root."
                    }
                },
                "required": ["folder_name"]
            }
        }

    def create_folder(self, folder_name: str, parent_folder_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Create a new folder in OneDrive.

        Args:
            folder_name: Name of the folder to create
            parent_folder_id: Optional parent folder ID

        Returns:
            Created folder metadata dictionary, or error dictionary
        """
        logger.info(f"Creating folder: {folder_name}, parent_folder: {parent_folder_id}")

        try:
            if not self.access_token and not self._authenticate():
                return {"error": "Failed to authenticate with Microsoft Graph API"}

            if not folder_name:
                return {"error": "folder_name is required"}

            # Build endpoint
            if parent_folder_id:
                endpoint = f"/me/drive/items/{parent_folder_id}/children"
            else:
                endpoint = "/me/drive/root/children"

            # Create folder payload
            folder_data = {
                "name": folder_name,
                "folder": {},
                "@microsoft.graph.conflictBehavior": "rename"
            }

            # Create the folder
            item = self._make_request("POST", endpoint, json=folder_data)

            # Format the response
            result = {
                "id": item.get('id'),
                "name": item.get('name'),
                "type": "folder",
                "modified_time": item.get('lastModifiedDateTime'),
                "created_time": item.get('createdDateTime'),
                "web_url": item.get('webUrl')
            }

            logger.info(f"Created folder: {folder_name} with ID: {item.get('id')}")
            return result

        except Exception as e:
            logger.error(f"Error creating folder: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
