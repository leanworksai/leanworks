import logging
from typing import List, Dict, Any, Optional
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from google.oauth2 import service_account
import json

logger = logging.getLogger(__name__)


class GoogleDriveTool:
    """
    Google Drive tool for accessing and managing files using service account authentication.

    Provides methods for listing, searching, downloading, uploading, and creating folders in Google Drive.
    Uses service account credentials which can only access files shared with the service account or files it owns.
    """

    def __init__(self, service_account_info: dict, scopes: List[str] = None):
        """
        Initialize GoogleDriveTool with service account credentials.

        Args:
            service_account_info: Dictionary containing service account JSON key data
            scopes: List of OAuth scopes (default: Google Drive scopes)
        """
        self.service_account_info = service_account_info

        if scopes is None:
            scopes = ['https://www.googleapis.com/auth/drive']

        self.scopes = scopes
        self.service = None
        self._authenticated = False

    def _authenticate(self):
        """Authenticate with Google Drive API using service account credentials."""
        try:
            if not self.service_account_info:
                logger.error("Service account info not provided")
                return False

            # Create credentials from service account info
            credentials = service_account.Credentials.from_service_account_info(
                self.service_account_info,
                scopes=self.scopes
            )

            # Build the Drive service
            self.service = build('drive', 'v3', credentials=credentials)
            self._authenticated = True
            logger.info("Google Drive authentication successful")
            return True

        except Exception as e:
            logger.error(f"Google Drive authentication failed: {str(e)}")
            return False

    @property
    def list_files_property(self):
        description = """
        List files and folders in a Google Drive folder.

        Lists all files and folders within a specified folder ID. If no folder_id is provided,
        lists files from the root directory. Returns file metadata including ID, name, type,
        size, and modification date.

        The service account can only access files that are shared with it or files it owns.
        """
        return {
            "type": "custom",
            "name": "google_drive_list_files",
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
                        "description": "Maximum number of files to return (default: 50, max: 100)",
                        "minimum": 1,
                        "maximum": 100,
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
        List files and folders in a Google Drive folder.

        Args:
            folder_id: ID of the folder to list files from (optional, defaults to root)
            max_results: Maximum number of files to return
            include_folders: Whether to include folders in results

        Returns:
            List of file/folder dictionaries, or error dictionary
        """
        logger.info(f"Listing files in folder: {folder_id}, max_results: {max_results}, include_folders: {include_folders}")

        try:
            if not self.service and not self._authenticate():
                return {"error": "Failed to authenticate with Google Drive API"}

            # Validate max_results
            max_results = min(max(1, max_results), 100)

            # Build query
            query_parts = []
            if folder_id:
                query_parts.append(f"'{folder_id}' in parents")
            if not include_folders:
                query_parts.append("mimeType != 'application/vnd.google-apps.folder'")

            query = " and ".join(query_parts) if query_parts else ""

            # Execute the query
            results = self.service.files().list(
                q=query,
                pageSize=max_results,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, createdTime, parents, webViewLink)",
                orderBy="modifiedTime desc"
            ).execute()

            files = results.get('files', [])

            # Format the response
            formatted_files = []
            for file in files:
                formatted_files.append({
                    "id": file.get('id'),
                    "name": file.get('name'),
                    "type": "folder" if file.get('mimeType') == 'application/vnd.google-apps.folder' else "file",
                    "mime_type": file.get('mimeType'),
                    "size": file.get('size'),
                    "modified_time": file.get('modifiedTime'),
                    "created_time": file.get('createdTime'),
                    "parent_id": file.get('parents', [None])[0] if file.get('parents') else None,
                    "web_view_link": file.get('webViewLink')
                })

            logger.info(f"Listed {len(formatted_files)} files/folders")
            return formatted_files

        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def search_files_property(self):
        description = """
        Search for files in Google Drive by name or content.

        Searches for files and folders using Google Drive's full-text search capabilities.
        Supports searching by filename, content, or both. Can limit search to specific folder.

        The service account can only find files that are shared with it or files it owns.
        """
        return {
            "type": "custom",
            "name": "google_drive_search_files",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. Can include filename, content keywords, or special operators like 'name:filename.txt'"
                    },
                    "folder_id": {
                        "type": "string",
                        "description": "Optional folder ID to limit search scope"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 50, max: 100)",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 50
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Filter by file type: 'file', 'folder', or 'any' (default: 'any')",
                        "enum": ["file", "folder", "any"],
                        "default": "any"
                    }
                },
                "required": ["query"]
            }
        }

    def search_files(self, query: str, folder_id: str = None, max_results: int = 50, file_type: str = "any", **kwargs) -> List[Dict[str, Any]]:
        """
        Search for files in Google Drive.

        Args:
            query: Search query
            folder_id: Optional folder ID to limit search scope
            max_results: Maximum number of results to return
            file_type: Filter by file type ('file', 'folder', or 'any')

        Returns:
            List of file/folder dictionaries, or error dictionary
        """
        logger.info(f"Searching files with query: '{query}', folder: {folder_id}, max_results: {max_results}, file_type: {file_type}")

        try:
            if not self.service and not self._authenticate():
                return {"error": "Failed to authenticate with Google Drive API"}

            # Validate max_results
            max_results = min(max(1, max_results), 100)

            # Build query
            query_parts = [f"fullText contains '{query}'"]

            if folder_id:
                query_parts.append(f"'{folder_id}' in parents")

            if file_type == "folder":
                query_parts.append("mimeType = 'application/vnd.google-apps.folder'")
            elif file_type == "file":
                query_parts.append("mimeType != 'application/vnd.google-apps.folder'")

            drive_query = " and ".join(query_parts)

            # Execute the search
            results = self.service.files().list(
                q=drive_query,
                pageSize=max_results,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, createdTime, parents, webViewLink)",
                orderBy="modifiedTime desc"
            ).execute()

            files = results.get('files', [])

            # Format the response
            formatted_files = []
            for file in files:
                formatted_files.append({
                    "id": file.get('id'),
                    "name": file.get('name'),
                    "type": "folder" if file.get('mimeType') == 'application/vnd.google-apps.folder' else "file",
                    "mime_type": file.get('mimeType'),
                    "size": file.get('size'),
                    "modified_time": file.get('modifiedTime'),
                    "created_time": file.get('createdTime'),
                    "parent_id": file.get('parents', [None])[0] if file.get('parents') else None,
                    "web_view_link": file.get('webViewLink')
                })

            logger.info(f"Search returned {len(formatted_files)} results")
            return formatted_files

        except Exception as e:
            logger.error(f"Error searching files: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def get_file_property(self):
        description = """
        Get metadata for a specific file or folder in Google Drive.

        Retrieves detailed information about a file or folder including name, size,
        modification date, MIME type, and download/view URLs.

        The service account can only access files that are shared with it or files it owns.
        """
        return {
            "type": "custom",
            "name": "google_drive_get_file",
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
            if not self.service and not self._authenticate():
                return {"error": "Failed to authenticate with Google Drive API"}

            if not file_id:
                return {"error": "file_id is required"}

            # Get file metadata
            file = self.service.files().get(
                fileId=file_id,
                fields="id, name, mimeType, size, modifiedTime, createdTime, parents, webViewLink, thumbnailLink, downloadUrl"
            ).execute()

            # Format the response
            result = {
                "id": file.get('id'),
                "name": file.get('name'),
                "type": "folder" if file.get('mimeType') == 'application/vnd.google-apps.folder' else "file",
                "mime_type": file.get('mimeType'),
                "size": file.get('size'),
                "modified_time": file.get('modifiedTime'),
                "created_time": file.get('createdTime'),
                "parent_id": file.get('parents', [None])[0] if file.get('parents') else None,
                "web_view_link": file.get('webViewLink'),
                "thumbnail_link": file.get('thumbnailLink'),
                "download_url": file.get('downloadUrl')
            }

            logger.info(f"Retrieved metadata for file: {file.get('name')}")
            return result

        except Exception as e:
            logger.error(f"Error getting file metadata: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def download_file_property(self):
        description = """
        Download the content of a file from Google Drive.

        Downloads the binary content of a file. This is primarily for non-Google file types
        (PDF, images, etc.). Google Docs files will be exported to appropriate formats.

        The service account can only download files that are shared with it or files it owns.
        """
        return {
            "type": "custom",
            "name": "google_drive_download_file",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "ID of the file to download"
                    },
                    "export_format": {
                        "type": "string",
                        "description": "For Google Docs files, export format (pdf, docx, txt, etc.). If not specified, uses default format."
                    }
                },
                "required": ["file_id"]
            }
        }

    def download_file(self, file_id: str, export_format: str = None, **kwargs) -> Dict[str, Any]:
        """
        Download the content of a file from Google Drive.

        Args:
            file_id: ID of the file to download
            export_format: Export format for Google Docs files

        Returns:
            Dictionary with content (base64 encoded) and metadata, or error dictionary
        """
        logger.info(f"Downloading file: {file_id}, export_format: {export_format}")

        try:
            if not self.service and not self._authenticate():
                return {"error": "Failed to authenticate with Google Drive API"}

            if not file_id:
                return {"error": "file_id is required"}

            # Get file metadata first to determine download method
            file_metadata = self.service.files().get(fileId=file_id, fields="name, mimeType, size").execute()
            mime_type = file_metadata.get('mimeType', '')

            # Handle Google Docs files (need export)
            if mime_type.startswith('application/vnd.google-apps.'):
                if not export_format:
                    # Default export formats for Google Docs types
                    export_formats = {
                        'application/vnd.google-apps.document': 'text/plain',
                        'application/vnd.google-apps.spreadsheet': 'text/csv',
                        'application/vnd.google-apps.presentation': 'application/pdf',
                        'application/vnd.google-apps.drawing': 'image/png'
                    }
                    export_format = export_formats.get(mime_type, 'application/pdf')

                # Export the file
                request = self.service.files().export_media(fileId=file_id, mimeType=export_format)
            else:
                # Download regular file
                request = self.service.files().get_media(fileId=file_id)

            # Execute download
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()

            content = fh.getvalue()

            # Base64 encode the content for JSON response
            import base64
            encoded_content = base64.b64encode(content).decode('utf-8')

            result = {
                "file_id": file_id,
                "file_name": file_metadata.get('name'),
                "mime_type": mime_type,
                "export_format": export_format,
                "content": encoded_content,
                "size_bytes": len(content)
            }

            logger.info(f"Downloaded file: {file_metadata.get('name')} ({len(content)} bytes)")
            return result

        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def upload_file_property(self):
        description = """
        Upload a file to Google Drive.

        Uploads a new file to Google Drive. The file content should be provided as base64-encoded data.
        Can optionally specify a parent folder. Returns the uploaded file's metadata.

        The service account can upload files to folders it has access to.
        """
        return {
            "type": "custom",
            "name": "google_drive_upload_file",
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
                    "mime_type": {
                        "type": "string",
                        "description": "MIME type of the file (e.g., 'text/plain', 'application/pdf')"
                    },
                    "parent_folder_id": {
                        "type": "string",
                        "description": "Optional ID of the parent folder to upload to. If not provided, uploads to root."
                    }
                },
                "required": ["file_name", "content", "mime_type"]
            }
        }

    def upload_file(self, file_name: str, content: str, mime_type: str, parent_folder_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Upload a file to Google Drive.

        Args:
            file_name: Name of the file to upload
            content: File content as base64-encoded string
            mime_type: MIME type of the file
            parent_folder_id: Optional parent folder ID

        Returns:
            Uploaded file metadata dictionary, or error dictionary
        """
        logger.info(f"Uploading file: {file_name}, mime_type: {mime_type}, parent_folder: {parent_folder_id}")

        try:
            if not self.service and not self._authenticate():
                return {"error": "Failed to authenticate with Google Drive API"}

            if not file_name or not content or not mime_type:
                return {"error": "file_name, content, and mime_type are required"}

            # Decode base64 content
            import base64
            file_content = base64.b64decode(content)

            # Create file metadata
            file_metadata = {'name': file_name}
            if parent_folder_id:
                file_metadata['parents'] = [parent_folder_id]

            # Create media object
            media = MediaIoBaseUpload(
                io.BytesIO(file_content),
                mimetype=mime_type,
                resumable=True
            )

            # Upload the file
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, mimeType, size, modifiedTime, createdTime, parents, webViewLink'
            ).execute()

            # Format the response
            result = {
                "id": file.get('id'),
                "name": file.get('name'),
                "mime_type": file.get('mimeType'),
                "size": file.get('size'),
                "modified_time": file.get('modifiedTime'),
                "created_time": file.get('createdTime'),
                "parent_id": file.get('parents', [None])[0] if file.get('parents') else None,
                "web_view_link": file.get('webViewLink')
            }

            logger.info(f"Uploaded file: {file_name} with ID: {file.get('id')}")
            return result

        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def create_folder_property(self):
        description = """
        Create a new folder in Google Drive.

        Creates a new folder with the specified name. Can optionally specify a parent folder.
        Returns the created folder's metadata.

        The service account can create folders in locations it has access to.
        """
        return {
            "type": "custom",
            "name": "google_drive_create_folder",
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
        Create a new folder in Google Drive.

        Args:
            folder_name: Name of the folder to create
            parent_folder_id: Optional parent folder ID

        Returns:
            Created folder metadata dictionary, or error dictionary
        """
        logger.info(f"Creating folder: {folder_name}, parent_folder: {parent_folder_id}")

        try:
            if not self.service and not self._authenticate():
                return {"error": "Failed to authenticate with Google Drive API"}

            if not folder_name:
                return {"error": "folder_name is required"}

            # Create folder metadata
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }

            if parent_folder_id:
                folder_metadata['parents'] = [parent_folder_id]

            # Create the folder
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id, name, mimeType, modifiedTime, createdTime, parents, webViewLink'
            ).execute()

            # Format the response
            result = {
                "id": folder.get('id'),
                "name": folder.get('name'),
                "type": "folder",
                "mime_type": folder.get('mimeType'),
                "modified_time": folder.get('modifiedTime'),
                "created_time": folder.get('createdTime'),
                "parent_id": folder.get('parents', [None])[0] if folder.get('parents') else None,
                "web_view_link": folder.get('webViewLink')
            }

            logger.info(f"Created folder: {folder_name} with ID: {folder.get('id')}")
            return result

        except Exception as e:
            logger.error(f"Error creating folder: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}