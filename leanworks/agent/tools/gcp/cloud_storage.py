import logging
from typing import List, Dict, Any, Optional, Union
from google.cloud import storage
from datetime import datetime, timedelta
import json
import re
import io
from leanworks.utils.env import get_storage_bucket, resolve_credential_path

logger = logging.getLogger(__name__)


class CloudStorageTool:
    """
    Google Cloud Storage tool for accessing files in Google Cloud Storage.

    Provides general file operations (list, upload, download, get metadata, signed URLs)
    while maintaining backward compatibility for chat image operations.
    Supports org-specific operations (chat images) and general file operations.
    """

    def __init__(self, storage_client, org_slug: str, bucket_name: Optional[str] = None, credential_path: Optional[str] = None):
        """
        Initialize CloudStorageTool with Storage client and org context.

        Args:
            storage_client: Google Cloud Storage client instance
            org_slug: Organization slug (e.g., 'leanworks.ai')
            bucket_name: GCS bucket name (default: environment-aware)
            credential_path: Path to GCP credential JSON file (default: environment-aware)
        """
        self.storage_client = storage_client
        self.org_slug = org_slug
        self.bucket_name = bucket_name or get_storage_bucket()
        self.credential_path = credential_path or resolve_credential_path()

        # URL expiration time (default: 1 year)
        self.url_expiration_days = 365

    def _get_bucket(self):
        """Get or create the storage bucket."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            # Check if bucket exists
            if not bucket.exists():
                logger.warning(f"Bucket {self.bucket_name} does not exist")
            return bucket
        except Exception as e:
            logger.error(f"Error accessing bucket {self.bucket_name}: {e}")
            raise

    def _generate_signed_url(self, storage_path: str, expiration_days: int = None, method: str = 'GET') -> str:
        """
        Generate a signed URL for accessing a file in Cloud Storage.

        Args:
            storage_path: Path to the file in the bucket
            expiration_days: Number of days until URL expires (default: 1 year)
            method: HTTP method (GET, PUT, etc.)

        Returns:
            Signed URL string
        """
        try:
            expiration_days = expiration_days or self.url_expiration_days
            bucket = self._get_bucket()
            blob = bucket.blob(storage_path)

            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(days=expiration_days)

            # Generate signed URL
            signed_url = blob.generate_signed_url(
                expiration=expires_at,
                method=method
            )

            return signed_url
        except Exception as e:
            logger.error(f"Error generating signed URL for {storage_path}: {e}")
            raise

    def _extract_storage_path(self, image_url_or_id: str, chat_id: str) -> Optional[str]:
        """
        Extract storage path from image URL or ID.

        Args:
            image_url_or_id: Either a signed URL or an image ID (UUID.jpg)
            chat_id: Chat ID for constructing path if image_id is provided

        Returns:
            Storage path or None if cannot be determined
        """
        # If it's already a storage path, return it
        if image_url_or_id.startswith('orgs/'):
            return image_url_or_id

        # If it's an imageId (UUID.jpg), construct the path
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jpg$', image_url_or_id, re.IGNORECASE):
            return f"orgs/{self.org_slug}/chat-images/{chat_id}/{image_url_or_id}"

        # Try to extract from signed URL
        try:
            from urllib.parse import urlparse
            url = urlparse(image_url_or_id)
            # Extract path from Google Cloud Storage signed URL
            # Format: https://storage.googleapis.com/bucket/path?signature=...
            if 'storage.googleapis.com' in url.netloc:
                path_match = url.path.split('/', 2)
                if len(path_match) >= 3:
                    return path_match[2]  # Path after /bucket/
        except Exception as e:
            logger.warning(f"Could not extract path from URL {image_url_or_id}: {e}")

        # Fallback: construct from chat_id
        if chat_id:
            return f"orgs/{self.org_slug}/chat-images/{chat_id}/{image_url_or_id}"

        return None

    # ============================================================================
    # GENERAL FILE OPERATIONS
    # ============================================================================

    @property
    def gcp_storage_list_files_property(self):
        description = """
        List files in Google Cloud Storage with optional prefix filtering.

        Lists all files (blobs) in the specified bucket that match the given prefix.
        Supports pagination and returns basic metadata for each file.

        Parameters:
        - prefix: Optional prefix to filter files (e.g., "orgs/myorg/" or "data/")
        - max_results: Maximum number of files to return (default: 100, max: 1000)
        - include_metadata: Whether to include file metadata (size, content_type, etc.)

        Returns:
        A list of dictionaries, each containing:
        - name: The file path/name
        - size: File size in bytes
        - content_type: MIME type of the file
        - created_at: ISO timestamp when the file was created
        - updated_at: ISO timestamp when the file was last modified
        - metadata: Additional file metadata (if include_metadata=true)
        """
        return {
            "type": "custom",
            "name": "gcp_storage_list_files",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "prefix": {
                        "type": "string",
                        "description": "Optional prefix to filter files (e.g., 'orgs/myorg/' or 'data/')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of files to return (default: 100, max: 1000)",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Whether to include detailed file metadata",
                        "default": False
                    }
                }
            }
        }

    def gcp_storage_list_files(
        self,
        prefix: str = "",
        max_results: int = 100,
        include_metadata: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        List files in Cloud Storage.

        Args:
            prefix: Optional prefix to filter files
            max_results: Maximum number of files to return
            include_metadata: Whether to include detailed metadata

        Returns:
            List of file dictionaries, or error dictionary
        """
        try:
            if not self.storage_client:
                return {"error": "Storage client not initialized"}

            # Validate max_results
            max_results = min(max(1, max_results), 1000)

            # List blobs
            bucket = self._get_bucket()
            blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)

            files = []
            for blob in blobs:
                file_info = {
                    "name": blob.name,
                    "size": blob.size,
                    "content_type": blob.content_type,
                    "created_at": blob.time_created.isoformat() + "Z" if blob.time_created else None,
                    "updated_at": blob.updated.isoformat() + "Z" if blob.updated else None
                }

                if include_metadata and blob.metadata:
                    file_info["metadata"] = blob.metadata

                files.append(file_info)

            logger.info(f"Listed {len(files)} files with prefix: '{prefix}'")
            return files

        except Exception as e:
            logger.error(f"Error listing files with prefix '{prefix}': {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def gcp_storage_get_signed_url_property(self):
        description = """
        Generate a signed URL for accessing any file in Google Cloud Storage.

        Creates a temporary signed URL that provides access to private files in Cloud Storage.
        The signed URL can be used for GET (download) or PUT (upload) operations.

        Parameters:
        - file_path: The full path to the file in the bucket
        - expiration_days: Number of days until URL expires (default: 365)
        - method: HTTP method for the URL ('GET' or 'PUT', default: 'GET')

        Returns:
        A dictionary containing:
        - signed_url: The signed URL for accessing the file
        - file_path: The file path
        - method: The HTTP method
        - expires_at: ISO timestamp when the URL expires
        """
        return {
            "type": "custom",
            "name": "gcp_storage_get_signed_url",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The full path to the file in the bucket"
                    },
                    "expiration_days": {
                        "type": "integer",
                        "description": "Number of days until URL expires (default: 365)",
                        "minimum": 1,
                        "maximum": 3650,
                        "default": 365
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method ('GET' or 'PUT')",
                        "enum": ["GET", "PUT"],
                        "default": "GET"
                    }
                },
                "required": ["file_path"]
            }
        }

    def gcp_storage_get_signed_url(
        self,
        file_path: str,
        expiration_days: int = 365,
        method: str = 'GET',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a signed URL for accessing a file.

        Args:
            file_path: The full path to the file in the bucket
            expiration_days: Number of days until URL expires
            method: HTTP method ('GET' or 'PUT')

        Returns:
            Dictionary with signed_url, file_path, method, and expires_at, or error dictionary
        """
        try:
            if not self.storage_client:
                return {"error": "Storage client not initialized"}

            if not file_path:
                return {"error": "file_path is required"}

            # Validate method
            if method not in ['GET', 'PUT']:
                return {"error": "method must be 'GET' or 'PUT'"}

            # Generate signed URL
            signed_url = self._generate_signed_url(file_path, expiration_days, method)

            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(days=expiration_days)

            logger.info(f"Generated signed {method} URL for file: {file_path}")

            return {
                "signed_url": signed_url,
                "file_path": file_path,
                "method": method,
                "expires_at": expires_at.isoformat() + "Z"
            }

        except Exception as e:
            logger.error(f"Error generating signed URL for {file_path}: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def gcp_storage_upload_file_property(self):
        description = """
        Upload a file to Google Cloud Storage.

        Uploads file content to the specified path in Cloud Storage. The file content
        should be provided as a base64-encoded string or raw bytes.

        Parameters:
        - file_path: The full path where the file should be stored in the bucket
        - content: The file content (base64-encoded string or raw string)
        - content_type: MIME type of the file (optional, auto-detected if not provided)
        - metadata: Optional dictionary of metadata to attach to the file

        Returns:
        A dictionary containing:
        - file_path: The path where the file was stored
        - size: Size of the uploaded file in bytes
        - content_type: MIME type of the uploaded file
        - uploaded_at: ISO timestamp when the file was uploaded
        """
        return {
            "type": "custom",
            "name": "gcp_storage_upload_file",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The full path where the file should be stored in the bucket"
                    },
                    "content": {
                        "type": "string",
                        "description": "The file content (base64-encoded string or raw string)"
                    },
                    "content_type": {
                        "type": "string",
                        "description": "MIME type of the file (optional, auto-detected if not provided)"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional dictionary of metadata to attach to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }

    def gcp_storage_upload_file(
        self,
        file_path: str,
        content: str,
        content_type: str = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Upload a file to Cloud Storage.

        Args:
            file_path: The full path where the file should be stored
            content: The file content (base64-encoded string or raw string)
            content_type: MIME type of the file
            metadata: Optional metadata dictionary

        Returns:
            Dictionary with upload information, or error dictionary
        """
        try:
            if not self.storage_client:
                return {"error": "Storage client not initialized"}

            if not file_path or not content:
                return {"error": "file_path and content are required"}

            # Try to decode base64 content
            try:
                import base64
                file_content = base64.b64decode(content)
            except:
                # If not base64, treat as raw string
                file_content = content.encode('utf-8')

            # Upload to Cloud Storage
            bucket = self._get_bucket()
            blob = bucket.blob(file_path)

            # Set content type if provided
            if content_type:
                blob.content_type = content_type

            # Set metadata if provided
            if metadata:
                blob.metadata = metadata

            # Upload the content
            blob.upload_from_string(file_content)

            logger.info(f"Uploaded file: {file_path} ({len(file_content)} bytes)")

            return {
                "file_path": file_path,
                "size": len(file_content),
                "content_type": blob.content_type,
                "uploaded_at": datetime.utcnow().isoformat() + "Z"
            }

        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def gcp_storage_download_file_property(self):
        description = """
        Download a file from Google Cloud Storage.

        Downloads the content of a file from Cloud Storage and returns it as a base64-encoded string.

        Parameters:
        - file_path: The full path to the file in the bucket

        Returns:
        A dictionary containing:
        - file_path: The path of the downloaded file
        - content: Base64-encoded file content
        - size: Size of the file in bytes
        - content_type: MIME type of the file
        - downloaded_at: ISO timestamp when the download occurred
        """
        return {
            "type": "custom",
            "name": "gcp_storage_download_file",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The full path to the file in the bucket"
                    }
                },
                "required": ["file_path"]
            }
        }

    def gcp_storage_download_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Download a file from Cloud Storage.

        Args:
            file_path: The full path to the file in the bucket

        Returns:
            Dictionary with file content and metadata, or error dictionary
        """
        try:
            if not self.storage_client:
                return {"error": "Storage client not initialized"}

            if not file_path:
                return {"error": "file_path is required"}

            # Download from Cloud Storage
            bucket = self._get_bucket()
            blob = bucket.blob(file_path)

            # Download the content
            content = blob.download_as_bytes()

            # Encode as base64 for JSON transport
            import base64
            encoded_content = base64.b64encode(content).decode('utf-8')

            logger.info(f"Downloaded file: {file_path} ({len(content)} bytes)")

            return {
                "file_path": file_path,
                "content": encoded_content,
                "size": len(content),
                "content_type": blob.content_type,
                "downloaded_at": datetime.utcnow().isoformat() + "Z"
            }

        except Exception as e:
            logger.error(f"Error downloading file {file_path}: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def gcp_storage_get_file_metadata_property(self):
        description = """
        Get metadata for a file in Google Cloud Storage.

        Retrieves detailed metadata about a file including size, content type,
        creation time, modification time, and custom metadata.

        Parameters:
        - file_path: The full path to the file in the bucket

        Returns:
        A dictionary containing:
        - file_path: The path of the file
        - size: Size of the file in bytes
        - content_type: MIME type of the file
        - created_at: ISO timestamp when the file was created
        - updated_at: ISO timestamp when the file was last modified
        - etag: ETag of the file
        - generation: Generation number of the file
        - metadata: Custom metadata attached to the file
        """
        return {
            "type": "custom",
            "name": "gcp_storage_get_file_metadata",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The full path to the file in the bucket"
                    }
                },
                "required": ["file_path"]
            }
        }

    def gcp_storage_get_file_metadata(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Get metadata for a file in Cloud Storage.

        Args:
            file_path: The full path to the file in the bucket

        Returns:
            Dictionary with file metadata, or error dictionary
        """
        try:
            if not self.storage_client:
                return {"error": "Storage client not initialized"}

            if not file_path:
                return {"error": "file_path is required"}

            # Get file metadata
            bucket = self._get_bucket()
            blob = bucket.blob(file_path)

            # Reload to get latest metadata
            blob.reload()

            metadata = {
                "file_path": file_path,
                "size": blob.size,
                "content_type": blob.content_type,
                "created_at": blob.time_created.isoformat() + "Z" if blob.time_created else None,
                "updated_at": blob.updated.isoformat() + "Z" if blob.updated else None,
                "etag": blob.etag,
                "generation": blob.generation,
                "metadata": blob.metadata or {}
            }

            logger.info(f"Retrieved metadata for file: {file_path}")
            return metadata

        except Exception as e:
            logger.error(f"Error getting metadata for file {file_path}: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    # ============================================================================
    # BACKWARD COMPATIBILITY: CHAT IMAGE OPERATIONS
    # ============================================================================

    @property
    def gcp_cloud_storage_get_image_url_property(self):
        description = f"""
        Get a signed URL for accessing an image from Google Cloud Storage for org `{self.org_slug}`.

        Images are stored in Google Cloud Storage at:
        orgs/{self.org_slug}/chat-images/{{chatId}}/{{imageId}}.jpg

        This tool generates signed URLs that provide temporary access to private images.
        Signed URLs expire after 1 year by default.

        Parameters:
        - imageId: The image ID (UUID.jpg format) or a signed URL that needs refreshing
        - chatId: The chat ID where the image was uploaded (required if imageId is provided)
        - expirationDays: Optional number of days until URL expires (default: 365)

        Examples:
        - Get URL for an image: {{"imageId": "123e4567-e89b-12d3-a456-426614174000.jpg", "chatId": "ai-assistant-user@example.com"}}
        - Refresh an expired URL: {{"imageId": "https://storage.googleapis.com/...", "chatId": "project-123"}}

        Returns:
        A dictionary with:
        - imageUrl: The signed URL for accessing the image
        - imageId: The image ID
        - expiresAt: ISO timestamp when the URL expires
        """
        return {
            "type": "custom",
            "name": "gcp_cloud_storage_get_image_url",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "imageId": {
                        "type": "string",
                        "description": "Image ID (UUID.jpg format) or existing signed URL that needs refreshing"
                    },
                    "chatId": {
                        "type": "string",
                        "description": "Chat ID where the image was uploaded (required if imageId is a UUID)"
                    },
                    "expirationDays": {
                        "type": "integer",
                        "description": "Number of days until URL expires (default: 365)",
                        "minimum": 1,
                        "maximum": 3650
                    }
                },
                "required": ["imageId", "chatId"]
            }
        }

    def gcp_cloud_storage_get_image_url(
        self,
        imageId: str,
        chatId: str,
        expirationDays: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get a signed URL for accessing an image.

        Args:
            imageId: Image ID (UUID.jpg) or existing signed URL
            chatId: Chat ID where image was uploaded
            expirationDays: Optional expiration days (default: 365)

        Returns:
            Dictionary with imageUrl, imageId, and expiresAt, or error dictionary
        """
        try:
            if not self.storage_client or not self.org_slug:
                return {"error": "Storage client or org_slug not initialized"}

            if not imageId or not chatId:
                return {"error": "imageId and chatId are required"}

            # Extract storage path
            storage_path = self._extract_storage_path(imageId, chatId)
            if not storage_path:
                return {"error": f"Could not determine storage path for imageId: {imageId}"}

            # Generate signed URL
            expiration_days = expirationDays or self.url_expiration_days
            signed_url = self._generate_signed_url(storage_path, expiration_days)

            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(days=expiration_days)

            # Extract image ID from path if needed
            actual_image_id = storage_path.split('/')[-1]

            logger.info(f"Generated signed URL for image: {actual_image_id} in chat: {chatId}")

            return {
                "imageUrl": signed_url,
                "imageId": actual_image_id,
                "expiresAt": expires_at.isoformat() + "Z"
            }

        except Exception as e:
            logger.error(f"Error getting image URL: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def gcp_cloud_storage_list_chat_images_property(self):
        description = f"""
        List all images in a chat from Google Cloud Storage for org `{self.org_slug}`.

        Lists all images stored for a specific chat ID and returns signed URLs for accessing them.
        Images are stored at: orgs/{self.org_slug}/chat-images/{{chatId}}/{{imageId}}.jpg

        Parameters:
        - chatId: The chat ID to list images for
        - limit: Maximum number of images to return (default: 50, max: 200)
        - expirationDays: Number of days until URLs expire (default: 365)

        Returns:
        A list of dictionaries, each containing:
        - imageUrl: Signed URL for accessing the image
        - imageId: The image ID (filename)
        - uploadedAt: ISO timestamp when the image was uploaded (if available)
        - expiresAt: ISO timestamp when the URL expires
        """
        return {
            "type": "custom",
            "name": "gcp_cloud_storage_list_chat_images",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "chatId": {
                        "type": "string",
                        "description": "Chat ID to list images for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of images to return (default: 50, max: 200)",
                        "minimum": 1,
                        "maximum": 200
                    },
                    "expirationDays": {
                        "type": "integer",
                        "description": "Number of days until URLs expire (default: 365)",
                        "minimum": 1,
                        "maximum": 3650
                    }
                },
                "required": ["chatId"]
            }
        }

    def gcp_cloud_storage_list_chat_images(
        self,
        chatId: str,
        limit: int = 50,
        expirationDays: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        List all images in a chat.

        Args:
            chatId: Chat ID to list images for
            limit: Maximum number of images (default: 50, max: 200)
            expirationDays: Optional expiration days (default: 365)

        Returns:
            List of image dictionaries, or error dictionary
        """
        try:
            if not self.storage_client or not self.org_slug:
                return {"error": "Storage client or org_slug not initialized"}

            if not chatId:
                return {"error": "chatId is required"}

            # Validate limit
            limit = min(max(1, limit), 200)

            # Build storage path prefix
            storage_prefix = f"orgs/{self.org_slug}/chat-images/{chatId}/"

            # List blobs in the prefix
            bucket = self._get_bucket()
            blobs = bucket.list_blobs(prefix=storage_prefix, max_results=limit)

            images = []
            expiration_days = expirationDays or self.url_expiration_days

            for blob in blobs:
                # Skip if not an image file
                if not blob.name.endswith('.jpg'):
                    continue

                # Generate signed URL
                try:
                    signed_url = self._generate_signed_url(blob.name, expiration_days)
                    expires_at = datetime.utcnow() + timedelta(days=expiration_days)

                    # Get upload time from blob metadata
                    uploaded_at = None
                    if blob.time_created:
                        uploaded_at = blob.time_created.isoformat() + "Z"

                    image_id = blob.name.split('/')[-1]

                    images.append({
                        "imageUrl": signed_url,
                        "imageId": image_id,
                        "uploadedAt": uploaded_at,
                        "expiresAt": expires_at.isoformat() + "Z"
                    })
                except Exception as e:
                    logger.warning(f"Error generating URL for {blob.name}: {e}")
                    continue

            logger.info(f"Listed {len(images)} images for chat: {chatId}")
            return images

        except Exception as e:
            logger.error(f"Error listing chat images: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
