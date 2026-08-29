"""
Base API client for leanworks-hub backend.
Provides environment-aware authentication (API key for local, bearer token for production).
"""
import requests
import os
import logging
from typing import Dict, Any, Optional

from leanworks.utils.env import get_hub_url

logger = logging.getLogger(__name__)

# MIME types for upload (must match leanworks-hub file-validation)
_UPLOAD_MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".csv": "text/csv",
}


class BaseAPIClient:
    """
    Base HTTP API client for leanworks-hub backend.
    Uses API key locally and bearer token in production.
    """
    
    def __init__(self, org_slug: str, user_id: Optional[str] = None):
        """
        Initialize base API client.
        
        Args:
            org_slug: Organization slug (e.g., 'leanworks.ai')
            user_id: Optional user ID/email for authentication
        """
        self.org_slug = org_slug
        self.user_id = user_id
        
        # Environment-aware configuration
        self.base_url = get_hub_url()
        self.api_key = os.getenv('LEANWORKS_API_KEY')
        self.bearer_token = os.getenv('LEANWORKS_BEARER_TOKEN')
        
        # If API key not set via environment variable, try to get it from Secret Manager
        # (same method used by ask API in local/dev)
        if not self.api_key:
            try:
                from app.services.client import get_cached_api_key
                # Try both naming conventions for API key (same as ask API middleware)
                for secret_name in ["api-key", "API_KEY"]:
                    try:
                        self.api_key = get_cached_api_key(secret_name)
                        if self.api_key:
                            logger.info(f"Retrieved API key from Secret Manager using secret name: {secret_name}")
                            break
                    except Exception as e:
                        logger.debug(
                            "Failed to get API key from Secret Manager "
                            "(error_type=%s)",
                            type(e).__name__,
                        )
                        continue
            except ImportError:
                # Not in backend context, Secret Manager not available
                logger.debug("Secret Manager not available (not in backend context)")
            except Exception as e:
                logger.debug(
                    "Error attempting to get API key from Secret Manager "
                    "(error_type=%s)",
                    type(e).__name__,
                )
        
        logger.info(f"BaseAPIClient initialized: base_url={self.base_url}, org_slug={org_slug}, has_api_key={bool(self.api_key)}, has_bearer_token={bool(self.bearer_token)}")
        
    def _get_headers(self) -> Dict[str, str]:
        """
        Get authentication headers based on environment.

        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            'Content-Type': 'application/json',
            'X-Org-Identifier': self.org_slug  # Organization identifier (slug or ID)
        }
        
        # Prefer API key for local development
        if self.api_key:
            headers['X-API-Key'] = self.api_key
            if self.user_id:
                headers['X-User-Email'] = self.user_id
        # Use bearer token for production
        elif self.bearer_token:
            headers['Authorization'] = f'Bearer {self.bearer_token}'
        else:
            logger.warning("No authentication credentials found (API key or bearer token)")
            
        return headers
    
    def _make_request(self, method: str, endpoint: str, raw: bool = False, **kwargs) -> Any:
        """
        Make HTTP request with proper error handling.
        
        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint path (e.g., '/api/tasks')
            raw: If True, return raw response content instead of JSON (default: False)
            **kwargs: Additional arguments passed to requests.request()
            
        Returns:
            Response JSON data (or raw bytes if raw=True)
            
        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        # Merge headers
        request_headers = self._get_headers()
        if 'headers' in kwargs:
            request_headers.update(kwargs.pop('headers'))
        
        # For raw responses, don't set JSON content-type
        if raw:
            request_headers.pop('Content-Type', None)
        
        kwargs['headers'] = request_headers
        
        try:
            logger.debug(f"Making {method} request to {url} (raw={raw})")
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            
            # Return raw bytes if requested
            if raw:
                return response.content
            
            # Return JSON if content exists
            if response.content:
                return response.json()
            return None
            
        except requests.HTTPError as e:
            logger.error(
                "HTTP request failed (method=%s, status=%s)",
                method, e.response.status_code,
            )
            raise
        except requests.RequestException as e:
            logger.error(f"Request error for {method} {endpoint}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {method} {endpoint}: {str(e)}")
            raise

    def _make_upload_request(
        self,
        endpoint: str,
        file_path: str,
        file_field_name: str = "file",
        extra_data: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        POST multipart/form-data with a file and optional form fields.
        Used for document upload (POST /api/docs/upload).

        Args:
            endpoint: API path (e.g. '/api/docs/upload')
            file_path: Local path to the file to upload
            file_field_name: Form field name for the file (default 'file')
            extra_data: Optional dict of form fields (e.g. title, projectId)

        Returns:
            Response JSON dict

        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        request_headers = self._get_headers()
        request_headers.pop("Content-Type", None)  # Let requests set multipart boundary

        ext = os.path.splitext(file_path)[1].lower()
        mime_type = _UPLOAD_MIME_BY_EXT.get(ext, "application/octet-stream")
        filename = os.path.basename(file_path)

        extra_data = extra_data or {}
        data = {k: (v if v is not None else "") for k, v in extra_data.items()}

        try:
            with open(file_path, "rb") as f:
                files = [(file_field_name, (filename, f, mime_type))]
                logger.debug(f"Making POST upload request to {url} (file={filename})")
                response = requests.post(
                    url,
                    headers=request_headers,
                    data=data,
                    files=files,
                )
            response.raise_for_status()
            if response.content:
                return response.json()
            return None
        except requests.HTTPError as e:
            logger.error("HTTP POST failed (status=%s)", e.response.status_code)
            raise
        except requests.RequestException as e:
            logger.error(f"Request error for POST {endpoint}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for POST {endpoint}: {str(e)}")
            raise
