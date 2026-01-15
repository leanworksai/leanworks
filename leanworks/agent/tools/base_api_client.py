"""
Base API client for leanworks-hub backend.
Provides environment-aware authentication (API key for local, bearer token for production).
"""
import requests
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


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
        self.base_url = os.getenv('LEANWORKS_HUB_URL', 'http://localhost:3001')
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
                        logger.debug(f"Failed to get API key from Secret Manager with secret name {secret_name}: {str(e)}")
                        continue
            except ImportError:
                # Not in backend context, Secret Manager not available
                logger.debug("Secret Manager not available (not in backend context)")
            except Exception as e:
                logger.debug(f"Error attempting to get API key from Secret Manager: {str(e)}")
        
        logger.info(f"BaseAPIClient initialized: base_url={self.base_url}, org_slug={org_slug}, has_api_key={bool(self.api_key)}, has_bearer_token={bool(self.bearer_token)}")
        
    def _get_headers(self) -> Dict[str, str]:
        """
        Get authentication headers based on environment.
        
        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            'Content-Type': 'application/json',
            'X-Org-Id': self.org_slug  # Always include org context
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
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Any:
        """
        Make HTTP request with proper error handling.
        
        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint path (e.g., '/api/tasks')
            **kwargs: Additional arguments passed to requests.request()
            
        Returns:
            Response JSON data
            
        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        # Merge headers
        request_headers = self._get_headers()
        if 'headers' in kwargs:
            request_headers.update(kwargs.pop('headers'))
        kwargs['headers'] = request_headers
        
        try:
            logger.debug(f"Making {method} request to {url}")
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            
            # Return JSON if content exists
            if response.content:
                return response.json()
            return None
            
        except requests.HTTPError as e:
            logger.error(f"HTTP error for {method} {endpoint}: {e.response.status_code} - {e.response.text}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request error for {method} {endpoint}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {method} {endpoint}: {str(e)}")
            raise
