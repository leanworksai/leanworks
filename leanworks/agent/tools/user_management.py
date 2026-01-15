"""
User Management Tool - Domain-specific tool for user operations via leanworks-hub API.
Replaces PostgresTool for user-related operations.
"""
from typing import Dict, List, Any, Optional
from .base_api_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class UserManagementTool(BaseAPIClient):
    """User management operations via leanworks-hub API."""
    
    @property
    def query_users_property(self):
        """Query users with flexible filtering."""
        return {
            "type": "custom",
            "name": "query_users",
            "description": """
Query organization users with flexible filtering.

Parameters:
- status: Filter by status (active, inactive)
- role: Filter by role (owner, admin, member)
- searchTerm: Search by name or email
- limit: Max results (default 100)

Examples:
- query_users(status='active')
- query_users(role='admin')
- query_users(searchTerm='john')
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by user status (active, inactive)"
                    },
                    "role": {
                        "type": "string",
                        "description": "Filter by role (owner, admin, member)"
                    },
                    "searchTerm": {
                        "type": "string",
                        "description": "Search by name or email"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 100)",
                        "default": 100
                    }
                }
            }
        }
    
    def query_users(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Query users via API.
        
        Args:
            **kwargs: Query parameters (status, role, searchTerm, limit)
            
        Returns:
            List of user dictionaries
        """
        try:
            result = self._make_request('GET', '/api/users', params=kwargs)
            logger.info(f"query_users returned {len(result) if isinstance(result, list) else 'unknown'} users")
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"query_users failed: {str(e)}")
            return {"error": str(e)}
