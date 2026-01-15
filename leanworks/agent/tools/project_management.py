"""
Project Management Tool - Domain-specific tool for project operations via leanworks-hub API.
Replaces PostgresTool for project-related operations.
"""
from typing import Dict, List, Any, Optional
from .base_api_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class ProjectManagementTool(BaseAPIClient):
    """Project management operations via leanworks-hub API."""
    
    @property
    def query_projects_property(self):
        """Query projects with flexible filtering."""
        return {
            "type": "custom",
            "name": "query_projects",
            "description": """
Query projects with flexible filtering.

Parameters:
- status: Filter by status (active, completed, on-hold, cancelled)
- owner: Filter by owner email
- visibility: Filter by visibility (all_members, specific_members)
- memberEmail: Filter projects where user is a member
- limit: Max results (default 100)
- sortBy: Sort field (created_at, name, status)
- sortOrder: asc or desc (default desc)

Examples:
- query_projects(status='active', limit=10)
- query_projects(owner='user@example.com')
- query_projects(memberEmail='user@example.com')
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by project status"
                    },
                    "owner": {
                        "type": "string",
                        "description": "Filter by owner email address"
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["all_members", "specific_members"],
                        "description": "Filter by visibility setting"
                    },
                    "memberEmail": {
                        "type": "string",
                        "description": "Filter projects where this user is a member"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 100)",
                        "default": 100
                    },
                    "sortBy": {
                        "type": "string",
                        "description": "Sort field (created_at, name, status)"
                    },
                    "sortOrder": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order (default desc)"
                    }
                }
            }
        }
    
    def query_projects(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Query projects via API.
        
        Args:
            **kwargs: Query parameters (status, owner, memberEmail, etc.)
            
        Returns:
            List of project dictionaries
        """
        try:
            result = self._make_request('GET', '/api/projects', params=kwargs)
            logger.info(f"query_projects returned {len(result) if isinstance(result, list) else 'unknown'} projects")
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"query_projects failed: {str(e)}")
            return {"error": str(e)}
