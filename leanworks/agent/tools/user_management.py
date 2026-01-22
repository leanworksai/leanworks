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

NOTE: For complex queries or joins with other tables, consider using execute_sql_query instead.

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
            logger.debug(f"query_users returned {len(result) if isinstance(result, list) else 'unknown'} users")
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"query_users failed: {str(e)}")
            return {"error": str(e)}

    @property
    def get_user_identification_instruction_property(self):
        """Property definition for get_user_identification_instruction tool."""
        return {
            "type": "custom",
            "name": "get_user_identification_instruction",
            "description": "Get detailed guidance on identifying and matching users across internal and external systems when user identity is ambiguous",
            "input_schema": {
                "type": "object",
                "properties": {}
            }
        }

    def get_user_identification_instruction(self) -> Dict[str, Any]:
        """
        Returns instructions for user identification and matching across systems.

        Returns:
            Dictionary with instructions for user identity matching
        """
        return {
            "instructions": """
USER IDENTITY MATCHING GUIDANCE:

Internal Tools (PostgreSQL, Search, DuckDB, Firestore):
- Use user_id from {USER_INFO} directly - no matching needed

External Tools (Outlook, Atlassian, GitHub, Linear):
- Verify user exists using search_users tool (jira_search_users, github_search_users, linear_search_users)
- If no match found, ask user for correct identifier - do NOT proceed

Confidence Thresholds:
- HIGH (≥0.9): Proceed automatically
- MEDIUM (0.7-0.9): Present options for confirmation
- LOW (<0.7) or NO MATCH: Ask user for identifier

Remember mappings for conversation duration. Skip re-verification if already confirmed.
            """
        }
