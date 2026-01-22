"""
Event Management Tool - DEPRECATED

This tool has been consolidated into ProjectManagementTool.
This file is kept for reference only and will be removed in a future version.

Use ProjectManagementTool instead:
- from leanworks.agent.tools.project_management import ProjectManagementTool
"""
import warnings
warnings.warn(
    "EventManagementTool is deprecated. Use ProjectManagementTool instead.",
    DeprecationWarning,
    stacklevel=2
)

from typing import Dict, List, Any, Optional
from .base_api_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class EventManagementTool(BaseAPIClient):
    """Event/calendar management operations via leanworks-hub API."""
    
    @property
    def query_events_property(self):
        """Query events with flexible filtering."""
        return {
            "type": "custom",
            "name": "query_events",
            "description": """
Query calendar events with flexible filtering.

NOTE: For complex queries or joins with other tables, consider using execute_sql_query instead.

Use this to check user availability, find free time slots, understand scheduling conflicts, 
and see upcoming meetings.

Parameters:
- userEmail: Filter events by user email (as attendee or creator)
- startDate: Filter events starting from this date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
- endDate: Filter events ending before this date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
- limit: Maximum number of events to return (default 100, max 500)

Examples:
- query_events(userEmail='user@example.com', startDate='2024-01-01', endDate='2024-01-31')
- query_events(userEmail='user@example.com', limit=20)
- query_events(startDate='2024-01-15T09:00:00', endDate='2024-01-15T17:00:00')
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "userEmail": {
                        "type": "string",
                        "description": "Filter by user email (as attendee or creator)"
                    },
                    "startDate": {
                        "type": "string",
                        "description": "Start date filter (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
                    },
                    "endDate": {
                        "type": "string",
                        "description": "End date filter (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of events (default 100, max 500)",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 500
                    }
                }
            }
        }
    
    def query_events(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Query events via API.
        
        Args:
            **kwargs: Query parameters (userEmail, startDate, endDate, limit)
            
        Returns:
            List of event dictionaries
        """
        try:
            result = self._make_request('GET', '/api/events', params=kwargs)
            logger.info(f"query_events returned {len(result) if isinstance(result, list) else 'unknown'} events")
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"query_events failed: {str(e)}")
            return {"error": str(e)}
