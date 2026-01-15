"""
Task Management Tool - Domain-specific tool for task operations via leanworks-hub API.
Replaces PostgresTool for task-related operations.
"""
from typing import Dict, List, Any, Optional
from .base_api_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class TaskManagementTool(BaseAPIClient):
    """Task management operations via leanworks-hub API."""
    
    @property
    def query_tasks_property(self):
        """Query tasks with flexible filtering."""
        return {
            "type": "custom",
            "name": "query_tasks",
            "description": """
Query tasks with flexible filtering.

Parameters:
- status: Filter by status (todo, in-progress, review, completed, blocked)
- priority: Filter by priority (low, medium, high, urgent)
- assignee: Filter by assignee email
- projectId: Filter by project ID
- createdAfter: ISO date string (YYYY-MM-DD)
- limit: Max results (default 100)
- sortBy: Sort field (created_at, due_date, priority, status)
- sortOrder: asc or desc (default desc)

Examples:
- query_tasks(status='completed', limit=10)
- query_tasks(assignee='user@example.com', priority='high')
- query_tasks(projectId='project-123', sortBy='due_date')
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status (todo, in-progress, review, completed, blocked)"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Filter by priority (low, medium, high, urgent)"
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Filter by assignee email address"
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Filter by project ID (UUID)"
                    },
                    "createdAfter": {
                        "type": "string",
                        "description": "Filter tasks created after this date (YYYY-MM-DD)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 100)",
                        "default": 100
                    },
                    "sortBy": {
                        "type": "string",
                        "description": "Sort field (created_at, due_date, priority, status)"
                    },
                    "sortOrder": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order (default desc)"
                    }
                }
            }
        }
    
    def query_tasks(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Query tasks via API.
        
        Args:
            **kwargs: Query parameters (status, priority, assignee, projectId, etc.)
            
        Returns:
            List of task dictionaries
        """
        try:
            result = self._make_request('GET', '/api/tasks', params=kwargs)
            logger.info(f"query_tasks returned {len(result) if isinstance(result, list) else 'unknown'} tasks")
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"query_tasks failed: {str(e)}")
            return {"error": str(e)}
    
    @property
    def create_task_property(self):
        """Create a new task."""
        return {
            "type": "custom",
            "name": "create_task",
            "description": """
Create a new task.

Parameters:
- title (required): Task title
- description: Task description
- projectId: Project ID (UUID)
- projectName: Project name (fallback if projectId not provided)
- assigneeId: Assignee email address (required format: valid email, e.g., 'user@example.com'). The API validates this must be a valid email format.
- status: Task status - one of 'todo', 'in-progress', 'review', 'completed', 'blocked' (default: 'todo')
- priority: Task priority - one of 'low', 'medium', 'high', 'urgent' (default: 'medium')
- dueDate: Due date in ISO format (YYYY-MM-DD)
- tags: Array of tag strings
- estimatedHours: Estimated hours (decimal number)
- visibility: 'all_members' or 'specific_members' (default: 'all_members')
- visibleToMembers: Array of email addresses (required if visibility='specific_members')

Returns:
- Success: Dictionary with task id and created fields
- Error: Dictionary with error message
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Task title (required)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Task description"
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Project ID (UUID)"
                    },
                    "projectName": {
                        "type": "string",
                        "description": "Project name"
                    },
                    "assigneeId": {
                        "type": "string",
                        "description": "Assignee email address (required format: valid email, e.g., 'user@example.com'). The API validates this must be a valid email format."
                    },
                    "status": {
                        "type": "string",
                        "enum": ["todo", "in-progress", "review", "completed", "blocked"],
                        "description": "Task status (default: 'todo')"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "description": "Task priority (default: 'medium')"
                    },
                    "dueDate": {
                        "type": "string",
                        "description": "Due date (YYYY-MM-DD)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of tag strings"
                    },
                    "estimatedHours": {
                        "type": "number",
                        "description": "Estimated hours"
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["all_members", "specific_members"],
                        "description": "Task visibility"
                    },
                    "visibleToMembers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of email addresses for specific_members visibility"
                    }
                },
                "required": ["title"]
            }
        }
    
    def create_task(self, **kwargs) -> Dict[str, Any]:
        """
        Create task via API.
        
        Args:
            **kwargs: Task properties (title, description, status, etc.)
            
        Returns:
            Dictionary with created task data or error
        """
        try:
            # Validate assigneeId format if provided
            if 'assigneeId' in kwargs and kwargs['assigneeId']:
                assignee_id = kwargs['assigneeId']
                # Check if it looks like an email (basic validation)
                if '@' not in assignee_id:
                    # Try to resolve display name to email by querying users
                    resolved_email = self._resolve_assignee_to_email(assignee_id)
                    if resolved_email:
                        kwargs['assigneeId'] = resolved_email
                        logger.info(f"Resolved assignee '{assignee_id}' to email '{resolved_email}'")
                    else:
                        return {
                            "error": f"assigneeId must be a valid email address (got: '{assignee_id}'). "
                                    f"Please provide an email address (e.g., 'user@example.com') instead of a display name."
                        }
            
            result = self._make_request('POST', '/api/tasks', json=kwargs)
            logger.info(f"create_task successful: {result.get('id')}")
            return result
        except Exception as e:
            logger.error(f"create_task failed: {str(e)}")
            return {"error": str(e)}
    
    def _resolve_assignee_to_email(self, assignee_input: str) -> Optional[str]:
        """
        Try to resolve assignee display name to email address by querying users API.
        
        Args:
            assignee_input: Display name or email address
            
        Returns:
            Email address if found, None otherwise
        """
        try:
            # If it already looks like an email, return it
            if '@' in assignee_input:
                return assignee_input.lower()
            
            # Query users endpoint to find matching user
            users = self._make_request('GET', '/api/users')
            if not isinstance(users, list):
                return None
            
            # Search for user by name (case-insensitive)
            assignee_lower = assignee_input.lower()
            for user in users:
                email = user.get('email', '').lower()
                first_name = user.get('first_name', '').lower()
                last_name = user.get('last_name', '').lower()
                full_name = f"{first_name} {last_name}".strip().lower()
                
                # Check if input matches email, first name, last name, or full name
                if (assignee_lower == email or 
                    assignee_lower == first_name or 
                    assignee_lower == last_name or
                    assignee_lower == full_name or
                    assignee_lower in full_name):
                    return email
            
            return None
        except Exception as e:
            logger.warning(f"Error resolving assignee '{assignee_input}': {str(e)}")
            return None
    
    @property  
    def update_task_property(self):
        """Update an existing task."""
        return {
            "type": "custom",
            "name": "update_task",
            "description": """
Update an existing task.

Parameters:
- taskId (required): Task ID to update
- title: Update title
- description: Update description
- status: Update status (todo, in-progress, review, completed, blocked)
- priority: Update priority (low, medium, high, urgent)
- assigneeId: Update assignee email address (required format: valid email, e.g., 'user@example.com'). The API validates this must be a valid email format.
- projectId: Update project ID
- dueDate: Update due date (YYYY-MM-DD)
- tags: Update tags array
- estimatedHours: Update estimated hours
- actualHours: Update actual hours
- visibility: Update visibility
- visibleToMembers: Update visible members list

Returns:
- Success: Dictionary with success: true
- Error: Dictionary with error message
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "taskId": {
                        "type": "string",
                        "description": "Task ID to update (required)"
                    },
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["todo", "in-progress", "review", "completed", "blocked"]
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"]
                    },
                    "assigneeId": {
                        "type": "string",
                        "description": "Assignee email address (required format: valid email, e.g., 'user@example.com'). The API validates this must be a valid email format."
                    },
                    "projectId": {"type": "string"},
                    "dueDate": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "estimatedHours": {"type": "number"},
                    "actualHours": {"type": "number"},
                    "visibility": {
                        "type": "string",
                        "enum": ["all_members", "specific_members"]
                    },
                    "visibleToMembers": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["taskId"]
            }
        }
    
    def update_task(self, taskId: str, **kwargs) -> Dict[str, Any]:
        """
        Update task via API.
        
        Args:
            taskId: Task ID to update
            **kwargs: Fields to update
            
        Returns:
            Dictionary with success status or error
        """
        try:
            # Validate assigneeId format if provided
            if 'assigneeId' in kwargs and kwargs['assigneeId']:
                assignee_id = kwargs['assigneeId']
                # Check if it looks like an email (basic validation)
                if '@' not in assignee_id:
                    # Try to resolve display name to email by querying users
                    resolved_email = self._resolve_assignee_to_email(assignee_id)
                    if resolved_email:
                        kwargs['assigneeId'] = resolved_email
                        logger.info(f"Resolved assignee '{assignee_id}' to email '{resolved_email}'")
                    else:
                        return {
                            "error": f"assigneeId must be a valid email address (got: '{assignee_id}'). "
                                    f"Please provide an email address (e.g., 'user@example.com') instead of a display name."
                        }
            
            result = self._make_request('PATCH', f'/api/tasks/{taskId}', json=kwargs)
            logger.info(f"update_task successful: {taskId}")
            return result if result else {"success": True}
        except Exception as e:
            logger.error(f"update_task failed: {str(e)}")
            return {"error": str(e)}
