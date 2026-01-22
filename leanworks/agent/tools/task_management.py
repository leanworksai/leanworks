"""
Task Management Tool - DEPRECATED

This tool has been consolidated into ProjectManagementTool.
This file is kept for reference only and will be removed in a future version.

Use ProjectManagementTool instead:
- from leanworks.agent.tools.project_management import ProjectManagementTool
"""
import warnings
warnings.warn(
    "TaskManagementTool is deprecated. Use ProjectManagementTool instead.",
    DeprecationWarning,
    stacklevel=2
)

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

NOTE: For complex queries or joins with other tables, consider using execute_sql_query instead.
Query tasks with flexible filtering.

Parameters:
- status: Filter by status (todo, in-progress, review, completed, blocked)
- priority: Filter by priority (low, medium, high, urgent)
- assignee: Filter by assignee email
- assigneeId: Filter by assignee ID (email)
- createdBy: Filter by creator email
- projectId: Filter by project ID
- projectName: Filter by project name
- teamId: Filter by team ID
- visibility: Filter by visibility (all_members, specific_members)
- dueDate: Filter by exact due date (YYYY-MM-DD)
- dueDateBefore: Filter tasks due before this date (YYYY-MM-DD)
- dueDateAfter: Filter tasks due after this date (YYYY-MM-DD)
- createdAfter: Filter tasks created after this date (YYYY-MM-DD)
- createdBefore: Filter tasks created before this date (YYYY-MM-DD)
- estimatedHoursMin: Filter by minimum estimated hours
- estimatedHoursMax: Filter by maximum estimated hours
- actualHoursMin: Filter by minimum actual hours
- actualHoursMax: Filter by maximum actual hours
- tags: Filter by tags (array of strings - tasks must have ALL specified tags)
- titleContains: Filter by title containing this text (case-insensitive)
- descriptionContains: Filter by description containing this text (case-insensitive)
- reasonContains: Filter by reason containing this text (case-insensitive)
- limit: Max results (default 100)
- sortBy: Sort field (created_at, created_date, due_date, priority, status, title, assignee, estimated_hours, actual_hours)
- sortOrder: asc or desc (default desc)

Examples:
- query_tasks(status='completed', limit=10)
- query_tasks(assignee='user@example.com', priority='high')
- query_tasks(projectId='project-123', sortBy='due_date')
- query_tasks(dueDateBefore='2026-02-01', priority='urgent')
- query_tasks(tags=['bug', 'urgent'], createdAfter='2026-01-01')
- query_tasks(titleContains='API', descriptionContains='endpoint')
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["todo", "in-progress", "review", "completed", "blocked"],
                        "description": "Filter by status"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "description": "Filter by priority"
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Filter by assignee email address"
                    },
                    "assigneeId": {
                        "type": "string",
                        "description": "Filter by assignee ID (email)"
                    },
                    "createdBy": {
                        "type": "string",
                        "description": "Filter by creator email address"
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Filter by project ID (UUID)"
                    },
                    "projectName": {
                        "type": "string",
                        "description": "Filter by project name"
                    },
                    "teamId": {
                        "type": "string",
                        "description": "Filter by team ID"
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["all_members", "specific_members"],
                        "description": "Filter by visibility"
                    },
                    "dueDate": {
                        "type": "string",
                        "description": "Filter by exact due date (YYYY-MM-DD)"
                    },
                    "dueDateBefore": {
                        "type": "string",
                        "description": "Filter tasks due before this date (YYYY-MM-DD)"
                    },
                    "dueDateAfter": {
                        "type": "string",
                        "description": "Filter tasks due after this date (YYYY-MM-DD)"
                    },
                    "createdAfter": {
                        "type": "string",
                        "description": "Filter tasks created after this date (YYYY-MM-DD)"
                    },
                    "createdBefore": {
                        "type": "string",
                        "description": "Filter tasks created before this date (YYYY-MM-DD)"
                    },
                    "estimatedHoursMin": {
                        "type": "number",
                        "description": "Filter by minimum estimated hours"
                    },
                    "estimatedHoursMax": {
                        "type": "number",
                        "description": "Filter by maximum estimated hours"
                    },
                    "actualHoursMin": {
                        "type": "number",
                        "description": "Filter by minimum actual hours"
                    },
                    "actualHoursMax": {
                        "type": "number",
                        "description": "Filter by maximum actual hours"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags (tasks must have ALL specified tags)"
                    },
                    "titleContains": {
                        "type": "string",
                        "description": "Filter by title containing this text (case-insensitive)"
                    },
                    "descriptionContains": {
                        "type": "string",
                        "description": "Filter by description containing this text (case-insensitive)"
                    },
                    "reasonContains": {
                        "type": "string",
                        "description": "Filter by reason containing this text (case-insensitive)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 100)",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 1000
                    },
                    "sortBy": {
                        "type": "string",
                        "enum": ["created_at", "created_date", "due_date", "priority", "status", "title", "assignee", "estimated_hours", "actual_hours"],
                        "description": "Sort field"
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
        Query tasks via API with comprehensive filtering.

        Args:
            **kwargs: Query parameters including:
                - status: Filter by status
                - priority: Filter by priority
                - assignee/assigneeId: Filter by assignee email
                - createdBy: Filter by creator email
                - projectId/projectName: Filter by project
                - teamId: Filter by team
                - visibility: Filter by visibility
                - dueDate/dueDateBefore/dueDateAfter: Filter by due dates
                - createdAfter/createdBefore: Filter by creation dates
                - estimatedHoursMin/estimatedHoursMax: Filter by estimated hours range
                - actualHoursMin/actualHoursMax: Filter by actual hours range
                - tags: Filter by tags (array)
                - titleContains/descriptionContains/reasonContains: Text search filters
                - limit: Maximum results (default 100)
                - sortBy/sortOrder: Sorting options

        Returns:
            List of task dictionaries
        """
        try:
            # Normalize parameter names to match API expectations
            params = {}
            for key, value in kwargs.items():
                if key == 'assigneeId':
                    # Map assigneeId to assignee for API compatibility
                    params['assignee'] = value
                elif key in ['dueDateBefore', 'dueDateAfter', 'createdBefore', 'estimatedHoursMin', 'estimatedHoursMax', 'actualHoursMin', 'actualHoursMax', 'titleContains', 'descriptionContains', 'reasonContains']:
                    # These are additional filters that may not be implemented in API yet
                    # Pass them through - API will ignore unknown parameters
                    params[key] = value
                else:
                    params[key] = value

            result = self._make_request('GET', '/api/tasks', params=params)
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

    @property
    def query_task_progress_updates_property(self):
        """Query task progress updates with filtering."""
        return {
            "type": "custom",
            "name": "query_task_progress_updates",
            "description": """
Query individual work updates and progress reports from team members.

Parameters:
- userId: Filter by user email who made the update
- projectId: Filter by project ID
- dateFrom: Start date (YYYY-MM-DD)
- dateTo: End date (YYYY-MM-DD)
- limit: Max results (default 100)
- sortOrder: asc or desc (default desc)

Examples:
- query_task_progress_updates(userId='user@example.com', limit=20)
- query_task_progress_updates(projectId='proj-123', dateFrom='2026-01-01')
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "userId": {
                        "type": "string",
                        "description": "Filter by user email"
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Filter by project ID"
                    },
                    "dateFrom": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD)"
                    },
                    "dateTo": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 100)",
                        "default": 100
                    },
                    "sortOrder": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order (default desc)"
                    }
                }
            }
        }

    def query_task_progress_updates(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Query task progress updates via API.

        Args:
            **kwargs: Query parameters (userId, projectId, dateFrom, dateTo, limit, sortOrder)

        Returns:
            List of task progress update dictionaries
        """
        try:
            # NOTE: API endpoint needs to be created in leanworks-hub
            # Expected endpoint: GET /api/task-progress-updates
            # Query params: userId, projectId, dateFrom, dateTo, limit, sortOrder
            result = self._make_request('GET', '/api/task-progress-updates', params=kwargs)
            logger.info(f"query_task_progress_updates returned {len(result) if isinstance(result, list) else 'unknown'} updates")
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"query_task_progress_updates failed: {str(e)}")
            return {"error": str(e)}
