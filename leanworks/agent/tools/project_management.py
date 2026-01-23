"""
Project Management Tool - Unified tool for all project management operations.
Handles tasks, projects, events, and SQL queries via leanworks-hub API.

This tool consolidates:
- Task management (CRUD, progress updates)
- Project management (queries, progress summaries)
- Event management (calendar queries)
- SQL query operations (direct database access)
"""
from typing import Dict, List, Any, Optional, Union
from .base_api_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class ProjectManagementTool(BaseAPIClient):
    """
    Unified project management operations via leanworks-hub API.

    This tool consolidates all project management functionality:
    - Task operations (query, create, update, progress updates)
    - Project operations (query, progress summaries)
    - Event operations (calendar queries)
    - SQL operations (direct database access)
    """

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
- taskId: Filter by task ID
- dateFrom: Start date (YYYY-MM-DD)
- dateTo: End date (YYYY-MM-DD)
- limit: Max results (default 100)
- sortOrder: asc or desc (default desc)

Examples:
- query_task_progress_updates(taskId='task-123', limit=10)
- query_task_progress_updates(userId='user@example.com', dateFrom='2026-01-01')
- query_task_progress_updates(projectId='proj-123', sortOrder='asc')
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "userId": {
                        "type": "string",
                        "description": "Filter by user email who made the update"
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Filter by project ID"
                    },
                    "taskId": {
                        "type": "string",
                        "description": "Filter by task ID"
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
            **kwargs: Query parameters (userId, projectId, taskId, dateFrom, dateTo, limit, sortOrder)

        Returns:
            List of task progress update dictionaries
        """
        try:
            result = self._make_request('GET', '/api/task-progress-updates', params=kwargs)
            logger.info(f"query_task_progress_updates returned {len(result) if isinstance(result, list) else 'unknown'} updates")
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"query_task_progress_updates failed: {str(e)}")
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

            # If it's empty or None, return None
            if not assignee_input or assignee_input.strip() == '':
                return None

            # Skip resolution for org slugs (they should not be used as assignees)
            assignee_lower = assignee_input.lower().strip()
            if assignee_lower.startswith('personal_') or len(assignee_lower) > 50:
                logger.warning(f"Skipping assignee resolution for suspected org slug: '{assignee_input}'")
                return None

            # Query users endpoint to find matching user
            users = self._make_request('GET', '/api/users')
            if not isinstance(users, list):
                return None

            # Search for user by name (case-insensitive)
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

    # ============================================================================
    # EVENT MANAGEMENT
    # ============================================================================

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

    # ============================================================================
    # SQL QUERY OPERATIONS
    # ============================================================================

    @property
    def execute_sql_query_property(self):
        """Execute SQL queries against project management data."""
        return {
            "type": "custom",
            "name": "execute_sql_query",
            "description": """
Execute SQL queries against project management data.

Use this tool to query tasks, projects, events, users, and related data using SQL. You can ONLY use this tool after you understand table schemas.
This provides flexible querying capabilities for complex data retrieval needs.

Parameters:
- sql (required): SQL SELECT or WITH query (max 10,000 characters)
- params: Array of parameterized query values (default: [])
- timeout: Query timeout in milliseconds (1000-60000, default: 30000)
- maxRows: Maximum rows to return (1-10000, default: 1000) - acts as upper bound for LIMIT clauses

Security:
- Only SELECT and WITH (CTE) queries allowed
- Parameterized queries recommended for dynamic values
- All LIMIT clauses are validated against maxRows parameter

SQL Query Best Practices:
1. Include LIMIT clauses in your SQL for precise control (will be validated against maxRows)
2. Use parameterized queries ($1, $2) for dynamic values
3. Prefer joining tables in a single SQL statement (using JOIN) over querying different tables separately, as this is more efficient and enables richer queries.
4. Use appropriate timeout for complex queries
5. Check for truncated results in metadata

Basic Examples:
- execute_sql_query(sql="SELECT * FROM tasks WHERE status = $1 LIMIT 10", params=["completed"])
- execute_sql_query(sql="SELECT * FROM tasks WHERE assignee_id = $1 LIMIT 5", params=["user@example.com"])
- execute_sql_query(sql="SELECT p.name, COUNT(t.id) as task_count FROM projects p LEFT JOIN tasks t ON p.id = t.project_id GROUP BY p.id, p.name ORDER BY task_count DESC LIMIT 5")
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT or WITH query (max 10,000 characters)"
                    },
                    "params": {
                        "type": "array",
                        "items": {"type": ["string", "number", "boolean", "null"]},
                        "description": "Parameterized query values (default: [])"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Query timeout in milliseconds (1000-60000, default: 30000)",
                        "minimum": 1000,
                        "maximum": 60000
                    },
                    "maxRows": {
                        "type": "integer",
                        "description": "Maximum rows to return (1-10000, default: 1000)",
                        "minimum": 1,
                        "maximum": 10000
                    }
                },
                "required": ["sql"]
            }
        }

    def execute_sql_query(self, sql: str, params: Optional[List] = None,
                         timeout: int = 30000, maxRows: int = 1000) -> Dict[str, Any]:
        """
        Execute SQL query via Query API.

        Args:
            sql: SQL SELECT or WITH query
            params: Parameterized query values
            timeout: Query timeout in milliseconds
            maxRows: Maximum rows to return

        Returns:
            Dictionary with success, data, and metadata
        """
        try:
            payload = {
                "sql": sql,
                "params": params or [],
                "options": {
                    "timeout": timeout,
                    "maxRows": maxRows,
                    "includeMetadata": True
                }
            }

            result = self._make_request('POST', '/api/query/execute', json=payload)

            if result.get('success'):
                logger.info(f"execute_sql_query successful: {result.get('metadata', {}).get('rowCount', 0)} rows")
                return result
            else:
                logger.error(f"execute_sql_query failed: {result.get('error', {}).get('message')}")
                return result

        except Exception as e:
            logger.error(f"execute_sql_query exception: {str(e)}")
            return {
                "success": False,
                "error": {
                    "code": "EXECUTION_ERROR",
                    "message": str(e)
                }
            }

    @property
    def get_table_schema_property(self):
        """Get schema information for queryable tables."""
        return {
            "type": "custom",
            "name": "get_table_schema",
            "description": """
Get schema information for queryable tables.

ALWAYS call this tool BEFORE execute_sql_query to understand table structures and column names/types.

RESPONSE FORMAT:

1. Without table parameter:
   {
     "success": true,
     "data": {
       "tables": ["users", "tasks", "projects", "task_progress_updates", "task_comments", "project_progress_updates", "project_members", "project_comments", "events"],
       "note": "Use ?table=<table_name> or ?table[]=table1&table[]=table2 to get detailed schema for specific table(s)"
     }
   }

2. With single table parameter (e.g., table="tasks"):
   {
     "success": true,
     "data": {
       "table": "tasks",
       "columns": [
         {"column_name": "id", "data_type": "TEXT", "is_nullable": false, "column_default": null, "column_description": "Task ID (primary key)"},
         {"column_name": "title", "data_type": "TEXT", "is_nullable": true, "column_default": null, "column_description": "Task name/title"},
         ...
       ]
     }
   }

3. With multiple tables parameter (e.g., table=["tasks", "projects"]):
   {
     "success": true,
     "data": {
       "tables": {
         "tasks": {
           "table": "tasks",
           "columns": [...]
         },
         "projects": {
           "table": "projects",
           "columns": [...]
         }
       },
       "note": "Retrieved schemas for 2 tables: tasks, projects"
     }
   }

USAGE STRATEGY:
1. Call without parameters to see list of available tables
2. Call with specific table name(s) to get detailed column information
3. For multiple tables, pass a list to minimize API calls
4. Use the column_name values when writing SQL queries
5. Check data_type and is_nullable to understand constraints

Parameters:
- table: Table name(s) to get schema for (optional)
  - If omitted: returns list of all available tables
  - If string: returns detailed schema for single table
  - If array: returns detailed schemas for multiple tables

Available Tables:
- users, tasks, projects, events
- task_progress_updates, task_comments
- project_progress_updates, project_members, project_comments

Examples:
- get_table_schema() - List all available tables
- get_table_schema(table="tasks") - Get schema for tasks table only
- get_table_schema(table=["tasks", "projects"]) - Get schemas for both tasks and projects tables
- get_table_schema(table=["project_progress_updates", "task_progress_updates"]) - Get schemas for progress update tables

IMPORTANT NOTES:
- Always verify column names from schema before writing SQL queries
- Use column_name exactly as returned (watch for snake_case like "project_id", "created_at", "date_id")
- Check data_type to use correct PostgreSQL functions (BIGINT vs DATE vs TEXT)
- If you see "BIGINT" type with milliseconds, use that for date filtering, NOT PostgreSQL date functions
- Pass multiple tables as array to reduce API round trips
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "table": {
                        "oneOf": [
                            {
                                "type": "string",
                                "description": "Single table name to get schema for"
                            },
                            {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of table names to get schemas for"
                            }
                        ],
                        "description": "Table name(s) to get schema for (optional - if omitted, returns list of all available tables)"
                    }
                }
            }
        }

    def get_table_schema(self, table: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Get table schema information via Query API.

        Args:
            table: Table name(s) - can be a single string, list of strings, or None

        Returns:
            Dictionary with schema information including column comments
        """
        try:
            params = {}
            if table:
                # Handle both single table string and list of tables
                if isinstance(table, list):
                    params["table"] = table
                else:
                    params["table"] = table

            # Always request column comments in the schema
            params["includeComments"] = True
            result = self._make_request('GET', '/api/query/schema', params=params)

            if result.get('success'):
                data = result.get('data', {})

                # Validate response structure
                if table:
                    if isinstance(table, list):
                        # When requesting multiple tables, we should get 'tables' object
                        if 'tables' not in data:
                            logger.error(f"get_table_schema returned invalid response for tables {table}: missing 'tables' field. Got: {list(data.keys())}")
                            return {
                                "success": False,
                                "error": {
                                    "code": "VALIDATION_ERROR",
                                    "message": f"API returned unexpected response format for tables {table}. Expected 'tables' field, got: {list(data.keys())}"
                                }
                            }
                    else:
                        # When requesting a single table, we should get 'columns' array
                        if 'columns' not in data:
                            logger.error(f"get_table_schema returned invalid response for table '{table}': missing 'columns' field. Got: {list(data.keys())}")
                            return {
                                "success": False,
                                "error": {
                                    "code": "VALIDATION_ERROR",
                                    "message": f"API returned unexpected response format for table '{table}'. Expected 'columns' field, got: {list(data.keys())}"
                                }
                            }
                else:
                    # When requesting all tables, we should get tables array
                    if 'tables' not in data:
                        logger.error(f"get_table_schema returned invalid response: missing 'tables' field. Got: {list(data.keys())}")
                        return {
                            "success": False,
                            "error": {
                                "code": "VALIDATION_ERROR",
                                "message": "API returned unexpected response format. Expected 'tables' field."
                            }
                        }

                table_desc = table if isinstance(table, str) else (f"{len(table)} tables" if isinstance(table, list) else "all")
                logger.info(f"get_table_schema successful for table: {table_desc}")
                return result
            else:
                logger.error(f"get_table_schema failed: {result.get('error', {}).get('message')}")
                return result

        except Exception as e:
            logger.error(f"get_table_schema exception: {str(e)}")
            return {
                "success": False,
                "error": {
                    "code": "EXECUTION_ERROR",
                    "message": str(e)
                }
            }
