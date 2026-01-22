"""
Query Management Tool - DEPRECATED

This tool has been consolidated into ProjectManagementTool.
This file is kept for reference only and will be removed in a future version.

Use ProjectManagementTool instead:
- from leanworks.agent.tools.project_management import ProjectManagementTool
"""
import warnings
warnings.warn(
    "QueryManagementTool is deprecated. Use ProjectManagementTool instead.",
    DeprecationWarning,
    stacklevel=2
)

from typing import Dict, List, Any, Optional
from .base_api_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class QueryManagementTool(BaseAPIClient):
    """Unified SQL query operations via leanworks-hub Query API."""

    @property
    def execute_sql_query_property(self):
        """Execute SQL queries against project management data."""
        return {
            "type": "custom",
            "name": "execute_sql_query",
            "description": """
Execute SQL queries against project management data.

Use this tool to query tasks, projects, events, users, and related data using SQL.
This provides flexible querying capabilities for complex data retrieval needs.

Parameters:
- sql (required): SQL SELECT or WITH query (max 10,000 characters)
- params: Array of parameterized query values (default: [])
- timeout: Query timeout in milliseconds (1000-60000, default: 30000)
- maxRows: Maximum rows to return (1-10000, default: 1000)

Available Tables:
- users: Organization user profiles and roles
- tasks: Task management data (status, priority, assignments)
- projects: Project information and metadata
- task_progress_updates: Task update history and progress notes
- task_comments: Comments on tasks
- project_progress_updates: Project update summaries
- project_members: Project membership and roles
- project_comments: Comments on projects
- events: Calendar events and meetings

Security:
- Only SELECT and WITH (CTE) queries allowed
- Parameterized queries recommended for dynamic values
- Rate limited: 100 queries per 15 minutes

Examples:
- execute_sql_query(sql="SELECT * FROM tasks WHERE status = $1 LIMIT 10", params=["completed"])
- execute_sql_query(sql="SELECT * FROM tasks WHERE assignee_id = $1", params=["user@example.com"])
- execute_sql_query(sql="SELECT p.name, COUNT(t.id) as task_count FROM projects p LEFT JOIN tasks t ON p.id = t.project_id GROUP BY p.id, p.name ORDER BY task_count DESC LIMIT 5")

Best Practices:
- Use LIMIT clauses to control result size
- Use parameterized queries ($1, $2) for dynamic values
- Use appropriate timeouts for complex queries
- Check schema first for complex queries
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
                "body": sql,
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

Use this tool to understand table structures before writing SQL queries.
Returns column names, data types, nullability, and defaults.

Parameters:
- table: Specific table name to get schema for (optional)

If table is specified, returns detailed column information.
If table is omitted, returns list of all available tables.

Available Tables:
- users, tasks, projects, events
- task_progress_updates, task_comments
- project_progress_updates, project_members, project_comments

Examples:
- get_table_schema() - List all available tables
- get_table_schema(table="tasks") - Get detailed schema for tasks table
- get_table_schema(table="users") - Get detailed schema for users table
            """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Specific table name to get schema for (optional)"
                    }
                }
            }
        }

    def get_table_schema(self, table: Optional[str] = None) -> Dict[str, Any]:
        """
        Get table schema information via Query API.

        Args:
            table: Specific table name (optional)

        Returns:
            Dictionary with schema information
        """
        try:
            params = {"table": table} if table else {}
            result = self._make_request('GET', '/api/query/schema', params=params)

            if result.get('success'):
                logger.info(f"get_table_schema successful for table: {table or 'all'}")
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