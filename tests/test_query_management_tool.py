"""
Tests for QueryManagementTool - Unified SQL query tool for project management data.
"""
import pytest
from unittest.mock import Mock, patch
from leanworks.agent.tools.query_management import QueryManagementTool


class TestQueryManagementTool:

    @pytest.fixture
    def tool(self):
        """Create a QueryManagementTool instance for testing."""
        return QueryManagementTool(org_slug="test-org", user_id="test@example.com")

    def test_execute_sql_query_success(self, tool):
        """Test successful SQL query execution."""
        mock_response = {
            "success": True,
            "data": [{"id": "task-1", "title": "Test Task"}],
            "metadata": {"rowCount": 1, "executionTimeMs": 45}
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.execute_sql_query(
                sql="SELECT * FROM tasks WHERE status = $1 LIMIT 5",
                params=["completed"]
            )

            assert result['success'] == True
            assert result['data'] == [{"id": "task-1", "title": "Test Task"}]
            assert result['metadata']['rowCount'] == 1
            mock_make_request.assert_called_once()

    def test_execute_sql_query_validation_error(self, tool):
        """Test SQL query with forbidden keywords."""
        mock_response = {
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Forbidden SQL keyword detected: INSERT"
            }
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.execute_sql_query(sql="INSERT INTO tasks VALUES (...)")

            assert result['success'] == False
            assert result['error']['code'] == 'VALIDATION_ERROR'
            mock_make_request.assert_called_once()

    def test_execute_sql_query_timeout(self, tool):
        """Test query timeout handling."""
        mock_response = {
            "success": False,
            "error": {
                "code": "TIMEOUT_ERROR",
                "message": "Query execution timed out after 30000ms"
            }
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.execute_sql_query(
                sql="SELECT * FROM tasks",
                timeout=1  # Very short timeout
            )

            assert result['success'] == False
            assert result['error']['code'] == 'TIMEOUT_ERROR'
            mock_make_request.assert_called_once()

    def test_get_table_schema_all(self, tool):
        """Test getting all table schemas."""
        mock_response = {
            "success": True,
            "data": {
                "tables": ["users", "tasks", "projects", "events"]
            }
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.get_table_schema()

            assert result['success'] == True
            assert "tables" in result['data']
            assert "users" in result['data']['tables']
            mock_make_request.assert_called_once()

    def test_get_table_schema_specific(self, tool):
        """Test getting specific table schema."""
        mock_response = {
            "success": True,
            "data": {
                "table": "tasks",
                "columns": [
                    {"column_name": "id", "data_type": "uuid", "is_nullable": "NO"},
                    {"column_name": "title", "data_type": "character varying", "is_nullable": "NO"}
                ]
            }
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.get_table_schema(table="tasks")

            assert result['success'] == True
            assert result['data']['table'] == 'tasks'
            assert len(result['data']['columns']) == 2
            mock_make_request.assert_called_once()

    def test_parameterized_query(self, tool):
        """Test parameterized query execution."""
        mock_response = {
            "success": True,
            "data": [{"assignee_id": "user@example.com", "task_count": 5}],
            "metadata": {"rowCount": 1}
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.execute_sql_query(
                sql="SELECT assignee_id, COUNT(*) as task_count FROM tasks WHERE assignee_id = $1 GROUP BY assignee_id",
                params=["user@example.com"]
            )

            assert result['success'] == True
            assert result['data'][0]['task_count'] == 5
            mock_make_request.assert_called_once()

    def test_complex_join_query(self, tool):
        """Test complex join query."""
        mock_response = {
            "success": True,
            "data": [{
                "task_title": "API Development",
                "project_name": "Backend System",
                "assignee_name": "John Doe"
            }],
            "metadata": {"rowCount": 1}
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            sql = """
                SELECT t.title as task_title, p.name as project_name, u.first_name || ' ' || u.last_name as assignee_name
                FROM tasks t
                LEFT JOIN projects p ON t.project_id = p.id
                LEFT JOIN users u ON t.assignee_id = u.email
                WHERE t.status = 'in-progress'
                LIMIT 10
            """
            result = tool.execute_sql_query(sql=sql)

            assert result['success'] == True
            assert result['data'][0]['project_name'] == 'Backend System'
            mock_make_request.assert_called_once()

    def test_tool_properties(self, tool):
        """Test that tool properties are properly defined."""
        # Test execute_sql_query property
        execute_prop = tool.execute_sql_query_property
        assert execute_prop['name'] == 'execute_sql_query'
        assert execute_prop['type'] == 'custom'
        assert 'SQL SELECT or WITH query' in execute_prop['description']

        # Test get_table_schema property
        schema_prop = tool.get_table_schema_property
        assert schema_prop['name'] == 'get_table_schema'
        assert schema_prop['type'] == 'custom'
        assert 'schema information' in schema_prop['description']

    def test_error_handling(self, tool):
        """Test error handling for network issues."""
        with patch.object(tool, '_make_request', side_effect=Exception("Network error")) as mock_make_request:
            result = tool.execute_sql_query(sql="SELECT * FROM tasks LIMIT 1")

            assert result['success'] == False
            assert result['error']['code'] == 'EXECUTION_ERROR'
            assert 'Network error' in result['error']['message']
            mock_make_request.assert_called_once()