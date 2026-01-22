"""
Tests for ProjectManagementTool - Unified tool for all project management operations.
"""
import pytest
from unittest.mock import Mock, patch
from leanworks.agent.tools.project_management import ProjectManagementTool


class TestProjectManagementTool:

    @pytest.fixture
    def tool(self):
        """Create a ProjectManagementTool instance for testing."""
        return ProjectManagementTool(org_slug="test-org", user_id="test@example.com")

    # Task Management Tests
    def test_query_tasks(self, tool):
        """Test task querying."""
        mock_response = {
            "success": True,
            "data": [{"id": "task-1", "title": "Test Task"}],
            "metadata": {"rowCount": 1}
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.query_tasks(status='completed', limit=10)

            assert result['success'] == True
            assert result['data'][0]['title'] == 'Test Task'
            mock_make_request.assert_called_once()

    def test_create_task(self, tool):
        """Test task creation."""
        mock_response = {"id": "task-123", "title": "New Task"}

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.create_task(title="New Task", description="Test description")

            assert result['id'] == 'task-123'
            assert result['title'] == 'New Task'
            mock_make_request.assert_called_once()

    def test_update_task(self, tool):
        """Test task updating."""
        mock_response = {"success": True}

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.update_task(taskId="task-123", status="completed")

            assert result['success'] == True
            mock_make_request.assert_called_once()

    def test_query_task_progress_updates(self, tool):
        """Test task progress updates querying."""
        mock_response = {
            "success": True,
            "data": [{"id": "update-1", "task_id": "task-123"}],
            "metadata": {"rowCount": 1}
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.query_task_progress_updates(taskId="task-123")

            assert result['success'] == True
            assert result['data'][0]['task_id'] == 'task-123'
            mock_make_request.assert_called_once()

    # Project Management Tests
    def test_query_projects(self, tool):
        """Test project querying."""
        mock_response = {
            "success": True,
            "data": [{"id": "proj-1", "name": "Test Project"}],
            "metadata": {"rowCount": 1}
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.query_projects(status='active', limit=10)

            assert result['success'] == True
            assert result['data'][0]['name'] == 'Test Project'
            mock_make_request.assert_called_once()

    def test_query_project_progress_updates(self, tool):
        """Test project progress updates querying."""
        mock_response = {
            "success": True,
            "data": [{"id": "update-1", "project_id": "proj-123"}],
            "metadata": {"rowCount": 1}
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.query_project_progress_updates(projectId="proj-123")

            assert result['success'] == True
            assert result['data'][0]['project_id'] == 'proj-123'
            mock_make_request.assert_called_once()

    # Event Management Tests
    def test_query_events(self, tool):
        """Test event querying."""
        mock_response = {
            "success": True,
            "data": [{"id": "event-1", "title": "Test Event"}],
            "metadata": {"rowCount": 1}
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.query_events(userEmail="user@example.com")

            assert result['success'] == True
            assert result['data'][0]['title'] == 'Test Event'
            mock_make_request.assert_called_once()

    # SQL Query Tests
    def test_execute_sql_query(self, tool):
        """Test SQL query execution."""
        mock_response = {
            "success": True,
            "data": [{"count": 5}],
            "metadata": {"rowCount": 1, "executionTimeMs": 45}
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.execute_sql_query(
                sql="SELECT COUNT(*) as count FROM tasks WHERE status = $1",
                params=["completed"]
            )

            assert result['success'] == True
            assert result['data'][0]['count'] == 5
            mock_make_request.assert_called_once()

    def test_get_table_schema(self, tool):
        """Test schema retrieval."""
        mock_response = {
            "success": True,
            "data": {
                "table": "tasks",
                "columns": [
                    {"column_name": "id", "data_type": "uuid", "is_nullable": "NO", "column_comment": "Unique identifier for the task"},
                    {"column_name": "title", "data_type": "character varying", "is_nullable": "NO", "column_comment": "Title of the task"}
                ]
            }
        }

        with patch.object(tool, '_make_request', return_value=mock_response) as mock_make_request:
            result = tool.get_table_schema(table="tasks")

            assert result['success'] == True
            assert result['data']['table'] == 'tasks'
            assert len(result['data']['columns']) == 2
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

    # Helper Method Tests
    def test_resolve_assignee_to_email(self, tool):
        """Test assignee resolution."""
        mock_users_response = [
            {"email": "john@example.com", "first_name": "John", "last_name": "Doe"}
        ]

        with patch.object(tool, '_make_request', return_value=mock_users_response) as mock_make_request:
            result = tool._resolve_assignee_to_email("john@example.com")

            assert result == "john@example.com"
            mock_make_request.assert_called_once()

    def test_resolve_assignee_to_email_by_name(self, tool):
        """Test assignee resolution by name."""
        mock_users_response = [
            {"email": "john@example.com", "first_name": "John", "last_name": "Doe"}
        ]

        with patch.object(tool, '_make_request', return_value=mock_users_response) as mock_make_request:
            result = tool._resolve_assignee_to_email("John")

            assert result == "john@example.com"
            mock_make_request.assert_called_once()

    # Error Handling Tests
    def test_query_tasks_error_handling(self, tool):
        """Test error handling in task queries."""
        with patch.object(tool, '_make_request', side_effect=Exception("Network error")) as mock_make_request:
            result = tool.query_tasks(status='completed')

            assert result['error'] == 'Network error'
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

    # Tool Properties Tests
    def test_tool_properties_exist(self, tool):
        """Test that all tool properties are properly defined."""
        # Task management properties
        assert hasattr(tool, 'query_tasks_property')
        assert hasattr(tool, 'create_task_property')
        assert hasattr(tool, 'update_task_property')
        assert hasattr(tool, 'query_task_progress_updates_property')

        # Project management properties
        assert hasattr(tool, 'query_projects_property')
        assert hasattr(tool, 'query_project_progress_updates_property')

        # Event management properties
        assert hasattr(tool, 'query_events_property')

        # SQL query properties
        assert hasattr(tool, 'execute_sql_query_property')
        assert hasattr(tool, 'get_table_schema_property')

    def test_property_descriptions(self, tool):
        """Test that tool properties have correct descriptions."""
        # Test a few key properties
        query_tasks_prop = tool.query_tasks_property
        assert query_tasks_prop['name'] == 'query_tasks'
        assert 'flexible filtering' in query_tasks_prop['description']

        execute_sql_prop = tool.execute_sql_query_property
        assert execute_sql_prop['name'] == 'execute_sql_query'
        assert 'SQL queries against project management data' in execute_sql_prop['description']

    # Complex Query Tests
    def test_complex_join_query(self, tool):
        """Test complex join query across multiple tables."""
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