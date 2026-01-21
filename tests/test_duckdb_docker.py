import pytest
import tempfile
import os
import json
import shutil
from unittest.mock import Mock, patch, MagicMock

# Import the Docker-based DuckDB classes and functions
from leanworks.agent.tools.duckdb import (
    DockerDuckDBTool,
    get_response_db_path_docker,
    get_container_workspace_dir,
    cleanup_responses_docker,
    clear_session_response_ids,
    add_session_response_id
)


class MockBashSession:
    """Mock bash session for testing"""
    def __init__(self, container_name="test-container", workspace_dir=None):
        self.container_name = container_name
        self.session_temp_dir = workspace_dir or tempfile.mkdtemp()
        self.container_workspace_path = "/workspace"


@pytest.fixture
def mock_bash_session():
    """Create a mock bash session with temp directory"""
    session = MockBashSession()
    yield session
    # Cleanup
    shutil.rmtree(session.session_temp_dir, ignore_errors=True)


@pytest.fixture
def docker_duckdb_tool(mock_bash_session):
    """Create a DockerDuckDBTool instance for testing"""
    response_id = "test-response-123"
    return DockerDuckDBTool(mock_bash_session, response_id=response_id)


class TestPathManagement:
    """Test path management functions"""

    def test_get_response_db_path_docker(self, mock_bash_session):
        """Test Docker path generation"""
        response_id = "test-123"
        path = get_response_db_path_docker(response_id, mock_bash_session.session_temp_dir)

        expected_dir = os.path.join(mock_bash_session.session_temp_dir, "duckdb", "responses", response_id)
        expected_file = os.path.join(expected_dir, "response.duckdb")

        assert path == expected_file
        assert os.path.exists(expected_dir)  # Directory should be created

    def test_get_container_workspace_dir(self):
        """Test container workspace path resolution"""
        workspace_dir = "/tmp/session_123"
        container_path = get_container_workspace_dir(workspace_dir)
        assert container_path == "/workspace"


class TestDockerDuckDBTool:
    """Test DockerDuckDBTool class"""

    @patch('subprocess.run')
    def test_execute_duckdb_command_success(self, mock_subprocess, docker_duckdb_tool):
        """Test successful DuckDB command execution"""
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout='{"result": "success"}',
            stderr=''
        )

        result = docker_duckdb_tool._execute_duckdb_command("/workspace/test.db", "SELECT 1")

        assert result["return_code"] == 0
        assert result["output"] == '{"result": "success"}'
        assert result["error"] == ""

        # Verify subprocess.run was called correctly
        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        assert cmd[0] == "docker"
        assert cmd[1] == "exec"
        assert docker_duckdb_tool.container_name in cmd
        assert "sh" in cmd
        assert "-c" in cmd
        assert "cd /workspace" in cmd[6]

    @patch('subprocess.run')
    def test_execute_duckdb_command_with_json_output(self, mock_subprocess, docker_duckdb_tool):
        """Test DuckDB command execution with JSON output flag"""
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout='[{"id": 1, "name": "test"}]',
            stderr=''
        )

        result = docker_duckdb_tool._execute_duckdb_command("/workspace/test.db", "SELECT * FROM test", use_json_output=True)

        assert result["return_code"] == 0
        assert result["output"] == '[{"id": 1, "name": "test"}]'

        # Verify -json flag was used
        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        assert "duckdb /workspace/test.db -json" in " ".join(cmd)

    @patch('subprocess.run')
    def test_execute_duckdb_command_timeout(self, mock_subprocess, docker_duckdb_tool):
        """Test DuckDB command execution timeout"""
        from subprocess import TimeoutExpired
        mock_subprocess.side_effect = TimeoutExpired("timeout", 30)

        result = docker_duckdb_tool._execute_duckdb_command("/workspace/test.db", "SELECT 1")

        assert result["return_code"] == -1
        assert result["error"] == "Command timed out"
        assert result["output"] == ""

    def test_write_json_to_workspace(self, docker_duckdb_tool):
        """Test writing JSON data to workspace"""
        test_data = [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}]

        container_path = docker_duckdb_tool._write_json_to_workspace(test_data)

        # Verify file was created in host workspace
        host_path = container_path.replace("/workspace", docker_duckdb_tool.workspace_dir)
        assert os.path.exists(host_path)

        # Verify JSON content
        with open(host_path, 'r') as f:
            loaded_data = json.load(f)
            assert loaded_data == test_data

        # Cleanup
        os.unlink(host_path)

    @patch('subprocess.run')
    def test_save_data_to_duckdb_simple(self, mock_subprocess, docker_duckdb_tool):
        """Test saving simple data to DuckDB"""
        mock_subprocess.return_value = Mock(returncode=0, stdout='', stderr='')

        test_data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        result = docker_duckdb_tool.save_data_to_duckdb(test_data, "users")

        assert result == "users"

        # Verify DuckDB command was executed
        assert mock_subprocess.call_count >= 1  # At least one command for CREATE TABLE

    @patch('subprocess.run')
    def test_save_data_to_duckdb_replace_mode(self, mock_subprocess, docker_duckdb_tool):
        """Test save_data_to_duckdb with replace mode"""
        mock_subprocess.return_value = Mock(returncode=0, stdout='', stderr='')

        # Test with replace mode (default)
        result = docker_duckdb_tool.save_data_to_duckdb([{"id": 1}], "test_table")
        assert result == "test_table"

    @patch('subprocess.run')
    def test_save_data_to_duckdb_append_mode(self, mock_subprocess, docker_duckdb_tool):
        """Test save_data_to_duckdb with append mode"""
        # Mock table exists check
        mock_subprocess.side_effect = [
            Mock(returncode=0, stdout='1', stderr=''),  # Table exists
            Mock(returncode=0, stdout='', stderr='')   # INSERT succeeds
        ]

        result = docker_duckdb_tool.save_data_to_duckdb([{"id": 2}], "test_table", if_exists="append")
        assert result == "test_table"

    def test_save_data_to_duckdb_no_response_id(self, mock_bash_session):
        """Test save_data_to_duckdb without response_id raises error"""
        tool = DockerDuckDBTool(mock_bash_session)  # No response_id

        with pytest.raises(ValueError, match="response_id must be set"):
            tool.save_data_to_duckdb([{"id": 1}], "test")

    @patch('subprocess.run')
    def test_query_duckdb_success(self, mock_subprocess, docker_duckdb_tool):
        """Test successful DuckDB query"""
        mock_json_output = '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout=mock_json_output,
            stderr=''
        )

        result = docker_duckdb_tool.query_duckdb("SELECT * FROM users")

        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[0]["name"] == "Alice"
        assert result[1]["id"] == 2
        assert result[1]["name"] == "Bob"

    @patch('subprocess.run')
    def test_query_duckdb_empty_result(self, mock_subprocess, docker_duckdb_tool):
        """Test DuckDB query with empty result"""
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout='',  # Empty output
            stderr=''
        )

        result = docker_duckdb_tool.query_duckdb("SELECT * FROM empty_table")
        assert result == []

    def test_query_duckdb_no_response_id(self, mock_bash_session):
        """Test query_duckdb without response_id raises error"""
        tool = DockerDuckDBTool(mock_bash_session)  # No response_id

        with pytest.raises(ValueError, match="response_id must be set"):
            tool.query_duckdb("SELECT 1")

    def test_query_duckdb_invalid_sql(self, docker_duckdb_tool):
        """Test query_duckdb with invalid SQL (non-SELECT)"""
        with pytest.raises(ValueError, match="Only read-only SELECT/WITH queries are allowed"):
            docker_duckdb_tool.query_duckdb("INSERT INTO test VALUES (1)")

    @patch('subprocess.run')
    def test_get_response_schema_success(self, mock_subprocess, docker_duckdb_tool):
        """Test successful schema retrieval"""
        # Mock table list query
        tables_json = '[{"table_name": "users"}, {"table_name": "products"}]'
        # Mock column queries for each table
        users_columns_json = '[{"column_name": "id", "data_type": "BIGINT", "is_nullable": false}, {"column_name": "name", "data_type": "VARCHAR", "is_nullable": true}]'
        products_columns_json = '[{"column_name": "product_id", "data_type": "BIGINT", "is_nullable": false}]'
        # Mock row count queries
        users_count_json = '[{"row_count": 5}]'
        products_count_json = '[{"row_count": 10}]'

        mock_subprocess.side_effect = [
            Mock(returncode=0, stdout=tables_json, stderr=''),      # Tables query
            Mock(returncode=0, stdout=users_columns_json, stderr=''), # Users columns
            Mock(returncode=0, stdout=users_count_json, stderr=''),   # Users count
            Mock(returncode=0, stdout=products_columns_json, stderr=''), # Products columns
            Mock(returncode=0, stdout=products_count_json, stderr='')  # Products count
        ]

        result = docker_duckdb_tool.get_response_schema()

        assert len(result) == 2

        # Check users table
        users_table = next(t for t in result if t["table"] == "users")
        assert users_table["row_count"] == 5
        assert len(users_table["columns"]) == 2
        assert users_table["columns"][0]["name"] == "id"
        assert users_table["columns"][0]["type"] == "BIGINT"
        assert users_table["columns"][0]["is_nullable"] == False

        # Check products table
        products_table = next(t for t in result if t["table"] == "products")
        assert products_table["row_count"] == 10
        assert len(products_table["columns"]) == 1

    def test_get_response_schema_no_response_id(self, mock_bash_session):
        """Test get_response_schema without response_id raises error"""
        tool = DockerDuckDBTool(mock_bash_session)  # No response_id

        with pytest.raises(ValueError, match="response_id must be set"):
            tool.get_response_schema()


class TestCleanupFunctions:
    """Test cleanup functions"""

    def test_cleanup_responses_docker(self, mock_bash_session):
        """Test Docker cleanup function"""
        # Create test response directories
        response_ids = {"resp1", "resp2"}
        for resp_id in response_ids:
            resp_dir = os.path.join(mock_bash_session.session_temp_dir, "duckdb", "responses", resp_id)
            os.makedirs(resp_dir, exist_ok=True)
            # Create a dummy file
            with open(os.path.join(resp_dir, "response.duckdb"), "w") as f:
                f.write("dummy")

        # Track response IDs
        clear_session_response_ids()
        for resp_id in response_ids:
            add_session_response_id(resp_id)

        # Run cleanup
        cleanup_responses_docker(mock_bash_session.session_temp_dir)

        # Verify directories are deleted
        for resp_id in response_ids:
            resp_dir = os.path.join(mock_bash_session.session_temp_dir, "duckdb", "responses", resp_id)
            assert not os.path.exists(resp_dir)

    def test_cleanup_responses_docker_no_responses(self, mock_bash_session):
        """Test cleanup when no response directories exist"""
        clear_session_response_ids()
        cleanup_responses_docker(mock_bash_session.session_temp_dir)
        # Should not raise any errors

    def test_cleanup_responses_docker_with_specific_ids(self, mock_bash_session):
        """Test cleanup with specific response IDs"""
        # Create only one response directory
        resp_dir = os.path.join(mock_bash_session.session_temp_dir, "duckdb", "responses", "specific-id")
        os.makedirs(resp_dir, exist_ok=True)

        cleanup_responses_docker(mock_bash_session.session_temp_dir, {"specific-id"})

        # Verify directory is deleted
        assert not os.path.exists(resp_dir)


class TestIntegrationScenarios:
    """Test integration scenarios"""

    @patch('subprocess.run')
    def test_nested_json_handling(self, mock_subprocess, docker_duckdb_tool):
        """Test handling of nested JSON structures"""
        mock_subprocess.return_value = Mock(returncode=0, stdout='', stderr='')

        nested_data = [
            {
                "id": 1,
                "user": {
                    "name": "Alice",
                    "profile": {
                        "age": 30,
                        "tags": ["developer", "python"]
                    }
                }
            }
        ]

        result = docker_duckdb_tool.save_data_to_duckdb(nested_data, "nested_users")
        assert result == "nested_users"

    @patch('subprocess.run')
    def test_complex_query_with_joins(self, mock_subprocess, docker_duckdb_tool):
        """Test complex queries that would be typical in DuckDB usage"""
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout='[{"user_name": "Alice", "total_orders": 5}]',
            stderr=''
        )

        # This would be a typical complex query
        sql = """
        SELECT u.name as user_name, COUNT(o.id) as total_orders
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        GROUP BY u.id, u.name
        HAVING COUNT(o.id) > 0
        """

        result = docker_duckdb_tool.query_duckdb(sql)
        assert len(result) == 1
        assert result[0]["user_name"] == "Alice"
        assert result[0]["total_orders"] == 5


if __name__ == "__main__":
    pytest.main([__file__])