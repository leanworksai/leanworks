import os
import uuid
import duckdb
import logging
import datetime
import json
import subprocess
import tempfile
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

# Global set to track response IDs created during the current session
_session_response_ids = set()

# Global root directory for DuckDB files
# Order of precedence: 1) Env var LEANWORKS_DUCKDB_ROOT, 2) Current working directory
ROOT_DIR = os.environ.get("LEANWORKS_DUCKDB_ROOT") or os.getcwd()


def get_root_dir() -> str:
    """
    Get the root directory for storing DuckDB files.
    """
    return ROOT_DIR


def set_root_dir(path: str) -> None:
    """
    Set the global root directory for storing DuckDB files.
    
    Args:
        path: Absolute path to the root directory
    """
    global ROOT_DIR
    ROOT_DIR = path


def clear_session_response_ids() -> None:
    """
    Clear the global set of response IDs tracked for the current session.
    This should be called at the start of a new conversation.
    """
    global _session_response_ids
    _session_response_ids.clear()
    logger.info("Cleared session response IDs tracking")


def add_session_response_id(response_id: str) -> None:
    """
    Add a response ID to the global tracking set.
    
    Args:
        response_id: The response ID to track
    """
    global _session_response_ids
    _session_response_ids.add(response_id)
    logger.debug(f"Added response ID to session tracking: {response_id}")


def get_session_response_ids() -> set:
    """
    Get the current set of tracked response IDs.
    
    Returns:
        Set of response IDs created during the current session
    """
    global _session_response_ids
    return _session_response_ids.copy()

def get_response_db_path(response_id: str) -> str:
    """
    Build the absolute DuckDB database file path for a specific response id.
    Location: <root>/duckdb/responses/<response_id>/response.duckdb

    NOTE: This function is for backwards compatibility with host-based DuckDB.
    For Docker-based operations, use get_response_db_path_docker instead.
    """
    base = get_root_dir()
    resp_dir = os.path.join(base, "duckdb", "responses", response_id)
    os.makedirs(resp_dir, exist_ok=True)
    return os.path.join(resp_dir, "response.duckdb")


def get_response_db_path_docker(response_id: str, workspace_dir: str) -> str:
    """
    Build the DuckDB database file path for Docker-based operations.
    Location: <workspace_dir>/duckdb/responses/<response_id>/response.duckdb

    Args:
        response_id: The response ID to build path for
        workspace_dir: The workspace directory (session temp dir on host)

    Returns:
        Path to the DuckDB database file
    """
    resp_dir = os.path.join(workspace_dir, "duckdb", "responses", response_id)
    os.makedirs(resp_dir, exist_ok=True)
    return os.path.join(resp_dir, "response.duckdb")


def get_container_workspace_dir(workspace_dir: str) -> str:
    """
    Get the container workspace path corresponding to a host workspace directory.

    Args:
        workspace_dir: Host workspace directory path

    Returns:
        Container workspace path (always /workspace)
    """
    return "/workspace"


class DuckDBTool:
    """
    A flexible DuckDB tool that supports saving various data types with auto-inferred schemas
    and running read-only SQL queries against them.

    Features:
    - Auto-schema inference from data (JSON, dicts, lists, etc.)
    - Full support for nested JSON structures (objects, arrays, deep nesting)
    - Dynamic table creation based on data structure
    - Support for replace/append operations
    - Read-only query execution with safety checks

    Nested JSON Examples:
        tool = DuckDBTool()
        
        # Nested objects
        tool.save_data_to_duckdb({"user": {"name": "John", "address": {"city": "NYC"}}}, "users")
        
        # Arrays of objects with nested data
        tool.save_data_to_duckdb([
            {"id": 1, "profile": {"age": 25, "tags": ["developer", "python"]}},
            {"id": 2, "profile": {"age": 30, "tags": ["manager", "sql"]}}
        ], "user_profiles")
        
        # Query nested data
        rows = tool.query_duckdb("SELECT id, profile.age, profile.tags[1] FROM user_profiles")
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize a DuckDB connection.

        Args:
            db_path: Path to a DuckDB database file. If None, an in-memory DB is used.
        """
        self.db_path = db_path
        # Ensure parent folder exists when using on-disk DB
        if isinstance(self.db_path, str):
            parent = os.path.dirname(self.db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        # When db_path is None, DuckDB creates an in-memory database
        self._conn = duckdb.connect(self.db_path or ":memory:", read_only=False)

    @classmethod
    def from_response(cls, response_id: str) -> "DuckDBTool":
        db_path = get_response_db_path(response_id=response_id)
        return cls(db_path=db_path)



    def save_data_to_duckdb(
        self,
        data: Any,
        table_name: str,
        if_exists: str = "replace"
    ) -> str:
        """
        Save data to DuckDB with auto-inferred schema, including support for nested JSON structures.
        
        Args:
            data: Data to save (dict, list of dicts, nested JSON, pandas DataFrame, etc.)
            table_name: Name of the table to create/update
            if_exists: 'replace' (default) or 'append'
            
        Returns:
            The table name used
            
        Examples:
            # Nested objects
            tool.save_data_to_duckdb({"user": {"name": "John", "address": {"city": "NYC"}}}, "users")
            
            # Arrays of objects
            tool.save_data_to_duckdb([{"id": 1, "tags": ["a", "b"]}, {"id": 2, "tags": ["c"]}], "items")
            
            # Complex nested structures
            tool.save_data_to_duckdb({"data": [{"nested": {"deep": {"value": 123}}}]}, "complex")
        """
        import json
        
        if if_exists not in {"replace", "append"}:
            logger.error(f"Invalid if_exists value: {if_exists}")
            return {"error": "if_exists must be 'replace' or 'append'"}
        
        # Ensure JSON extension is available for nested JSON support
        self._ensure_json_extension()
            
        # Convert data to a format DuckDB can auto-infer with nested support
        if isinstance(data, dict):
            # Single record - convert to list of dicts (preserves nested structure)
            records = [data]
        elif isinstance(data, list):
            # Handle different types of lists
            if all(isinstance(item, dict) for item in data):
                # List of dicts - use as is (preserves nested structures in each dict)
                records = data
            elif len(data) > 0:
                # Mixed list or list of primitives - wrap in a structure
                records = [{"items": data, "created_at": datetime.datetime.now().isoformat()}]
            else:
                # Empty list
                records = [{"items": [], "created_at": datetime.datetime.now().isoformat()}]
        elif isinstance(data, str):
            try:
                # Try to parse as JSON (handles nested JSON strings)
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    records = [parsed]
                elif isinstance(parsed, list):
                    # Handle nested arrays properly
                    if all(isinstance(item, dict) for item in parsed):
                        records = parsed
                    else:
                        records = [{"items": parsed, "created_at": datetime.datetime.now().isoformat()}]
                else:
                    # Scalar value from JSON
                    records = [{"value": parsed, "created_at": datetime.datetime.now().isoformat()}]
            except json.JSONDecodeError:
                # Not JSON, treat as plain text
                records = [{"content": data, "created_at": datetime.datetime.now().isoformat()}]
        else:
            # Handle other data types (objects with __dict__, etc.)
            try:
                if hasattr(data, '__dict__'):
                    # Convert object to dict (preserves nested attributes)
                    records = [self._obj_to_dict(data)]
                elif hasattr(data, 'keys'):
                    # Dict-like object
                    records = [dict(data)]
                else:
                    # Fallback for other types
                    records = [{"data": str(data), "type": type(data).__name__, "created_at": datetime.datetime.now().isoformat()}]
            except Exception as e:
                logger.warning("Failed to convert data to dict: %s", str(e))
                records = [{"data": str(data), "created_at": datetime.datetime.now().isoformat()}]
        
        logger.info("Saving %d records to table '%s' with nested JSON support", len(records), table_name)
        
        # Create table from records with auto-inferred schema
        if if_exists == "replace":
            # Drop existing table and create new one
            self._conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
        # Use DuckDB's direct Python object insertion
        # This is more reliable than JSON string parsing
        try:
            if if_exists == "replace" or not self._table_exists(table_name):
                # For replace mode, drop the table first
                self._conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                # Create table from first record to establish schema
                if records:
                    # Get column names from first record
                    first_record = records[0]
                    columns = list(first_record.keys())
                    
                    # Create table with appropriate column types
                    column_defs = []
                    for col in columns:
                        # Use VARCHAR as default type for flexibility with nested data
                        column_defs.append(f'"{col}" VARCHAR')
                    
                    create_sql = f"CREATE TABLE {table_name} ({', '.join(column_defs)})"
                    self._conn.execute(create_sql)
            
            # Insert records one by one to handle nested structures properly
            if records:
                first_record = records[0]
                columns = list(first_record.keys())
                placeholders = ', '.join(['?' for _ in columns])
                quoted_columns = [f'"{col}"' for col in columns]
                insert_sql = f'INSERT INTO {table_name} ({", ".join(quoted_columns)}) VALUES ({placeholders})'
                
                for record in records:
                    # Convert nested objects to JSON strings for storage
                    values = []
                    for col in columns:
                        value = record.get(col)
                        if isinstance(value, (dict, list)):
                            # Convert nested structures to JSON strings
                            values.append(json.dumps(value, default=str, ensure_ascii=False))
                        elif value is None:
                            values.append(None)
                        else:
                            values.append(str(value))
                    
                    self._conn.execute(insert_sql, values)
                    
        except Exception as e:
            logger.warning("Direct insertion approach failed, using JSON text fallback: %s", str(e))
            
            # Final fallback: create a simple table with the JSON as text
            if if_exists == "replace" or not self._table_exists(table_name):
                self._conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                self._conn.execute(f"""
                    CREATE TABLE {table_name} (
                        data_json TEXT,
                        record_count INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            
            json_data = json.dumps(records, default=str, ensure_ascii=False)
            self._conn.execute(f"INSERT INTO {table_name} (data_json, record_count) VALUES (?, ?)", 
                             [json_data, len(records)])
            
            logger.info("Successfully saved data to table '%s'", table_name)
            return table_name
        except Exception as e:
            logger.error(f"DuckDB save_data_to_duckdb failed: {str(e)}")
            # Return only the error message without full details
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    def _ensure_json_extension(self) -> None:
        """
        Ensure the JSON extension is loaded for better nested JSON support.
        """
        try:
            self._conn.execute("INSTALL json")
        except Exception:
            # Ignore install errors; extension might already be available
            pass
        try:
            self._conn.execute("LOAD json")
        except Exception:
            # Ignore load errors; basic JSON support should still work
            pass
    
    def _obj_to_dict(self, obj: Any) -> Dict[str, Any]:
        """
        Convert an object to a dictionary, handling nested objects recursively.
        """
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._obj_to_dict(value)
                elif isinstance(value, (list, tuple)):
                    result[key] = [self._obj_to_dict(item) if hasattr(item, '__dict__') else item for item in value]
                else:
                    result[key] = value
            return result
        else:
            return {"value": obj}
    
    def _table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        """
        result = self._conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", 
            [table_name]
        ).fetchone()
        return result[0] > 0

    def query_duckdb(self, sql: str) -> List[Dict[str, Any]]:
        try:
            if not isinstance(sql, str) or sql.strip() == "":
                raise ValueError("sql must be a non-empty string")

            sql_lower = sql.strip().lower()
            forbidden = [
                "insert",
                "update",
                "delete",
                "merge",
                "truncate",
                "create",
                "drop",
                "alter",
                "pragma",
                "copy",
                "attach",
                "detach",
                "replace",
            ]
            if not (sql_lower.startswith("select") or sql_lower.startswith("with")) or any(
                w in sql_lower for w in forbidden
            ):
                raise ValueError("Only read-only SELECT/WITH queries are allowed")

            logger.info("Executing DuckDB query: %s", sql.replace("\n", " ")[:500])
            cur = self._conn.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchall()

            result: List[Dict[str, Any]] = []
            for row in rows:
                item: Dict[str, Any] = {}
                for name, value in zip(cols, row):
                    if isinstance(value, (datetime.date, datetime.datetime)):
                        item[name] = value.isoformat()
                    else:
                        item[name] = value
                result.append(item)
            return result
        except Exception as e:
            logger.error(f"DuckDB query failed: {str(e)}")
            # Return only the error message without full details
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


class DockerDuckDBTool:
    """
    A Docker-based DuckDB tool that executes operations via DuckDB CLI in a Docker container.

    This tool shares the same Docker container as the bash tool and operates within the
    /workspace directory for consistent file access and isolation.
    """

    def __init__(self, bash_session, response_id: Optional[str] = None):
        """
        Initialize a Docker-based DuckDB tool.

        Args:
            bash_session: The bash session object containing container info
            response_id: Optional response ID for database operations
        """
        self.bash_session = bash_session
        self.response_id = response_id
        self.container_name = bash_session.container_name
        self.workspace_dir = bash_session.session_temp_dir  # Host workspace directory
        self.container_workspace_path = bash_session.container_workspace_path  # Usually /workspace

    def _execute_duckdb_command(self, db_path: str, sql: str, use_json_output: bool = False) -> dict:
        """
        Execute a DuckDB command in the Docker container.

        Args:
            db_path: Path to the DuckDB database file (container path)
            sql: SQL command to execute
            use_json_output: Whether to use -json flag for JSON output

        Returns:
            dict with 'output', 'error', and 'return_code' keys
        """
        try:
            # Build the DuckDB CLI command
            if use_json_output:
                duckdb_cmd = f'duckdb {db_path} -json "{sql}"'
            else:
                duckdb_cmd = f'duckdb {db_path} "{sql}"'

            # Execute command in Docker container
            exec_cmd = [
                'docker', 'exec',
                self.container_name,
                'sh', '-c', f'cd {self.container_workspace_path} && {duckdb_cmd}'
            ]

            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )

            return {
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            logger.error(f"DuckDB command timed out: {sql}")
            return {
                "output": "",
                "error": "Command timed out",
                "return_code": -1
            }
        except Exception as e:
            logger.error(f"Error executing DuckDB command: {str(e)}")
            return {
                "output": "",
                "error": str(e),
                "return_code": -1
            }

    def _write_json_to_workspace(self, data: Any) -> str:
        """
        Write data as JSON to a temp file in the workspace directory.

        Args:
            data: Data to write as JSON

        Returns:
            Path to the JSON file (container path)
        """
        # Convert data to JSON
        json_data = json.dumps(data, default=str, ensure_ascii=False)

        # Write to temp file in workspace directory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=self.workspace_dir, delete=False) as f:
            f.write(json_data)
            temp_file_path = f.name

        # Return container path (relative to workspace)
        container_path = os.path.join(self.container_workspace_path, os.path.basename(temp_file_path))
        return container_path

    def save_data_to_duckdb(self, data: Any, table_name: str, if_exists: str = "replace") -> str:
        """
        Save data to DuckDB using CLI commands in Docker.

        Args:
            data: Data to save (dict, list of dicts, etc.)
            table_name: Name of the table to create/update
            if_exists: 'replace' or 'append'

        Returns:
            The table name used
        """
        if not self.response_id:
            raise ValueError("response_id must be set to save data")

        # Get database path
        db_path = get_response_db_path_docker(self.response_id, self.workspace_dir)
        container_db_path = db_path.replace(self.workspace_dir, self.container_workspace_path)

        # Convert data to appropriate format
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                records = data
            else:
                records = [{"items": data, "created_at": datetime.datetime.now().isoformat()}]
        else:
            records = [{"data": str(data), "created_at": datetime.datetime.now().isoformat()}]

        # Write records to JSON file in workspace
        json_file_path = self._write_json_to_workspace(records)

        try:
            # Create table from JSON file
            sql = f"CREATE TABLE {table_name} AS SELECT * FROM read_json('{json_file_path}');"

            if if_exists == "append":
                # For append, we need to handle existing tables
                # First check if table exists, then insert
                check_result = self._execute_duckdb_command(
                    container_db_path,
                    f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
                )

                if check_result["return_code"] == 0 and "1" in check_result["output"]:
                    # Table exists, use INSERT
                    sql = f"INSERT INTO {table_name} SELECT * FROM read_json('{json_file_path}');"
                # If table doesn't exist, CREATE will work

            result = self._execute_duckdb_command(container_db_path, sql)

            if result["return_code"] != 0:
                logger.error(f"Failed to save data to DuckDB: {result['error']}")
                return {"error": result["error"]}

            logger.info(f"Successfully saved {len(records)} records to table '{table_name}'")
            return table_name

        finally:
            # Clean up temp JSON file
            try:
                host_json_path = json_file_path.replace(self.container_workspace_path, self.workspace_dir)
                os.unlink(host_json_path)
            except Exception:
                pass  # Ignore cleanup errors

    def query_duckdb(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute a read-only SQL query using DuckDB CLI in Docker.

        Args:
            sql: Read-only SQL query (SELECT/WITH only)

        Returns:
            List of dictionaries representing query results
        """
        if not self.response_id:
            raise ValueError("response_id must be set to query data")

        # Validate query is read-only
        sql_lower = sql.strip().lower()
        forbidden = [
            "insert", "update", "delete", "merge", "truncate", "create", "drop",
            "alter", "pragma", "copy", "attach", "detach", "replace"
        ]

        if not (sql_lower.startswith("select") or sql_lower.startswith("with")) or any(
            w in sql_lower for w in forbidden
        ):
            raise ValueError("Only read-only SELECT/WITH queries are allowed")

        # Get database path
        db_path = get_response_db_path_docker(self.response_id, self.workspace_dir)
        container_db_path = db_path.replace(self.workspace_dir, self.container_workspace_path)

        # Execute query with JSON output
        result = self._execute_duckdb_command(container_db_path, sql, use_json_output=True)

        if result["return_code"] != 0:
            logger.error(f"DuckDB query failed: {result['error']}")
            return {"error": result["error"]}

        try:
            # Parse JSON output
            if result["output"].strip():
                rows = json.loads(result["output"])
                # Convert datetime strings back if needed
                for row in rows:
                    for key, value in row.items():
                        if isinstance(value, str):
                            # Try to parse ISO datetime strings
                            try:
                                # This is a simplified check - could be enhanced
                                if 'T' in value and ('-' in value or ':' in value):
                                    datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
                                    row[key] = value  # Keep as string for now
                            except:
                                pass
                return rows
            else:
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse DuckDB JSON output: {e}")
            return {"error": f"Failed to parse query results: {e}"}

    def get_response_schema(self) -> List[Dict[str, Any]]:
        """
        Get table schema information for the response database.

        Returns:
            List of table schema dictionaries
        """
        if not self.response_id:
            raise ValueError("response_id must be set to get schema")

        db_path = get_response_db_path_docker(self.response_id, self.workspace_dir)
        container_db_path = db_path.replace(self.workspace_dir, self.container_workspace_path)

        schema = []

        # Get table list
        tables_result = self._execute_duckdb_command(
            container_db_path,
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name",
            use_json_output=True
        )

        if tables_result["return_code"] != 0:
            logger.error(f"Failed to get tables: {tables_result['error']}")
            return {"error": tables_result["error"]}

        try:
            tables = json.loads(tables_result["output"]) if tables_result["output"].strip() else []
        except json.JSONDecodeError:
            return {"error": "Failed to parse tables list"}

        for table_row in tables:
            table_name = table_row.get("table_name")
            if not table_name:
                continue

            # Get column information
            cols_result = self._execute_duckdb_command(
                container_db_path,
                f"SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema = 'main' AND table_name = '{table_name}' ORDER BY ordinal_position",
                use_json_output=True
            )

            if cols_result["return_code"] != 0:
                logger.warning(f"Failed to get columns for table {table_name}: {cols_result['error']}")
                continue

            try:
                cols = json.loads(cols_result["output"]) if cols_result["output"].strip() else []
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse columns for table {table_name}")
                continue

            # Get row count
            count_result = self._execute_duckdb_command(
                container_db_path,
                f"SELECT COUNT(*) as row_count FROM {table_name}",
                use_json_output=True
            )

            row_count = 0
            if count_result["return_code"] == 0:
                try:
                    count_data = json.loads(count_result["output"])
                    row_count = count_data[0].get("row_count", 0) if count_data else 0
                except:
                    pass

            columns = []
            for col in cols:
                columns.append({
                    "name": col.get("column_name"),
                    "type": col.get("data_type"),
                    "is_nullable": bool(col.get("is_nullable", True))
                })

            schema.append({
                "table": table_name,
                "row_count": row_count,
                "columns": columns
            })

        return schema


def query_response_duckdb_property() -> Dict[str, Any]:
    description = """
    Run a read-only SQL query against a response-scoped DuckDB database. 
    Tables and schemas are auto-inferred from the data that was saved, including full support for nested JSON structures.
    
    Nested JSON querying examples:
    - Access nested objects: SELECT user.name, user.address.city FROM table_name
    - Access array elements: SELECT tags[1], tags[2] FROM table_name  
    - Unnest arrays: SELECT unnest(tags) as tag FROM table_name
    - Use SHOW TABLES or DESCRIBE table_name to explore the schema.

    Use this tool when previous tool result statistics is insufficient to answer the question.
    Only SELECT/WITH queries are allowed. DML/DDL/PRAGMA/COPY are rejected. Don't query the whole table, only query the data or statistics you need.
    """
    return {
        "type": "custom",
        "name": "query_response_duckdb",
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "response_id": {"type": "string", "description": "Identifier of the saved response"},
                "sql": {"type": "string", "description": "Read-only SQL (SELECT/WITH)"}
            },
            "required": ["response_id", "sql"]
        }
    }


def get_response_schema_property() -> Dict[str, Any]:
    description = """
    Get the table schema information for a response-scoped DuckDB database.
    This tool helps you understand the structure of tables and their columns before writing queries.
    
    Returns detailed schema information including:
    - Table names available in the database
    - Column names, data types, and nullable status for each table
    - Support for nested JSON structures and complex data types
    
    Use this tool when previous tool result statistics is insufficient to answer the question. First to explore the data structure, then use query_response_duckdb to run queries.
    """
    return {
        "type": "custom",
        "name": "get_response_schema",
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "response_id": {"type": "string", "description": "Identifier of the saved response database"}
            },
            "required": ["response_id"]
        }
    }

# (module-level query_duckdb and close removed; now instance methods on DuckDBTool)


def save_data(
    data: Any,
    table_name: str,
    response_id: Optional[str] = None,
    if_exists: str = "replace"
) -> str:
    """
    Save data into DuckDB with auto-inferred schema. Returns the response_id used.
    
    Args:
        data: Data to save (dict, list of dicts, JSON string, etc.)
        table_name: Name of the table to create
        response_id: Optional response ID, generates UUID if not provided
        if_exists: 'replace' (default) or 'append'
    
    Returns:
        The response_id used for the database
    """
    rid = response_id or str(uuid.uuid4())
    
    # Track this response ID for cleanup
    add_session_response_id(rid)
    
    tool = DuckDBTool.from_response(response_id=rid)
    try:
        tool.save_data_to_duckdb(data=data, table_name=table_name, if_exists=if_exists)
        return rid
    finally:
        tool.close()


def query_response_duckdb(response_id: str, sql: str) -> List[Dict[str, Any]]:
    """
    Execute a read-only SQL query against the saved response database identified by response_id.
    
    Args:
        response_id: Identifier of the saved response
        sql: Read-only SQL query (SELECT/WITH only)
    
    Returns:
        List of dictionaries representing query results
    """
    tool = DuckDBTool.from_response(response_id=response_id)
    try:
        return tool.query_duckdb(sql)
    finally:
        tool.close()

def get_response_schema(response_id: str) -> List[Dict[str, Any]]:
    """
    Inspect the DuckDB database for a given response_id and return detailed table/column schema information.
    This function is essential for AI agents to understand data structure before writing queries.
    
    Args:
        response_id: Identifier of the saved response database
    
    Returns:
        List of table schema dictionaries with the following structure:
        [
            {
                "table": "table_name",
                "row_count": 1234,
                "columns": [
                    {
                        "name": "column_name", 
                        "type": "VARCHAR|INTEGER|JSON|STRUCT|LIST|etc", 
                        "is_nullable": true|false
                    },
                    ...
                ]
            },
            ...
        ]
        
    Example output for nested JSON data:
        [
            {
                "table": "users",
                "row_count": 10,
                "columns": [
                    {"name": "id", "type": "BIGINT", "is_nullable": false},
                    {"name": "user", "type": "STRUCT(name VARCHAR, address STRUCT(city VARCHAR))", "is_nullable": true},
                    {"name": "tags", "type": "VARCHAR[]", "is_nullable": true}
                ]
            }
        ]
    """
    tool = DuckDBTool.from_response(response_id=response_id)
    try:
        tables = tool.query_duckdb(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        )
        schema: List[Dict[str, Any]] = []
        for row in tables:
            table_name = row.get("table_name")
            if not table_name:
                continue
                
            # Get column information
            cols = tool.query_duckdb(
                f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'main' AND table_name = '{table_name}'
                ORDER BY ordinal_position
                """
            )
            columns: List[Dict[str, Any]] = []
            for c in cols:
                columns.append(
                    {
                        "name": c.get("column_name"),
                        "type": c.get("data_type"),
                        "is_nullable": bool(c.get("is_nullable")),
                    }
                )
            
            # Get row count for the table
            try:
                count_result = tool.query_duckdb(f"SELECT COUNT(*) as row_count FROM {table_name}")
                row_count = count_result[0].get("row_count", 0) if count_result else 0
            except Exception:
                row_count = 0  # Fallback if count fails
                
            schema.append({
                "table": table_name, 
                "row_count": row_count,
                "columns": columns
            })
        return schema
    finally:
        tool.close()


def cleanup_responses(response_ids: Optional[set] = None) -> None:
    """
    Clean up DuckDB response databases for the given response IDs.
    If no response_ids provided, uses the tracked session response IDs.
    
    Args:
        response_ids: Set of response IDs to clean up. If None, uses tracked session IDs.
    """
    import shutil
    
    # Use tracked response IDs if none provided
    if response_ids is None:
        response_ids = get_session_response_ids()
    
    if not response_ids:
        logger.info("No DuckDB response databases to clean up")
        return
    
    try:
        base_dir = get_root_dir()
        responses_dir = os.path.join(base_dir, "duckdb", "responses")
        
        if not os.path.exists(responses_dir):
            logger.info(f"No DuckDB responses directory found at {responses_dir}")
            return
        
        deleted_count = 0
        
        # Clean up each tracked response ID
        for response_id in response_ids:
            response_dir = os.path.join(responses_dir, response_id)
            
            if os.path.exists(response_dir) and os.path.isdir(response_dir):
                try:
                    shutil.rmtree(response_dir)
                    deleted_count += 1
                    logger.info(f"Deleted DuckDB response directory: {response_dir}")
                except Exception as e:
                    logger.error(f"Failed to delete DuckDB response directory {response_dir}: {str(e)}")
        
        # Clear the tracked response IDs after cleanup
        clear_session_response_ids()
        
        if deleted_count > 0:
            logger.info(f"Response cleanup completed: deleted {deleted_count} DuckDB response database(s)")
        else:
            logger.info("Response cleanup completed: no DuckDB response databases found to delete")
            
    except Exception as e:
        logger.error(f"Error during DuckDB response cleanup: {str(e)}")


def cleanup_responses_docker(workspace_dir: str, response_ids: Optional[set] = None) -> None:
    """
    Clean up DuckDB response databases in a Docker workspace directory.

    Args:
        workspace_dir: The workspace directory (host path) where DuckDB files are stored
        response_ids: Set of response IDs to clean up. If None, uses tracked session IDs.
    """
    import shutil

    # Use tracked response IDs if none provided
    if response_ids is None:
        response_ids = get_session_response_ids()

    if not response_ids:
        logger.info("No DuckDB response databases to clean up")
        return

    try:
        responses_dir = os.path.join(workspace_dir, "duckdb", "responses")

        if not os.path.exists(responses_dir):
            logger.info(f"No DuckDB responses directory found at {responses_dir}")
            return

        deleted_count = 0

        # Clean up each tracked response ID
        for response_id in response_ids:
            response_dir = os.path.join(responses_dir, response_id)

            if os.path.exists(response_dir) and os.path.isdir(response_dir):
                try:
                    shutil.rmtree(response_dir)
                    deleted_count += 1
                    logger.info(f"Deleted DuckDB response directory: {response_dir}")
                except Exception as e:
                    logger.error(f"Failed to delete DuckDB response directory {response_dir}: {str(e)}")

        # Clear the tracked response IDs after cleanup
        clear_session_response_ids()

        if deleted_count > 0:
            logger.info(f"Docker response cleanup completed: deleted {deleted_count} DuckDB response database(s)")
        else:
            logger.info("Docker response cleanup completed: no DuckDB response databases found to delete")

    except Exception as e:
        logger.error(f"Error during Docker DuckDB response cleanup: {str(e)}")





