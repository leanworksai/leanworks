import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import json
import logging
import datetime
import re
import os
import threading
from google.cloud import secretmanager
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PostgresTool:
    """
    PostgreSQL tool for querying Cloud SQL databases.
    
    Fully automated deployment detection:
    - k8s deployment: Auto-detects /cloudsql directory and uses Unix socket path
                      (/cloudsql/PROJECT_ID:REGION:INSTANCE_NAME) via Cloud SQL Proxy sidecar
    - Local development: Auto-detects absence of /cloudsql and uses TCP connection
                         (127.0.0.1:5432) via local Cloud SQL Proxy
    
    Environment variables (all optional - auto-detection handles defaults):
    - DB_HOST: Database host (optional - auto-detected based on environment)
    - DB_PORT: Database port (default: 5432)
    - DB_USER: Database user (default: postgres)
    - DB_REGION: Cloud SQL region (default: us-west1)
    - DB_INSTANCE: Cloud SQL instance name (default: leanworks-prod)
    - DB_PASSWORD: Database password (fallback if Secret Manager fails)
    
    Note: No manual configuration needed - the tool automatically detects k8s vs local environment.
    """
    # Class-level PostgreSQL connection pools (per database/domain)
    _pools: Dict[str, pool.ThreadedConnectionPool] = {}
    _pools_lock = threading.Lock()
    _cached_password: Optional[str] = None
    _secret_manager_client: Optional[secretmanager.SecretManagerServiceClient] = None
    _project_id: Optional[str] = None
    
    @classmethod
    def set_secret_manager(cls, secret_manager_client: secretmanager.SecretManagerServiceClient, credential_path: str = "gcp_credential.json"):
        """Set the Secret Manager client and read project ID from credential file."""
        cls._secret_manager_client = secret_manager_client
        try:
            with open(credential_path, "r") as f:
                credential_data = json.load(f)
            cls._project_id = credential_data.get("project_id")
        except Exception as e:
            logger.error(f"Failed to read project_id from {credential_path}: {e}")
            raise
    
    # Shared database name (for users, organizations, org_members)
    SHARED_DB_NAME = 'shared'
    
    # Tables that we need to fetch schemas for (only these tables are used in queries)
    REQUIRED_TABLES = [
        'users',
        'tasks',
        'task_progress_updates',
        'project_progress_updates',
        'projects',
        'teams',
        'integrations',
        'docs',
        'project_comments',
        'project_members',
        'task_comments',
    ]
    
    # Comprehensive table descriptions to help the agent understand when to use each table
    TABLE_DESCRIPTIONS = {
        'users': (
            'Stores organization-specific user profiles and contact information. '
            'Contains user details like names, emails, job titles, responsibilities, and timezones. '
            'The email field is the primary key and is used as user_id throughout the system. '
            'Note: This contains org-specific profile data; global user accounts are in the shared database.'
        ),
        'tasks': (
            'Stores all tasks and action items within projects. '
            'Contains task details including status, priority, assignments, due dates, and progress tracking. '
            'Tasks are linked to projects via project_id and to users via assignee_id (email). '
            'This is the primary table for task management queries.'
        ),
        'task_progress_updates': (
            'Stores individual work updates and daily progress reports from team members. '
            'Contains detailed work descriptions, associated tasks, and timestamps. '
            'Use this for individual team member updates. '
            'This is different from project_progress_updates which contains aggregated summaries.'
        ),
        'project_progress_updates': (
            'Stores AI-generated daily aggregated summaries of all updates for each project. '
            'Contains one summary per project per day, combining all individual updates. '
            'Use this for high-level project status queries rather than detailed individual work reports.'
        ),
        'projects': (
            'Stores project information and metadata. '
            'Contains project details like name, description, status, priority, owner, and dates. '
            'Projects are the top-level organizational unit; tasks belong to projects. '
            'The id field is the primary key, but projects are often referenced by name in queries.'
        ),
        'teams': (
            'Stores team information and organizational structure. '
            'Contains team details like name, description, and owner. '
            'Teams can have associated projects and members. '
            'Note: Team membership details may be in related tables like project_members or team_members.'
        ),
        'integrations': (
            'Stores configuration and connection status for third-party integrations. '
            'Tracks which external tools (GitLab, Jira, Atlassian, etc.) are connected and their settings.'
        ),
        'docs': (
            'Stores documents, notes, and rich text content within the organization. '
            'Contains document content (HTML), metadata, and associations with projects or teams. '
            'Documents can be associated with projects or teams via project_id/team_id.'
        ),
        'project_comments': (
            'Stores comments and discussions on projects. '
            'Each comment is linked to a project via project_id and includes comment text, author info, and date.'
        ),
        'project_members': (
            'Stores project membership records linking users to projects. '
            'This is a junction table connecting projects (project_id) to users (user_email). '
            'Use this to find project teams and membership relationships.'
        ),
        'task_comments': (
            'Stores comments and discussions on individual tasks. '
            'Each comment is linked to a task via task_id and includes comment text, author info, and date.'
        ),
    }
    
    @classmethod
    def _get_database_name(cls, org_slug: str) -> str:
        """Get org database name from org_slug (with org_ prefix).
        
        Note: org_slug is expected to be already sanitized (lowercase, no special characters).
        """
        return f'org_{org_slug}'
    
    @classmethod
    def get_shared_pool(cls, credential_path: str = "gcp_credential.json") -> pool.ThreadedConnectionPool:
        """Get or create the shared database connection pool (for users, organizations, org_members)."""
        return cls._get_connection_pool(cls.SHARED_DB_NAME, credential_path)
    
    @classmethod
    def _get_postgres_password(cls, credential_path: str = "gcp_credential.json") -> str:
        """Get PostgreSQL password from Secret Manager (cached)."""
        if cls._cached_password:
            return cls._cached_password
        
        try:
            if cls._secret_manager_client and cls._project_id:
                full_name = f"projects/{cls._project_id}/secrets/postgresdb-password/versions/latest"
                response = cls._secret_manager_client.access_secret_version(name=full_name)
                cls._cached_password = response.payload.data.decode("UTF-8")
                logger.info("PostgreSQL password fetched from Secret Manager")
                return cls._cached_password
            else:
                raise ValueError("Secret Manager client not set")
        except Exception as e:
            logger.warning(f"Failed to fetch password from Secret Manager: {e}, trying environment variable")
            # Fallback to environment variable
            cls._cached_password = os.getenv("DB_PASSWORD", "")
            return cls._cached_password
    
    @classmethod
    def _get_connection_pool(cls, database_name: str, credential_path: str = "gcp_credential.json") -> pool.ThreadedConnectionPool:
        """Get or create a connection pool for a specific database."""
        if database_name in cls._pools:
            return cls._pools[database_name]
        
        with cls._pools_lock:
            # Double-check after acquiring lock
            if database_name in cls._pools:
                return cls._pools[database_name]
            
            try:
                # Read GCP credentials to get project ID
                with open(credential_path, "r") as f:
                    credential = json.load(f)
                project_id = credential.get("project_id", "leanworks-474204")
                
                # Get connection parameters
                password = cls._get_postgres_password(credential_path)
                region = os.getenv("DB_REGION", "us-west1")
                instance_name = os.getenv("DB_INSTANCE", "leanworks-prod")
                
                # Auto-detect connection method (k8s vs local)
                # Priority: 1) Explicit DB_HOST env var, 2) Auto-detect k8s (Unix socket), 3) Local (TCP)
                db_host = os.getenv("DB_HOST")
                
                if db_host:
                    # User explicitly set DB_HOST, use it
                    logger.info(f"Using DB_HOST from environment: {db_host}")
                else:
                    # Auto-detect: Check if we're in k8s (Unix socket available) or local
                    unix_socket_path = f"/cloudsql/{project_id}:{region}:{instance_name}"
                    
                    # Check if Unix socket directory exists (k8s with Cloud SQL Proxy sidecar)
                    if os.path.exists("/cloudsql") and os.path.isdir("/cloudsql"):
                        # Check if the specific socket file/directory exists
                        # Cloud SQL Proxy creates a socket file in /cloudsql/
                        socket_exists = False
                        try:
                            # List files in /cloudsql to see if our instance socket exists
                            if os.path.exists(unix_socket_path):
                                socket_exists = True
                            else:
                                # Sometimes the socket might be in a subdirectory or have a different name
                                # Check if any file/dir in /cloudsql matches our instance pattern
                                for item in os.listdir("/cloudsql"):
                                    if f"{project_id}:{region}:{instance_name}" in item:
                                        unix_socket_path = f"/cloudsql/{item}"
                                        socket_exists = True
                                        break
                        except (OSError, PermissionError):
                            pass
                        
                        if socket_exists:
                            db_host = unix_socket_path
                            logger.info(f"Auto-detected k8s environment: Using Unix socket connection: {db_host}")
                        else:
                            # /cloudsql exists but our socket doesn't - might be starting up, try anyway
                            db_host = unix_socket_path
                            logger.info(f"Auto-detected k8s environment (socket may be initializing): Using Unix socket: {db_host}")
                    else:
                        # No /cloudsql directory - we're likely in local development
                        db_host = "127.0.0.1"
                        logger.info(f"Auto-detected local environment: Using TCP connection: {db_host}:{os.getenv('DB_PORT', '5432')}")
                        logger.info("Note: Ensure Cloud SQL Proxy is running locally if connecting to Cloud SQL")
                
                db_port = int(os.getenv("DB_PORT", "5432"))
                db_user = os.getenv("DB_USER", "postgres")
                
                # Create connection pool
                # For k8s: uses Unix socket (/cloudsql/...) - no SSL needed
                # For local: uses TCP (127.0.0.1) via Cloud SQL Proxy - no SSL needed
                # SSL is only needed for direct TCP connections to Cloud SQL (not used here)
                pool_kwargs = {
                    'minconn': 1,
                    'maxconn': 10,
                    'host': db_host,
                    'port': db_port,
                    'database': database_name,
                    'user': db_user,
                    'password': password,
                    'cursor_factory': RealDictCursor,
                }
                
                # SSL is not needed for:
                # - Unix socket connections (k8s deployment via Cloud SQL Proxy sidecar)
                # - Local TCP connections via Cloud SQL Proxy
                # SSL would only be needed for direct TCP connections to Cloud SQL (not used)
                # psycopg2 will handle Unix socket paths automatically when host starts with '/'
                
                connection_pool = pool.ThreadedConnectionPool(**pool_kwargs)
                
                cls._pools[database_name] = connection_pool
                logger.info(f"PostgreSQL connection pool created for database: {database_name}")
                return connection_pool
                
            except Exception as e:
                logger.error(f"Failed to create PostgreSQL connection pool for {database_name}: {e}")
                raise
    
    def __init__(self, postgres_client_wrapper):
        """
        Initialize PostgresTool with a PostgreSQL client wrapper.
        
        Args:
            postgres_client_wrapper: An object with attributes `org_slug` (organization name like 'leanworks.ai')
                                    and optionally `client_name`.
        """
        self.postgres_client_wrapper = postgres_client_wrapper
        
        # Get org_slug from wrapper (use client_name as fallback)
        self.org_slug = getattr(self.postgres_client_wrapper, 'org_slug', None)
        if not self.org_slug:
            # Fallback: construct org_slug from client_name if available
            client_name = getattr(self.postgres_client_wrapper, 'client_name', 'unknown')
            # Try to construct a reasonable org_slug (this is a fallback, should provide actual org_slug)
            self.org_slug = f"{client_name}.ai" if client_name != 'unknown' else 'leanworks.ai'
            logger.warning(f"org_slug not provided in wrapper, using fallback: {self.org_slug}")
        
        # Get database name from org_slug
        self.database_name = self._get_database_name(self.org_slug)
        
        # Get credential path
        credential_path = getattr(self.postgres_client_wrapper, 'credential_path', 'gcp_credential.json')
        
        # Get connection pool for this domain's database
        self.pool = self._get_connection_pool(self.database_name, credential_path)
        
        # Fetch table schemas directly from PostgreSQL database
        try:
            self.schemas = self._fetch_schemas_from_db()
        except Exception as e:
            logger.warning(f"Failed to fetch schemas from PostgreSQL database: {str(e)}")
            self.schemas = ""
    
    def _fetch_schemas_from_db(self) -> str:
        """
        Fetch table schemas directly from PostgreSQL database using information_schema.
        Only fetches schemas for tables that are actually needed (REQUIRED_TABLES).
        
        Returns:
            Formatted string with table schemas
        """
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                # Only get the tables we actually need (not all tables in the database)
                # Use a parameterized query to filter by table names
                if not self.REQUIRED_TABLES:
                    return ""
                
                # Build parameterized query for specific tables
                placeholders = ','.join(['%s'] * len(self.REQUIRED_TABLES))
                cursor.execute(f"""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_type = 'BASE TABLE'
                      AND table_name IN ({placeholders})
                    ORDER BY table_name
                """, tuple(self.REQUIRED_TABLES))
                tables = cursor.fetchall()
                
                schema_parts = []
                
                for table_row in tables:
                    table_name = table_row.get('table_name')
                    if not table_name:
                        continue
                    
                    # Get column information
                    cursor.execute("""
                        SELECT 
                            column_name,
                            data_type,
                            is_nullable,
                            column_default,
                            character_maximum_length,
                            numeric_precision,
                            numeric_scale
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = %s
                        ORDER BY ordinal_position
                    """, (table_name,))
                    columns = cursor.fetchall()
                    
                    # Get primary key information
                    cursor.execute("""
                        SELECT column_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                          ON tc.constraint_name = kcu.constraint_name
                          AND tc.table_schema = kcu.table_schema
                        WHERE tc.table_schema = 'public'
                          AND tc.table_name = %s
                          AND tc.constraint_type = 'PRIMARY KEY'
                        ORDER BY kcu.ordinal_position
                    """, (table_name,))
                    pk_result = cursor.fetchall()
                    primary_keys = [row.get('column_name') for row in pk_result]
                    
                    # Use our custom comprehensive table descriptions
                    # These are specifically written to help the agent understand when and how to use each table
                    table_description = PostgresTool.TABLE_DESCRIPTIONS.get(
                        table_name, 
                        f'Table: {table_name} (no description available)'
                    )
                    
                    # Format table schema with prominent description
                    schema_parts.append(f"**Table: {table_name}**")
                    schema_parts.append(f"  Description: {table_description}")
                    
                    # Add primary key info
                    if primary_keys:
                        pk_str = ', '.join(primary_keys)
                        schema_parts.append(f"  Primary Key: {pk_str}")
                    
                    # Add columns
                    for col in columns:
                        col_name = col.get('column_name')
                        data_type = col.get('data_type', '')
                        is_nullable = col.get('is_nullable', 'YES') == 'YES'
                        col_default = col.get('column_default')
                        max_length = col.get('character_maximum_length')
                        precision = col.get('numeric_precision')
                        scale = col.get('numeric_scale')
                        
                        # Format data type with additional info
                        type_str = data_type.upper()
                        if max_length:
                            type_str += f"({max_length})"
                        elif precision is not None:
                            if scale is not None and scale > 0:
                                type_str += f"({precision},{scale})"
                            else:
                                type_str += f"({precision})"
                        
                        # Build column description
                        col_desc = f"  - {col_name} ({type_str})"
                        if not is_nullable:
                            col_desc += " - NOT NULL"
                        if col_default:
                            col_desc += f" - Default: {col_default}"
                        
                        schema_parts.append(col_desc)
                    
                    schema_parts.append("")  # Empty line between tables
                
                return "\n".join(schema_parts).strip()
                
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def _validate_sql_query(self, sql: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL query for safety - only allow read-only SELECT/WITH queries.
        
        Args:
            sql: SQL query string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(sql, str) or sql.strip() == "":
            return False, "SQL must be a non-empty string"
        
        sql_lower = sql.strip().lower()
        
        # List of forbidden SQL keywords that could modify data or schema
        forbidden_keywords = [
            "insert",
            "update",
            "delete",
            "merge",
            "truncate",
            "create",
            "drop",
            "alter",
            "grant",
            "revoke",
            "copy",
            "attach",
            "detach",
            "replace",
            "exec",
            "execute",
            "call",
        ]
        
        # Check if SQL starts with SELECT or WITH (CTE)
        if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
            return False, "Only SELECT and WITH (CTE) queries are allowed. Query must start with SELECT or WITH."
        
        # Check for forbidden keywords
        for keyword in forbidden_keywords:
            # Use word boundaries to avoid false positives (e.g., "description" shouldn't match "drop")
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, sql_lower):
                return False, f"Forbidden SQL keyword detected: {keyword}. Only read-only queries are allowed."
        
        return True, None
    
    def _execute_sql_query(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query and return results.
        
        Args:
            sql: SQL query string (must be SELECT or WITH statement)
            
        Returns:
            List of rows as dictionaries, or error dictionary
        """
        # Validate SQL query
        is_valid, error_msg = self._validate_sql_query(sql)
        if not is_valid:
            logger.error(f"SQL validation failed: {error_msg}")
            return {"error": error_msg}
        
        # Execute query
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                # Log query for audit (truncate for logging)
                logger.info(f"Executing PostgreSQL query: {sql.replace(chr(10), ' ')[:500]}")
                
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                # Convert RealDictRow to regular dict
                results = [dict(row) for row in rows]
                
                # Ensure 'id' field exists (use primary key if available)
                for result in results:
                    if "id" not in result:
                        # Try common ID field names
                        for id_field in ["id", "email", "update_id", "integration_id"]:
                            if id_field in result:
                                result["id"] = result[id_field]
                                break
                
                logger.info(f"PostgreSQL query completed, returned {len(results)} results (database: {self.database_name})")
                return results
                
        except Exception as e:
            logger.error(f"PostgreSQL query execution failed: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @property
    def query_postgres_property(self):
        description = f"""
        Query PostgreSQL database for org `{self.org_slug}` (database: {self.database_name}).
        
        This tool is strictly READ-ONLY. It executes SQL SELECT queries and returns results in document format.
        Only SELECT and WITH (CTE) queries are allowed. All write operations (INSERT, UPDATE, DELETE, etc.) are blocked.
        
        Provide `sql`: A SQL SELECT or WITH query string to execute.
        
        SQL Query Examples:
        
        Simple queries:
        - SELECT * FROM tasks WHERE status = 'completed'
        - SELECT * FROM users WHERE email = 'john@example.com'
        - SELECT * FROM tasks WHERE project_id = 'ProjectX' ORDER BY created_at DESC LIMIT 10
        
        Queries with JOINs:
        - SELECT t.*, p.name as project_name FROM tasks t JOIN projects p ON t.project_id = p.id WHERE t.status = 'in-progress'
        - SELECT u.email, u.first_name, u.last_name, COUNT(t.id) as task_count FROM users u LEFT JOIN tasks t ON u.email = t.assignee_id GROUP BY u.email, u.first_name, u.last_name
        
        Queries with CTEs (Common Table Expressions):
        - WITH recent_tasks AS (SELECT * FROM tasks WHERE created_at > EXTRACT(EPOCH FROM NOW() - INTERVAL '7 days') * 1000) SELECT * FROM recent_tasks WHERE status = 'todo'
        - WITH project_stats AS (SELECT project_id, COUNT(*) as task_count, SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_count FROM tasks GROUP BY project_id) SELECT p.name, ps.task_count, ps.completed_count FROM projects p JOIN project_stats ps ON p.id = ps.project_id
        
        Queries with aggregations:
        - SELECT project_id, COUNT(*) as total_tasks, AVG(estimated_hours) as avg_hours FROM tasks GROUP BY project_id
        - SELECT status, COUNT(*) as count FROM tasks GROUP BY status ORDER BY count DESC
        
        Table Selection Guide:
        Each table in the schema below includes a Description that explains when to use it. 
        ALWAYS read the table Description before writing SQL to ensure you're querying the correct table.
        
        Quick reference (see detailed descriptions in schemas below):
        - "users": User profiles, names, emails, job titles, responsibilities, timezones
        - "tasks": Tasks, action items, assignments, task status, due dates, priorities
        - "task_progress_updates": Individual work updates, daily progress reports, what team members worked on
        - "project_progress_updates": Daily aggregated project summaries, project-level progress overviews
        - "projects": Project information, project names, descriptions, collaborators, project metadata
        - "teams": Teams, team members, team leads, team projects
        - "integrations": External integrations (GitLab, Jira, Atlassian, etc.) and their configurations
        - "docs": Documents, notes, documentation, rich text content, pinned docs
        - "project_comments": Comments on projects, project discussions, project feedback
        - "project_members": Project membership, who is assigned to projects, project collaborators
        - "task_comments": Comments on tasks, task discussions, task feedback
        
        Important Notes:
        - ALWAYS read the table Description in the schemas below to understand when to use each table
        - ALWAYS check the table schemas below to understand the exact column names and data types before writing SQL
        - Field names in PostgreSQL are typically snake_case (e.g., created_at, updated_at, project_id)
        - Timestamps are stored as BIGINT (milliseconds since epoch)
        - Use the actual PostgreSQL table names directly (e.g., "task_progress_updates" not "updates")
        - Most of the time, user won't directly give you any 'id' but rather a 'name'. You should try to get the mapping from name to id first (for example, project name to project id and user name to user id), and then filter using the id.
        - When filtering by a 'name' field, you can use LIKE operator with % wildcard for prefix matching: WHERE name LIKE 'Project%'
        - If your response is empty, it means either you are filtering using a wrong value or the result is empty. In either case, you should try to query the first 5 rows to see if the result is empty. If it is not empty, then use those sample data to have a better understanding of the schema.
        - Read-only: Only SELECT and WITH queries are allowed. All write operations are automatically blocked.
        
        Table Schemas (REQUIRED READING - check these to understand table structure and column names):
        {self.schemas}
        """
        return {
            "type": "custom",
            "name": "query_postgres",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT or WITH query to execute (required). Only read-only queries are allowed.",
                    }
                },
                "required": ["sql"]
            }
        }
    
    def query_postgres(self, sql=None, **kwargs):
        """
        Query PostgreSQL database using a SQL query.
        
        Args:
            sql: SQL SELECT or WITH query string
        
        Returns:
            List of documents, or error dictionary
        """
        try:
            # Handle case where sql might be passed in kwargs
            if sql is None and 'sql' in kwargs:
                sql = kwargs['sql']
            elif sql is None:
                raise ValueError("sql parameter is required")
            
            if not isinstance(sql, str):
                raise ValueError("sql must be a string")
            
            # Execute SQL query
            start_time = datetime.datetime.now()
            results = self._execute_sql_query(sql)
            duration_ms = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
            
            if isinstance(results, dict) and "error" in results:
                return results
            
            logger.info(f"PostgreSQL query completed in {duration_ms}ms, returned {len(results)} results (database: {self.database_name})")
            
            return results
        
        except Exception as e:
            logger.error(f"PostgreSQL tool failed: org_slug={self.org_slug}, database={self.database_name}, error={str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

