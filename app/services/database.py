"""
PostgreSQL database connection and query utilities
Multi-tenant support with separate databases per organization
"""
import json
import logging
import os
import re
import time
import subprocess
import socket
import atexit
from typing import Optional, Dict, Any, Tuple
from psycopg2 import pool, connect
from psycopg2.extras import RealDictCursor
from google.cloud import secretmanager
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# Global connection pools
_org_pools: Dict[str, pool.ThreadedConnectionPool] = {}  # org_<slug> -> pool
_tenant_pools: Dict[str, pool.ThreadedConnectionPool] = {}  # Legacy: domain -> pool (deprecated)
_shared_pool: Optional[pool.ThreadedConnectionPool] = None  # Shared database pool (for users table)
_cached_password: Optional[str] = None
_secret_manager_client: Optional[secretmanager.SecretManagerServiceClient] = None
_project_id: Optional[str] = None
_credentials = None
_cloud_sql_proxy_process: Optional[subprocess.Popen] = None  # Track proxy process


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open and accepting connections"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _ensure_cloud_sql_proxy_running() -> bool:
    """
    Ensure Cloud SQL proxy is running locally.
    Starts it if not already running.
    
    Returns:
        True if proxy is running (or was started), False otherwise
    """
    global _cloud_sql_proxy_process
    
    # Only run in local environment
    is_kubernetes = os.environ.get("KUBERNETES_SERVICE_HOST") is not None
    socket_path = "/cloudsql/leanworks-474204:us-west1:leanworks-prod/.s.PGSQL.5432"
    is_gcp = os.path.exists(os.path.dirname(socket_path))
    
    if is_kubernetes or is_gcp:
        # Not local, don't start proxy
        return True
    
    # Check if port 5432 is already open
    db_port = int(os.environ.get("DB_PORT", "5432"))
    if _is_port_open("127.0.0.1", db_port):
        logger.info(f"✅ Cloud SQL proxy already running on port {db_port}")
        return True
    
    # Check if we already started a process
    if _cloud_sql_proxy_process is not None:
        # Check if process is still alive
        if _cloud_sql_proxy_process.poll() is None:
            logger.info("✅ Cloud SQL proxy process already running")
            return True
        else:
            # Process died, reset
            _cloud_sql_proxy_process = None
    
    # Need to start Cloud SQL proxy
    logger.info("🚀 Starting Cloud SQL proxy locally...")
    
    # Get connection string from environment or use default
    connection_name = os.environ.get(
        "CLOUD_SQL_CONNECTION_NAME",
        "leanworks-474204:us-west1:leanworks-prod"
    )
    
    # Get credentials file path
    credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_credential.json")
    if not os.path.exists(credentials_file):
        logger.error(f"❌ Credentials file not found: {credentials_file}")
        logger.error("   Cannot start Cloud SQL proxy without credentials")
        return False
    
    # Find cloud-sql-proxy binary
    proxy_binary = os.environ.get("CLOUD_SQL_PROXY_BINARY", "cloud-sql-proxy")
    
    # Check if binary exists
    try:
        result = subprocess.run(
            ["which", proxy_binary],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            logger.error(f"❌ Cloud SQL proxy binary not found: {proxy_binary}")
            logger.error("   Install it with: gcloud components install cloud-sql-proxy")
            logger.error("   Or download from: https://cloud.google.com/sql/docs/postgres/sql-proxy")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking for Cloud SQL proxy binary: {str(e)}")
        return False
    
    # Start Cloud SQL proxy
    try:
        # Use cloud-sql-proxy v2 syntax: INSTANCE_CONNECTION_NAME with --port flag
        # By default it binds to 127.0.0.1, so we just need to specify the port
        cmd = [
            proxy_binary,
            connection_name,
            f"--port={db_port}",
            f"--credentials-file={credentials_file}"
        ]
        
        logger.info(f"   Command: {' '.join(cmd)}")
        logger.info(f"   Connection: {connection_name}")
        logger.info(f"   Port: {db_port}")
        
        _cloud_sql_proxy_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True  # Detach from parent process
        )
        
        # Register cleanup function
        atexit.register(_cleanup_cloud_sql_proxy)
        
        # Wait for proxy to start (check port availability)
        max_wait = 30  # seconds
        check_interval = 0.5
        elapsed = 0
        
        logger.info(f"⏳ Waiting for Cloud SQL proxy to start (max {max_wait}s)...")
        while elapsed < max_wait:
            if _is_port_open("127.0.0.1", db_port):
                logger.info(f"✅ Cloud SQL proxy started successfully after {elapsed:.1f}s")
                return True
            
            # Check if process died
            if _cloud_sql_proxy_process.poll() is not None:
                stdout, stderr = _cloud_sql_proxy_process.communicate()
                logger.error(f"❌ Cloud SQL proxy process exited with code {_cloud_sql_proxy_process.returncode}")
                if stderr:
                    logger.error(f"   Error output: {stderr.decode('utf-8', errors='ignore')}")
                _cloud_sql_proxy_process = None
                return False
            
            time.sleep(check_interval)
            elapsed += check_interval
        
        logger.error(f"❌ Timeout waiting for Cloud SQL proxy to start after {elapsed:.1f}s")
        return False
        
    except Exception as e:
        logger.error(f"❌ Failed to start Cloud SQL proxy: {str(e)}")
        _cloud_sql_proxy_process = None
        return False


def _cleanup_cloud_sql_proxy():
    """Cleanup Cloud SQL proxy process on exit"""
    global _cloud_sql_proxy_process
    if _cloud_sql_proxy_process is not None:
        try:
            logger.info("🛑 Stopping Cloud SQL proxy...")
            _cloud_sql_proxy_process.terminate()
            try:
                _cloud_sql_proxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _cloud_sql_proxy_process.kill()
            _cloud_sql_proxy_process = None
            logger.info("✅ Cloud SQL proxy stopped")
        except Exception as e:
            logger.warning(f"Error stopping Cloud SQL proxy: {str(e)}")


def initialize_database():
    """Initialize database connection infrastructure"""
    global _secret_manager_client, _project_id, _credentials
    
    try:
        logger.info("Initializing database connection infrastructure...")
        
        # Check if credential file exists - use it if available, otherwise use ADC
        credential_file_exists = os.path.exists("gcp_credential.json")
        
        if credential_file_exists:
            # Use credential file if it exists (works in both local and Cloud Run)
            logger.info("Using service account file: gcp_credential.json")
            _credentials = service_account.Credentials.from_service_account_file("gcp_credential.json")
            
            # Load project_id from credentials
            with open("gcp_credential.json", "r") as f:
                credential_data = json.load(f)
            _project_id = credential_data["project_id"]
            
            # Initialize Secret Manager client
            _secret_manager_client = secretmanager.SecretManagerServiceClient(credentials=_credentials)
            
            # Ensure Cloud SQL proxy is running if in local environment
            is_cloud_run = os.environ.get("K_SERVICE") is not None
            if not is_cloud_run:
                _ensure_cloud_sql_proxy_running()
        else:
            # Fallback to Application Default Credentials (ADC) if file doesn't exist
            logger.info("gcp_credential.json not found, using Application Default Credentials")
            from google.auth import default
            _credentials, _project_id = default()
            # Initialize Secret Manager client with ADC
            _secret_manager_client = secretmanager.SecretManagerServiceClient(credentials=_credentials)
        
        logger.info(f"Database infrastructure initialized (project: {_project_id})")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database infrastructure: {str(e)}")
        raise


def get_postgres_password() -> str:
    """Get PostgreSQL password from Secret Manager (cached)"""
    global _cached_password, _secret_manager_client, _project_id
    
    if _cached_password:
        return _cached_password
    
    if not _secret_manager_client or not _project_id:
        # Fallback to environment variable
        return os.environ.get("DB_PASSWORD", "")
    
    try:
        secret_name = f"projects/{_project_id}/secrets/postgresdb-password/versions/latest"
        response = _secret_manager_client.access_secret_version(name=secret_name)
        _cached_password = response.payload.data.decode("UTF-8").strip()
        logger.info("✅ PostgreSQL password fetched from Secret Manager")
        return _cached_password
    except Exception as e:
        logger.error(f"❌ Failed to fetch password from Secret Manager: {str(e)}")
        return os.environ.get("DB_PASSWORD", "")


def get_domain_from_email(email: str) -> str:
    """Extract domain from email address.
    
    Returns the domain part of the email (e.g., 'leanworks.ai' from 'user@leanworks.ai')
    """
    if "@" not in email:
        raise ValueError(f"Invalid email format: {email}")
    domain = email.split("@")[1]
    if not domain:
        raise ValueError(f"Invalid email format: {email}")
    return domain.lower()




def _determine_db_host() -> Tuple[str, int]:
    """
    Determine database host and port based on environment.
    Returns (host, port) tuple.
    
    Logic:
    - If DB_HOST is explicitly set, use it (this is the preferred method)
    - If running on Kubernetes, default to shared Cloud SQL Proxy service
    - If running on GCP (socket directory exists), use Unix socket
    - Otherwise (local development), use TCP/IP connection to 127.0.0.1
    """
    db_host_env = os.environ.get("DB_HOST")
    if db_host_env:
        # User explicitly set DB_HOST, use it (e.g., cloud-sql-proxy-service in Kubernetes)
        db_port = int(os.environ.get("DB_PORT", "5432"))
        logger.info(f"🔧 Using DB_HOST from environment: {db_host_env}:{db_port}")
        return db_host_env, db_port
    
    # Check if we're in Kubernetes environment
    # Kubernetes sets KUBERNETES_SERVICE_HOST environment variable
    is_kubernetes = os.environ.get("KUBERNETES_SERVICE_HOST") is not None
    
    if is_kubernetes:
        # In Kubernetes, default to shared Cloud SQL Proxy service (TCP/IP)
        db_host = "cloud-sql-proxy-service"  # Kubernetes service DNS name
        db_port = int(os.environ.get("DB_PORT", "5432"))
        logger.info(f"🔧 Detected Kubernetes environment, using shared Cloud SQL Proxy service: {db_host}:{db_port}")
        return db_host, db_port
    
    # Check if Unix socket directory exists (for GCP App Engine or other GCP services)
    socket_path = "/cloudsql/leanworks-474204:us-west1:leanworks-prod/.s.PGSQL.5432"
    if os.path.exists(os.path.dirname(socket_path)):
        db_host = "/cloudsql/leanworks-474204:us-west1:leanworks-prod"
        logger.info("🔧 Detected GCP environment, using Unix socket connection")
        return db_host, 5432  # Port not used for Unix sockets
    
    # Local development - use Cloud SQL Proxy default port (TCP/IP)
    db_host = "127.0.0.1"
    db_port = int(os.environ.get("DB_PORT", "5432"))
    logger.info("🔧 Detected local environment, using TCP/IP connection (Cloud SQL Proxy)")
    return db_host, db_port


def _wait_for_unix_socket(socket_dir: str, max_wait_time: int = 60, check_interval: int = 1) -> None:
    """
    Wait for Unix socket to be available (for Kubernetes Cloud SQL Proxy sidecar).
    
    Args:
        socket_dir: Directory where the socket should be created (e.g., '/cloudsql/...')
        max_wait_time: Maximum time to wait in seconds (default: 60)
        check_interval: Time between checks in seconds (default: 1)
    """
    socket_file = f"{socket_dir}/.s.PGSQL.5432"
    socket_dir_path = socket_dir
    
    # Check if socket directory exists
    if not os.path.exists(socket_dir_path):
        logger.info(f"⏳ Waiting for Unix socket directory to be created: {socket_dir_path}")
        logger.info(f"   Max wait time: {max_wait_time}s, checking every {check_interval}s")
        
        start_time = time.time()
        last_log_time = start_time
        check_count = 0
        
        while not os.path.exists(socket_dir_path):
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                logger.error(f"❌ Timeout after {elapsed:.1f}s waiting for socket directory: {socket_dir_path}")
                logger.error(f"   Directory does not exist. Check Cloud SQL Proxy sidecar status.")
                raise TimeoutError(f"Timeout waiting for socket directory: {socket_dir_path}")
            
            # Log progress every 5 seconds
            if time.time() - last_log_time >= 5:
                check_count += 1
                logger.info(f"   Still waiting... ({elapsed:.1f}s elapsed, {check_count} checks)")
                last_log_time = time.time()
            
            time.sleep(check_interval)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Socket directory created after {elapsed:.1f}s: {socket_dir_path}")
    
    # Check if socket file exists
    if not os.path.exists(socket_file):
        logger.info(f"⏳ Waiting for Unix socket file to be available: {socket_file}")
        logger.info(f"   Max wait time: {max_wait_time}s, checking every {check_interval}s")
        start_time = time.time()
        last_log_time = start_time
        check_count = 0
        
        while not os.path.exists(socket_file):
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                logger.error(f"❌ Timeout after {elapsed:.1f}s waiting for socket file: {socket_file}")
                logger.error(f"   Socket file does not exist. Check Cloud SQL Proxy sidecar status.")
                raise TimeoutError(f"Timeout waiting for Unix socket: {socket_file}")
            
            # Log progress every 5 seconds
            if time.time() - last_log_time >= 5:
                check_count += 1
                logger.info(f"   Still waiting for socket file... ({elapsed:.1f}s elapsed, {check_count} checks)")
                last_log_time = time.time()
            
            time.sleep(check_interval)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Unix socket is now available after {elapsed:.1f}s: {socket_file}")


def get_org_db_name(org_slug: str) -> str:
    """Get the database name for an organization from its slug.
    
    Args:
        org_slug: Organization slug (e.g., 'leanworks')
        
    Returns:
        Database name (e.g., 'org_leanworks')
    """
    return f'org_{org_slug}'


# ============================================================================
# ORG DATABASE POOL
# ============================================================================

def get_org_pool(org_slug: str) -> pool.ThreadedConnectionPool:
    """Get or create connection pool for an organization's database.
    
    Args:
        org_slug: Organization slug (e.g., 'leanworks')
        
    Returns:
        Connection pool for the org's database (org_<slug>)
    """
    global _org_pools
    
    # Get database name from slug
    db_name = get_org_db_name(org_slug)
    
    # Return cached pool if exists
    if db_name in _org_pools:
        return _org_pools[db_name]
    
    logger.info(f"🔌 Creating connection pool for org database: {db_name} (slug: {org_slug})")
    
    password = get_postgres_password()
    db_host, db_port = _determine_db_host()
    db_user = os.environ.get("DB_USER", "postgres")
    
    # Ensure database exists
    ensure_database_exists(db_name, password)
    
    try:
        pool_params = {
            'minconn': 1,
            'maxconn': 10,
            'database': db_name,
            'user': db_user,
            'password': password,
        }
        if db_host.startswith('/'):
            pool_params['host'] = db_host
            logger.info(f"🔌 Using Unix socket connection: {db_host}, database: {db_name}")
        else:
            pool_params['host'] = db_host
            pool_params['port'] = db_port
            logger.info(f"🔌 Using TCP/IP connection: {db_host}:{db_port}, database: {db_name}")
        
        connection_pool = pool.ThreadedConnectionPool(**pool_params)
        
        # Cache the pool
        _org_pools[db_name] = connection_pool
        logger.info(f"✅ Connection pool created for org slug: {org_slug} (db: {db_name})")
        return connection_pool
    except Exception as e:
        logger.error(f"❌ Failed to create connection pool for org slug {org_slug}: {str(e)}")
        raise


def query_org(org_slug: str, query: str, params: Optional[tuple] = None) -> list:
    """Execute a query on an organization's database and return results.
    
    Args:
        org_slug: Organization slug (e.g., 'leanworks')
        query: SQL query to execute
        params: Query parameters
        
    Returns:
        List of result rows as dictionaries
    """
    conn = None
    org_pool = None
    try:
        org_pool = get_org_pool(org_slug)
        conn = org_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Org database query error for slug {org_slug}: {str(e)}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        raise
    finally:
        if conn and org_pool:
            org_pool.putconn(conn)


def query_org_one(org_slug: str, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
    """Execute a query on an organization's database and return first row or None."""
    results = query_org(org_slug, query, params)
    return results[0] if results else None


def execute_org(org_slug: str, query: str, params: Optional[tuple] = None) -> None:
    """Execute a non-SELECT query (UPDATE, INSERT, DELETE) on an organization's database.
    
    This function is for queries that don't return results. It commits the transaction.
    
    Args:
        org_slug: Organization slug (e.g., 'leanworks')
        query: SQL query to execute (UPDATE, INSERT, DELETE)
        params: Query parameters
    """
    conn = None
    org_pool = None
    try:
        org_pool = get_org_pool(org_slug)
        conn = org_pool.getconn()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()  # Commit the transaction for UPDATE/INSERT/DELETE
        cursor.close()
    except Exception as e:
        if conn:
            conn.rollback()  # Rollback on error
        logger.error(f"Org database execute error for slug {org_slug}: {str(e)}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        raise
    finally:
        if conn and org_pool:
            org_pool.putconn(conn)


# ============================================================================
# LEGACY TENANT POOL (for backward compatibility with user_email based access)
# ============================================================================

def ensure_database_exists(db_name: str, password: str) -> None:
    """Ensure database exists, create if it doesn't"""
    db_host, db_port = _determine_db_host()
    db_user = os.environ.get("DB_USER", "postgres")
    
    # Wait for Unix socket if needed (only for Unix socket connections)
    if db_host.startswith('/cloudsql/'):
        _wait_for_unix_socket(db_host)
    
    # Try to connect to the target database
    try:
        # For Unix sockets, pass host as socket path, no port
        # For TCP connections, pass host and port
        connect_params = {
            'database': db_name,
            'user': db_user,
            'password': password,
        }
        if db_host.startswith('/'):
            # Unix socket connection
            connect_params['host'] = db_host
        else:
            # TCP connection
            connect_params['host'] = db_host
            connect_params['port'] = db_port
        
        conn = connect(**connect_params)
        conn.close()
        logger.info(f"✅ Database '{db_name}' exists")
        return
    except Exception as e:
        # If database doesn't exist (error code 3D000), create it
        if hasattr(e, 'pgcode') and e.pgcode == '3D000':
            logger.info(f"📝 Creating database '{db_name}' for new tenant...")
            
            # Connect to default postgres database to create new database
            admin_connect_params = {
                'database': 'postgres',
                'user': db_user,
                'password': password,
            }
            if db_host.startswith('/'):
                admin_connect_params['host'] = db_host
            else:
                admin_connect_params['host'] = db_host
                admin_connect_params['port'] = db_port
            
            admin_conn = connect(**admin_connect_params)
            admin_conn.autocommit = True
            admin_cursor = admin_conn.cursor()
            admin_cursor.execute(f'CREATE DATABASE "{db_name}"')
            admin_cursor.close()
            admin_conn.close()
            logger.info(f"✅ Database '{db_name}' created successfully")
        else:
            # Other connection errors
            raise


def get_tenant_pool(user_email: str) -> pool.ThreadedConnectionPool:
    """Get or create connection pool for a specific tenant (domain)"""
    global _tenant_pools
    
    domain = get_domain_from_email(user_email)
    db_name = domain
    
    # Return cached pool if exists
    if db_name in _tenant_pools:
        return _tenant_pools[db_name]
    
    logger.info(f"🔌 Creating connection pool for tenant: {db_name} (from {user_email})")
    
    password = get_postgres_password()
    db_host, db_port = _determine_db_host()
    db_user = os.environ.get("DB_USER", "postgres")
    
    # Ensure database exists
    ensure_database_exists(db_name, password)
    
    # Create connection pool for this tenant
    try:
        # For Unix sockets, pass host as socket path, no port
        # For TCP connections, pass host and port
        pool_params = {
            'minconn': 1,
            'maxconn': 10,
            'database': db_name,
            'user': db_user,
            'password': password,
        }
        if db_host.startswith('/'):
            # Unix socket connection
            pool_params['host'] = db_host
            logger.info(f"🔌 Using Unix socket connection: {db_host}, database: {db_name}")
        else:
            # TCP connection
            pool_params['host'] = db_host
            pool_params['port'] = db_port
            logger.info(f"🔌 Using TCP/IP connection: {db_host}:{db_port}, database: {db_name}")
        
        connection_pool = pool.ThreadedConnectionPool(**pool_params)
        
        # Cache the pool
        _tenant_pools[db_name] = connection_pool
        logger.info(f"✅ Connection pool created for tenant: {db_name}")
        return connection_pool
    except Exception as e:
        logger.error(f"❌ Failed to create connection pool for tenant {db_name}: {str(e)}")
        logger.error(f"   Connection params: host={db_host}, port={db_port if not db_host.startswith('/') else 'N/A (Unix socket)'}, database={db_name}, user={db_user}")
        raise


def get_tenant_connection(user_email: str):
    """Get a connection from the tenant's pool"""
    pool = get_tenant_pool(user_email)
    return pool.getconn()


def return_tenant_connection(user_email: str, conn):
    """Return a connection to the tenant's pool"""
    pool = get_tenant_pool(user_email)
    pool.putconn(conn)


def query_tenant(user_email: str, query: str, params: Optional[tuple] = None) -> list:
    """Execute a query for a specific tenant and return results"""
    conn = None
    connection_pool = None
    try:
        connection_pool = get_tenant_pool(user_email)
        conn = connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Database query error for tenant {get_domain_from_email(user_email)}: {str(e)}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        raise
    finally:
        if conn and connection_pool:
            connection_pool.putconn(conn)


def query_tenant_one(user_email: str, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
    """Execute a query for a specific tenant and return first row or None"""
    results = query_tenant(user_email, query, params)
    return results[0] if results else None


# ============================================================================
# SHARED DATABASE POOL (for users table and other shared data)
# ============================================================================

def get_shared_pool() -> pool.ThreadedConnectionPool:
    """Get or create connection pool for the shared database (contains users table)"""
    global _shared_pool
    
    if _shared_pool:
        return _shared_pool
    
    logger.info("🔌 Creating connection pool for shared database")
    
    password = get_postgres_password()
    db_host, db_port = _determine_db_host()
    db_user = os.environ.get("DB_USER", "postgres")
    db_name = "shared"
    
    # Ensure database exists
    ensure_database_exists(db_name, password)
    
    try:
        pool_params = {
            'minconn': 1,
            'maxconn': 10,
            'database': db_name,
            'user': db_user,
            'password': password,
        }
        if db_host.startswith('/'):
            pool_params['host'] = db_host
            logger.info(f"🔌 Using Unix socket connection: {db_host}, database: {db_name}")
        else:
            pool_params['host'] = db_host
            pool_params['port'] = db_port
            logger.info(f"🔌 Using TCP/IP connection: {db_host}:{db_port}, database: {db_name}")
        
        _shared_pool = pool.ThreadedConnectionPool(**pool_params)
        logger.info(f"✅ Connection pool created for shared database")
        return _shared_pool
    except Exception as e:
        logger.error(f"❌ Failed to create connection pool for shared database: {str(e)}")
        raise


def query_shared_one(query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
    """Execute a query on the shared database and return first row or None.
    
    Args:
        query: SQL query to execute
        params: Query parameters
        
    Returns:
        First result row as dictionary or None if no results
    """
    conn = None
    shared_pool = None
    try:
        shared_pool = get_shared_pool()
        conn = shared_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        return dict(results[0]) if results else None
    except Exception as e:
        logger.error(f"Shared database query error: {str(e)}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        raise
    finally:
        if conn and shared_pool:
            shared_pool.putconn(conn)


# Initialize on import
try:
    initialize_database()
except Exception as e:
    logger.warning(f"Database initialization failed: {str(e)}")

