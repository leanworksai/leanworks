from leanworks.agent.tools.postgres import PostgresTool
from leanworks.agent.tools.doc_management import DocManagementTool
from leanworks.agent.tools.search import SearchTool
from leanworks.agent.tools.outlook import OutlookTool
from leanworks.agent.tools.duckdb import DuckDBTool
from leanworks.agent.tools.firestore import FirestoreTool
from leanworks.agent.tools.cloud_storage import CloudStorageTool
from leanworks.agent.tools.atlassian import AtlassianTool
from leanworks.agent.tools.github import GitHubTool
from leanworks.agent.tools.notion import NotionTool
from leanworks.agent.tools.clickup import ClickUpTool
from leanworks.agent.tools.linear import LinearTool
from leanworks.agent.helpers import AgentHelpers
from google.cloud import storage
import logging
import json
import subprocess
import tempfile
import os
import shutil
import resource
import signal
import queue
import threading
import time
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class ToolUse:
    def __init__(self, org_slug=None, firestore_client=None, secret_manager_client=None, read_document_ids=None, tools=None, root_dir=None, user_id=None, session_id=None, credential_path: str = "gcp_credential.json"):
        """
        Initialize ToolUse with various client connections using lazy loading.
        
        Args:
            org_slug: Organization name (e.g., 'leanworks.ai') extracted from user_id. Used to determine database and client_name.
            firestore_client: Firestore client
            secret_manager_client: Secret Manager client
            read_document_ids: Set of document IDs already read for deduplication
            tools: List of tools to enable. Internal tools ['search', 'postgres', 'duckdb'] are always available.
                   External tools (e.g., 'outlook') should be explicitly provided in this list.
            credential_path: Path to GCP credential JSON file (default: "gcp_credential.json")
        """
        # Store initialization parameters for lazy loading
        self.org_slug = org_slug
        # Create postgres_client_wrapper internally if org_slug is provided
        if org_slug:
            class PostgresClientWrapper:
                def __init__(self, org_slug):
                    self.org_slug = org_slug
            self.postgres_client_wrapper = PostgresClientWrapper(org_slug)
        else:
            self.postgres_client_wrapper = None
        self.firestore_client = firestore_client
        self.secret_manager_client = secret_manager_client
        self.credential_path = credential_path
        self.project_id = AgentHelpers.get_project_id_from_credentials(credential_path)
        self.read_document_ids = read_document_ids if read_document_ids is not None else set()
        self.user_id = user_id
        self.session_id = session_id
        
        # Internal tools that are always available
        internal_tools = ['search', 'postgres', 'duckdb', 'firestore']
        
        # Set default tools if not provided
        if tools is None:
            requested_tools = internal_tools
        else:
            # Add provided tools to default tools (with deduplication)
            default_tools = internal_tools
            requested_tools = list(set(default_tools + tools))  # Remove duplicates while preserving functionality
        
        self.requested_tools = requested_tools
        logger.info(f"Final enabled tools: {self.requested_tools}")
        
        # Tool instance cache - tools are initialized only when first accessed
        self._tool_cache = {}
        
        # RAG storage tool cache (lazy initialization)
        self._rag_storage = None
        
        # Track which tools are actually enabled (successfully initialized)
        self.enabled_tools = []
        
        # Initialize cached properties
        self._tools_cache = None
        self._function_map_cache = None
        
        # Register DuckDB tools immediately (stateless wrappers; no instance required)
        if 'duckdb' in self.requested_tools:
            self.enabled_tools.append('duckdb')
            logger.info("DuckDB tools registered (stateless)")
        
        # Log initialization completion
        logger.info(f"ToolUse initialized with lazy loading for tools: {self.requested_tools}")
    

    # Lazy loading properties for individual tools
    @property
    def postgres_tool(self):
        """Lazy-load PostgreSQL tool on first access."""
        if 'postgres_tool' not in self._tool_cache:
            if 'postgres' in self.requested_tools and self.postgres_client_wrapper:
                try:
                    # Set Secret Manager client for PostgresTool
                    if self.secret_manager_client:
                        PostgresTool.set_secret_manager(self.secret_manager_client, self.credential_path)
                    self._tool_cache['postgres_tool'] = PostgresTool(self.postgres_client_wrapper, user_id=self.user_id)
                    if 'postgres' not in self.enabled_tools:
                        self.enabled_tools.append('postgres')
                    logger.info("PostgresTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize PostgresTool: {str(e)}")
                    self._tool_cache['postgres_tool'] = None
            elif 'postgres' in self.requested_tools:
                logger.warning("PostgresTool not initialized: missing postgres_client_wrapper")
                self._tool_cache['postgres_tool'] = None
            else:
                self._tool_cache['postgres_tool'] = None
        return self._tool_cache['postgres_tool']
    
    @property
    def doc_management_tool(self):
        """Lazy-load DocManagementTool on first access."""
        if 'doc_management_tool' not in self._tool_cache:
            if 'postgres' in self.requested_tools and self.postgres_client_wrapper:
                try:
                    # Set Secret Manager client for PostgresTool (needed for connection pool)
                    if self.secret_manager_client:
                        PostgresTool.set_secret_manager(self.secret_manager_client, self.credential_path)
                    self._tool_cache['doc_management_tool'] = DocManagementTool(
                        self.postgres_client_wrapper,
                        user_id=self.user_id
                    )
                    if 'doc_management' not in self.enabled_tools:
                        self.enabled_tools.append('doc_management')
                    logger.info("DocManagementTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize DocManagementTool: {str(e)}")
                    self._tool_cache['doc_management_tool'] = None
            elif 'postgres' in self.requested_tools:
                logger.warning("DocManagementTool not initialized: missing postgres_client_wrapper")
                self._tool_cache['doc_management_tool'] = None
            else:
                self._tool_cache['doc_management_tool'] = None
        return self._tool_cache['doc_management_tool']
    
    @property
    def search_tool(self):
        """Lazy-load Search tool on first access."""
        if 'search_tool' not in self._tool_cache:
            if 'search' in self.requested_tools and self.firestore_client and self.secret_manager_client and self.project_id:
                try:
                    if not self.org_slug:
                        raise ValueError("org_slug is required for SearchTool initialization")
                    
                    self._tool_cache['search_tool'] = SearchTool(
                        self.firestore_client,
                        self.org_slug,
                        self.secret_manager_client,
                        read_document_ids=self.read_document_ids,
                        credential_path=self.credential_path
                    )
                    if 'search' not in self.enabled_tools:
                        self.enabled_tools.append('search')
                    logger.info("SearchTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize SearchTool: {str(e)}")
                    self._tool_cache['search_tool'] = None
            elif 'search' in self.requested_tools:
                logger.warning("SearchTool not initialized: missing firestore_client, secret_manager_client, or project_id")
                self._tool_cache['search_tool'] = None
            else:
                self._tool_cache['search_tool'] = None
        return self._tool_cache['search_tool']
    
    
    @property
    def outlook_tool(self):
        """Lazy-load Outlook tool on first access."""
        if 'outlook_tool' not in self._tool_cache:
            if 'outlook' in self.requested_tools and self.secret_manager_client and self.project_id and self.org_slug:
                try:
                    # Helper function to get secret
                    def get_secret(name):
                        full_name = f"projects/{self.project_id}/secrets/{name}/versions/latest"
                        response = self.secret_manager_client.access_secret_version(name=full_name)
                        return response.payload.data.decode("UTF-8")
                    
                    # Construct secret name from org_slug
                    # Convert underscores to hyphens for secret name
                    org_slug_for_secret = self.org_slug.replace('_', '-')
                    secret_name = f"integrations-{org_slug_for_secret}-outlook"
                    
                    # Retrieve and parse JSON secret
                    secret_json = get_secret(secret_name)
                    outlook_credentials = json.loads(secret_json)
                    
                    self._tool_cache['outlook_tool'] = OutlookTool(
                        client_id=outlook_credentials.get('clientId'),
                        client_secret=outlook_credentials.get('clientSecret'),
                        tenant_id=outlook_credentials.get('tenantId')
                    )
                    if 'outlook' not in self.enabled_tools:
                        self.enabled_tools.append('outlook')
                    logger.info("OutlookTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize OutlookTool: {str(e)}")
                    self._tool_cache['outlook_tool'] = None
            elif 'outlook' in self.requested_tools:
                logger.warning("OutlookTool not initialized: missing secret_client, project_id, or org_slug")
                self._tool_cache['outlook_tool'] = None
            else:
                self._tool_cache['outlook_tool'] = None
        return self._tool_cache['outlook_tool']
    
    @property
    def firestore_tool(self):
        """Lazy-load Firestore tool on first access."""
        if 'firestore_tool' not in self._tool_cache:
            if 'firestore' in self.requested_tools and self.firestore_client and self.org_slug:
                try:
                    self._tool_cache['firestore_tool'] = FirestoreTool(
                        self.firestore_client,
                        self.org_slug,
                        user_id=self.user_id
                    )
                    if 'firestore' not in self.enabled_tools:
                        self.enabled_tools.append('firestore')
                    logger.info("FirestoreTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize FirestoreTool: {str(e)}")
                    self._tool_cache['firestore_tool'] = None
            elif 'firestore' in self.requested_tools:
                logger.warning("FirestoreTool not initialized: missing firestore_client or org_slug")
                self._tool_cache['firestore_tool'] = None
            else:
                self._tool_cache['firestore_tool'] = None
        return self._tool_cache['firestore_tool']
    
    @property
    def cloud_storage_tool(self):
        """Lazy-load Cloud Storage tool on first access."""
        if 'cloud_storage_tool' not in self._tool_cache:
            if 'cloud_storage' in self.requested_tools and self.org_slug:
                try:
                    # Initialize Storage client
                    storage_client = storage.Client.from_service_account_json(self.credential_path)
                    
                    self._tool_cache['cloud_storage_tool'] = CloudStorageTool(
                        storage_client,
                        self.org_slug,
                        credential_path=self.credential_path
                    )
                    if 'cloud_storage' not in self.enabled_tools:
                        self.enabled_tools.append('cloud_storage')
                    logger.info("CloudStorageTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize CloudStorageTool: {str(e)}")
                    self._tool_cache['cloud_storage_tool'] = None
            elif 'cloud_storage' in self.requested_tools:
                logger.warning("CloudStorageTool not initialized: missing org_slug")
                self._tool_cache['cloud_storage_tool'] = None
            else:
                self._tool_cache['cloud_storage_tool'] = None
        return self._tool_cache['cloud_storage_tool']
    
    @property
    def atlassian_tool(self):
        """Lazy-load Atlassian tool on first access."""
        if 'atlassian_tool' not in self._tool_cache:
            if ('jira' in self.requested_tools or 'atlassian' in self.requested_tools) and self.secret_manager_client and self.project_id and self.org_slug:
                try:
                    # Helper function to get secret
                    def get_secret(name):
                        full_name = f"projects/{self.project_id}/secrets/{name}/versions/latest"
                        response = self.secret_manager_client.access_secret_version(name=full_name)
                        return response.payload.data.decode("UTF-8")
                    
                    # Construct secret name from org_slug
                    # Convert underscores to hyphens for secret name
                    org_slug_for_secret = self.org_slug.replace('_', '-')
                    secret_name = f"integrations-{org_slug_for_secret}-atlassian"
                    
                    # Retrieve and parse JSON secret
                    secret_json = get_secret(secret_name)
                    atlassian_credentials = json.loads(secret_json)
                    
                    self._tool_cache['atlassian_tool'] = AtlassianTool(
                        email=atlassian_credentials.get('email'),
                        domain=atlassian_credentials.get('domain'),
                        api_token=atlassian_credentials.get('apiToken')
                    )
                    if 'jira' not in self.enabled_tools:
                        self.enabled_tools.append('jira')
                    logger.info("AtlassianTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize AtlassianTool: {str(e)}")
                    self._tool_cache['atlassian_tool'] = None
            elif 'jira' in self.requested_tools or 'atlassian' in self.requested_tools:
                logger.warning("AtlassianTool not initialized: missing secret_client, project_id, or org_slug")
                self._tool_cache['atlassian_tool'] = None
            else:
                self._tool_cache['atlassian_tool'] = None
        return self._tool_cache['atlassian_tool']
    
    @property
    def github_tool(self):
        """Lazy-load GitHub tool on first access."""
        if 'github_tool' not in self._tool_cache:
            if 'github' in self.requested_tools and self.secret_manager_client and self.project_id and self.org_slug:
                try:
                    # Helper function to get secret
                    def get_secret(name):
                        full_name = f"projects/{self.project_id}/secrets/{name}/versions/latest"
                        response = self.secret_manager_client.access_secret_version(name=full_name)
                        return response.payload.data.decode("UTF-8")
                    
                    # Construct secret name from org_slug
                    # Convert underscores to hyphens for secret name
                    org_slug_for_secret = self.org_slug.replace('_', '-')
                    secret_name = f"integrations-{org_slug_for_secret}-github"
                    
                    # Retrieve and parse JSON secret for installation data
                    secret_json = get_secret(secret_name)
                    github_installation = json.loads(secret_json)
                    installation_id = github_installation.get('installationId')
                    
                    # Retrieve GitHub App credentials (global secrets, not per-org)
                    app_id = get_secret('github-app-id')
                    private_key = get_secret('github-app-private-key')
                    
                    self._tool_cache['github_tool'] = GitHubTool(
                        installation_id=int(installation_id) if installation_id else None,
                        app_id=app_id,
                        private_key=private_key
                    )
                    if 'github' not in self.enabled_tools:
                        self.enabled_tools.append('github')
                    logger.info("GitHubTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize GitHubTool: {str(e)}")
                    self._tool_cache['github_tool'] = None
            elif 'github' in self.requested_tools:
                logger.warning("GitHubTool not initialized: missing secret_client, project_id, or org_slug")
                self._tool_cache['github_tool'] = None
            else:
                self._tool_cache['github_tool'] = None
        return self._tool_cache['github_tool']
    
    @property
    def notion_tool(self):
        """Lazy-load Notion tool on first access."""
        if 'notion_tool' not in self._tool_cache:
            if 'notion' in self.requested_tools and self.secret_manager_client and self.project_id and self.org_slug:
                try:
                    # Helper function to get secret
                    def get_secret(name):
                        full_name = f"projects/{self.project_id}/secrets/{name}/versions/latest"
                        response = self.secret_manager_client.access_secret_version(name=full_name)
                        return response.payload.data.decode("UTF-8")
                    
                    # Construct secret name from org_slug
                    # Convert underscores to hyphens for secret name
                    org_slug_for_secret = self.org_slug.replace('_', '-')
                    secret_name = f"integrations-{org_slug_for_secret}-notion"
                    
                    # Retrieve and parse JSON secret
                    secret_json = get_secret(secret_name)
                    notion_credentials = json.loads(secret_json)
                    
                    self._tool_cache['notion_tool'] = NotionTool(
                        integration_token=notion_credentials.get('integrationToken')
                    )
                    if 'notion' not in self.enabled_tools:
                        self.enabled_tools.append('notion')
                    logger.info("NotionTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize NotionTool: {str(e)}")
                    self._tool_cache['notion_tool'] = None
            elif 'notion' in self.requested_tools:
                logger.warning("NotionTool not initialized: missing secret_client, project_id, or org_slug")
                self._tool_cache['notion_tool'] = None
            else:
                self._tool_cache['notion_tool'] = None
        return self._tool_cache['notion_tool']
    
    @property
    def clickup_tool(self):
        """Lazy-load ClickUp tool on first access."""
        if 'clickup_tool' not in self._tool_cache:
            if 'clickup' in self.requested_tools and self.secret_manager_client and self.project_id and self.org_slug:
                try:
                    # Helper function to get secret
                    def get_secret(name):
                        full_name = f"projects/{self.project_id}/secrets/{name}/versions/latest"
                        response = self.secret_manager_client.access_secret_version(name=full_name)
                        return response.payload.data.decode("UTF-8")
                    
                    # Construct secret name from org_slug
                    # Convert underscores to hyphens for secret name
                    org_slug_for_secret = self.org_slug.replace('_', '-')
                    secret_name = f"integrations-{org_slug_for_secret}-clickup"
                    
                    # Retrieve and parse JSON secret
                    secret_json = get_secret(secret_name)
                    clickup_credentials = json.loads(secret_json)
                    
                    self._tool_cache['clickup_tool'] = ClickUpTool(
                        api_token=clickup_credentials.get('apiToken')
                    )
                    if 'clickup' not in self.enabled_tools:
                        self.enabled_tools.append('clickup')
                    logger.info("ClickUpTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize ClickUpTool: {str(e)}")
                    self._tool_cache['clickup_tool'] = None
            elif 'clickup' in self.requested_tools:
                logger.warning("ClickUpTool not initialized: missing secret_client, project_id, or org_slug")
                self._tool_cache['clickup_tool'] = None
            else:
                self._tool_cache['clickup_tool'] = None
        return self._tool_cache['clickup_tool']
    
    @property
    def linear_tool(self):
        """Lazy-load Linear tool on first access."""
        if 'linear_tool' not in self._tool_cache:
            if 'linear' in self.requested_tools and self.secret_manager_client and self.project_id and self.org_slug:
                try:
                    # Helper function to get secret
                    def get_secret(name):
                        full_name = f"projects/{self.project_id}/secrets/{name}/versions/latest"
                        response = self.secret_manager_client.access_secret_version(name=full_name)
                        return response.payload.data.decode("UTF-8")
                    
                    # Construct secret name from org_slug
                    # Convert underscores to hyphens for secret name
                    org_slug_for_secret = self.org_slug.replace('_', '-')
                    secret_name = f"integrations-{org_slug_for_secret}-linear"
                    
                    # Retrieve and parse JSON secret
                    secret_json = get_secret(secret_name)
                    linear_credentials = json.loads(secret_json)
                    
                    self._tool_cache['linear_tool'] = LinearTool(
                        api_key=linear_credentials.get('apiKey')
                    )
                    if 'linear' not in self.enabled_tools:
                        self.enabled_tools.append('linear')
                    logger.info("LinearTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize LinearTool: {str(e)}")
                    self._tool_cache['linear_tool'] = None
            elif 'linear' in self.requested_tools:
                logger.warning("LinearTool not initialized: missing secret_client, project_id, or org_slug")
                self._tool_cache['linear_tool'] = None
            else:
                self._tool_cache['linear_tool'] = None
        return self._tool_cache['linear_tool']
    
    def _create_bash_session(self):
        """Create a persistent bash session using Docker."""
        import uuid
        
        class DockerBashSession:
            def __init__(self):
                # Generate unique container name
                self.container_name = f"bash-session-{uuid.uuid4().hex[:12]}"
                self.container_id = None
                
                # Get system temp directory to mount into container
                # This allows files created on the host to be accessible from inside the container
                host_temp_dir = tempfile.gettempdir()
                # Mount temp directory at /host-tmp in container
                container_mount_path = '/host-tmp'
                
                # Create and start Docker container
                try:
                    # Use a lightweight base image (alpine with bash)
                    create_cmd = [
                        'docker', 'run', '-d',
                        '--name', self.container_name,
                        '--rm',  # Auto-remove when stopped
                        '--network', 'none',  # No network access for security
                        '--memory', '256m',  # Limit memory to 256MB
                        '--cpus', '1.0',  # Limit to 1 CPU
                        '--pids-limit', '100',  # Limit number of processes
                        '--read-only',  # Read-only root filesystem
                        '--tmpfs', '/tmp:rw,noexec,nosuid,size=100m',  # Writable /tmp
                        '--tmpfs', '/home:rw,noexec,nosuid,size=100m',  # Writable /home
                        '-v', f'{host_temp_dir}:{container_mount_path}:ro',  # Mount temp dir as read-only
                        'alpine:latest',
                        'sh', '-c', 'tail -f /dev/null'  # Keep container running
                    ]
                    
                    result = subprocess.run(
                        create_cmd,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode != 0:
                        raise Exception(f"Failed to create Docker container: {result.stderr}")
                    
                    self.container_id = result.stdout.strip()
                    self.host_temp_dir = host_temp_dir
                    self.container_mount_path = container_mount_path
                    logger.info(f"Created Docker container {self.container_name} ({self.container_id[:12]}) with temp dir mounted at {container_mount_path}")
                    
                except FileNotFoundError:
                    raise Exception("Docker is not installed or not in PATH")
                except subprocess.TimeoutExpired:
                    raise Exception("Docker container creation timed out")
                except Exception as e:
                    logger.error(f"Error creating Docker container: {e}")
                    raise
        
        return DockerBashSession()
    
    def _translate_path_for_container(self, command: str, session) -> str:
        """
        Translate file paths in command from host temp directory to container mount path.
        
        Args:
            command: The bash command with potential file paths
            session: DockerBashSession instance with mount info
            
        Returns:
            Command with translated paths
        """
        if not hasattr(session, 'host_temp_dir') or not hasattr(session, 'container_mount_path'):
            return command
        
        host_temp_dir = session.host_temp_dir
        container_mount_path = session.container_mount_path
        
        # Normalize paths for comparison
        host_temp_dir_norm = os.path.normpath(host_temp_dir)
        
        # Simple approach: replace temp directory path with container mount path
        if host_temp_dir_norm in command:
            # Escape special regex characters in the temp directory path
            escaped_temp_dir = re.escape(host_temp_dir_norm)
            
            def replace_temp_path(match):
                matched_path = match.group(0)
                # Get the relative path from temp directory
                if matched_path.startswith(host_temp_dir_norm):
                    rel_path = matched_path[len(host_temp_dir_norm):].lstrip(os.sep)
                    if rel_path:
                        # Construct container path
                        container_path = os.path.join(container_mount_path, rel_path).replace('\\', '/')
                    else:
                        container_path = container_mount_path
                    return container_path
                return matched_path
            
            # Replace temp directory paths (match the temp dir followed by any path characters)
            # This pattern matches the temp dir path and everything after it until a space, quote, or special char
            translated_command = re.sub(
                escaped_temp_dir + r'[^\s"\'<>|&;()]*',
                replace_temp_path,
                command
            )
            return translated_command
        
        return command
    
    def _execute_bash_command_in_session(self, command: str, timeout: int = 30) -> dict:
        """
        Execute a bash command in the Docker container.
        
        Args:
            command: The bash command to execute
            timeout: Maximum execution time in seconds (default: 30)
            
        Returns:
            dict with 'output', 'error', and 'return_code' keys
        """
        try:
            # Validate command for dangerous patterns
            dangerous_patterns = ['rm -rf /', 'format', ':(){:|:&};:']
            for pattern in dangerous_patterns:
                if pattern in command:
                    return {
                        "output": "",
                        "error": f"Command contains dangerous pattern: {pattern}",
                        "return_code": -1
                    }
            
            session = self._bash_session
            
            # Check if container is still running
            check_cmd = ['docker', 'inspect', '--format', '{{.State.Running}}', session.container_name]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
            
            if check_result.returncode != 0 or check_result.stdout.strip() != 'true':
                # Container stopped, create a new one
                logger.warning(f"Container {session.container_name} is not running, recreating...")
                try:
                    # Try to remove old container if it exists
                    subprocess.run(['docker', 'rm', '-f', session.container_name], 
                                 capture_output=True, timeout=5)
                except:
                    pass
                self._bash_session = self._create_bash_session()
                session = self._bash_session
            
            # Translate file paths from host temp directory to container mount path
            translated_command = self._translate_path_for_container(command, session)
            
            # Execute command in Docker container
            # Use sh -c to execute the command (alpine uses sh, not bash)
            exec_cmd = [
                'docker', 'exec',
                session.container_name,
                'sh', '-c', translated_command
            ]
            
            try:
                result = subprocess.run(
                    exec_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                return {
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_code": result.returncode
                }
            except subprocess.TimeoutExpired:
                # Kill the command if it times out
                try:
                    subprocess.run(['docker', 'exec', session.container_name, 'pkill', '-9', 'sh'],
                                 capture_output=True, timeout=5)
                except:
                    pass
                
                return {
                    "output": "",
                    "error": f"Command timed out after {timeout} seconds",
                    "return_code": -1
                }
                
        except Exception as e:
            logger.error(f"Error executing bash command in Docker: {e}")
            return {
                "output": "",
                "error": f"Error executing command: {str(e)}",
                "return_code": -1
            }
    
    def _set_resource_limits(self):
        """Set resource limits for sandboxed execution."""
        try:
            # Limit CPU time (30 seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
            # Limit memory (256 MB)
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
            # Limit file size (10 MB)
            resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        except Exception as e:
            logger.warning(f"Could not set all resource limits: {e}")
    
    def _execute_code(self, code: str, language: str = "python", timeout: int = 30) -> dict:
        """
        Execute code in a sandboxed environment.
        
        Args:
            code: The code to execute
            language: Programming language (python, javascript, etc.)
            timeout: Maximum execution time in seconds (default: 30)
            
        Returns:
            dict with 'output' and 'error' keys
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                if language.lower() == "python":
                    # Write code to temporary file
                    code_file = os.path.join(temp_dir, "code.py")
                    with open(code_file, 'w') as f:
                        f.write(code)
                    
                    # Execute Python code
                    process = subprocess.Popen(
                        ["python3", code_file],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=temp_dir,
                        preexec_fn=lambda: self._set_resource_limits()
                    )
                    
                    try:
                        stdout, stderr = process.communicate(timeout=timeout)
                        return {
                            "output": stdout.decode('utf-8', errors='replace'),
                            "error": stderr.decode('utf-8', errors='replace'),
                            "return_code": process.returncode
                        }
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        return {
                            "output": "",
                            "error": f"Code execution timed out after {timeout} seconds",
                            "return_code": -1
                        }
                else:
                    return {
                        "output": "",
                        "error": f"Unsupported language: {language}. Currently only 'python' is supported.",
                        "return_code": -1
                    }
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return {
                "output": "",
                "error": f"Error executing code: {str(e)}",
                "return_code": -1
            }
    
    def _handle_text_editor(self, action: str, file_path: str = None, content: str = None, start_line: int = None, end_line: int = None) -> dict:
        """
        Handle text editor operations.
        
        Args:
            action: Operation to perform (read, write, edit, list)
            file_path: Path to the file (relative to safe directory)
            content: Content to write or insert
            start_line: Start line for edit operations
            end_line: End line for edit operations
            
        Returns:
            dict with operation result
        """
        try:
            # Define safe directory (current working directory or temp)
            safe_dir = os.getcwd()
            if not os.path.exists(safe_dir):
                safe_dir = tempfile.gettempdir()
            
            if action == "read":
                if not file_path:
                    return {"error": "file_path is required for read operation"}
                
                # Ensure path is within safe directory
                full_path = os.path.join(safe_dir, file_path)
                if not os.path.abspath(full_path).startswith(os.path.abspath(safe_dir)):
                    return {"error": "File path is outside safe directory"}
                
                if not os.path.exists(full_path):
                    return {"error": f"File not found: {file_path}"}
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return {"content": content, "file_path": file_path}
            
            elif action == "write":
                if not file_path or content is None:
                    return {"error": "file_path and content are required for write operation"}
                
                full_path = os.path.join(safe_dir, file_path)
                if not os.path.abspath(full_path).startswith(os.path.abspath(safe_dir)):
                    return {"error": "File path is outside safe directory"}
                
                # Create directory if needed
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return {"success": True, "file_path": file_path, "message": f"File written successfully"}
            
            elif action == "edit":
                if not file_path or content is None:
                    return {"error": "file_path and content are required for edit operation"}
                
                full_path = os.path.join(safe_dir, file_path)
                if not os.path.abspath(full_path).startswith(os.path.abspath(safe_dir)):
                    return {"error": "File path is outside safe directory"}
                
                if not os.path.exists(full_path):
                    return {"error": f"File not found: {file_path}"}
                
                # Read existing content
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Edit lines if specified
                if start_line is not None and end_line is not None:
                    # Replace lines start_line to end_line (1-indexed)
                    lines[start_line-1:end_line] = [content + '\n']
                else:
                    # Append content
                    lines.append(content + '\n')
                
                # Write back
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return {"success": True, "file_path": file_path, "message": "File edited successfully"}
            
            elif action == "list":
                # List files in safe directory
                files = []
                for root, dirs, filenames in os.walk(safe_dir):
                    # Limit depth to prevent excessive listing
                    depth = root[len(safe_dir):].count(os.sep)
                    if depth > 2:
                        dirs[:] = []
                        continue
                    
                    for filename in filenames:
                        rel_path = os.path.relpath(os.path.join(root, filename), safe_dir)
                        files.append(rel_path)
                
                return {"files": files[:100]}  # Limit to 100 files
            
            else:
                return {"error": f"Unknown action: {action}. Supported actions: read, write, edit, list"}
        
        except Exception as e:
            logger.error(f"Error in text editor operation: {e}")
            return {"error": f"Error: {str(e)}"}
    
    @property
    def bash_tool_property(self):
        """Bash tool specification (Anthropic-defined schema-less tool)."""
        return {
            "type": "bash_20250124",
            "name": "bash"
        }
    
    def bash(self, command: str = None, restart: bool = False) -> str:
        """Execute a bash command or restart the session."""
        if restart:
            # Restart the bash session (stop and remove Docker container)
            if hasattr(self, '_bash_session') and self._bash_session is not None:
                try:
                    session = self._bash_session
                    # Stop and remove the container
                    subprocess.run(['docker', 'stop', session.container_name],
                                 capture_output=True, timeout=10)
                    subprocess.run(['docker', 'rm', '-f', session.container_name],
                                 capture_output=True, timeout=10)
                    logger.info(f"Stopped and removed Docker container {session.container_name}")
                except Exception as e:
                    logger.warning(f"Error stopping Docker container: {e}")
            self._bash_session = None
            return "Bash session restarted"
        
        if not command:
            return "Error: command is required unless restart is true"
        
        # Use persistent session if available, otherwise create one
        if not hasattr(self, '_bash_session') or self._bash_session is None:
            try:
                self._bash_session = self._create_bash_session()
            except Exception as e:
                return f"Error creating Docker container: {str(e)}. Make sure Docker is installed and running."
        
        result = self._execute_bash_command_in_session(command)
        if result["return_code"] == 0:
            return result["output"]
        else:
            error_msg = result['error'] if result['error'] else "No error message"
            output_msg = result['output'] if result['output'] else "No output"
            return f"Error (return code {result['return_code']}): {error_msg}\nOutput: {output_msg}"
    
    @property
    def text_editor_tool_property(self):
        """Text editor tool specification (Anthropic-defined schema-less tool)."""
        # Use text_editor_20250728 which is supported by claude-haiku-4-5-20251001
        # The name must be 'str_replace_based_edit_tool' for this version
        return {
            "type": "text_editor_20250728",
            "name": "str_replace_based_edit_tool"
        }
    
    def text_editor(self, **kwargs) -> str:
        """
        Handle text editor operations.
        The text_editor tool is a client tool that requires implementation.
        Parameters are provided by Claude based on the tool's built-in schema.
        """
        # Extract parameters from kwargs (Claude provides these based on tool schema)
        # The text_editor tool has a built-in schema, so we handle the operations
        result = self._handle_text_editor_from_kwargs(**kwargs)
        if "error" in result:
            return f"Error: {result['error']}"
        elif "success" in result:
            return result.get("message", "Operation completed successfully")
        elif "content" in result:
            return result["content"]
        elif "files" in result:
            return "\n".join(result["files"])
        else:
            return str(result)
    
    def _handle_text_editor_from_kwargs(self, **kwargs) -> dict:
        """
        Handle text editor operations from Claude's tool call parameters.
        The text_editor_20250728 tool uses specific command names: view, create, str_replace, insert
        """
        # Check which command is present in kwargs
        # The text_editor_20250728 tool uses command names directly, not an "action" parameter
        # Priority: create > str_replace > insert > view (check for specific params first)
        
        # Handle 'create' command (has file_text)
        if 'file_text' in kwargs:
            path = kwargs.get('path')
            file_text = kwargs.get('file_text')
            
            if not path:
                return {"error": "path is required for create operation"}
            
            # Determine safe directory
            safe_dir = os.getcwd()
            if not os.path.exists(safe_dir):
                safe_dir = tempfile.gettempdir()
            
            # Handle absolute paths or relative paths
            if os.path.isabs(path):
                temp_dir = tempfile.gettempdir()
                if path.startswith(temp_dir):
                    full_path = path
                else:
                    full_path = os.path.join(safe_dir, os.path.basename(path))
            else:
                full_path = os.path.join(safe_dir, path)
            
            # Security check
            if not os.path.abspath(full_path).startswith(os.path.abspath(safe_dir)) and not full_path.startswith(tempfile.gettempdir()):
                return {"error": "File path is outside safe directory"}
            
            # Create directory if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(file_text)
            
            return {"success": True, "file_path": path, "message": f"File created successfully"}
        
        # Handle 'str_replace' command
        if 'old_str' in kwargs and 'new_str' in kwargs:
            path = kwargs.get('path')
            old_str = kwargs.get('old_str')
            new_str = kwargs.get('new_str')
            
            if not path:
                return {"error": "path is required for str_replace operation"}
            
            # Determine safe directory
            safe_dir = os.getcwd()
            if not os.path.exists(safe_dir):
                safe_dir = tempfile.gettempdir()
            
            # Handle absolute paths or relative paths
            if os.path.isabs(path):
                temp_dir = tempfile.gettempdir()
                if path.startswith(temp_dir):
                    full_path = path
                else:
                    full_path = os.path.join(safe_dir, os.path.basename(path))
            else:
                full_path = os.path.join(safe_dir, path)
            
            # Security check
            if not os.path.abspath(full_path).startswith(os.path.abspath(safe_dir)) and not full_path.startswith(tempfile.gettempdir()):
                return {"error": "File path is outside safe directory"}
            
            if not os.path.exists(full_path):
                return {"error": f"File not found: {path}"}
            
            # Read file
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count occurrences
            count = content.count(old_str)
            if count == 0:
                return {"error": f"No matches found for the string to replace in {path}"}
            if count > 1:
                return {"error": f"Multiple matches ({count}) found for the string to replace. Please be more specific."}
            
            # Replace
            new_content = content.replace(old_str, new_str, 1)  # Replace only first occurrence
            
            # Write back
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {"success": True, "file_path": path, "message": f"String replaced successfully in {path}"}
        
        # Handle 'insert' command
        if 'insert_line' in kwargs and 'new_str' in kwargs:
            path = kwargs.get('path')
            insert_line = kwargs.get('insert_line')
            new_str = kwargs.get('new_str')
            
            if not path:
                return {"error": "path is required for insert operation"}
            
            # Determine safe directory
            safe_dir = os.getcwd()
            if not os.path.exists(safe_dir):
                safe_dir = tempfile.gettempdir()
            
            # Handle absolute paths or relative paths
            if os.path.isabs(path):
                temp_dir = tempfile.gettempdir()
                if path.startswith(temp_dir):
                    full_path = path
                else:
                    full_path = os.path.join(safe_dir, os.path.basename(path))
            else:
                full_path = os.path.join(safe_dir, path)
            
            # Security check
            if not os.path.abspath(full_path).startswith(os.path.abspath(safe_dir)) and not full_path.startswith(tempfile.gettempdir()):
                return {"error": "File path is outside safe directory"}
            
            if not os.path.exists(full_path):
                return {"error": f"File not found: {path}"}
            
            # Read file
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Insert at specified line (1-indexed, insert after the line)
            if insert_line < 0:
                insert_line = 0
            if insert_line > len(lines):
                insert_line = len(lines)
            
            # Insert the new string (add newline if new_str doesn't end with one)
            insert_text = new_str if new_str.endswith('\n') else new_str + '\n'
            lines.insert(insert_line, insert_text)
            
            # Write back
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return {"success": True, "file_path": path, "message": f"Text inserted successfully at line {insert_line + 1}"}
        
        # Handle 'view' command (default if only path is provided)
        if 'path' in kwargs:
            path = kwargs.get('path')
            view_range = kwargs.get('view_range')  # Optional: [start_line, end_line]
            max_characters = kwargs.get('max_characters')  # Optional
            
            # Determine safe directory
            safe_dir = os.getcwd()
            if not os.path.exists(safe_dir):
                safe_dir = tempfile.gettempdir()
            
            # Handle absolute paths or relative paths
            if os.path.isabs(path):
                # If absolute path, check if it's in temp directory (for doc management files)
                temp_dir = tempfile.gettempdir()
                if path.startswith(temp_dir):
                    # It's a temp file, use it directly
                    full_path = path
                else:
                    # Absolute path outside temp - treat as relative to safe_dir
                    full_path = os.path.join(safe_dir, os.path.basename(path))
            else:
                # Relative path
                full_path = os.path.join(safe_dir, path)
            
            # Security check
            if not os.path.abspath(full_path).startswith(os.path.abspath(safe_dir)) and not full_path.startswith(tempfile.gettempdir()):
                return {"error": "File path is outside safe directory"}
            
            if not os.path.exists(full_path):
                return {"error": f"File not found: {path}"}
            
            # Read file content
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Apply view_range if specified
            if view_range and len(view_range) == 2:
                start_line, end_line = view_range
                # Convert to 0-indexed and handle bounds
                start_idx = max(0, start_line - 1) if start_line > 0 else 0
                end_idx = min(len(lines), end_line) if end_line > 0 else len(lines)
                content = ''.join(lines[start_idx:end_idx])
            else:
                content = ''.join(lines)
            
            # Apply max_characters limit if specified
            if max_characters and len(content) > max_characters:
                content = content[:max_characters] + "\n... [truncated]"
            
            return {"content": content, "file_path": path}
        
        # If no recognized command, return error
        return {"error": "Unknown text editor command. Supported commands: view (path), create (path, file_text), str_replace (path, old_str, new_str), insert (path, insert_line, new_str)"}

    @property
    def tools(self):
        """Build tools list on demand using lazy-loaded tool instances."""
        if self._tools_cache is None:
            self._tools_cache = []
            
            # Add PostgreSQL tools if available
            if self.postgres_tool:
                self._tools_cache.extend([
                    self.postgres_tool.query_postgres_property,
                    self.postgres_tool.create_task_property,
                    self.postgres_tool.update_task_property,
                    self.postgres_tool.list_events_property,
                ])
                logger.info("PostgreSQL tools added to tools list (lazy)")
            
            # Add Doc Management tools if available
            if self.doc_management_tool:
                self._tools_cache.extend([
                    self.doc_management_tool.create_doc_property,
                    self.doc_management_tool.update_doc_property,
                    self.doc_management_tool.get_doc_property,
                    self.doc_management_tool.list_docs_property,
                    self.doc_management_tool.get_doc_markdown_path_property,
                    self.doc_management_tool.create_doc_from_markdown_file_property,
                    self.doc_management_tool.update_doc_from_markdown_file_property,
                ])
                logger.info("Doc Management tools added to tools list (lazy)")
            
            # Add Search tools if available
            if self.search_tool:
                self._tools_cache.append(self.search_tool.search_documents_property)
                logger.info("search_documents tool added to tools list (lazy)")
                
                
            # Add Outlook tools if available
            if self.outlook_tool:
                self._tools_cache.extend([
                    self.outlook_tool.list_upcoming_meetings_property,
                    self.outlook_tool.find_available_slots_property
                ])
                logger.info("Outlook tools added to tools list (lazy)")

            # Add Firestore tools if available
            if self.firestore_tool:
                self._tools_cache.extend([
                    self.firestore_tool.query_messages_property,
                    self.firestore_tool.send_message_property,
                ])
                logger.info("Firestore tools added to tools list (lazy)")

            # Add Cloud Storage tools if available
            if self.cloud_storage_tool:
                self._tools_cache.extend([
                    self.cloud_storage_tool.get_image_url_property,
                    self.cloud_storage_tool.list_chat_images_property
                ])
                logger.info("Cloud Storage tools added to tools list (lazy)")

            # Add Atlassian tools if available
            if self.atlassian_tool:
                self._tools_cache.extend([
                    self.atlassian_tool.search_issues_property,
                    self.atlassian_tool.get_issue_property,
                    self.atlassian_tool.create_issue_property,
                    self.atlassian_tool.update_issue_property,
                    self.atlassian_tool.add_comment_property,
                    self.atlassian_tool.search_users_property
                ])
                logger.info("Atlassian tools added to tools list (lazy)")

            # Add GitHub tools if available
            if self.github_tool:
                self._tools_cache.extend([
                    self.github_tool.list_repositories_property,
                    self.github_tool.get_repository_property,
                    self.github_tool.search_issues_property,
                    self.github_tool.get_issue_property,
                    self.github_tool.create_issue_property,
                    self.github_tool.update_issue_property,
                    self.github_tool.add_issue_comment_property,
                    self.github_tool.list_pull_requests_property,
                    self.github_tool.get_pull_request_property,
                    self.github_tool.create_pull_request_property,
                    self.github_tool.list_commits_property,
                    self.github_tool.get_commit_property,
                    self.github_tool.get_pull_request_commits_property,
                    self.github_tool.search_users_property
                ])
                logger.info("GitHub tools added to tools list (lazy)")

            # Add Notion tools if available
            if self.notion_tool:
                self._tools_cache.extend([
                    self.notion_tool.search_pages_property,
                    self.notion_tool.get_page_property,
                    self.notion_tool.create_page_property,
                    self.notion_tool.update_page_property,
                    self.notion_tool.archive_page_property,
                    self.notion_tool.query_database_property,
                    self.notion_tool.get_database_property,
                    self.notion_tool.create_database_entry_property,
                    self.notion_tool.update_database_entry_property
                ])
                logger.info("Notion tools added to tools list (lazy)")

            # Add ClickUp tools if available
            if self.clickup_tool:
                self._tools_cache.extend([
                    self.clickup_tool.search_tasks_property,
                    self.clickup_tool.get_task_property,
                    self.clickup_tool.create_task_property,
                    self.clickup_tool.update_task_property,
                    self.clickup_tool.add_comment_property,
                    self.clickup_tool.list_spaces_property,
                    self.clickup_tool.list_lists_property
                ])
                logger.info("ClickUp tools added to tools list (lazy)")

            # Add Linear tools if available
            if self.linear_tool:
                self._tools_cache.extend([
                    self.linear_tool.list_issues_property,
                    self.linear_tool.get_issue_property,
                    self.linear_tool.create_issue_property,
                    self.linear_tool.update_issue_property,
                    self.linear_tool.search_issues_property,
                    self.linear_tool.list_projects_property,
                    self.linear_tool.get_project_property,
                    self.linear_tool.list_teams_property,
                    self.linear_tool.search_users_property
                ])
                logger.info("Linear tools added to tools list (lazy)")

            # Add DuckDB tools (response-scoped tools only)
            if 'duckdb' in self.requested_tools:
                from leanworks.agent.tools.duckdb import query_response_duckdb_property, get_response_schema_property
                self._tools_cache.extend([
                    query_response_duckdb_property(),
                    get_response_schema_property()
                ])
                logger.info("DuckDB tools added to tools list (query_response, get_schema)")
            
            # Add client tools (bash and text_editor - code_execution is a server tool)
            # These are always available as they don't require external dependencies
            self._tools_cache.extend([
                self.bash_tool_property,
                self.text_editor_tool_property
            ])
            logger.info("Client tools added to tools list (bash, text_editor)")
            
            logger.info(f"Tools list built with {len(self._tools_cache)} tools")
        
        return self._tools_cache

    @property
    def function_map(self):
        """Build function map on demand using lazy-loaded tool instances."""
        if self._function_map_cache is None:
            self._function_map_cache = {}
            
            # Add PostgreSQL functions if available
            if self.postgres_tool:
                self._function_map_cache.update({
                    "query_postgres": self.postgres_tool.query_postgres,
                    "create_task": self.postgres_tool.create_task,
                    "update_task": self.postgres_tool.update_task,
                    "list_events": self.postgres_tool.list_events,
                })
                logger.info("PostgreSQL functions added to function_map (lazy)")
            
            # Add Doc Management functions if available
            if self.doc_management_tool:
                self._function_map_cache.update({
                    "create_doc": self.doc_management_tool.create_doc,
                    "update_doc": self.doc_management_tool.update_doc,
                    "get_doc": self.doc_management_tool.get_doc,
                    "list_docs": self.doc_management_tool.list_docs,
                    "get_doc_markdown_path": self.doc_management_tool.get_doc_markdown_path,
                    "create_doc_from_markdown_file": self.doc_management_tool.create_doc_from_markdown_file,
                    "update_doc_from_markdown_file": self.doc_management_tool.update_doc_from_markdown_file,
                })
                logger.info("Doc Management functions added to function_map (lazy)")
            
            # Add search function if available
            if self.search_tool:
                self._function_map_cache["search_documents"] = self.search_tool.search_documents
                logger.info("search_documents function added to function_map (lazy)")
                
                
            # Add Outlook functions if available
            if self.outlook_tool:
                self._function_map_cache.update({
                    "list_upcoming_meetings": self.outlook_tool.list_upcoming_meetings,
                    "find_available_slots": self.outlook_tool.find_available_slots
                })
                logger.info("Outlook functions added to function_map (lazy)")

            # Add Firestore functions if available
            if self.firestore_tool:
                self._function_map_cache.update({
                    "query_messages": self.firestore_tool.query_messages,
                    "send_message": self.firestore_tool.send_message,
                })
                logger.info("Firestore functions added to function_map (lazy)")

            # Add Cloud Storage functions if available
            if self.cloud_storage_tool:
                self._function_map_cache.update({
                    "get_image_url": self.cloud_storage_tool.get_image_url,
                    "list_chat_images": self.cloud_storage_tool.list_chat_images
                })
                logger.info("Cloud Storage functions added to function_map (lazy)")

            # Add Atlassian functions if available
            if self.atlassian_tool:
                self._function_map_cache.update({
                    "search_issues": self.atlassian_tool.search_issues,
                    "get_issue": self.atlassian_tool.get_issue,
                    "create_issue": self.atlassian_tool.create_issue,
                    "update_issue": self.atlassian_tool.update_issue,
                    "add_comment": self.atlassian_tool.add_comment,
                    "jira_search_users": self.atlassian_tool.search_users
                })
                logger.info("Atlassian functions added to function_map (lazy)")

            # Add GitHub functions if available
            if self.github_tool:
                self._function_map_cache.update({
                    "github_list_repositories": self.github_tool.list_repositories,
                    "github_get_repository": self.github_tool.get_repository,
                    "github_search_issues": self.github_tool.search_issues,
                    "github_get_issue": self.github_tool.get_issue,
                    "github_create_issue": self.github_tool.create_issue,
                    "github_update_issue": self.github_tool.update_issue,
                    "github_add_issue_comment": self.github_tool.add_issue_comment,
                    "github_list_pull_requests": self.github_tool.list_pull_requests,
                    "github_get_pull_request": self.github_tool.get_pull_request,
                    "github_create_pull_request": self.github_tool.create_pull_request,
                    "github_list_commits": self.github_tool.list_commits,
                    "github_get_commit": self.github_tool.get_commit,
                    "github_get_pull_request_commits": self.github_tool.get_pull_request_commits,
                    "github_search_users": self.github_tool.search_users
                })
                logger.info("GitHub functions added to function_map (lazy)")

            # Add Notion functions if available
            if self.notion_tool:
                self._function_map_cache.update({
                    "notion_search_pages": self.notion_tool.search_pages,
                    "notion_get_page": self.notion_tool.get_page,
                    "notion_create_page": self.notion_tool.create_page,
                    "notion_update_page": self.notion_tool.update_page,
                    "notion_archive_page": self.notion_tool.archive_page,
                    "notion_query_database": self.notion_tool.query_database,
                    "notion_get_database": self.notion_tool.get_database,
                    "notion_create_database_entry": self.notion_tool.create_database_entry,
                    "notion_update_database_entry": self.notion_tool.update_database_entry
                })
                logger.info("Notion functions added to function_map (lazy)")

            # Add ClickUp functions if available
            if self.clickup_tool:
                self._function_map_cache.update({
                    "clickup_search_tasks": self.clickup_tool.search_tasks,
                    "clickup_get_task": self.clickup_tool.get_task,
                    "clickup_create_task": self.clickup_tool.create_task,
                    "clickup_update_task": self.clickup_tool.update_task,
                    "clickup_add_comment": self.clickup_tool.add_comment,
                    "clickup_list_spaces": self.clickup_tool.list_spaces,
                    "clickup_list_lists": self.clickup_tool.list_lists
                })
                logger.info("ClickUp functions added to function_map (lazy)")

            # Add Linear functions if available
            if self.linear_tool:
                self._function_map_cache.update({
                    "linear_list_issues": self.linear_tool.list_issues,
                    "linear_get_issue": self.linear_tool.get_issue,
                    "linear_create_issue": self.linear_tool.create_issue,
                    "linear_update_issue": self.linear_tool.update_issue,
                    "linear_search_issues": self.linear_tool.search_issues,
                    "linear_list_projects": self.linear_tool.list_projects,
                    "linear_get_project": self.linear_tool.get_project,
                    "linear_list_teams": self.linear_tool.list_teams,
                    "linear_search_users": self.linear_tool.search_users
                })
                logger.info("Linear functions added to function_map (lazy)")

            # Add DuckDB function mapping (response-scoped functions only)
            if 'duckdb' in self.requested_tools:
                from leanworks.agent.tools.duckdb import query_response_duckdb, get_response_schema
                self._function_map_cache.update({
                    "query_response_duckdb": query_response_duckdb,
                    "get_response_schema": get_response_schema
                })

            # Add client tool function mappings (always available)
            # Note: bash and text_editor are client tools, code_execution is a server tool
            # text_editor tool uses name "str_replace_based_edit_tool" in the API for version 20250728
            self._function_map_cache.update({
                "bash": self.bash,
                "str_replace_based_edit_tool": self.text_editor
            })
            logger.info("Client tool functions added to function_map (bash, str_replace_based_edit_tool)")

            logger.info(f"Function map built with {len(self._function_map_cache)} functions")
            logger.info(f"Available functions: {list(self._function_map_cache.keys())}")
        
        return self._function_map_cache
    
    def clear_cache(self):
        """Clear all cached tools and rebuild on next access."""
        self._tool_cache.clear()
        self._tools_cache = None
        self._function_map_cache = None
        self.enabled_tools = []
        # Re-register DuckDB tools immediately (stateless)
        if 'duckdb' in self.requested_tools:
            self.enabled_tools.append('duckdb')
    
    def _get_rag_storage_tool(self):
        """
        Lazy initialize RAG storage tool for storing unstructured large responses.
        Reuses existing SearchTool's vector DB client if available.
        
        Returns:
            RAGStorageTool instance or None if not available
        """
        if self._rag_storage is None:
            # Check if search tool is available (which has vector DB client)
            if 'search' in self.requested_tools and self.search_tool:
                try:
                    from leanworks.agent.tools.rag_storage import RAGStorageTool
                    from leanworks.setting import LARGE_RESPONSE_CONFIG
                    
                    # Reuse search tool's vector DB and embedding clients
                    vectordb_client = self.search_tool.chat.vectordb_client
                    embedding_client = self.search_tool.chat.vectordb_client.embedding_model_client
                    
                    # Get chunk settings from config
                    chunk_size = LARGE_RESPONSE_CONFIG.get("rag_chunk_size", 512)
                    chunk_overlap = LARGE_RESPONSE_CONFIG.get("rag_chunk_overlap", 128)
                    
                    self._rag_storage = RAGStorageTool(
                        vectordb_client=vectordb_client,
                        embedding_client=embedding_client,
                        org_slug=self.org_slug,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    logger.info("RAGStorageTool initialized successfully (reusing SearchTool clients)")
                except Exception as e:
                    logger.warning(f"Failed to initialize RAGStorageTool: {e}")
                    self._rag_storage = False  # Mark as unavailable
            else:
                logger.debug("RAGStorageTool not available: search tool not enabled or not initialized")
                self._rag_storage = False  # Mark as unavailable
        
        return self._rag_storage if self._rag_storage else None
        logger.info("Tool cache cleared - tools will be reinitialized on next access")
    
    def cleanup_bash_session(self):
        """Clean up Docker container for bash session."""
        if hasattr(self, '_bash_session') and self._bash_session is not None:
            try:
                session = self._bash_session
                # Stop and remove the container
                subprocess.run(['docker', 'stop', session.container_name],
                             capture_output=True, timeout=10)
                subprocess.run(['docker', 'rm', '-f', session.container_name],
                             capture_output=True, timeout=10)
                logger.info(f"Cleaned up Docker container {session.container_name}")
                self._bash_session = None
            except Exception as e:
                logger.warning(f"Error cleaning up Docker container: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup_bash_session()
        except:
            pass  # Ignore errors during cleanup