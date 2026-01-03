from leanworks.agent.tools.postgres import PostgresTool
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
                    self._tool_cache['postgres_tool'] = PostgresTool(self.postgres_client_wrapper)
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

    @property
    def tools(self):
        """Build tools list on demand using lazy-loaded tool instances."""
        if self._tools_cache is None:
            self._tools_cache = []
            
            # Add PostgreSQL tools if available
            if self.postgres_tool:
                self._tools_cache.extend([
                    self.postgres_tool.query_postgres_property,
                ])
                logger.info("PostgreSQL tools added to tools list (lazy)")
            
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
                self._tools_cache.append(self.firestore_tool.query_messages_property)
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
                })
                logger.info("PostgreSQL functions added to function_map (lazy)")
            
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
                self._function_map_cache["query_messages"] = self.firestore_tool.query_messages
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
        logger.info("Tool cache cleared - tools will be reinitialized on next access")