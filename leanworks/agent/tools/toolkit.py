
from leanworks.agent.tools.bigquery import BigQueryTool
from leanworks.agent.tools.search import SearchTool
from leanworks.agent.tools.gitlab import GitlabTool
from leanworks.agent.tools.outlook import OutlookTool
from leanworks.agent.tools.duckdb import DuckDBTool
import logging

logger = logging.getLogger(__name__)

class ToolUse:
    def __init__(self, bq_client_wrapper=None, storage_client=None, secret_client=None, read_document_ids=None, tools=None, root_dir=None, user_id=None, session_id=None):
        """
        Initialize ToolUse with various client connections.
        
        Args:
            bq_client_wrapper: BigQuery client wrapper that has dataset_id attribute
            storage_client: Google Cloud Storage client
            secret_client: Secret management client
            read_document_ids: Set of document IDs already read for deduplication
            tools: List of additional tools to enable. These will be added to the default tools ['search', 'bigquery', 'duckdb']
        """
        # Set default tools if not provided
        if tools is None:
            requested_tools = ['search', 'bigquery', 'duckdb']
        else:
            # Add provided tools to default tools (with deduplication)
            default_tools = ['search', 'bigquery', 'duckdb']
            requested_tools = list(set(default_tools + tools))  # Remove duplicates while preserving functionality
        
        # Track which tools are actually enabled (successfully initialized)
        self.enabled_tools = []

        # Persist session context for tools that can leverage it (e.g., DuckDB-backed persistence)
        self.user_id = user_id
        self.session_id = session_id
        
        # Initialize tool instances based on requested tools and available clients
        self.bigquery_tool = None

        if 'bigquery' in requested_tools and bq_client_wrapper:
            try:
                self.bigquery_tool = BigQueryTool(bq_client_wrapper)
                self.enabled_tools.append('bigquery')
                logger.info("BigQueryTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize BigQueryTool: {str(e)}")
                self.bigquery_tool = None
        elif 'bigquery' in requested_tools:
            logger.warning("BigQueryTool not initialized: missing bq_client_wrapper")
        
        # Initialize SearchTool with error handling
        self.search_tool = None
        if 'search' in requested_tools and storage_client and secret_client:
            try:
                # Pass shared read_document_ids so deduplication persists across searches
                # Ensure read_document_ids is initialized before passing
                self.read_document_ids = read_document_ids if read_document_ids is not None else set()
                self.search_tool = SearchTool(storage_client, secret_client, read_document_ids=self.read_document_ids)
                self.enabled_tools.append('search')
                logger.info("SearchTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SearchTool: {str(e)}")
                self.search_tool = None
        elif 'search' in requested_tools:
            logger.warning("SearchTool not initialized: missing storage_client or secret_client")
            
        # Initialize GitlabTool with error handling - get credentials from secret_client
        self.gitlab_tool = None
        if 'gitlab' in requested_tools and secret_client:
            try:
                gitlab_auth = {
                    'gitlab_url': secret_client.get('GITLAB_DOMAIN'),
                    'gitlab_token': secret_client.get('GITLAB_KEY')
                }
                self.gitlab_tool = GitlabTool(gitlab_auth)
                self.enabled_tools.append('gitlab')
                logger.info("GitlabTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GitlabTool: {str(e)}")
                self.gitlab_tool = None
        elif 'gitlab' in requested_tools:
            logger.warning("GitlabTool not initialized: missing secret_client")
            
        # Initialize OutlookTool with error handling - get credentials from secret_client
        self.outlook_tool = None
        if 'outlook' in requested_tools and secret_client:
            try:
                outlook_auth = {
                    'azure_client_id': secret_client.get('AD_CLIENT_ID'),
                    'azure_client_secret': secret_client.get('AD_CLIENT_SECRET'),
                    'azure_tenant_id': secret_client.get('AD_TENANT_ID')
                }
                self.outlook_tool = OutlookTool(
                    client_id=outlook_auth.get('azure_client_id'),
                    client_secret=outlook_auth.get('azure_client_secret'),
                    tenant_id=outlook_auth.get('azure_tenant_id')
                )
                self.enabled_tools.append('outlook')
                logger.info("OutlookTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OutlookTool: {str(e)}")
                self.outlook_tool = None
        elif 'outlook' in requested_tools:
            logger.warning("OutlookTool not initialized: missing secret_client")

        # Register DuckDB tools (stateless wrappers; no instance required)
        if 'duckdb' in requested_tools:
            self.enabled_tools.append('duckdb')
            logger.info("DuckDB tools registered (stateless)")
            
        # Ensure read_document_ids exists even if search tool not initialized above
        if not hasattr(self, 'read_document_ids'):
            self.read_document_ids = read_document_ids if read_document_ids is not None else set()
        
        # Initialize tools list based on successfully initialized tools
        self.tools = []
        if self.bigquery_tool:
            self.tools.extend([
                self.bigquery_tool.query_bigquery_property,
            ])
        
        if self.search_tool:
            self.tools.append(self.search_tool.search_documents_property)
            logger.info("search_documents tool added to tools list")
            
        # Add GitLab tools if available and enabled
        if self.gitlab_tool:
            self.tools.extend([
                self.gitlab_tool.list_gitlab_projects_property,
                self.gitlab_tool.list_gitlab_issues_property,
                self.gitlab_tool.list_gitlab_milestones_property,
                self.gitlab_tool.find_gitlab_user_by_email_property,
                self.gitlab_tool.list_gitlab_project_members_property,
                self.gitlab_tool.get_gitlab_project_detail_property,
                self.gitlab_tool.list_gitlab_groups_property,
                self.gitlab_tool.get_gitlab_group_detail_property,
                self.gitlab_tool.get_issue_detail_property
            ])
            
        # Add Outlook tools if available and enabled
        if self.outlook_tool:
            self.tools.extend([
                self.outlook_tool.list_upcoming_meetings_property,
                self.outlook_tool.find_available_slots_property
            ])
            logger.info("Outlook tools added to tools list")

        # Add DuckDB tools (response-scoped tools only)
        if 'duckdb' in requested_tools:
            from leanworks.agent.tools.duckdb import query_response_duckdb_property, get_response_schema_property
            self.tools.extend([
                query_response_duckdb_property(),
                get_response_schema_property()
            ])
            logger.info("DuckDB tools added to tools list (query_response, get_schema)")
        
        # Define function map based on successfully initialized tools
        self.function_map = {}
        
        if self.bigquery_tool:
            self.function_map.update({
                "query_bigquery": self.bigquery_tool.query_bigquery,
            })
        
        # Add search function if available and enabled
        if self.search_tool:
            self.function_map["search_documents"] = self.search_tool.search_documents
            logger.info("search_documents function added to function_map (direct)")
            
        # Add GitLab functions if available and enabled
        if self.gitlab_tool:
            # Wrapper to auto-inject session info for large results persistence
            def _list_gitlab_issues_with_session(**kwargs):
                try:
                    save_flag = kwargs.get("save_large_to_duckdb", True)
                    if save_flag:
                        # Do not expose session identifiers via tool args; instead, set them on the tool instance
                        if hasattr(self.gitlab_tool, "_session_context") is False:
                            self.gitlab_tool._session_context = {}
                        self.gitlab_tool._session_context.update({
                            "user_id": self.user_id,
                            "session_id": self.session_id,
                        })
                except Exception:
                    # Best-effort injection; continue without blocking
                    pass
                return self.gitlab_tool.list_gitlab_issues(**kwargs)

            self.function_map.update({
                "list_gitlab_projects": self.gitlab_tool.list_gitlab_projects,
                "list_gitlab_issues": _list_gitlab_issues_with_session,
                "list_gitlab_milestones": self.gitlab_tool.list_gitlab_milestones,
                "find_gitlab_user_by_email": self.gitlab_tool.find_gitlab_user_by_email,
                "list_gitlab_project_members": self.gitlab_tool.list_gitlab_project_members,
                "get_gitlab_project_detail": self.gitlab_tool.get_gitlab_project_detail,
                "list_gitlab_groups": self.gitlab_tool.list_gitlab_groups,
                "get_gitlab_group_detail": self.gitlab_tool.get_gitlab_group_detail,
                "get_issue_detail": self.gitlab_tool.get_issue_detail
            })
            
        # Add Outlook functions if available and enabled
        if self.outlook_tool:
            self.function_map.update({
                "list_upcoming_meetings": self.outlook_tool.list_upcoming_meetings,
                "find_available_slots": self.outlook_tool.find_available_slots
            })
            logger.info("Outlook functions added to function_map")

        # Add DuckDB function mapping (response-scoped functions only)
        if 'duckdb' in requested_tools:
            from leanworks.agent.tools.duckdb import query_response_duckdb, get_response_schema
            self.function_map.update({
                "query_response_duckdb": query_response_duckdb,
                "get_response_schema": get_response_schema
            })

        # Log final tool availability for debugging
        logger.info(f"Requested tools: {requested_tools}")
        logger.info(f"Successfully enabled tools: {self.enabled_tools}")
        logger.info(f"Available tools: {[tool.get('name', 'unknown') for tool in self.tools]}")
        logger.info(f"Available functions: {list(self.function_map.keys())}")

    
    