from leanworks.agent.tools.postgres import PostgresTool
from leanworks.agent.tools.search import SearchTool
from leanworks.agent.tools.outlook import OutlookTool
from leanworks.agent.tools.duckdb import DuckDBTool
from leanworks.agent.helpers import AgentHelpers
import logging

logger = logging.getLogger(__name__)

class ToolUse:
    def __init__(self, domain=None, firestore_client=None, secret_manager_client=None, read_document_ids=None, tools=None, root_dir=None, user_id=None, session_id=None, credential_path: str = "gcp_credential.json"):
        """
        Initialize ToolUse with various client connections using lazy loading.
        
        Args:
            domain: Client domain (e.g., 'leanworks.ai') extracted from user_id. Used to determine database and client_name.
            firestore_client: Firestore client
            secret_manager_client: Secret Manager client
            read_document_ids: Set of document IDs already read for deduplication
            tools: List of tools to enable. Internal tools ['search', 'postgres', 'duckdb'] are always available.
                   External tools (e.g., 'outlook') should be explicitly provided in this list.
            credential_path: Path to GCP credential JSON file (default: "gcp_credential.json")
        """
        # Store initialization parameters for lazy loading
        self.domain = domain
        # Create postgres_client_wrapper internally if domain is provided
        if domain:
            class PostgresClientWrapper:
                def __init__(self, domain):
                    self.domain = domain
            self.postgres_client_wrapper = PostgresClientWrapper(domain)
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
        internal_tools = ['search', 'postgres', 'duckdb']
        
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
                    # Get client_name from domain
                    client_name = None
                    if self.domain:
                        # Extract client_name from domain by removing non-alphanumeric characters
                        import re
                        client_name = re.sub(r'[^a-zA-Z0-9]', '', self.domain)
                    
                    if not client_name:
                        raise ValueError("Cannot determine client_name from domain")
                    
                    self._tool_cache['search_tool'] = SearchTool(
                        self.firestore_client,
                        self.domain,
                        self.secret_manager_client,
                        client_name,
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
            if 'outlook' in self.requested_tools and self.secret_manager_client and self.project_id:
                try:
                    # Helper function to get secret
                    def get_secret(name):
                        full_name = f"projects/{self.project_id}/secrets/{name}/versions/latest"
                        response = self.secret_manager_client.access_secret_version(name=full_name)
                        return response.payload.data.decode("UTF-8")
                    
                    outlook_auth = {
                        'azure_client_id': get_secret('AD_CLIENT_ID'),
                        'azure_client_secret': get_secret('AD_CLIENT_SECRET'),
                        'azure_tenant_id': get_secret('AD_TENANT_ID')
                    }
                    self._tool_cache['outlook_tool'] = OutlookTool(
                        client_id=outlook_auth.get('azure_client_id'),
                        client_secret=outlook_auth.get('azure_client_secret'),
                        tenant_id=outlook_auth.get('azure_tenant_id')
                    )
                    if 'outlook' not in self.enabled_tools:
                        self.enabled_tools.append('outlook')
                    logger.info("OutlookTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize OutlookTool: {str(e)}")
                    self._tool_cache['outlook_tool'] = None
            elif 'outlook' in self.requested_tools:
                logger.warning("OutlookTool not initialized: missing secret_client")
                self._tool_cache['outlook_tool'] = None
            else:
                self._tool_cache['outlook_tool'] = None
        return self._tool_cache['outlook_tool']

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