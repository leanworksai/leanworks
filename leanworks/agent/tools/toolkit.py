from leanworks.agent.tools.firestore import FirestoreTool
from leanworks.agent.tools.search import SearchTool
from leanworks.agent.tools.outlook import OutlookTool
from leanworks.agent.tools.duckdb import DuckDBTool
import logging

logger = logging.getLogger(__name__)

class ToolUse:
    def __init__(self, firestore_client_wrapper=None, storage_client=None, secret_client=None, read_document_ids=None, tools=None, root_dir=None, user_id=None, session_id=None):
        """
        Initialize ToolUse with various client connections using lazy loading.
        
        Args:
            firestore_client_wrapper: Firestore client wrapper that has domain attribute
            storage_client: Google Cloud Storage client
            secret_client: Secret management client
            read_document_ids: Set of document IDs already read for deduplication
            tools: List of additional tools to enable. These will be added to the internal tools ['search', 'firestore', 'duckdb'] which are always available.
                   External tools (e.g., 'outlook') are only enabled if they appear in the Firestore integrations collection.
        """
        # Store initialization parameters for lazy loading
        self.firestore_client_wrapper = firestore_client_wrapper
        self.storage_client = storage_client
        self.secret_client = secret_client
        self.read_document_ids = read_document_ids if read_document_ids is not None else set()
        self.user_id = user_id
        self.session_id = session_id
        
        # Get available integrations from Firestore
        available_integrations = self._get_available_integrations()
        logger.info(f"Available integrations from Firestore: {available_integrations}")
        
        # Internal tools that are always available
        internal_tools = ['search', 'firestore', 'duckdb']
        
        # Set default tools if not provided
        if tools is None:
            requested_tools = internal_tools
        else:
            # Add provided tools to default tools (with deduplication)
            default_tools = internal_tools
            requested_tools = list(set(default_tools + tools))  # Remove duplicates while preserving functionality
        
        # Filter requested tools based on available integrations
        # Always include internal tools (firestore, search, duckdb), filter others
        filtered_tools = []
        for tool in requested_tools:
            if tool in internal_tools:
                # Always include internal tools
                filtered_tools.append(tool)
            elif tool in available_integrations:
                # Include external tools only if they're in the integration list
                filtered_tools.append(tool)
                logger.info(f"Tool '{tool}' enabled (found in integrations)")
            else:
                # Skip tools not in integrations
                logger.info(f"Tool '{tool}' disabled (not found in integrations)")
        
        self.requested_tools = filtered_tools
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
    
    def _get_available_integrations(self):
        """
        Query Firestore integrations collection to get list of available integrations.
        
        Returns:
            list: List of integration types (e.g., ['outlook', 'duckdb', 'gitlab', 'jira'])
        """
        try:
            if not self.firestore_client_wrapper:
                logger.warning("No Firestore client wrapper available, cannot fetch integrations")
                return []
            
            # Get domain from wrapper
            domain = getattr(self.firestore_client_wrapper, 'domain', None)
            if not domain:
                logger.warning("Domain not available in Firestore client wrapper")
                return []
            
            # Initialize Firestore client using FirestoreTool's method
            from leanworks.setting import _get_firestore_client
            db = _get_firestore_client()
            
            # Query integrations collection
            collection_path = f"domains/{domain}/integrations"
            integrations_ref = db.collection(collection_path)
            integrations_docs = integrations_ref.stream()
            
            # Extract integration types
            available_integrations = []
            for doc in integrations_docs:
                doc_data = doc.to_dict()
                if doc_data:
                    # Get integration type (e.g., 'outlook', 'gitlab', 'jira')
                    integration_type = doc_data.get('type') or doc_data.get('integrationType')
                    if integration_type:
                        available_integrations.append(integration_type.lower())
            
            logger.info(f"Found {len(available_integrations)} integrations in Firestore for domain {domain}")
            return available_integrations
            
        except Exception as e:
            logger.error(f"Failed to fetch integrations from Firestore: {str(e)}")
            return []

    # Lazy loading properties for individual tools
    @property
    def firestore_tool(self):
        """Lazy-load Firestore tool on first access."""
        if 'firestore_tool' not in self._tool_cache:
            if 'firestore' in self.requested_tools and self.firestore_client_wrapper:
                try:
                    self._tool_cache['firestore_tool'] = FirestoreTool(self.firestore_client_wrapper)
                    if 'firestore' not in self.enabled_tools:
                        self.enabled_tools.append('firestore')
                    logger.info("FirestoreTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize FirestoreTool: {str(e)}")
                    self._tool_cache['firestore_tool'] = None
            elif 'firestore' in self.requested_tools:
                logger.warning("FirestoreTool not initialized: missing firestore_client_wrapper")
                self._tool_cache['firestore_tool'] = None
            else:
                self._tool_cache['firestore_tool'] = None
        return self._tool_cache['firestore_tool']
    
    @property
    def search_tool(self):
        """Lazy-load Search tool on first access."""
        if 'search_tool' not in self._tool_cache:
            if 'search' in self.requested_tools and self.storage_client and self.secret_client:
                try:
                    # Get client_name from firestore_client_wrapper
                    client_name = None
                    if self.firestore_client_wrapper:
                        if hasattr(self.firestore_client_wrapper, 'client_name'):
                            client_name = self.firestore_client_wrapper.client_name
                        elif hasattr(self.firestore_client_wrapper, 'domain'):
                            # Extract client_name from domain by removing non-alphanumeric characters
                            import re
                            client_name = re.sub(r'[^a-zA-Z0-9]', '', self.firestore_client_wrapper.domain)
                    
                    if not client_name:
                        raise ValueError("Cannot determine client_name from firestore_client_wrapper")
                    
                    self._tool_cache['search_tool'] = SearchTool(
                        self.storage_client, 
                        self.secret_client,
                        client_name,
                        read_document_ids=self.read_document_ids
                    )
                    if 'search' not in self.enabled_tools:
                        self.enabled_tools.append('search')
                    logger.info("SearchTool initialized successfully (lazy)")
                except Exception as e:
                    logger.error(f"Failed to initialize SearchTool: {str(e)}")
                    self._tool_cache['search_tool'] = None
            elif 'search' in self.requested_tools:
                logger.warning("SearchTool not initialized: missing storage_client or secret_client")
                self._tool_cache['search_tool'] = None
            else:
                self._tool_cache['search_tool'] = None
        return self._tool_cache['search_tool']
    
    
    @property
    def outlook_tool(self):
        """Lazy-load Outlook tool on first access."""
        if 'outlook_tool' not in self._tool_cache:
            if 'outlook' in self.requested_tools and self.secret_client:
                try:
                    outlook_auth = {
                        'azure_client_id': self.secret_client.get('AD_CLIENT_ID'),
                        'azure_client_secret': self.secret_client.get('AD_CLIENT_SECRET'),
                        'azure_tenant_id': self.secret_client.get('AD_TENANT_ID')
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
            
            # Add Firestore tools if available
            if self.firestore_tool:
                self._tools_cache.extend([
                    self.firestore_tool.query_firestore_property,
                ])
                logger.info("Firestore tools added to tools list (lazy)")
            
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
            
            # Add Firestore functions if available
            if self.firestore_tool:
                self._function_map_cache.update({
                    "query_firestore": self.firestore_tool.query_firestore,
                })
                logger.info("Firestore functions added to function_map (lazy)")
            
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