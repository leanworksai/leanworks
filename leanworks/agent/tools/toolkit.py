from leanworks.agent.tools.leanworks import LeanworksTool
from leanworks.agent.tools.search import SearchTool
from leanworks.agent.tools.gitlab import GitlabTool
import logging

logger = logging.getLogger(__name__)

class ToolUse:
    def __init__(self, bq_client_wrapper=None, storage_client=None, secret_client=None, read_document_ids=None, gitlab_auth=None, tools=None):
        """
        Initialize ToolUse with various client connections.
        
        Args:
            bq_client_wrapper: BigQuery client wrapper that has dataset_id attribute
            storage_client: Google Cloud Storage client
            secret_client: Secret management client
            read_document_ids: Set of document IDs already read for deduplication
            gitlab_auth: Dictionary containing gitlab_url and gitlab_token
            tools: List of additional tools to enable. These will be added to the default tools ['leanworks', 'search']
        """
        # Set default tools if not provided
        if tools is None:
            requested_tools = ['leanworks', 'search']
        else:
            # Add provided tools to default tools (with deduplication)
            default_tools = ['leanworks', 'search']
            requested_tools = list(set(default_tools + tools))  # Remove duplicates while preserving functionality
        
        # Track which tools are actually enabled (successfully initialized)
        self.enabled_tools = []
        
        # Initialize tool instances based on requested tools and available clients
        self.leanworks_tool = None
        if 'leanworks' in requested_tools and bq_client_wrapper:
            self.leanworks_tool = LeanworksTool(bq_client_wrapper)
            self.enabled_tools.append('leanworks')
        elif 'leanworks' in requested_tools:
            logger.warning("LeanworksTool not initialized: missing bq_client_wrapper")
        
        # Initialize SearchTool with error handling
        self.search_tool = None
        if 'search' in requested_tools and storage_client and secret_client:
            try:
                self.search_tool = SearchTool(storage_client, secret_client)
                self.enabled_tools.append('search')
                logger.info("SearchTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SearchTool: {str(e)}")
                self.search_tool = None
        elif 'search' in requested_tools:
            logger.warning("SearchTool not initialized: missing storage_client or secret_client")
            
        # Initialize GitlabTool with error handling  
        self.gitlab_tool = None
        if 'gitlab' in requested_tools and gitlab_auth:
            try:
                self.gitlab_tool = GitlabTool(gitlab_auth)
                self.enabled_tools.append('gitlab')
                logger.info("GitlabTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GitlabTool: {str(e)}")
                self.gitlab_tool = None
        elif 'gitlab' in requested_tools:
            logger.warning("GitlabTool not initialized: missing gitlab_auth")
            
        self.read_document_ids = read_document_ids if read_document_ids is not None else set()
        
        # Initialize tools list based on successfully initialized tools
        self.tools = []
        if self.leanworks_tool:
            self.tools.extend([
                self.leanworks_tool.list_projects_property,
                self.leanworks_tool.list_tasks_property,
                self.leanworks_tool.list_progress_updates_property,
                self.leanworks_tool.add_task_property,
                self.leanworks_tool.list_users_property,
            ])
        
        if self.search_tool:
            self.tools.append(self.search_tool.search_knowledge_property)
            logger.info("search_knowledge tool added to tools list")
            
        # Add GitLab tools if available and enabled
        if self.gitlab_tool:
            self.tools.extend([
                self.gitlab_tool.list_gitlab_projects_property,
                self.gitlab_tool.list_gitlab_issues_property,
                self.gitlab_tool.find_gitlab_user_by_email_property,
                self.gitlab_tool.list_gitlab_project_members_property,
                self.gitlab_tool.get_gitlab_project_detail_property,
                self.gitlab_tool.list_gitlab_groups_property,
                self.gitlab_tool.get_gitlab_group_detail_property,
                self.gitlab_tool.get_issue_detail_property
            ])
        
        # Define function map based on successfully initialized tools
        self.function_map = {}
        
        # Add leanworks functions if available and enabled
        if self.leanworks_tool:
            self.function_map.update({
                "list_projects": self.leanworks_tool.list_projects,
                "list_tasks": self.leanworks_tool.list_tasks,
                "list_progress_updates": self.leanworks_tool.list_progress_updates,
                "add_task": self.leanworks_tool.add_task,
                "list_users": self.leanworks_tool.list_users,
            })
        
        # Add search function if available and enabled
        if self.search_tool:
            self.function_map["search_knowledge"] = self._search_knowledge_with_deduplication
            logger.info("search_knowledge function added to function_map")
            
        # Add GitLab functions if available and enabled
        if self.gitlab_tool:
            self.function_map.update({
                "list_gitlab_projects": self.gitlab_tool.list_gitlab_projects,
                "list_gitlab_issues": self.gitlab_tool.list_gitlab_issues,
                "find_gitlab_user_by_email": self.gitlab_tool.find_gitlab_user_by_email,
                "list_gitlab_project_members": self.gitlab_tool.list_gitlab_project_members,
                "get_gitlab_project_detail": self.gitlab_tool.get_gitlab_project_detail,
                "list_gitlab_groups": self.gitlab_tool.list_gitlab_groups,
                "get_gitlab_group_detail": self.gitlab_tool.get_gitlab_group_detail,
                "get_issue_detail": self.gitlab_tool.get_issue_detail
            })

        # Log final tool availability for debugging
        logger.info(f"Requested tools: {requested_tools}")
        logger.info(f"Successfully enabled tools: {self.enabled_tools}")
        logger.info(f"Available tools: {[tool.get('name', 'unknown') for tool in self.tools]}")
        logger.info(f"Available functions: {list(self.function_map.keys())}")

    
    def _search_knowledge_with_deduplication(self, query: str):
        """
        Wrapper for search_knowledge that passes the read document IDs for deduplication.
        
        Args:
            query: The search query
            
        Returns:
            SearchResult object with formatted context and data sources
        """
        return self.search_tool.search_knowledge(query, self.read_document_ids)