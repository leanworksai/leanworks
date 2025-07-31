from leanworks.agent.tools.leanworks import LeanworksTool
from leanworks.agent.tools.search import SearchTool
from leanworks.agent.tools.gitlab import GitlabTool
import logging

logger = logging.getLogger(__name__)

class ToolUse:
    def __init__(self, bq_client_wrapper=None, storage_client=None, secret_client=None, read_document_ids=None, gitlab_auth=None):
        """
        Initialize ToolUse with various client connections.
        
        Args:
            bq_client_wrapper: BigQuery client wrapper that has dataset_id attribute
            storage_client: Google Cloud Storage client
            secret_client: Secret management client
            read_document_ids: Set of document IDs already read for deduplication
            gitlab_auth: Dictionary containing gitlab_url and gitlab_token
        """
        self.leanworks_tool = LeanworksTool(bq_client_wrapper) if bq_client_wrapper else None
        
        # Initialize SearchTool with error handling
        self.search_tool = None
        if storage_client and secret_client:
            try:
                self.search_tool = SearchTool(storage_client, secret_client)
                logger.info("SearchTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SearchTool: {str(e)}")
                self.search_tool = None
        else:
            logger.warning("SearchTool not initialized: missing storage_client or secret_client")
            
        # Initialize GitlabTool with error handling  
        self.gitlab_tool = None
        if gitlab_auth:
            try:
                self.gitlab_tool = GitlabTool(gitlab_auth)
                logger.info("GitlabTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GitlabTool: {str(e)}")
                self.gitlab_tool = None
        else:
            logger.warning("GitlabTool not initialized: missing gitlab_auth")
            
        self.read_document_ids = read_document_ids if read_document_ids is not None else set()
        
        # Initialize tools list with leanworks and search tools
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
        else:
            logger.warning("search_knowledge tool not available")
            
        # Add GitLab tools if available
        if self.gitlab_tool:
            self.tools.extend([
                self.gitlab_tool.list_gitlab_projects_property,
                self.gitlab_tool.list_gitlab_issues_property,
                self.gitlab_tool.find_gitlab_user_by_email_property,
                self.gitlab_tool.list_gitlab_project_members_property,
                self.gitlab_tool.get_gitlab_project_detail_property
            ])
        
        # Define function map
        self.function_map = {}
        
        # Add leanworks functions if available
        if self.leanworks_tool:
            self.function_map.update({
                "list_projects": self.leanworks_tool.list_projects,
                "list_tasks": self.leanworks_tool.list_tasks,
                "list_progress_updates": self.leanworks_tool.list_progress_updates,
                "add_task": self.leanworks_tool.add_task,
                "list_users": self.leanworks_tool.list_users,
            })
        
        # Add search function if available
        if self.search_tool:
            self.function_map["search_knowledge"] = self._search_knowledge_with_deduplication
            logger.info("search_knowledge function added to function_map")
        else:
            logger.warning("search_knowledge function not available")
            
        # Add GitLab functions if available
        if self.gitlab_tool:
            self.function_map.update({
                "list_gitlab_projects": self.gitlab_tool.list_gitlab_projects,
                "list_gitlab_issues": self.gitlab_tool.list_gitlab_issues,
                "find_gitlab_user_by_email": self.gitlab_tool.find_gitlab_user_by_email,
                "list_gitlab_project_members": self.gitlab_tool.list_gitlab_project_members,
                "get_gitlab_project_detail": self.gitlab_tool.get_gitlab_project_detail
            })

        # Log final tool availability for debugging
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