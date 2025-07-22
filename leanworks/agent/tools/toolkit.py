from leanworks.agent.tools.project import ProjectTool
from leanworks.agent.tools.search import SearchTool

class ToolUse:
    def __init__(self, bq_client_wrapper=None, storage_client=None, secret_client=None, read_document_ids=None):
        """
        Initialize ToolUse with a BigQuery client.
        
        Args:
            bq_client: BigQuery client object that has dataset_id attribute
            read_document_ids: Set of document IDs already read for deduplication
        """
        self.project_tool = ProjectTool(bq_client_wrapper) if bq_client_wrapper else None
        self.search_tool = SearchTool(storage_client, secret_client)
        self.read_document_ids = read_document_ids if read_document_ids is not None else set()
        
        self.tools = [
            self.project_tool.list_projects_property,
            self.project_tool.list_tasks_property,
            self.project_tool.list_progress_updates_property,
            self.project_tool.add_task_property,
            self.project_tool.list_users_property,
            self.search_tool.search_knowledge_property
        ]
        
        # Define function map as variable
        self.function_map = {
            "list_projects": self.project_tool.list_projects,
            "list_tasks": self.project_tool.list_tasks,
            "list_progress_updates": self.project_tool.list_progress_updates,
            "add_task": self.project_tool.add_task,
            "list_users": self.project_tool.list_users,
            "search_knowledge": self._search_knowledge_with_deduplication
        }

    
    def _search_knowledge_with_deduplication(self, query: str):
        """
        Wrapper for search_knowledge that passes the read document IDs for deduplication.
        
        Args:
            query: The search query
            
        Returns:
            SearchResult object with formatted context and data sources
        """
        return self.search_tool.search_knowledge(query, self.read_document_ids)