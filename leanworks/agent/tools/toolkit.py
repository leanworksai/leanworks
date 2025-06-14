from leanworks.agent.tools.project import ProjectTool
from leanworks.agent.tools.search import SearchTool

class ToolUse:
    def __init__(self, bq_client=None):
        """
        Initialize ToolUse with a BigQuery client.
        
        Args:
            bq_client: BigQuery client object that has dataset_id attribute
        """
        self.project_tool = ProjectTool(bq_client) if bq_client else None
        self.search_tool = SearchTool()
        
        # Define tools as variables
        if self.project_tool:
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
                "search_knowledge": self.search_tool.search_knowledge
            }
        else:
            # If no bq_client is provided, only include search tool
            self.tools = [
                self.search_tool.search_knowledge_property
            ]
            
            self.function_map = {
                "search_knowledge": self.search_tool.search_knowledge
            }