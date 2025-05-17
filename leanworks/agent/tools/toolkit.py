from leanworks.agent.tools.project import ProjectTool
from leanworks.agent.tools.search import SearchTool

class ToolUse:
    def __init__(self):
        self.project_tool = ProjectTool()
        self.search_tool = SearchTool()
        
        # Define tools as variables
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