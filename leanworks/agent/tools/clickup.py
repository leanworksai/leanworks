import logging
from typing import List, Dict, Optional
import requests
import json

logger = logging.getLogger(__name__)

class ClickUpTool:
    def __init__(self, api_token: str = None):
        """
        Initialize ClickUpTool with ClickUp API credentials.
        
        Args:
            api_token: ClickUp API token
        """
        self.api_token = api_token
        self.base_url = "https://api.clickup.com/api/v2"
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Make an HTTP request to the ClickUp API.
        
        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            endpoint: API endpoint (relative to base_url)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response data as dictionary or error dictionary
        """
        if not self.api_token:
            return {"error": "ClickUp credentials not configured"}
        
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            headers = kwargs.pop('headers', {})
            # ClickUp API uses token directly without "Bearer" prefix
            headers.setdefault('Authorization', self.api_token)
            headers.setdefault('Content-Type', 'application/json')
            
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                **kwargs
            )
            
            if response.status_code >= 400:
                error_msg = f"ClickUp API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('err', error_data.get('error', error_msg))
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get('message', str(error_msg))
                except:
                    error_msg = response.text or error_msg
                logger.error(f"{error_msg} - {response.text[:200]}")
                return {"error": error_msg}
            
            if response.content:
                return response.json()
            return {}
            
        except Exception as e:
            logger.error(f"Error making ClickUp API request: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def search_tasks_property(self):
        description = """
        Search ClickUp tasks with filters. Returns a list of tasks matching the criteria.
        This clickup tool should be called when you need to find tasks based on various criteria like list, assignees, status, etc.
        """
        return {
            "type": "custom",
            "name": "clickup_search_tasks",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "list_id": {
                        "type": "string",
                        "description": "List ID to search tasks in (optional)"
                    },
                    "assignees": {
                        "type": "string",
                        "description": "Comma-separated list of assignee user IDs (optional)"
                    },
                    "statuses": {
                        "type": "string",
                        "description": "Comma-separated list of status names (optional)"
                    },
                    "include_closed": {
                        "type": "boolean",
                        "description": "Whether to include closed tasks. Defaults to false."
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number for pagination. Defaults to 0."
                    }
                },
                "required": []
            }
        }
    
    def search_tasks(self, list_id: str = None, assignees: str = None, statuses: str = None,
                    include_closed: bool = False, page: int = 0) -> List[Dict]:
        """
        Search ClickUp tasks with filters.
        
        Args:
            list_id: List ID to search tasks in (optional)
            assignees: Comma-separated list of assignee user IDs (optional)
            statuses: Comma-separated list of status names (optional)
            include_closed: Whether to include closed tasks
            page: Page number for pagination
            
        Returns:
            List of task dictionaries
        """
        logger.info(f"Executing search_tasks with list_id: {list_id}, assignees: {assignees}, statuses: {statuses}")
        try:
            if list_id:
                # Get tasks from a specific list
                params = {
                    'archived': str(include_closed).lower(),
                    'page': page
                }
                if assignees:
                    params['assignees[]'] = [a.strip() for a in assignees.split(',')]
                if statuses:
                    params['statuses[]'] = [s.strip() for s in statuses.split(',')]
                
                result = self._make_request('GET', f'/list/{list_id}/task', params=params)
            else:
                # Search across all tasks (requires team_id, but we'll use get_authorized_user to get team)
                # For now, return error if no list_id provided
                return {"error": "list_id is required for task search. Use list_spaces and list_lists to find list IDs."}
            
            if 'error' in result:
                return result
            
            tasks = result.get('tasks', [])
            formatted_tasks = []
            
            for task in tasks:
                status_data = task.get('status', {})
                assignees_data = task.get('assignees', [])
                formatted_task = {
                    'id': task.get('id'),
                    'name': task.get('name'),
                    'description': task.get('description'),
                    'status': status_data.get('status') if status_data else None,
                    'assignees': [a.get('username') for a in assignees_data if a.get('username')],
                    'due_date': task.get('due_date'),
                    'date_created': task.get('date_created'),
                    'date_updated': task.get('date_updated'),
                    'url': task.get('url'),
                    'priority': task.get('priority', {}).get('priority') if task.get('priority') else None
                }
                formatted_tasks.append(formatted_task)
            
            return formatted_tasks
            
        except Exception as e:
            logger.error(f"Error searching tasks: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def get_task_property(self):
        description = """
        Get detailed information about a specific ClickUp task by its ID.
        This clickup tool returns complete task details including description, status, assignees, comments, and other fields.
        """
        return {
            "type": "custom",
            "name": "clickup_get_task",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ClickUp task ID"
                    }
                },
                "required": ["task_id"]
            }
        }
    
    def get_task(self, task_id: str) -> Dict:
        """
        Get detailed information about a specific task.
        
        Args:
            task_id: ClickUp task ID
            
        Returns:
            Complete task details
        """
        logger.info(f"Executing get_task for task_id: {task_id}")
        try:
            result = self._make_request('GET', f'/task/{task_id}')
            
            if 'error' in result:
                return result
            
            status_data = result.get('status', {})
            assignees_data = result.get('assignees', [])
            
            # Get comments
            comments_result = self._make_request('GET', f'/task/{task_id}/comment')
            comments = []
            if 'error' not in comments_result:
                comments_list = comments_result.get('comments', [])
                for comment in comments_list:
                    comments.append({
                        'id': comment.get('id'),
                        'user': comment.get('user', {}).get('username') if comment.get('user') else None,
                        'comment': comment.get('comment', []),
                        'date': comment.get('date')
                    })
            
            formatted_task = {
                'id': result.get('id'),
                'name': result.get('name'),
                'description': result.get('description'),
                'status': status_data.get('status') if status_data else None,
                'assignees': [a.get('username') for a in assignees_data if a.get('username')],
                'due_date': result.get('due_date'),
                'date_created': result.get('date_created'),
                'date_updated': result.get('date_updated'),
                'url': result.get('url'),
                'priority': result.get('priority', {}).get('priority') if result.get('priority') else None,
                'tags': [tag.get('name') for tag in result.get('tags', [])],
                'comments': comments,
                'checklists': result.get('checklists', [])
            }
            
            return formatted_task
            
        except Exception as e:
            logger.error(f"Error getting task: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def create_task_property(self):
        description = """
        Create a new ClickUp task in a list. This clickup tool returns the created task ID and details.
        """
        return {
            "type": "custom",
            "name": "clickup_create_task",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "list_id": {
                        "type": "string",
                        "description": "List ID where the task will be created"
                    },
                    "name": {
                        "type": "string",
                        "description": "Task name/title"
                    },
                    "description": {
                        "type": "string",
                        "description": "Task description (optional)"
                    },
                    "assignees": {
                        "type": "string",
                        "description": "Comma-separated list of assignee user IDs (optional)"
                    },
                    "status": {
                        "type": "string",
                        "description": "Status name (optional)"
                    },
                    "due_date": {
                        "type": "integer",
                        "description": "Due date as Unix timestamp in milliseconds (optional)"
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority level: 1 (urgent), 2 (high), 3 (normal), 4 (low). Optional."
                    }
                },
                "required": ["list_id", "name"]
            }
        }
    
    def create_task(self, list_id: str, name: str, description: str = None,
                   assignees: str = None, status: str = None, due_date: int = None,
                   priority: int = None) -> Dict:
        """
        Create a new ClickUp task.
        
        Args:
            list_id: List ID where the task will be created
            name: Task name/title
            description: Task description (optional)
            assignees: Comma-separated list of assignee user IDs (optional)
            status: Status name (optional)
            due_date: Due date as Unix timestamp in milliseconds (optional)
            priority: Priority level (1=urgent, 2=high, 3=normal, 4=low)
            
        Returns:
            Created task details
        """
        logger.info(f"Executing create_task for list_id: {list_id}, name: {name}")
        try:
            payload = {
                "name": name
            }
            
            if description:
                payload["description"] = description
            
            if assignees:
                assignee_list = [a.strip() for a in assignees.split(',')]
                payload["assignees"] = assignee_list
            
            if status:
                payload["status"] = status
            
            if due_date:
                payload["due_date"] = due_date
            
            if priority:
                payload["priority"] = priority
            
            result = self._make_request('POST', f'/list/{list_id}/task', json=payload)
            
            if 'error' in result:
                return result
            
            # Get the created task details
            task_id = result.get('id')
            if task_id:
                return self.get_task(task_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating task: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def update_task_property(self):
        description = """
        Update an existing ClickUp task. This clickup tool allows you to update name, description, assignees, status, due_date, or priority.
        """
        return {
            "type": "custom",
            "name": "clickup_update_task",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ClickUp task ID"
                    },
                    "name": {
                        "type": "string",
                        "description": "Updated task name/title (optional)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Updated description (optional)"
                    },
                    "assignees": {
                        "type": "string",
                        "description": "Comma-separated list of assignee user IDs (optional)"
                    },
                    "status": {
                        "type": "string",
                        "description": "Status name to update to (optional)"
                    },
                    "due_date": {
                        "type": "integer",
                        "description": "Due date as Unix timestamp in milliseconds (optional)"
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority level: 1 (urgent), 2 (high), 3 (normal), 4 (low). Optional."
                    }
                },
                "required": ["task_id"]
            }
        }
    
    def update_task(self, task_id: str, name: str = None, description: str = None,
                   assignees: str = None, status: str = None, due_date: int = None,
                   priority: int = None) -> Dict:
        """
        Update an existing ClickUp task.
        
        Args:
            task_id: ClickUp task ID
            name: Updated task name (optional)
            description: Updated description (optional)
            assignees: Comma-separated list of assignee user IDs (optional)
            status: Status name to update to (optional)
            due_date: Due date as Unix timestamp in milliseconds (optional)
            priority: Priority level (optional)
            
        Returns:
            Updated task details
        """
        logger.info(f"Executing update_task for task_id: {task_id}")
        try:
            payload = {}
            
            if name:
                payload["name"] = name
            
            if description:
                payload["description"] = description
            
            if assignees:
                assignee_list = [a.strip() for a in assignees.split(',')]
                payload["assignees"] = assignee_list
            
            if status:
                payload["status"] = status
            
            if due_date is not None:
                payload["due_date"] = due_date
            
            if priority:
                payload["priority"] = priority
            
            if not payload:
                return {"error": "At least one field must be provided for update"}
            
            result = self._make_request('PUT', f'/task/{task_id}', json=payload)
            
            if 'error' in result:
                return result
            
            # Return updated task details
            return self.get_task(task_id)
            
        except Exception as e:
            logger.error(f"Error updating task: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def add_comment_property(self):
        description = """
        Add a comment to a ClickUp task. This clickup tool allows you to add comments to existing tasks.
        """
        return {
            "type": "custom",
            "name": "clickup_add_comment",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ClickUp task ID"
                    },
                    "comment_text": {
                        "type": "string",
                        "description": "Comment text to add"
                    }
                },
                "required": ["task_id", "comment_text"]
            }
        }
    
    def add_comment(self, task_id: str, comment_text: str) -> Dict:
        """
        Add a comment to a ClickUp task.
        
        Args:
            task_id: ClickUp task ID
            comment_text: Comment text
            
        Returns:
            Comment details
        """
        logger.info(f"Executing add_comment for task_id: {task_id}")
        try:
            payload = {
                "comment_text": comment_text
            }
            
            result = self._make_request('POST', f'/task/{task_id}/comment', json=payload)
            
            if 'error' in result:
                return result
            
            formatted_comment = {
                'id': result.get('id'),
                'user': result.get('user', {}).get('username') if result.get('user') else None,
                'comment': result.get('comment', []),
                'date': result.get('date')
            }
            
            return formatted_comment
            
        except Exception as e:
            logger.error(f"Error adding comment: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def list_spaces_property(self):
        description = """
        List all spaces in the ClickUp workspace. This clickup tool is useful for finding space IDs to navigate to lists and tasks.
        """
        return {
            "type": "custom",
            "name": "clickup_list_spaces",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "team_id": {
                        "type": "string",
                        "description": "Team/workspace ID (optional, will use first team if not provided)"
                    }
                },
                "required": []
            }
        }
    
    def list_spaces(self, team_id: str = None) -> List[Dict]:
        """
        List all spaces in the workspace.
        
        Args:
            team_id: Team/workspace ID (optional)
            
        Returns:
            List of space dictionaries
        """
        logger.info(f"Executing list_spaces with team_id: {team_id}")
        try:
            # If no team_id provided, get the first team
            if not team_id:
                teams_result = self._make_request('GET', '/team')
                if 'error' in teams_result:
                    return teams_result
                teams = teams_result.get('teams', [])
                if not teams:
                    return {"error": "No teams found in workspace"}
                team_id = teams[0].get('id')
            
            result = self._make_request('GET', f'/team/{team_id}/space')
            
            if 'error' in result:
                return result
            
            spaces = result.get('spaces', [])
            formatted_spaces = []
            
            for space in spaces:
                formatted_space = {
                    'id': space.get('id'),
                    'name': space.get('name'),
                    'private': space.get('private', False),
                    'archived': space.get('archived', False)
                }
                formatted_spaces.append(formatted_space)
            
            return formatted_spaces
            
        except Exception as e:
            logger.error(f"Error listing spaces: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def list_lists_property(self):
        description = """
        List all lists in a ClickUp space or folder. This clickup tool is useful for finding list IDs to create or search tasks.
        """
        return {
            "type": "custom",
            "name": "clickup_list_lists",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "space_id": {
                        "type": "string",
                        "description": "Space ID (required if folder_id not provided)"
                    },
                    "folder_id": {
                        "type": "string",
                        "description": "Folder ID (optional, if provided will list lists in folder)"
                    },
                    "archived": {
                        "type": "boolean",
                        "description": "Whether to include archived lists. Defaults to false."
                    }
                },
                "required": []
            }
        }
    
    def list_lists(self, space_id: str = None, folder_id: str = None, archived: bool = False) -> List[Dict]:
        """
        List all lists in a space or folder.
        
        Args:
            space_id: Space ID (required if folder_id not provided)
            folder_id: Folder ID (optional)
            archived: Whether to include archived lists
            
        Returns:
            List of list dictionaries
        """
        logger.info(f"Executing list_lists with space_id: {space_id}, folder_id: {folder_id}")
        try:
            if folder_id:
                result = self._make_request('GET', f'/folder/{folder_id}/list', params={'archived': str(archived).lower()})
            elif space_id:
                result = self._make_request('GET', f'/space/{space_id}/list', params={'archived': str(archived).lower()})
            else:
                return {"error": "Either space_id or folder_id must be provided"}
            
            if 'error' in result:
                return result
            
            lists = result.get('lists', [])
            formatted_lists = []
            
            for list_item in lists:
                formatted_list = {
                    'id': list_item.get('id'),
                    'name': list_item.get('name'),
                    'archived': list_item.get('archived', False),
                    'status': list_item.get('status', {})
                }
                formatted_lists.append(formatted_list)
            
            return formatted_lists
            
        except Exception as e:
            logger.error(f"Error listing lists: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

