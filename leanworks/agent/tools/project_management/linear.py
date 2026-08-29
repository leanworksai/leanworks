import logging
from typing import List, Dict, Optional
import requests
import json

logger = logging.getLogger(__name__)

class LinearTool:
    def __init__(self, api_key: str = None):
        """
        Initialize LinearTool with Linear API credentials.
        
        Args:
            api_key: Linear personal API key
        """
        self.api_key = api_key
        self.base_url = "https://api.linear.app/graphql"
        self.headers = {
            'Authorization': api_key if api_key else None,
            'Content-Type': 'application/json'
        }
        
    def _make_request(self, query: str, variables: Dict = None) -> Dict:
        """
        Make a GraphQL request to the Linear API.
        
        Args:
            query: GraphQL query or mutation string
            variables: Optional variables for the query
            
        Returns:
            Response data as dictionary or error dictionary
        """
        if not self.api_key:
            return {"error": "Linear API key not configured"}
        
        try:
            payload = {
                'query': query
            }
            if variables:
                payload['variables'] = variables
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code >= 400:
                error_msg = f"Linear API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'errors' in error_data:
                        errors = error_data['errors']
                        if errors:
                            error_msg = errors[0].get('message', error_msg)
                    else:
                        error_msg = error_data.get('message', error_msg)
                except:
                    error_msg = response.text or error_msg
                logger.error("Linear API request failed (status=%s)", response.status_code)
                return {"error": error_msg}
            
            result = response.json()
            
            # Check for GraphQL errors in response
            if 'errors' in result:
                error_msg = result['errors'][0].get('message', 'GraphQL error')
                logger.error("Linear GraphQL response contained an error")
                return {"error": error_msg}
            
            return result.get('data', {})
            
        except Exception as e:
            logger.error(f"Error making Linear API request: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def list_issues_property(self):
        description = """
        List Linear issues with optional filters. Returns a list of issues.
        """
        return {
            "type": "custom",
            "name": "linear_list_issues",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "team_id": {
                        "type": "string",
                        "description": "Filter by team ID (optional)"
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Filter by project ID (optional)"
                    },
                    "assignee_id": {
                        "type": "string",
                        "description": "Filter by assignee user ID (optional)"
                    },
                    "state": {
                        "type": "string",
                        "description": "Filter by state name (e.g., 'In Progress', 'Done', 'Backlog'). Optional."
                    },
                    "first": {
                        "type": "integer",
                        "description": "Number of issues to return. Defaults to 50 if not specified (max: 100)."
                    }
                },
                "required": []
            }
        }
    
    def list_issues(self, team_id: str = None, project_id: str = None, assignee_id: str = None, 
                   state: str = None, first: int = 50) -> List[Dict]:
        """
        List Linear issues with optional filters.
        
        Args:
            team_id: Filter by team ID (optional)
            project_id: Filter by project ID (optional)
            assignee_id: Filter by assignee user ID (optional)
            state: Filter by state name (optional)
            first: Number of issues to return (default: 50, max: 100)
            
        Returns:
            List of issue dictionaries
        """
        logger.info(f"Executing list_issues, team_id: {team_id}, project_id: {project_id}, assignee_id: {assignee_id}, state: {state}")
        try:
            # Build filter conditions
            filter_parts = []
            if team_id:
                filter_parts.append(f'team: {{ id: {{ eq: "{team_id}" }} }}')
            if project_id:
                filter_parts.append(f'project: {{ id: {{ eq: "{project_id}" }} }}')
            if assignee_id:
                filter_parts.append(f'assignee: {{ id: {{ eq: "{assignee_id}" }} }}')
            if state:
                filter_parts.append(f'state: {{ name: {{ eq: "{state}" }} }}')
            
            filter_str = f'filter: {{ {", ".join(filter_parts)} }}' if filter_parts else ''
            
            query = f"""
            query {{
                issues({filter_str} first: {min(first, 100)}) {{
                    nodes {{
                        id
                        identifier
                        title
                        description
                        state {{
                            id
                            name
                            type
                        }}
                        assignee {{
                            id
                            name
                            email
                        }}
                        creator {{
                            id
                            name
                            email
                        }}
                        team {{
                            id
                            key
                            name
                        }}
                        project {{
                            id
                            name
                        }}
                        priority
                        labels {{
                            nodes {{
                                id
                                name
                                color
                            }}
                        }}
                        createdAt
                        updatedAt
                        completedAt
                        dueDate
                        url
                    }}
                }}
            }}
            """
            
            result = self._make_request(query)
            
            if 'error' in result:
                return result
            
            issues = result.get('issues', {}).get('nodes', [])
            formatted_issues = []
            
            for issue in issues:
                formatted_issue = {
                    'id': issue.get('id'),
                    'identifier': issue.get('identifier'),
                    'title': issue.get('title'),
                    'description': issue.get('description'),
                    'state': issue.get('state', {}).get('name') if issue.get('state') else None,
                    'state_type': issue.get('state', {}).get('type') if issue.get('state') else None,
                    'assignee': issue.get('assignee', {}).get('name') if issue.get('assignee') else None,
                    'assignee_id': issue.get('assignee', {}).get('id') if issue.get('assignee') else None,
                    'assignee_email': issue.get('assignee', {}).get('email') if issue.get('assignee') else None,
                    'creator': issue.get('creator', {}).get('name') if issue.get('creator') else None,
                    'team': issue.get('team', {}).get('name') if issue.get('team') else None,
                    'team_key': issue.get('team', {}).get('key') if issue.get('team') else None,
                    'project': issue.get('project', {}).get('name') if issue.get('project') else None,
                    'priority': issue.get('priority'),
                    'labels': [label.get('name') for label in issue.get('labels', {}).get('nodes', [])],
                    'created_at': issue.get('createdAt'),
                    'updated_at': issue.get('updatedAt'),
                    'completed_at': issue.get('completedAt'),
                    'due_date': issue.get('dueDate'),
                    'url': issue.get('url')
                }
                formatted_issues.append(formatted_issue)
            
            return formatted_issues
            
        except Exception as e:
            logger.error(f"Error listing issues: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def get_issue_property(self):
        description = """
        Get detailed information about a specific Linear issue by its ID or identifier (e.g., ENG-123).
        """
        return {
            "type": "custom",
            "name": "linear_get_issue",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "issue_id": {
                        "type": "string",
                        "description": "Issue ID or identifier (e.g., 'ENG-123' or UUID)"
                    }
                },
                "required": ["issue_id"]
            }
        }
    
    def get_issue(self, issue_id: str) -> Dict:
        """
        Get detailed information about a specific issue.
        
        Args:
            issue_id: Issue ID or identifier (e.g., "ENG-123" or UUID)
            
        Returns:
            Complete issue details
        """
        logger.info(f"Executing get_issue for id: {issue_id}")
        try:
            # Check if it's an identifier (e.g., ENG-123) or UUID
            # UUIDs are typically 36 characters with dashes in specific positions
            is_uuid = (len(issue_id) == 36 and issue_id.count('-') == 4 and 
                      issue_id[8] == '-' and issue_id[13] == '-' and 
                      issue_id[18] == '-' and issue_id[23] == '-')
            
            if is_uuid:
                # Use id filter for UUID
                query = f"""
                query {{
                    issue(id: "{issue_id}") {{
                        id
                        identifier
                        title
                        description
                        state {{
                            id
                            name
                            type
                        }}
                        assignee {{
                            id
                            name
                            email
                        }}
                        creator {{
                            id
                            name
                            email
                        }}
                        team {{
                            id
                            key
                            name
                        }}
                        project {{
                            id
                            name
                        }}
                        priority
                        labels {{
                            nodes {{
                                id
                                name
                                color
                            }}
                        }}
                        comments {{
                            nodes {{
                                id
                                body
                                user {{
                                    id
                                    name
                                    email
                                }}
                                createdAt
                                updatedAt
                            }}
                        }}
                        createdAt
                        updatedAt
                        completedAt
                        dueDate
                        url
                    }}
                }}
                """
            else:
                # Likely an identifier (e.g., LEA-15), search for it
                # Linear doesn't support identifier filter directly, so we list issues and filter client-side
                # We'll search a reasonable number of issues to find the matching identifier
                query = f"""
                query {{
                    issues(first: 100) {{
                        nodes {{
                            id
                            identifier
                            title
                            description
                            state {{
                                id
                                name
                                type
                            }}
                            assignee {{
                                id
                                name
                                email
                            }}
                            creator {{
                                id
                                name
                                email
                            }}
                            team {{
                                id
                                key
                                name
                            }}
                            project {{
                                id
                                name
                            }}
                            priority
                            labels {{
                                nodes {{
                                    id
                                    name
                                    color
                                }}
                            }}
                            comments {{
                                nodes {{
                                    id
                                    body
                                    user {{
                                        id
                                        name
                                        email
                                    }}
                                    createdAt
                                    updatedAt
                                }}
                            }}
                            createdAt
                            updatedAt
                            completedAt
                            dueDate
                            url
                        }}
                    }}
                }}
                """
            
            result = self._make_request(query)
            
            if 'error' in result:
                return result
            
            # Handle both direct issue query and issues list query
            if 'issue' in result:
                issue = result.get('issue')
            elif 'issues' in result:
                issues = result.get('issues', {}).get('nodes', [])
                if not issues:
                    return {"error": f"Issue '{issue_id}' not found"}
                # If we searched by identifier, find the matching issue
                if not is_uuid:
                    matching_issue = None
                    for i in issues:
                        if i.get('identifier') == issue_id:
                            matching_issue = i
                            break
                    if not matching_issue:
                        return {"error": f"Issue '{issue_id}' not found"}
                    issue = matching_issue
                else:
                    issue = issues[0]
            else:
                return {"error": f"Issue '{issue_id}' not found"}
            
            if not issue:
                return {"error": f"Issue '{issue_id}' not found"}
            
            comments = []
            comment_data = issue.get('comments', {}).get('nodes', [])
            for comment in comment_data:
                comments.append({
                    'id': comment.get('id'),
                    'body': comment.get('body'),
                    'user': comment.get('user', {}).get('name') if comment.get('user') else None,
                    'user_email': comment.get('user', {}).get('email') if comment.get('user') else None,
                    'created_at': comment.get('createdAt'),
                    'updated_at': comment.get('updatedAt')
                })
            
            formatted_issue = {
                'id': issue.get('id'),
                'identifier': issue.get('identifier'),
                'title': issue.get('title'),
                'description': issue.get('description'),
                'state': issue.get('state', {}).get('name') if issue.get('state') else None,
                'state_type': issue.get('state', {}).get('type') if issue.get('state') else None,
                'assignee': issue.get('assignee', {}).get('name') if issue.get('assignee') else None,
                'assignee_id': issue.get('assignee', {}).get('id') if issue.get('assignee') else None,
                'assignee_email': issue.get('assignee', {}).get('email') if issue.get('assignee') else None,
                'creator': issue.get('creator', {}).get('name') if issue.get('creator') else None,
                'team': issue.get('team', {}).get('name') if issue.get('team') else None,
                'team_key': issue.get('team', {}).get('key') if issue.get('team') else None,
                'project': issue.get('project', {}).get('name') if issue.get('project') else None,
                'priority': issue.get('priority'),
                'labels': [label.get('name') for label in issue.get('labels', {}).get('nodes', [])],
                'comments': comments,
                'created_at': issue.get('createdAt'),
                'updated_at': issue.get('updatedAt'),
                'completed_at': issue.get('completedAt'),
                'due_date': issue.get('dueDate'),
                'url': issue.get('url')
            }
            
            return formatted_issue
            
        except Exception as e:
            logger.error(f"Error getting issue: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def create_issue_property(self):
        description = """
        Create a new Linear issue. Returns the created issue details.
        """
        return {
            "type": "custom",
            "name": "linear_create_issue",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "team_id": {
                        "type": "string",
                        "description": "Team ID (required)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Issue title (required)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Issue description (optional)"
                    },
                    "assignee_id": {
                        "type": "string",
                        "description": "Assignee user ID (optional)"
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Project ID (optional)"
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority: 0 (No priority), 1 (Urgent), 2 (High), 3 (Medium), 4 (Low). Optional."
                    },
                    "state_id": {
                        "type": "string",
                        "description": "Initial state ID (optional, defaults to team's default state)"
                    },
                    "label_ids": {
                        "type": "string",
                        "description": "Comma-separated list of label IDs (optional)"
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date in ISO 8601 format (optional)"
                    }
                },
                "required": ["team_id", "title"]
            }
        }
    
    def create_issue(self, team_id: str, title: str, description: str = None, assignee_id: str = None,
                    project_id: str = None, priority: int = None, state_id: str = None,
                    label_ids: str = None, due_date: str = None) -> Dict:
        """
        Create a new Linear issue.
        
        Args:
            team_id: Team ID (required)
            title: Issue title (required)
            description: Issue description (optional)
            assignee_id: Assignee user ID (optional)
            project_id: Project ID (optional)
            priority: Priority level 0-4 (optional)
            state_id: Initial state ID (optional)
            label_ids: Comma-separated label IDs (optional)
            due_date: Due date in ISO 8601 format (optional)
            
        Returns:
            Created issue details
        """
        logger.info(f"Executing create_issue for team: {team_id}, title: {title}")
        try:
            # Build input object
            escaped_title = title.replace('"', '\\"')
            input_fields = [f'teamId: "{team_id}"', f'title: "{escaped_title}"']
            
            if description:
                escaped_desc = description.replace('"', '\\"').replace(chr(10), "\\n").replace(chr(13), "")
                input_fields.append(f'description: "{escaped_desc}"')
            if assignee_id:
                input_fields.append(f'assigneeId: "{assignee_id}"')
            if project_id:
                input_fields.append(f'projectId: "{project_id}"')
            if priority is not None:
                input_fields.append(f'priority: {priority}')
            if state_id:
                input_fields.append(f'stateId: "{state_id}"')
            if label_ids:
                label_list = [f'"{lid.strip()}"' for lid in label_ids.split(',')]
                input_fields.append(f'labelIds: [{", ".join(label_list)}]')
            if due_date:
                input_fields.append(f'dueDate: "{due_date}"')
            
            mutation = f"""
            mutation {{
                issueCreate(input: {{
                    {', '.join(input_fields)}
                }}) {{
                    success
                    issue {{
                        id
                        identifier
                        title
                        description
                        state {{
                            id
                            name
                        }}
                        assignee {{
                            id
                            name
                            email
                        }}
                        team {{
                            id
                            key
                            name
                        }}
                        project {{
                            id
                            name
                        }}
                        priority
                        url
                    }}
                }}
            }}
            """
            
            result = self._make_request(mutation)
            
            if 'error' in result:
                return result
            
            issue_create = result.get('issueCreate', {})
            if not issue_create.get('success'):
                return {"error": "Failed to create issue"}
            
            issue = issue_create.get('issue', {})
            if not issue:
                return {"error": "Issue created but details not returned"}
            
            # Get full issue details
            issue_id = issue.get('id')
            if issue_id:
                return self.get_issue(issue_id)
            
            # Fallback to basic info
            formatted_issue = {
                'id': issue.get('id'),
                'identifier': issue.get('identifier'),
                'title': issue.get('title'),
                'description': issue.get('description'),
                'state': issue.get('state', {}).get('name') if issue.get('state') else None,
                'assignee': issue.get('assignee', {}).get('name') if issue.get('assignee') else None,
                'team': issue.get('team', {}).get('name') if issue.get('team') else None,
                'project': issue.get('project', {}).get('name') if issue.get('project') else None,
                'priority': issue.get('priority'),
                'url': issue.get('url')
            }
            
            return formatted_issue
            
        except Exception as e:
            logger.error(f"Error creating issue: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def update_issue_property(self):
        description = """
        Update an existing Linear issue. Allows updating title, description, assignee, state, priority, and other fields.
        """
        return {
            "type": "custom",
            "name": "linear_update_issue",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "issue_id": {
                        "type": "string",
                        "description": "Issue ID or identifier (e.g., 'ENG-123' or UUID)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Updated title (optional)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Updated description (optional)"
                    },
                    "assignee_id": {
                        "type": "string",
                        "description": "Assignee user ID (optional, use empty string to unassign)"
                    },
                    "state_id": {
                        "type": "string",
                        "description": "State ID to transition to (optional)"
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority: 0 (No priority), 1 (Urgent), 2 (High), 3 (Medium), 4 (Low). Optional."
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Project ID (optional, use empty string to remove from project)"
                    },
                    "label_ids": {
                        "type": "string",
                        "description": "Comma-separated list of label IDs (optional)"
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date in ISO 8601 format (optional, use empty string to remove)"
                    }
                },
                "required": ["issue_id"]
            }
        }
    
    def update_issue(self, issue_id: str, title: str = None, description: str = None,
                    assignee_id: str = None, state_id: str = None, priority: int = None,
                    project_id: str = None, label_ids: str = None, due_date: str = None) -> Dict:
        """
        Update an existing Linear issue.
        
        Args:
            issue_id: Issue ID or identifier
            title: Updated title (optional)
            description: Updated description (optional)
            assignee_id: Assignee user ID (optional, use empty string to unassign)
            state_id: State ID to transition to (optional)
            priority: Priority level 0-4 (optional)
            project_id: Project ID (optional, use empty string to remove from project)
            label_ids: Comma-separated label IDs (optional)
            due_date: Due date in ISO 8601 format (optional, use empty string to remove)
            
        Returns:
            Updated issue details
        """
        logger.info(f"Executing update_issue for id: {issue_id}")
        try:
            # First get the issue to convert identifier to UUID if needed
            issue_info = self.get_issue(issue_id)
            if 'error' in issue_info:
                return issue_info
            
            actual_issue_id = issue_info.get('id')
            if not actual_issue_id:
                return {"error": "Could not determine issue ID"}
            
            # Build input object
            input_fields = []
            
            if title is not None:
                escaped_title = title.replace('"', '\\"')
                input_fields.append(f'title: "{escaped_title}"')
            if description is not None:
                escaped_desc = description.replace('"', '\\"').replace(chr(10), "\\n").replace(chr(13), "")
                input_fields.append(f'description: "{escaped_desc}"')
            if assignee_id is not None:
                if assignee_id == "":
                    input_fields.append('assigneeId: null')
                else:
                    input_fields.append(f'assigneeId: "{assignee_id}"')
            if state_id is not None:
                input_fields.append(f'stateId: "{state_id}"')
            if priority is not None:
                input_fields.append(f'priority: {priority}')
            if project_id is not None:
                if project_id == "":
                    input_fields.append('projectId: null')
                else:
                    input_fields.append(f'projectId: "{project_id}"')
            if label_ids is not None:
                label_list = [f'"{lid.strip()}"' for lid in label_ids.split(',')]
                input_fields.append(f'labelIds: [{", ".join(label_list)}]')
            if due_date is not None:
                if due_date == "":
                    input_fields.append('dueDate: null')
                else:
                    input_fields.append(f'dueDate: "{due_date}"')
            
            if not input_fields:
                # Nothing to update, return current issue
                return issue_info
            
            mutation = f"""
            mutation {{
                issueUpdate(id: "{actual_issue_id}", input: {{
                    {', '.join(input_fields)}
                }}) {{
                    success
                    issue {{
                        id
                    }}
                }}
            }}
            """
            
            result = self._make_request(mutation)
            
            if 'error' in result:
                return result
            
            issue_update = result.get('issueUpdate', {})
            if not issue_update.get('success'):
                return {"error": "Failed to update issue"}
            
            # Return updated issue details
            return self.get_issue(actual_issue_id)
            
        except Exception as e:
            logger.error(f"Error updating issue: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def search_issues_property(self):
        description = """
        Search Linear issues using Linear's query syntax. Returns a list of issues matching the query.
        """
        return {
            "type": "custom",
            "name": "linear_search_issues",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string (e.g., 'title:bug assignee:john')"
                    },
                    "first": {
                        "type": "integer",
                        "description": "Number of issues to return. Defaults to 50 if not specified (max: 100)."
                    }
                },
                "required": ["query"]
            }
        }
    
    def search_issues(self, query: str, first: int = 50) -> List[Dict]:
        """
        Search Linear issues using query syntax.
        
        Args:
            query: Search query string (Linear query syntax, e.g., "title:bug assignee:john")
            first: Number of issues to return (default: 50, max: 100)
            
        Returns:
            List of issue dictionaries
        """
        logger.info("Executing Linear search_issues (query_chars=%d)", len(query))
        try:
            # Use the issues query with filter instead of deprecated issueSearch
            # If query is empty, just list all issues
            if not query or query.strip() == "":
                # Empty query - just list issues
                graphql_query = f"""
                query {{
                    issues(first: {min(first, 100)}) {{
                        nodes {{
                            id
                            identifier
                            title
                            description
                            state {{
                                id
                                name
                                type
                            }}
                            assignee {{
                                id
                                name
                                email
                            }}
                            creator {{
                                id
                                name
                                email
                            }}
                            team {{
                                id
                                key
                                name
                            }}
                            project {{
                                id
                                name
                            }}
                            priority
                            labels {{
                                nodes {{
                                    id
                                    name
                                    color
                                }}
                            }}
                            createdAt
                            updatedAt
                            completedAt
                            dueDate
                            url
                        }}
                    }}
                }}
                """
            else:
                # Parse query and build filter
                # Linear query syntax: "title:bug assignee:john" etc.
                # For now, we'll use a simple text search on title/description
                # Note: Linear's filter API is complex, so we'll do a basic implementation
                escaped_query = query.replace('"', '\\"')
                # Use a simple contains filter on title
                graphql_query = f"""
                query {{
                    issues(filter: {{ title: {{ containsIgnoreCase: "{escaped_query}" }} }}, first: {min(first, 100)}) {{
                        nodes {{
                            id
                            identifier
                            title
                            description
                            state {{
                                id
                                name
                                type
                            }}
                            assignee {{
                                id
                                name
                                email
                            }}
                            creator {{
                                id
                                name
                                email
                            }}
                            team {{
                                id
                                key
                                name
                            }}
                            project {{
                                id
                                name
                            }}
                            priority
                            labels {{
                                nodes {{
                                    id
                                    name
                                    color
                                }}
                            }}
                            createdAt
                            updatedAt
                            completedAt
                            dueDate
                            url
                        }}
                    }}
                }}
                """
            
            result = self._make_request(graphql_query)
            
            if 'error' in result:
                return result
            
            issues = result.get('issues', {}).get('nodes', [])
            formatted_issues = []
            
            for issue in issues:
                formatted_issue = {
                    'id': issue.get('id'),
                    'identifier': issue.get('identifier'),
                    'title': issue.get('title'),
                    'description': issue.get('description'),
                    'state': issue.get('state', {}).get('name') if issue.get('state') else None,
                    'state_type': issue.get('state', {}).get('type') if issue.get('state') else None,
                    'assignee': issue.get('assignee', {}).get('name') if issue.get('assignee') else None,
                    'assignee_id': issue.get('assignee', {}).get('id') if issue.get('assignee') else None,
                    'assignee_email': issue.get('assignee', {}).get('email') if issue.get('assignee') else None,
                    'creator': issue.get('creator', {}).get('name') if issue.get('creator') else None,
                    'team': issue.get('team', {}).get('name') if issue.get('team') else None,
                    'team_key': issue.get('team', {}).get('key') if issue.get('team') else None,
                    'project': issue.get('project', {}).get('name') if issue.get('project') else None,
                    'priority': issue.get('priority'),
                    'labels': [label.get('name') for label in issue.get('labels', {}).get('nodes', [])],
                    'created_at': issue.get('createdAt'),
                    'updated_at': issue.get('updatedAt'),
                    'completed_at': issue.get('completedAt'),
                    'due_date': issue.get('dueDate'),
                    'url': issue.get('url')
                }
                formatted_issues.append(formatted_issue)
            
            return formatted_issues
            
        except Exception as e:
            logger.error(f"Error searching issues: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def list_projects_property(self):
        description = """
        List projects in the Linear workspace. Returns a list of projects with key information.
        """
        return {
            "type": "custom",
            "name": "linear_list_projects",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "first": {
                        "type": "integer",
                        "description": "Number of projects to return. Defaults to 50 if not specified (max: 100)."
                    }
                },
                "required": []
            }
        }
    
    def list_projects(self, first: int = 50) -> List[Dict]:
        """
        List projects in the workspace.
        
        Args:
            first: Number of projects to return (default: 50, max: 100)
            
        Returns:
            List of project dictionaries
        """
        logger.info(f"Executing list_projects")
        try:
            query = f"""
            query {{
                projects(first: {min(first, 100)}) {{
                    nodes {{
                        id
                        name
                        description
                        state
                        progress
                        startDate
                        targetDate
                        teams {{
                            nodes {{
                                id
                                key
                                name
                            }}
                        }}
                        createdAt
                        updatedAt
                        url
                    }}
                }}
            }}
            """
            
            result = self._make_request(query)
            
            if 'error' in result:
                return result
            
            projects = result.get('projects', {}).get('nodes', [])
            formatted_projects = []
            
            for project in projects:
                formatted_project = {
                    'id': project.get('id'),
                    'name': project.get('name'),
                    'description': project.get('description'),
                    'state': project.get('state'),
                    'progress': project.get('progress'),
                    'start_date': project.get('startDate'),
                    'target_date': project.get('targetDate'),
                    'teams': [team.get('name') for team in project.get('teams', {}).get('nodes', [])],
                    'created_at': project.get('createdAt'),
                    'updated_at': project.get('updatedAt'),
                    'url': project.get('url')
                }
                formatted_projects.append(formatted_project)
            
            return formatted_projects
            
        except Exception as e:
            logger.error(f"Error listing projects: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def get_project_property(self):
        description = """
        Get detailed information about a specific Linear project by its ID.
        """
        return {
            "type": "custom",
            "name": "linear_get_project",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Project ID"
                    }
                },
                "required": ["project_id"]
            }
        }
    
    def get_project(self, project_id: str) -> Dict:
        """
        Get detailed information about a specific project.
        
        Args:
            project_id: Project ID
            
        Returns:
            Complete project details
        """
        logger.info(f"Executing get_project for id: {project_id}")
        try:
            query = f"""
            query {{
                project(id: "{project_id}") {{
                    id
                    name
                    description
                    state
                    progress
                    startDate
                    targetDate
                    teams {{
                        nodes {{
                            id
                            key
                            name
                        }}
                    }}
                    createdAt
                    updatedAt
                    url
                }}
            }}
            """
            
            result = self._make_request(query)
            
            if 'error' in result:
                return result
            
            project = result.get('project')
            if not project:
                return {"error": f"Project '{project_id}' not found"}
            
            formatted_project = {
                'id': project.get('id'),
                'name': project.get('name'),
                'description': project.get('description'),
                'state': project.get('state'),
                'progress': project.get('progress'),
                'start_date': project.get('startDate'),
                'target_date': project.get('targetDate'),
                'teams': [team.get('name') for team in project.get('teams', {}).get('nodes', [])],
                'created_at': project.get('createdAt'),
                'updated_at': project.get('updatedAt'),
                'url': project.get('url')
            }
            
            return formatted_project
            
        except Exception as e:
            logger.error(f"Error getting project: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def list_teams_property(self):
        description = """
        List teams in the Linear workspace. Returns a list of teams with key information.
        """
        return {
            "type": "custom",
            "name": "linear_list_teams",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "first": {
                        "type": "integer",
                        "description": "Number of teams to return. Defaults to 50 if not specified (max: 100)."
                    }
                },
                "required": []
            }
        }
    
    def list_teams(self, first: int = 50) -> List[Dict]:
        """
        List teams in the workspace.
        
        Args:
            first: Number of teams to return (default: 50, max: 100)
            
        Returns:
            List of team dictionaries
        """
        logger.info(f"Executing list_teams")
        try:
            query = f"""
            query {{
                teams(first: {min(first, 100)}) {{
                    nodes {{
                        id
                        key
                        name
                        description
                        createdAt
                        updatedAt
                    }}
                }}
            }}
            """
            
            result = self._make_request(query)
            
            if 'error' in result:
                return result
            
            teams = result.get('teams', {}).get('nodes', [])
            formatted_teams = []
            
            for team in teams:
                formatted_team = {
                    'id': team.get('id'),
                    'key': team.get('key'),
                    'name': team.get('name'),
                    'description': team.get('description'),
                    'created_at': team.get('createdAt'),
                    'updated_at': team.get('updatedAt')
                }
                formatted_teams.append(formatted_team)
            
            return formatted_teams
            
        except Exception as e:
            logger.error(f"Error listing teams: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def search_users_property(self):
        description = """
        Search Linear users by name or email. Returns a list of matching users with their IDs and display names.
        This tool is useful for finding the correct Linear user identifier when you have a partial name, email, or slightly different identifier.
        
        IMPORTANT: If this tool returns zero results, always suggest the user confirm the correct Linear user identifier. 
        The user may need to provide the exact user ID or check their Linear profile.
        """
        return {
            "type": "custom",
            "name": "linear_search_users",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (name or email)"
                    },
                    "first": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Defaults to 10 if not specified (max: 50)."
                    }
                },
                "required": ["query"]
            }
        }
    
    def search_users(self, query: str, first: int = 10) -> List[Dict]:
        """
        Search Linear users by name or email.
        
        Args:
            query: Search query (name or email). If empty, returns all users.
            first: Maximum number of results to return (default: 10, max: 50)
            
        Returns:
            List of user dictionaries with id, name, email, etc.
        """
        logger.info("Executing Linear search_users (query_chars=%d)", len(query))
        try:
            # Linear doesn't have a direct user search, so we'll list all users and filter
            # This is a limitation but follows the pattern of other tools
            graphql_query = f"""
            query {{
                users(first: {min(first * 3, 100)}) {{
                    nodes {{
                        id
                        name
                        email
                        displayName
                        active
                    }}
                }}
            }}
            """
            
            result = self._make_request(graphql_query)
            
            if 'error' in result:
                return result
            
            all_users = result.get('users', {}).get('nodes', [])
            
            # If query is empty, return all users (up to first limit)
            if not query or query.strip() == "":
                matching_users = []
                for user in all_users[:first]:
                    matching_users.append({
                        'id': user.get('id'),
                        'name': user.get('name'),
                        'email': user.get('email'),
                        'display_name': user.get('displayName'),
                        'active': user.get('active', True)
                    })
                return matching_users
            
            query_lower = query.lower().strip()
            
            # Filter users based on query
            matching_users = []
            for user in all_users:
                name = user.get('name', '').lower() if user.get('name') else ''
                email = user.get('email', '').lower() if user.get('email') else ''
                display_name = user.get('displayName', '').lower() if user.get('displayName') else ''
                user_id = user.get('id', '').lower()
                
                # Check if query matches any field
                if (query_lower in name or 
                    query_lower in email or 
                    query_lower in display_name or
                    query_lower in user_id):
                    matching_users.append({
                        'id': user.get('id'),
                        'name': user.get('name'),
                        'email': user.get('email'),
                        'display_name': user.get('displayName'),
                        'active': user.get('active', True)
                    })
                    
                    if len(matching_users) >= first:
                        break
            
            if len(matching_users) == 0:
                return {
                    "error": f"No Linear users found matching '{query}'",
                    "message": f"No Linear users found whose name or email contains '{query}'. Please check the spelling or try a different search term.",
                    "users": []
                }
            
            return matching_users
            
        except Exception as e:
            logger.error(f"Error searching users: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
