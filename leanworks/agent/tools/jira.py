import logging
from typing import List, Dict, Optional
import requests
from requests.auth import HTTPBasicAuth
import json

logger = logging.getLogger(__name__)

class JiraTool:
    def __init__(self, email: str = None, domain: str = None, api_token: str = None):
        """
        Initialize JiraTool with Jira API credentials.
        
        Args:
            email: Jira account email address
            domain: Jira domain (e.g., "your-domain.atlassian.net")
            api_token: Jira API token
        """
        self.email = email
        self.domain = domain
        self.api_token = api_token
        self.base_url = f"https://{domain}/rest/api/3" if domain else None
        self.auth = HTTPBasicAuth(email, api_token) if email and api_token else None
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Make an HTTP request to the Jira API.
        
        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            endpoint: API endpoint (relative to base_url)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response data as dictionary or error dictionary
        """
        if not self.auth or not self.base_url:
            return {"error": "Jira credentials not configured"}
        
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            headers = kwargs.pop('headers', {})
            headers.setdefault('Content-Type', 'application/json')
            headers.setdefault('Accept', 'application/json')
            
            response = requests.request(
                method=method,
                url=url,
                auth=self.auth,
                headers=headers,
                **kwargs
            )
            
            if response.status_code >= 400:
                error_msg = f"Jira API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('errorMessages', [error_msg])[0]
                except:
                    error_msg = response.text or error_msg
                logger.error(f"{error_msg} - {response.text[:200]}")
                return {"error": error_msg}
            
            if response.content:
                return response.json()
            return {}
            
        except Exception as e:
            logger.error(f"Error making Jira API request: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def search_issues_property(self):
        description = """
        Search Jira issues using JQL (Jira Query Language). Returns a list of issues matching the query.
        This tool should be called when you need to find issues based on various criteria like project, status, assignee, etc.
        """
        return {
            "type": "custom",
            "name": "search_issues",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "jql": {
                        "type": "string",
                        "description": "JQL query string (e.g., 'project = PROJ AND status = Open')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of issues to return. Defaults to 50 if not specified."
                    },
                    "fields": {
                        "type": "string",
                        "description": "Comma-separated list of fields to return (e.g., 'summary,status,assignee'). If not specified, returns common fields."
                    }
                },
                "required": ["jql"]
            }
        }
    
    def search_issues(self, jql: str, max_results: int = 50, fields: str = None) -> List[Dict]:
        """
        Search Jira issues using JQL.
        
        Args:
            jql: JQL query string (must include a restriction like project, updated date, etc.)
            max_results: Maximum number of issues to return
            fields: Comma-separated list of fields to return
            
        Returns:
            List of issue dictionaries
        """
        logger.info(f"Executing search_issues with JQL: {jql}, max_results: {max_results}")
        try:
            # Use the new /search/jql endpoint (migration from deprecated /search)
            # The endpoint requires bounded JQL queries (with restrictions)
            # Reference: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issue-search/#api-group-issue-search
            # Build fields list
            if fields:
                # Convert comma-separated string to list
                field_list = [f.strip() for f in fields.split(',')]
            else:
                # Default fields to return
                field_list = ['summary', 'status', 'assignee', 'reporter', 'created', 'updated', 'priority', 'issuetype', 'project']
            
            # The /search/jql endpoint payload structure (does not use startAt, uses nextPageToken for pagination)
            payload = {
                'jql': jql,
                'maxResults': max_results,
                'fields': field_list
            }
            
            result = self._make_request('POST', '/search/jql', json=payload)
            
            if 'error' in result:
                return result
            
            issues = result.get('issues', [])
            formatted_issues = []
            
            for issue in issues:
                fields_data = issue.get('fields', {})
                formatted_issue = {
                    'key': issue.get('key'),
                    'summary': fields_data.get('summary'),
                    'status': fields_data.get('status', {}).get('name') if fields_data.get('status') else None,
                    'assignee': fields_data.get('assignee', {}).get('displayName') if fields_data.get('assignee') else None,
                    'reporter': fields_data.get('reporter', {}).get('displayName') if fields_data.get('reporter') else None,
                    'created': fields_data.get('created'),
                    'updated': fields_data.get('updated'),
                    'priority': fields_data.get('priority', {}).get('name') if fields_data.get('priority') else None,
                    'issue_type': fields_data.get('issuetype', {}).get('name') if fields_data.get('issuetype') else None,
                    'project': fields_data.get('project', {}).get('key') if fields_data.get('project') else None,
                    'description': fields_data.get('description'),
                }
                formatted_issues.append(formatted_issue)
            
            return formatted_issues
            
        except Exception as e:
            logger.error(f"Error searching issues: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def get_issue_property(self):
        description = """
        Get detailed information about a specific Jira issue by its key (e.g., PROJ-123).
        Returns complete issue details including description, status, assignee, comments, and other fields.
        """
        return {
            "type": "custom",
            "name": "get_issue",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Jira issue key (e.g., 'PROJ-123')"
                    }
                },
                "required": ["issue_key"]
            }
        }
    
    def get_issue(self, issue_key: str) -> Dict:
        """
        Get detailed information about a specific issue.
        
        Args:
            issue_key: Jira issue key (e.g., "PROJ-123")
            
        Returns:
            Complete issue details
        """
        logger.info(f"Executing get_issue for key: {issue_key}")
        try:
            result = self._make_request('GET', f'/issue/{issue_key}')
            
            if 'error' in result:
                return result
            
            fields_data = result.get('fields', {})
            
            # Get comments
            comments = []
            comment_data = fields_data.get('comment', {})
            if comment_data:
                comment_list = comment_data.get('comments', [])
                for comment in comment_list:
                    comments.append({
                        'author': comment.get('author', {}).get('displayName') if comment.get('author') else None,
                        'body': comment.get('body'),
                        'created': comment.get('created'),
                        'updated': comment.get('updated')
                    })
            
            formatted_issue = {
                'key': result.get('key'),
                'summary': fields_data.get('summary'),
                'description': fields_data.get('description'),
                'status': fields_data.get('status', {}).get('name') if fields_data.get('status') else None,
                'assignee': fields_data.get('assignee', {}).get('displayName') if fields_data.get('assignee') else None,
                'reporter': fields_data.get('reporter', {}).get('displayName') if fields_data.get('reporter') else None,
                'created': fields_data.get('created'),
                'updated': fields_data.get('updated'),
                'priority': fields_data.get('priority', {}).get('name') if fields_data.get('priority') else None,
                'issue_type': fields_data.get('issuetype', {}).get('name') if fields_data.get('issuetype') else None,
                'project': fields_data.get('project', {}).get('key') if fields_data.get('project') else None,
                'project_name': fields_data.get('project', {}).get('name') if fields_data.get('project') else None,
                'labels': fields_data.get('labels', []),
                'comments': comments,
                'resolution': fields_data.get('resolution', {}).get('name') if fields_data.get('resolution') else None,
            }
            
            return formatted_issue
            
        except Exception as e:
            logger.error(f"Error getting issue: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def create_issue_property(self):
        description = """
        Create a new Jira issue. Returns the created issue key and details.
        """
        return {
            "type": "custom",
            "name": "create_issue",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_key": {
                        "type": "string",
                        "description": "Project key (e.g., 'PROJ')"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Issue summary/title"
                    },
                    "issue_type": {
                        "type": "string",
                        "description": "Issue type (e.g., 'Task', 'Bug', 'Story', 'Epic'). Defaults to 'Task' if not specified."
                    },
                    "description": {
                        "type": "string",
                        "description": "Issue description (optional)"
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Assignee email address or account ID (optional)"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority name (e.g., 'Highest', 'High', 'Medium', 'Low', 'Lowest'). Optional."
                    }
                },
                "required": ["project_key", "summary"]
            }
        }
    
    def create_issue(self, project_key: str, summary: str, issue_type: str = "Task", 
                     description: str = None, assignee: str = None, priority: str = None) -> Dict:
        """
        Create a new Jira issue.
        
        Args:
            project_key: Project key (e.g., "PROJ")
            summary: Issue summary/title
            issue_type: Issue type (defaults to "Task")
            description: Issue description (optional)
            assignee: Assignee email or account ID (optional)
            priority: Priority name (optional)
            
        Returns:
            Created issue key and details
        """
        logger.info(f"Executing create_issue for project: {project_key}, summary: {summary}")
        try:
            payload = {
                "fields": {
                    "project": {
                        "key": project_key
                    },
                    "summary": summary,
                    "issuetype": {
                        "name": issue_type
                    }
                }
            }
            
            if description:
                payload["fields"]["description"] = {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": description
                                }
                            ]
                        }
                    ]
                }
            
            if assignee:
                # Support both email addresses and account IDs
                if "@" in assignee:
                    payload["fields"]["assignee"] = {
                        "emailAddress": assignee
                    }
                else:
                    payload["fields"]["assignee"] = {
                        "accountId": assignee
                    }
            
            if priority:
                payload["fields"]["priority"] = {
                    "name": priority
                }
            
            result = self._make_request('POST', '/issue', json=payload)
            
            if 'error' in result:
                return result
            
            # Get the created issue details
            issue_key = result.get('key')
            if issue_key:
                return self.get_issue(issue_key)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating issue: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def update_issue_property(self):
        description = """
        Update an existing Jira issue. You can update summary, description, assignee, priority, or status.
        """
        return {
            "type": "custom",
            "name": "update_issue",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Jira issue key (e.g., 'PROJ-123')"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Updated summary/title (optional)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Updated description (optional)"
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Assignee email address or account ID (optional)"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority name (e.g., 'Highest', 'High', 'Medium', 'Low', 'Lowest'). Optional."
                    },
                    "status": {
                        "type": "string",
                        "description": "Status name to transition to (e.g., 'In Progress', 'Done'). Optional."
                    }
                },
                "required": ["issue_key"]
            }
        }
    
    def update_issue(self, issue_key: str, summary: str = None, description: str = None,
                    assignee: str = None, priority: str = None, status: str = None) -> Dict:
        """
        Update an existing Jira issue.
        
        Args:
            issue_key: Jira issue key (e.g., "PROJ-123")
            summary: Updated summary (optional)
            description: Updated description (optional)
            assignee: Assignee email or account ID (optional)
            priority: Priority name (optional)
            status: Status name to transition to (optional)
            
        Returns:
            Updated issue details
        """
        logger.info(f"Executing update_issue for key: {issue_key}")
        try:
            payload = {
                "fields": {}
            }
            
            if summary:
                payload["fields"]["summary"] = summary
            
            if description:
                payload["fields"]["description"] = {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": description
                                }
                            ]
                        }
                    ]
                }
            
            if assignee:
                # Support both email addresses and account IDs
                if "@" in assignee:
                    payload["fields"]["assignee"] = {
                        "emailAddress": assignee
                    }
                else:
                    payload["fields"]["assignee"] = {
                        "accountId": assignee
                    }
            
            if priority:
                payload["fields"]["priority"] = {
                    "name": priority
                }
            
            # Update fields first
            if payload["fields"]:
                result = self._make_request('PUT', f'/issue/{issue_key}', json=payload)
                if 'error' in result:
                    return result
            
            # Handle status transition separately if provided
            if status:
                # First, get available transitions
                transitions_result = self._make_request('GET', f'/issue/{issue_key}/transitions')
                if 'error' not in transitions_result:
                    transitions = transitions_result.get('transitions', [])
                    target_transition = None
                    for transition in transitions:
                        if transition.get('to', {}).get('name') == status:
                            target_transition = transition
                            break
                    
                    if target_transition:
                        transition_payload = {
                            "transition": {
                                "id": target_transition.get('id')
                            }
                        }
                        transition_result = self._make_request('POST', f'/issue/{issue_key}/transitions', json=transition_payload)
                        if 'error' in transition_result:
                            return transition_result
                    else:
                        logger.warning(f"Status transition to '{status}' not available for issue {issue_key}")
            
            # Return updated issue details
            return self.get_issue(issue_key)
            
        except Exception as e:
            logger.error(f"Error updating issue: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def add_comment_property(self):
        description = """
        Add a comment to a Jira issue.
        """
        return {
            "type": "custom",
            "name": "add_comment",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "issue_key": {
                        "type": "string",
                        "description": "Jira issue key (e.g., 'PROJ-123')"
                    },
                    "comment_body": {
                        "type": "string",
                        "description": "Comment text to add"
                    }
                },
                "required": ["issue_key", "comment_body"]
            }
        }
    
    def add_comment(self, issue_key: str, comment_body: str) -> Dict:
        """
        Add a comment to a Jira issue.
        
        Args:
            issue_key: Jira issue key (e.g., "PROJ-123")
            comment_body: Comment text
            
        Returns:
            Comment details
        """
        logger.info(f"Executing add_comment for key: {issue_key}")
        try:
            payload = {
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": comment_body
                                }
                            ]
                        }
                    ]
                }
            }
            
            result = self._make_request('POST', f'/issue/{issue_key}/comment', json=payload)
            
            if 'error' in result:
                return result
            
            formatted_comment = {
                'id': result.get('id'),
                'author': result.get('author', {}).get('displayName') if result.get('author') else None,
                'body': result.get('body'),
                'created': result.get('created'),
                'updated': result.get('updated')
            }
            
            return formatted_comment
            
        except Exception as e:
            logger.error(f"Error adding comment: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

