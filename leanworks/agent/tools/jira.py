import logging
from typing import List, Dict, Optional, Tuple
import requests
from requests.auth import HTTPBasicAuth
import json
from difflib import SequenceMatcher

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
        
        IMPORTANT: If this tool returns zero results when filtering by assignee or reporter, it may mean the Jira user identifier is incorrect. 
        Always suggest the user confirm the correct Jira username/account ID and consider using jira_search_users to find the correct user identifier.
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
                # Try approximate matching if exact match might fail
                match_result = self.find_matching_user(assignee, max_variations=3)
                
                if match_result['match'] and match_result['confidence'] >= 0.9:
                    # High confidence - use the matched user
                    matched_user = match_result['match']
                    if "@" in matched_user.get('emailAddress', ''):
                        payload["fields"]["assignee"] = {
                            "emailAddress": matched_user['emailAddress']
                        }
                    elif matched_user.get('accountId'):
                        payload["fields"]["assignee"] = {
                            "accountId": matched_user['accountId']
                        }
                    else:
                        # Fallback to original
                        if "@" in assignee:
                            payload["fields"]["assignee"] = {
                                "emailAddress": assignee
                            }
                        else:
                            payload["fields"]["assignee"] = {
                                "accountId": assignee
                            }
                elif match_result['match'] and match_result['confidence'] >= 0.7:
                    # Medium confidence - return error with suggestion
                    matched_user = match_result['match']
                    alternatives = [matched_user.get('displayName', matched_user.get('emailAddress', ''))] + \
                                  [alt.get('displayName', alt.get('emailAddress', '')) for alt in match_result['alternatives'][:2]]
                    return {
                        "error": f"Could not find exact match for assignee '{assignee}'. Found possible match: {matched_user.get('displayName', matched_user.get('emailAddress', ''))}",
                        "suggestion": f"Did you mean: {', '.join(alternatives)}?",
                        "match_result": match_result
                    }
                else:
                    # Low confidence or no match - try original, but return error with suggestions if it fails
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
                # Try approximate matching if exact match might fail
                match_result = self.find_matching_user(assignee, max_variations=3)
                
                if match_result['match'] and match_result['confidence'] >= 0.9:
                    # High confidence - use the matched user
                    matched_user = match_result['match']
                    if "@" in matched_user.get('emailAddress', ''):
                        payload["fields"]["assignee"] = {
                            "emailAddress": matched_user['emailAddress']
                        }
                    elif matched_user.get('accountId'):
                        payload["fields"]["assignee"] = {
                            "accountId": matched_user['accountId']
                        }
                    else:
                        # Fallback to original
                        if "@" in assignee:
                            payload["fields"]["assignee"] = {
                                "emailAddress": assignee
                            }
                        else:
                            payload["fields"]["assignee"] = {
                                "accountId": assignee
                            }
                elif match_result['match'] and match_result['confidence'] >= 0.7:
                    # Medium confidence - return error with suggestion
                    matched_user = match_result['match']
                    alternatives = [matched_user.get('displayName', matched_user.get('emailAddress', ''))] + \
                                  [alt.get('displayName', alt.get('emailAddress', '')) for alt in match_result['alternatives'][:2]]
                    return {
                        "error": f"Could not find exact match for assignee '{assignee}'. Found possible match: {matched_user.get('displayName', matched_user.get('emailAddress', ''))}",
                        "suggestion": f"Did you mean: {', '.join(alternatives)}?",
                        "match_result": match_result
                    }
                else:
                    # Low confidence or no match - try original, but return error with suggestions if it fails
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
    
    def _normalize_identifier(self, identifier: str) -> str:
        """Normalize identifier for comparison (lowercase, remove special chars)."""
        if not identifier:
            return ""
        normalized = identifier.lower().strip()
        # Remove common special characters for comparison
        normalized = normalized.replace('.', '').replace('-', '').replace('_', '')
        return normalized
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity score between two strings (0.0 to 1.0)."""
        if not str1 or not str2:
            return 0.0
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _generate_identifier_variations(self, identifier: str, max_variations: int = 3) -> List[str]:
        """Generate identifier variations for approximate matching."""
        if not identifier:
            return []
        
        variations = []
        seen = set()
        
        # Add normalized version
        normalized = self._normalize_identifier(identifier)
        if normalized and normalized not in seen:
            variations.append(normalized)
            seen.add(normalized)
        
        # Add lowercase version
        lower = identifier.lower().strip()
        if lower != identifier and lower not in seen and len(variations) < max_variations:
            variations.append(lower)
            seen.add(lower)
        
        # Remove dots
        no_dots = identifier.replace('.', '')
        if no_dots != identifier and no_dots not in seen and len(variations) < max_variations:
            variations.append(no_dots)
            seen.add(no_dots)
        
        # Remove hyphens
        no_hyphens = identifier.replace('-', '')
        if no_hyphens != identifier and no_hyphens not in seen and len(variations) < max_variations:
            variations.append(no_hyphens)
            seen.add(no_hyphens)
        
        # First part before @ (for emails)
        if '@' in identifier:
            username_part = identifier.split('@')[0]
            if username_part not in seen and len(variations) < max_variations:
                variations.append(username_part)
                seen.add(username_part)
        
        return variations[:max_variations]
    
    @property
    def search_users_property(self):
        description = """
        Search Jira users by name, email, or username. Returns a list of matching users with their account IDs and display names.
        This tool is useful for finding the correct Jira user identifier when you have a partial name, email, or slightly different identifier.
        
        IMPORTANT: If this tool returns zero results, always suggest the user confirm the correct Jira username/account ID. 
        The user may need to provide the exact username, email, or account ID, or check their Jira profile.
        """
        return {
            "type": "custom",
            "name": "jira_search_users",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (name, email, or username)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Defaults to 10 if not specified (max: 50)."
                    }
                },
                "required": ["query"]
            }
        }
    
    def search_users(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search Jira users by name, email, or username.
        Uses GET /rest/api/2/user for exact matches, then GET /rest/api/2/users to list and filter users.
        
        Args:
            query: Search query (name, email, or username) - will try exact match first, then list and filter
            max_results: Maximum number of results to return (default: 10, max: 50)
            
        Returns:
            List of user dictionaries with accountId, displayName, emailAddress, etc.
        """
        logger.info(f"Executing search_users with query: {query}")
        try:
            if not query:
                return []
            
            # Clean the query - remove whitespace
            query = query.strip()
            query_lower = query.lower()
            
            # First, try to get the user directly by accountId or username if it looks like an exact identifier
            # Jira accountIds are typically long alphanumeric strings, usernames can vary
            # Try accountId first (if it looks like one), then username
            is_likely_account_id = (
                len(query) > 10 and 
                not ' ' in query and 
                not '@' in query
            )
            
            if is_likely_account_id:
                logger.info(f"Query looks like accountId, trying GET /rest/api/2/user?accountId={query}")
                user_result = self._make_request('GET', '/user', params={'accountId': query})
                
                if 'error' not in user_result and user_result.get('accountId'):
                    # Successfully found user by accountId
                    formatted_user = {
                        'accountId': user_result.get('accountId'),
                        'displayName': user_result.get('displayName'),
                        'emailAddress': user_result.get('emailAddress'),
                        'active': user_result.get('active', True)
                    }
                    logger.info(f"Found user by accountId: {formatted_user['accountId']}")
                    return [formatted_user]
            
            # Try username if it doesn't look like an accountId
            if not is_likely_account_id and not '@' in query:
                logger.info(f"Trying GET /rest/api/2/user?username={query}")
                user_result = self._make_request('GET', '/user', params={'username': query})
                
                if 'error' not in user_result and user_result.get('accountId'):
                    # Successfully found user by username
                    formatted_user = {
                        'accountId': user_result.get('accountId'),
                        'displayName': user_result.get('displayName'),
                        'emailAddress': user_result.get('emailAddress'),
                        'active': user_result.get('active', True)
                    }
                    logger.info(f"Found user by username: {formatted_user['accountId']}")
                    return [formatted_user]
            
            # Use GET /rest/api/2/users to list users and filter client-side
            logger.info(f"Using GET /rest/api/2/users to list and filter users for query: {query}")
            all_matches = []
            max_iterations = 10  # Limit to 10 iterations (up to 1000 users) for performance
            max_results_param = 100  # Maximum per page for GET /rest/api/2/users
            start_at = 0
            
            for iteration in range(max_iterations):
                params = {
                    'startAt': start_at,
                    'maxResults': max_results_param
                }
                
                result = self._make_request('GET', '/users', params=params)
                
                if 'error' in result:
                    # If we get an error and we haven't found any matches yet, return the error
                    if iteration == 0 and len(all_matches) == 0:
                        return result
                    # Otherwise, break and return what we have
                    break
                
                users = result if isinstance(result, list) else []
                
                if not users:
                    # No more users to process
                    break
                
                # Update startAt for next iteration
                start_at += len(users)
                
                # Score and filter users based on query
                for user in users:
                    account_id = user.get('accountId', '')
                    display_name = user.get('displayName', '')
                    email = user.get('emailAddress', '')
                    
                    if not account_id:
                        continue
                    
                    # Calculate match score
                    score = 0
                    display_name_lower = display_name.lower() if display_name else ''
                    email_lower = email.lower() if email else ''
                    account_id_lower = account_id.lower()
                    
                    # Exact match gets highest score
                    if account_id_lower == query_lower:
                        score = 1000
                    elif display_name and display_name_lower == query_lower:
                        score = 900
                    elif email and email_lower == query_lower:
                        score = 800
                    # Starts with query
                    elif display_name and display_name_lower.startswith(query_lower):
                        score = 500 - len(display_name)
                    elif email and email_lower.startswith(query_lower):
                        score = 400
                    # Contains query
                    elif display_name and query_lower in display_name_lower:
                        score = 200 - len(display_name)
                    elif email and query_lower in email_lower:
                        score = 150
                    elif query_lower in account_id_lower:
                        score = 100
                    else:
                        continue  # Skip users that don't match at all
                    
                    formatted_user = {
                        'accountId': account_id,
                        'displayName': display_name,
                        'emailAddress': email,
                        'active': user.get('active', True),
                        '_score': score  # Store score for sorting
                    }
                    all_matches.append(formatted_user)
                
                # If we have enough high-quality matches, we can stop early
                if len(all_matches) >= max_results * 2:
                    # Check if we have enough high-scoring matches
                    high_score_matches = [m for m in all_matches if m.get('_score', 0) >= 200]
                    if len(high_score_matches) >= max_results:
                        break
            
            # Sort by score (descending) and remove score from output
            all_matches.sort(key=lambda x: x.get('_score', 0), reverse=True)
            formatted_users = []
            for user in all_matches[:max_results]:
                user.pop('_score', None)  # Remove internal score field
                formatted_users.append(user)
            
            # If no users found, return a clear message
            if len(formatted_users) == 0:
                return {
                    "error": f"No Jira users found matching '{query}'",
                    "message": f"No Jira users found whose username, display name, or email contains '{query}'. Please check the spelling or try a different search term.",
                    "users": []
                }
            
            return formatted_users
            
        except Exception as e:
            logger.error(f"Error searching users: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    def find_matching_user(self, given_identifier: str, max_variations: int = 3) -> Dict:
        """
        Find matching Jira user by searching for users whose username, display name, or email contains the given identifier.
        
        Args:
            given_identifier: The name/email/username provided by the user
            max_variations: Maximum number of variations to try (default: 3, not used in simplified version)
            
        Returns:
            Dictionary with 'match', 'confidence', 'action', and 'alternatives' keys
            - match: The matched user dict (or None)
            - confidence: Confidence score 0.0 to 1.0
            - action: 'proceed', 'confirm', or 'ask_user'
            - alternatives: List of alternative matches (for confirmation)
        """
        if not given_identifier:
            return {
                "match": None,
                "confidence": 0.0,
                "action": "ask_user",
                "alternatives": []
            }
        
        # Search for users containing the identifier
        search_results = self.search_users(given_identifier, max_results=10)
        
        # Check if search returned an error (e.g., no users found)
        if isinstance(search_results, dict) and 'error' in search_results:
            return {
                "match": None,
                "confidence": 0.0,
                "action": "ask_user",
                "alternatives": [],
                "message": search_results.get("message", search_results.get("error", "No users found"))
            }
        
        if isinstance(search_results, list) and len(search_results) > 0:
            # Score results based on match quality
            scored_results = []
            given_lower = given_identifier.lower()
            
            for user in search_results:
                email = user.get('emailAddress', '').lower() if user.get('emailAddress') else ''
                display_name = user.get('displayName', '').lower() if user.get('displayName') else ''
                account_id = user.get('accountId', '').lower() if user.get('accountId') else ''
                
                # Calculate confidence based on match quality
                confidence = 0.0
                
                # Exact match - highest confidence
                if (given_lower == email or 
                    given_lower == display_name or 
                    given_lower == account_id):
                    confidence = 1.0
                # Starts or ends with query - high confidence
                elif (email.startswith(given_lower) or email.endswith(given_lower) or
                      display_name.startswith(given_lower) or display_name.endswith(given_lower) or
                      account_id.startswith(given_lower) or account_id.endswith(given_lower)):
                    confidence = 0.9
                # Contains query - medium-high confidence
                elif (given_lower in email or 
                      given_lower in display_name or 
                      given_lower in account_id):
                    confidence = 0.8
                
                if confidence > 0:
                    scored_results.append({
                        "user": user,
                        "confidence": confidence
                    })
            
            if scored_results:
                # Sort by confidence (highest first)
                scored_results.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Verify users exist before proceeding
                verified_results = []
                for result in scored_results:
                    user = result["user"]
                    account_id = user.get('accountId')
                    email = user.get('emailAddress')
                    
                    # Verify user exists by trying to get user info
                    verified = False
                    if account_id:
                        try:
                            # Try to get user by account ID
                            user_result = self._make_request('GET', f'/user?accountId={account_id}')
                            if 'error' not in user_result:
                                verified = True
                        except:
                            pass
                    
                    # If account ID verification failed, try email
                    if not verified and email:
                        try:
                            # Try to search for user by email to verify using search_users
                            verify_search = self.search_users(email, max_results=1)
                            if isinstance(verify_search, list) and len(verify_search) > 0:
                                verify_user = verify_search[0]
                                if verify_user.get('accountId') == account_id:
                                    verified = True
                        except:
                            pass
                    
                    result["verified"] = verified
                    verified_results.append(result)
                
                # Prefer verified results, but include unverified if no verified ones found
                verified_matches = [r for r in verified_results if r.get("verified", False)]
                if verified_matches:
                    best_match = verified_matches[0]
                elif verified_results:
                    best_match = verified_results[0]
                else:
                    best_match = scored_results[0]
                
                # Determine action based on confidence and verification
                if best_match['confidence'] >= 0.9 and best_match.get("verified", False):
                    action = "proceed"
                elif best_match['confidence'] >= 0.9:
                    # High confidence but not verified - still proceed but log warning
                    action = "proceed"
                    logger.warning(f"High confidence match '{best_match['user'].get('displayName', 'unknown')}' not verified")
                elif best_match['confidence'] >= 0.7:
                    action = "confirm"
                else:
                    action = "ask_user"
                
                # Get top alternatives for confirmation (prefer verified ones)
                alternatives = []
                for r in verified_results[:4]:  # Get more to filter
                    if r["user"] != best_match["user"]:
                        alternatives.append(r["user"])
                        if len(alternatives) >= 3:
                            break
                
                return {
                    "match": best_match["user"],
                    "confidence": best_match['confidence'],
                    "action": action,
                    "alternatives": alternatives,
                    "verified": best_match.get("verified", False)
                }
        
        # No match found
        return {
            "match": None,
            "confidence": 0.0,
            "action": "ask_user",
            "alternatives": [],
            "message": f"No Jira users found matching '{given_identifier}'. Please check the spelling or try a different identifier."
        }

