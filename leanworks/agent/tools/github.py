import logging
import time
import re
from typing import List, Dict, Optional, Tuple
import requests
import json
import jwt
from datetime import datetime, timedelta
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class GitHubTool:
    def __init__(self, installation_id: int = None, app_id: str = None, private_key: str = None):
        """
        Initialize GitHubTool with GitHub App credentials.
        
        Args:
            installation_id: GitHub App installation ID
            app_id: GitHub App ID
            private_key: GitHub App private key (PEM format)
        """
        self.installation_id = installation_id
        self.app_id = app_id
        self.private_key = private_key
        self.base_url = "https://api.github.com"
        self._installation_token = None
        self._token_expires_at = None
        
    def _generate_jwt_token(self) -> str:
        """
        Generate a JWT token for GitHub App authentication.
        JWT tokens expire in 10 minutes.
        
        Returns:
            JWT token string
        """
        if not self.app_id or not self.private_key:
            raise ValueError("GitHub App ID and private key are required")
        
        # JWT expires in 10 minutes (GitHub requirement: max 10 minutes)
        now = int(time.time())
        payload = {
            'iat': now - 60,  # Issued at time (1 minute ago to account for clock skew)
            'exp': now + (10 * 60) - 60,  # Expires in 9 minutes
            'iss': self.app_id  # Issuer (App ID)
        }
        
        try:
            token = jwt.encode(payload, self.private_key, algorithm='RS256')
            return token
        except Exception as e:
            logger.error(f"Failed to generate JWT token: {str(e)}")
            raise
    
    def _get_installation_token(self) -> str:
        """
        Get or refresh the installation access token.
        Tokens are cached and refreshed when expired.
        
        Returns:
            Installation access token
        """
        # Check if we have a valid cached token
        if self._installation_token and self._token_expires_at:
            if time.time() < self._token_expires_at - 60:  # Refresh 1 minute before expiry
                return self._installation_token
        
        # Generate new token
        if not self.installation_id:
            raise ValueError("Installation ID is required")
        
        jwt_token = self._generate_jwt_token()
        
        # Exchange JWT for installation access token
        url = f"{self.base_url}/app/installations/{self.installation_id}/access_tokens"
        headers = {
            'Authorization': f'Bearer {jwt_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.post(url, headers=headers)
            
            if response.status_code != 201:
                error_msg = f"GitHub API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('message', error_msg)
                except:
                    error_msg = response.text or error_msg
                logger.error(f"{error_msg} - {response.text[:200]}")
                raise Exception(error_msg)
            
            token_data = response.json()
            self._installation_token = token_data.get('token')
            expires_at = token_data.get('expires_at')
            
            if expires_at:
                # Parse ISO format timestamp
                expires_dt = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                self._token_expires_at = expires_dt.timestamp()
            else:
                # Default to 1 hour if not provided
                self._token_expires_at = time.time() + 3600
            
            logger.info("GitHub installation token refreshed successfully")
            return self._installation_token
            
        except Exception as e:
            logger.error(f"Failed to get installation token: {str(e)}")
            raise
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Make an HTTP request to the GitHub API.
        
        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE)
            endpoint: API endpoint (relative to base_url)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response data as dictionary or error dictionary
        """
        try:
            # Get installation token (will refresh if needed)
            token = self._get_installation_token()
            
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            headers = kwargs.pop('headers', {})
            headers.setdefault('Authorization', f'Bearer {token}')
            headers.setdefault('Accept', 'application/vnd.github.v3+json')
            headers.setdefault('Content-Type', 'application/json')
            
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                **kwargs
            )
            
            if response.status_code >= 400:
                error_msg = f"GitHub API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('message', error_msg)
                    if 'errors' in error_data:
                        errors = error_data['errors']
                        if errors:
                            error_msg += f" - {errors[0].get('message', '')}"
                except:
                    error_msg = response.text or error_msg
                logger.error(f"{error_msg} - {response.text[:200]}")
                return {"error": error_msg}
            
            if response.content:
                return response.json()
            return {}
            
        except Exception as e:
            logger.error(f"Error making GitHub API request: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def list_repositories_property(self):
        description = """
        List repositories accessible to the GitHub App installation. Returns a list of repositories with key information.
        """
        return {
            "type": "custom",
            "name": "github_list_repositories",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "organization": {
                        "type": "string",
                        "description": "Organization name to filter repositories (optional)"
                    },
                    "type": {
                        "type": "string",
                        "description": "Repository type filter: 'all', 'owner', 'member'. Defaults to 'all' if not specified."
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort field: 'created', 'updated', 'pushed', 'full_name'. Defaults to 'full_name'."
                    },
                    "direction": {
                        "type": "string",
                        "description": "Sort direction: 'asc' or 'desc'. Defaults to 'asc'."
                    }
                },
                "required": []
            }
        }
    
    def list_repositories(self, organization: str = None, type: str = "all", sort: str = "full_name", direction: str = "asc") -> List[Dict]:
        """
        List repositories accessible to the installation.
        
        Args:
            organization: Organization name to filter (optional)
            type: Repository type filter (default: "all")
            sort: Sort field (default: "full_name")
            direction: Sort direction (default: "asc")
            
        Returns:
            List of repository dictionaries
        """
        logger.info(f"Executing list_repositories, organization: {organization}, type: {type}")
        try:
            if organization:
                endpoint = f"/orgs/{organization}/repos"
            else:
                endpoint = "/installation/repositories"
            
            params = {
                'type': type,
                'sort': sort,
                'direction': direction,
                'per_page': 100
            }
            
            result = self._make_request('GET', endpoint, params=params)
            
            if 'error' in result:
                return result
            
            # Handle different response formats
            if isinstance(result, dict) and 'repositories' in result:
                repos = result['repositories']
            elif isinstance(result, list):
                repos = result
            else:
                repos = []
            
            formatted_repos = []
            for repo in repos:
                formatted_repo = {
                    'id': repo.get('id'),
                    'name': repo.get('name'),
                    'full_name': repo.get('full_name'),
                    'description': repo.get('description'),
                    'private': repo.get('private'),
                    'language': repo.get('language'),
                    'stars': repo.get('stargazers_count', 0),
                    'forks': repo.get('forks_count', 0),
                    'open_issues': repo.get('open_issues_count', 0),
                    'default_branch': repo.get('default_branch'),
                    'created_at': repo.get('created_at'),
                    'updated_at': repo.get('updated_at'),
                    'pushed_at': repo.get('pushed_at'),
                    'url': repo.get('html_url'),
                    'owner': repo.get('owner', {}).get('login') if isinstance(repo.get('owner'), dict) else (repo.get('owner') if repo.get('owner') else None)
                }
                formatted_repos.append(formatted_repo)
            
            return formatted_repos
            
        except Exception as e:
            logger.error(f"Error listing repositories: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def get_repository_property(self):
        description = """
        Get detailed information about a specific GitHub repository.
        """
        return {
            "type": "custom",
            "name": "github_get_repository",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    }
                },
                "required": ["owner", "repo"]
            }
        }
    
    def get_repository(self, owner: str, repo: str) -> Dict:
        """
        Get detailed information about a specific repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Complete repository details
        """
        logger.info(f"Executing get_repository for {owner}/{repo}")
        try:
            result = self._make_request('GET', f'/repos/{owner}/{repo}')
            
            if 'error' in result:
                return result
            
            formatted_repo = {
                'id': result.get('id'),
                'name': result.get('name'),
                'full_name': result.get('full_name'),
                'description': result.get('description'),
                'private': result.get('private'),
                'language': result.get('language'),
                'stars': result.get('stargazers_count', 0),
                'forks': result.get('forks_count', 0),
                'watchers': result.get('watchers_count', 0),
                'open_issues': result.get('open_issues_count', 0),
                'default_branch': result.get('default_branch'),
                'created_at': result.get('created_at'),
                'updated_at': result.get('updated_at'),
                'pushed_at': result.get('pushed_at'),
                'url': result.get('html_url'),
                'clone_url': result.get('clone_url'),
                'owner': result.get('owner', {}).get('login') if result.get('owner') else None,
                'topics': result.get('topics', []),
                'archived': result.get('archived', False),
                'disabled': result.get('disabled', False)
            }
            
            return formatted_repo
            
        except Exception as e:
            logger.error(f"Error getting repository: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def search_issues_property(self):
        description = """
        Search GitHub issues using GitHub's search syntax. Returns a list of issues matching the query.
        Example queries: 'is:issue is:open repo:owner/repo', 'author:username', 'label:bug'.
        """
        return {
            "type": "custom",
            "name": "github_search_issues",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query using GitHub search syntax (e.g., 'is:issue is:open repo:owner/repo')"
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort field: 'comments', 'reactions', 'interactions', 'created', 'updated'. Defaults to 'best match'."
                    },
                    "order": {
                        "type": "string",
                        "description": "Sort order: 'desc' or 'asc'. Defaults to 'desc'."
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Number of results per page. Defaults to 30 if not specified (max 100)."
                    }
                },
                "required": ["query"]
            }
        }
    
    def search_issues(self, query: str, sort: str = None, order: str = "desc", per_page: int = 30) -> List[Dict]:
        """
        Search GitHub issues using query syntax.
        
        Args:
            query: Search query (e.g., "is:issue is:open repo:owner/repo")
            sort: Sort field (optional)
            order: Sort order (default: "desc")
            per_page: Results per page (default: 30, max: 100)
            
        Returns:
            List of issue dictionaries
        """
        logger.info(f"Executing search_issues with query: {query}")
        try:
            params = {
                'q': query,
                'order': order,
                'per_page': min(per_page, 100)
            }
            
            if sort:
                params['sort'] = sort
            
            result = self._make_request('GET', '/search/issues', params=params)
            
            if 'error' in result:
                return result
            
            issues = result.get('items', [])
            formatted_issues = []
            
            for issue in issues:
                formatted_issue = {
                    'number': issue.get('number'),
                    'title': issue.get('title'),
                    'body': issue.get('body'),
                    'state': issue.get('state'),
                    'user': issue.get('user', {}).get('login') if issue.get('user') else None,
                    'labels': [label.get('name') for label in issue.get('labels', [])],
                    'assignees': [assignee.get('login') for assignee in issue.get('assignees', [])],
                    'comments': issue.get('comments', 0),
                    'created_at': issue.get('created_at'),
                    'updated_at': issue.get('updated_at'),
                    'closed_at': issue.get('closed_at'),
                    'url': issue.get('html_url'),
                    'repository': issue.get('repository_url', '').split('/repos/')[-1] if issue.get('repository_url') else None
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
        Get detailed information about a specific GitHub issue by its number.
        """
        return {
            "type": "custom",
            "name": "github_get_issue",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    },
                    "issue_number": {
                        "type": "integer",
                        "description": "Issue number"
                    }
                },
                "required": ["owner", "repo", "issue_number"]
            }
        }
    
    def get_issue(self, owner: str, repo: str, issue_number: int) -> Dict:
        """
        Get detailed information about a specific issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_number: Issue number
            
        Returns:
            Complete issue details
        """
        logger.info(f"Executing get_issue for {owner}/{repo}#{issue_number}")
        try:
            result = self._make_request('GET', f'/repos/{owner}/{repo}/issues/{issue_number}')
            
            if 'error' in result:
                return result
            
            # Get comments
            comments_result = self._make_request('GET', f'/repos/{owner}/{repo}/issues/{issue_number}/comments')
            comments = []
            if 'error' not in comments_result and isinstance(comments_result, list):
                for comment in comments_result:
                    comments.append({
                        'id': comment.get('id'),
                        'user': comment.get('user', {}).get('login') if comment.get('user') else None,
                        'body': comment.get('body'),
                        'created_at': comment.get('created_at'),
                        'updated_at': comment.get('updated_at')
                    })
            
            formatted_issue = {
                'number': result.get('number'),
                'title': result.get('title'),
                'body': result.get('body'),
                'state': result.get('state'),
                'user': result.get('user', {}).get('login') if result.get('user') else None,
                'labels': [label.get('name') for label in result.get('labels', [])],
                'assignees': [assignee.get('login') for assignee in result.get('assignees', [])],
                'comments_count': result.get('comments', 0),
                'comments': comments,
                'created_at': result.get('created_at'),
                'updated_at': result.get('updated_at'),
                'closed_at': result.get('closed_at'),
                'url': result.get('html_url'),
                'milestone': result.get('milestone', {}).get('title') if result.get('milestone') else None
            }
            
            return formatted_issue
            
        except Exception as e:
            logger.error(f"Error getting issue: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def create_issue_property(self):
        description = """
        Create a new GitHub issue in a repository.
        """
        return {
            "type": "custom",
            "name": "github_create_issue",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    },
                    "title": {
                        "type": "string",
                        "description": "Issue title"
                    },
                    "body": {
                        "type": "string",
                        "description": "Issue body/description (optional)"
                    },
                    "labels": {
                        "type": "string",
                        "description": "Comma-separated list of label names (optional)"
                    },
                    "assignees": {
                        "type": "string",
                        "description": "Comma-separated list of assignee usernames (optional)"
                    }
                },
                "required": ["owner", "repo", "title"]
            }
        }
    
    def create_issue(self, owner: str, repo: str, title: str, body: str = None, labels: str = None, assignees: str = None) -> Dict:
        """
        Create a new GitHub issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Issue title
            body: Issue body (optional)
            labels: Comma-separated label names (optional)
            assignees: Comma-separated assignee usernames (optional)
            
        Returns:
            Created issue details
        """
        logger.info(f"Executing create_issue for {owner}/{repo}, title: {title}")
        try:
            payload = {
                'title': title
            }
            
            if body:
                payload['body'] = body
            
            if labels:
                payload['labels'] = [label.strip() for label in labels.split(',')]
            
            if assignees:
                payload['assignees'] = [assignee.strip() for assignee in assignees.split(',')]
            
            result = self._make_request('POST', f'/repos/{owner}/{repo}/issues', json=payload)
            
            if 'error' in result:
                return result
            
            formatted_issue = {
                'number': result.get('number'),
                'title': result.get('title'),
                'body': result.get('body'),
                'state': result.get('state'),
                'user': result.get('user', {}).get('login') if result.get('user') else None,
                'labels': [label.get('name') for label in result.get('labels', [])],
                'assignees': [assignee.get('login') for assignee in result.get('assignees', [])],
                'created_at': result.get('created_at'),
                'url': result.get('html_url')
            }
            
            return formatted_issue
            
        except Exception as e:
            logger.error(f"Error creating issue: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def update_issue_property(self):
        description = """
        Update an existing GitHub issue. You can update title, body, state, labels, or assignees.
        """
        return {
            "type": "custom",
            "name": "github_update_issue",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    },
                    "issue_number": {
                        "type": "integer",
                        "description": "Issue number"
                    },
                    "title": {
                        "type": "string",
                        "description": "Updated title (optional)"
                    },
                    "body": {
                        "type": "string",
                        "description": "Updated body (optional)"
                    },
                    "state": {
                        "type": "string",
                        "description": "State: 'open' or 'closed' (optional)"
                    },
                    "labels": {
                        "type": "string",
                        "description": "Comma-separated list of label names (optional)"
                    },
                    "assignees": {
                        "type": "string",
                        "description": "Comma-separated list of assignee usernames (optional)"
                    }
                },
                "required": ["owner", "repo", "issue_number"]
            }
        }
    
    def update_issue(self, owner: str, repo: str, issue_number: int, title: str = None, body: str = None,
                    state: str = None, labels: str = None, assignees: str = None) -> Dict:
        """
        Update an existing GitHub issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_number: Issue number
            title: Updated title (optional)
            body: Updated body (optional)
            state: State ("open" or "closed") (optional)
            labels: Comma-separated label names (optional)
            assignees: Comma-separated assignee usernames (optional)
            
        Returns:
            Updated issue details
        """
        logger.info(f"Executing update_issue for {owner}/{repo}#{issue_number}")
        try:
            payload = {}
            
            if title:
                payload['title'] = title
            if body:
                payload['body'] = body
            if state:
                payload['state'] = state
            if labels:
                payload['labels'] = [label.strip() for label in labels.split(',')]
            if assignees:
                payload['assignees'] = [assignee.strip() for assignee in assignees.split(',')]
            
            if not payload:
                # If nothing to update, just return the current issue
                return self.get_issue(owner, repo, issue_number)
            
            result = self._make_request('PATCH', f'/repos/{owner}/{repo}/issues/{issue_number}', json=payload)
            
            if 'error' in result:
                return result
            
            formatted_issue = {
                'number': result.get('number'),
                'title': result.get('title'),
                'body': result.get('body'),
                'state': result.get('state'),
                'labels': [label.get('name') for label in result.get('labels', [])],
                'assignees': [assignee.get('login') for assignee in result.get('assignees', [])],
                'updated_at': result.get('updated_at'),
                'url': result.get('html_url')
            }
            
            return formatted_issue
            
        except Exception as e:
            logger.error(f"Error updating issue: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def add_issue_comment_property(self):
        description = """
        Add a comment to a GitHub issue.
        """
        return {
            "type": "custom",
            "name": "github_add_issue_comment",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    },
                    "issue_number": {
                        "type": "integer",
                        "description": "Issue number"
                    },
                    "body": {
                        "type": "string",
                        "description": "Comment text"
                    }
                },
                "required": ["owner", "repo", "issue_number", "body"]
            }
        }
    
    def add_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> Dict:
        """
        Add a comment to a GitHub issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_number: Issue number
            body: Comment text
            
        Returns:
            Comment details
        """
        logger.info(f"Executing add_issue_comment for {owner}/{repo}#{issue_number}")
        try:
            payload = {
                'body': body
            }
            
            result = self._make_request('POST', f'/repos/{owner}/{repo}/issues/{issue_number}/comments', json=payload)
            
            if 'error' in result:
                return result
            
            formatted_comment = {
                'id': result.get('id'),
                'user': result.get('user', {}).get('login') if result.get('user') else None,
                'body': result.get('body'),
                'created_at': result.get('created_at'),
                'updated_at': result.get('updated_at'),
                'url': result.get('html_url')
            }
            
            return formatted_comment
            
        except Exception as e:
            logger.error(f"Error adding comment: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def list_pull_requests_property(self):
        description = """
        List pull requests for a repository. Returns a list of pull requests with key information.
        """
        return {
            "type": "custom",
            "name": "github_list_pull_requests",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    },
                    "state": {
                        "type": "string",
                        "description": "PR state: 'open', 'closed', or 'all'. Defaults to 'open' if not specified."
                    },
                    "head": {
                        "type": "string",
                        "description": "Filter by head branch (optional)"
                    },
                    "base": {
                        "type": "string",
                        "description": "Filter by base branch (optional)"
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort field: 'created', 'updated', 'popularity', 'long-running'. Defaults to 'created'."
                    },
                    "direction": {
                        "type": "string",
                        "description": "Sort direction: 'asc' or 'desc'. Defaults to 'desc'."
                    }
                },
                "required": ["owner", "repo"]
            }
        }
    
    def list_pull_requests(self, owner: str, repo: str, state: str = "open", head: str = None, base: str = None,
                          sort: str = "created", direction: str = "desc", per_page: int = 100) -> List[Dict]:
        """
        List pull requests for a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: PR state (default: "open")
            head: Filter by head branch (optional)
            base: Filter by base branch (optional)
            sort: Sort field (default: "created")
            direction: Sort direction (default: "desc")
            
        Returns:
            List of pull request dictionaries
        """
        logger.info(f"Executing list_pull_requests for {owner}/{repo}, state: {state}")
        try:
            params = {
                'state': state,
                'sort': sort,
                'direction': direction,
                'per_page': min(per_page, 100)
            }
            
            if head:
                params['head'] = head
            if base:
                params['base'] = base
            
            result = self._make_request('GET', f'/repos/{owner}/{repo}/pulls', params=params)
            
            if 'error' in result:
                return result
            
            prs = result if isinstance(result, list) else []
            formatted_prs = []
            
            for pr in prs:
                formatted_pr = {
                    'number': pr.get('number'),
                    'title': pr.get('title'),
                    'body': pr.get('body'),
                    'state': pr.get('state'),
                    'user': pr.get('user', {}).get('login') if pr.get('user') else None,
                    'head': pr.get('head', {}).get('ref') if pr.get('head') else None,
                    'base': pr.get('base', {}).get('ref') if pr.get('base') else None,
                    'merged': pr.get('merged', False),
                    'mergeable': pr.get('mergeable'),
                    'draft': pr.get('draft', False),
                    'labels': [label.get('name') for label in pr.get('labels', [])],
                    'assignees': [assignee.get('login') for assignee in pr.get('assignees', [])],
                    'created_at': pr.get('created_at'),
                    'updated_at': pr.get('updated_at'),
                    'closed_at': pr.get('closed_at'),
                    'merged_at': pr.get('merged_at'),
                    'url': pr.get('html_url')
                }
                formatted_prs.append(formatted_pr)
            
            return formatted_prs
            
        except Exception as e:
            logger.error(f"Error listing pull requests: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def get_pull_request_property(self):
        description = """
        Get detailed information about a specific pull request by its number.
        """
        return {
            "type": "custom",
            "name": "github_get_pull_request",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    },
                    "pr_number": {
                        "type": "integer",
                        "description": "Pull request number"
                    }
                },
                "required": ["owner", "repo", "pr_number"]
            }
        }
    
    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> Dict:
        """
        Get detailed information about a specific pull request.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            
        Returns:
            Complete PR details
        """
        logger.info(f"Executing get_pull_request for {owner}/{repo}#{pr_number}")
        try:
            result = self._make_request('GET', f'/repos/{owner}/{repo}/pulls/{pr_number}')
            
            if 'error' in result:
                return result
            
            formatted_pr = {
                'number': result.get('number'),
                'title': result.get('title'),
                'body': result.get('body'),
                'state': result.get('state'),
                'user': result.get('user', {}).get('login') if result.get('user') else None,
                'head': result.get('head', {}).get('ref') if result.get('head') else None,
                'base': result.get('base', {}).get('ref') if result.get('base') else None,
                'merged': result.get('merged', False),
                'mergeable': result.get('mergeable'),
                'mergeable_state': result.get('mergeable_state'),
                'draft': result.get('draft', False),
                'labels': [label.get('name') for label in result.get('labels', [])],
                'assignees': [assignee.get('login') for assignee in result.get('assignees', [])],
                'requested_reviewers': [reviewer.get('login') for reviewer in result.get('requested_reviewers', [])],
                'comments': result.get('comments', 0),
                'review_comments': result.get('review_comments', 0),
                'commits': result.get('commits', 0),
                'additions': result.get('additions'),
                'deletions': result.get('deletions'),
                'changed_files': result.get('changed_files'),
                'created_at': result.get('created_at'),
                'updated_at': result.get('updated_at'),
                'closed_at': result.get('closed_at'),
                'merged_at': result.get('merged_at'),
                'url': result.get('html_url')
            }
            
            return formatted_pr
            
        except Exception as e:
            logger.error(f"Error getting pull request: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def create_pull_request_property(self):
        description = """
        Create a new pull request in a repository.
        """
        return {
            "type": "custom",
            "name": "github_create_pull_request",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    },
                    "title": {
                        "type": "string",
                        "description": "Pull request title"
                    },
                    "head": {
                        "type": "string",
                        "description": "Head branch (the branch with changes)"
                    },
                    "base": {
                        "type": "string",
                        "description": "Base branch (the branch to merge into)"
                    },
                    "body": {
                        "type": "string",
                        "description": "Pull request description (optional)"
                    },
                    "draft": {
                        "type": "boolean",
                        "description": "Whether to create as draft PR. Defaults to false if not specified."
                    }
                },
                "required": ["owner", "repo", "title", "head", "base"]
            }
        }
    
    def create_pull_request(self, owner: str, repo: str, title: str, head: str, base: str,
                           body: str = None, draft: bool = False) -> Dict:
        """
        Create a new pull request.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: PR title
            head: Head branch (branch with changes)
            base: Base branch (branch to merge into)
            body: PR description (optional)
            draft: Whether to create as draft (default: False)
            
        Returns:
            Created PR details
        """
        logger.info(f"Executing create_pull_request for {owner}/{repo}, head: {head}, base: {base}")
        try:
            payload = {
                'title': title,
                'head': head,
                'base': base
            }
            
            if body:
                payload['body'] = body
            if draft:
                payload['draft'] = draft
            
            result = self._make_request('POST', f'/repos/{owner}/{repo}/pulls', json=payload)
            
            if 'error' in result:
                return result
            
            formatted_pr = {
                'number': result.get('number'),
                'title': result.get('title'),
                'body': result.get('body'),
                'state': result.get('state'),
                'head': result.get('head', {}).get('ref') if result.get('head') else None,
                'base': result.get('base', {}).get('ref') if result.get('base') else None,
                'draft': result.get('draft', False),
                'created_at': result.get('created_at'),
                'url': result.get('html_url')
            }
            
            return formatted_pr
            
        except Exception as e:
            logger.error(f"Error creating pull request: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def list_commits_property(self):
        description = """
        List commits for a repository. Returns a list of commits with key information. Can filter by branch, path, author, or date range.
        
        IMPORTANT: If this tool returns zero results when filtering by author, it may mean the GitHub username is incorrect. 
        Always suggest the user confirm the correct GitHub username/handle and consider using github_search_users to find the correct username.
        """
        return {
            "type": "custom",
            "name": "github_list_commits",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    },
                    "sha": {
                        "type": "string",
                        "description": "Branch, tag, or commit SHA to list commits from (optional, defaults to default branch)"
                    },
                    "path": {
                        "type": "string",
                        "description": "File path to filter commits by (optional)"
                    },
                    "author": {
                        "type": "string",
                        "description": "Author username to filter commits by (optional)"
                    },
                    "since": {
                        "type": "string",
                        "description": "Only show commits after this date (ISO 8601 format, optional)"
                    },
                    "until": {
                        "type": "string",
                        "description": "Only show commits before this date (ISO 8601 format, optional)"
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Number of results per page. Defaults to 30 if not specified (max 100)."
                    }
                },
                "required": ["owner", "repo"]
            }
        }
    
    def list_commits(self, owner: str, repo: str, sha: str = None, path: str = None, author: str = None,
                    since: str = None, until: str = None, per_page: int = 30, try_approximate_match: bool = True) -> List[Dict]:
        """
        List commits for a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            sha: Branch, tag, or commit SHA (optional)
            path: File path filter (optional)
            author: Author username filter (optional)
            since: Only show commits after this date (ISO 8601, optional)
            until: Only show commits before this date (ISO 8601, optional)
            per_page: Results per page (default: 30, max: 100)
            try_approximate_match: If True, try to find matching username if exact match fails (default: True)
            
        Returns:
            List of commit dictionaries, or error dictionary with matching suggestions
        """
        logger.info(f"Executing list_commits for {owner}/{repo}, author: {author}")
        try:
            params = {
                'per_page': min(per_page, 100)
            }
            
            if sha:
                params['sha'] = sha
            if path:
                params['path'] = path
            if author:
                params['author'] = author
            if since:
                params['since'] = since
            if until:
                params['until'] = until
            
            result = self._make_request('GET', f'/repos/{owner}/{repo}/commits', params=params)
            
            # If error and we have an author, try approximate matching
            if 'error' in result and author and try_approximate_match:
                logger.info(f"Exact author match failed for '{author}', trying approximate matching...")
                match_result = self.find_matching_username(author, owner, repo, max_variations=3)
                
                if match_result['match'] and match_result['confidence'] >= 0.9:
                    # High confidence match - proceed automatically
                    logger.info(f"Found high-confidence match: '{author}' -> '{match_result['match']}' (confidence: {match_result['confidence']:.2f})")
                    params['author'] = match_result['match']
                    result = self._make_request('GET', f'/repos/{owner}/{repo}/commits', params=params)
                    if 'error' not in result:
                        # Success - add metadata about the match
                        if isinstance(result, list):
                            return result
                elif match_result['match'] and match_result['confidence'] >= 0.7:
                    # Medium confidence - return error with suggestion
                    alternatives = [match_result['match']] + match_result['alternatives'][:2]
                    return {
                        "error": f"Could not find commits for author '{author}'. Found possible match: {match_result['match']}",
                        "suggestion": f"Did you mean: {', '.join(alternatives)}?",
                        "match_result": match_result
                    }
                elif match_result['match']:
                    # Low confidence - return error with all options
                    alternatives = [match_result['match']] + match_result['alternatives']
                    return {
                        "error": f"Could not find commits for author '{author}'. Found possible matches: {', '.join(alternatives)}",
                        "suggestion": "Please provide the exact GitHub username, or use github_search_users to find the correct username.",
                        "match_result": match_result
                    }
                else:
                    # No match found
                    return {
                        "error": f"Could not find commits for author '{author}'. No matching GitHub username found.",
                        "suggestion": "Please provide the exact GitHub username, or use github_search_users to find the correct username."
                    }
            
            if 'error' in result:
                return result
            
            commits = result if isinstance(result, list) else []
            formatted_commits = []
            
            for commit in commits:
                commit_data = commit.get('commit', {})
                author_info = commit_data.get('author', {})
                formatted_commit = {
                    'sha': commit.get('sha'),
                    'message': commit_data.get('message'),
                    'author': author_info.get('name') if author_info else None,
                    'author_email': author_info.get('email') if author_info else None,
                    'author_date': author_info.get('date') if author_info else None,
                    'committer': commit_data.get('committer', {}).get('name') if commit_data.get('committer') else None,
                    'committer_date': commit_data.get('committer', {}).get('date') if commit_data.get('committer') else None,
                    'url': commit.get('html_url'),
                    'stats': commit.get('stats', {}),
                    'files_count': len(commit.get('files', []))
                }
                formatted_commits.append(formatted_commit)
            
            return formatted_commits
            
        except Exception as e:
            logger.error(f"Error listing commits: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def get_commit_property(self):
        description = """
        Get detailed information about a specific commit by its SHA. Returns complete commit details including message, author, files changed, and statistics.
        """
        return {
            "type": "custom",
            "name": "github_get_commit",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    },
                    "sha": {
                        "type": "string",
                        "description": "Commit SHA (full or short)"
                    }
                },
                "required": ["owner", "repo", "sha"]
            }
        }
    
    def get_commit(self, owner: str, repo: str, sha: str) -> Dict:
        """
        Get detailed information about a specific commit.
        
        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA
            
        Returns:
            Complete commit details
        """
        logger.info(f"Executing get_commit for {owner}/{repo}, sha: {sha}")
        try:
            result = self._make_request('GET', f'/repos/{owner}/{repo}/commits/{sha}')
            
            if 'error' in result:
                return result
            
            commit_data = result.get('commit', {})
            author_info = commit_data.get('author', {})
            committer_info = commit_data.get('committer', {})
            stats = result.get('stats', {})
            files = result.get('files', [])
            
            formatted_files = []
            for file in files:
                formatted_files.append({
                    'filename': file.get('filename'),
                    'status': file.get('status'),
                    'additions': file.get('additions', 0),
                    'deletions': file.get('deletions', 0),
                    'changes': file.get('changes', 0)
                    # Note: patch/diff is excluded to avoid large responses
                })
            
            formatted_commit = {
                'sha': result.get('sha'),
                'message': commit_data.get('message'),
                'author': author_info.get('name') if author_info else None,
                'author_email': author_info.get('email') if author_info else None,
                'author_date': author_info.get('date') if author_info else None,
                'committer': committer_info.get('name') if committer_info else None,
                'committer_email': committer_info.get('email') if committer_info else None,
                'committer_date': committer_info.get('date') if committer_info else None,
                'url': result.get('html_url'),
                'stats': {
                    'additions': stats.get('additions', 0),
                    'deletions': stats.get('deletions', 0),
                    'total': stats.get('total', 0)
                },
                'files': formatted_files,
                'files_count': len(files),
                'parents': [parent.get('sha') for parent in result.get('parents', [])]
            }
            
            return formatted_commit
            
        except Exception as e:
            logger.error(f"Error getting commit: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def get_pull_request_commits_property(self):
        description = """
        List commits included in a pull request. Returns a list of commits that are part of the PR.
        """
        return {
            "type": "custom",
            "name": "github_get_pull_request_commits",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name"
                    },
                    "pr_number": {
                        "type": "integer",
                        "description": "Pull request number"
                    }
                },
                "required": ["owner", "repo", "pr_number"]
            }
        }
    
    def get_pull_request_commits(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        """
        List commits included in a pull request.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            
        Returns:
            List of commit dictionaries
        """
        logger.info(f"Executing get_pull_request_commits for {owner}/{repo}#{pr_number}")
        try:
            result = self._make_request('GET', f'/repos/{owner}/{repo}/pulls/{pr_number}/commits')
            
            if 'error' in result:
                return result
            
            commits = result if isinstance(result, list) else []
            formatted_commits = []
            
            for commit in commits:
                commit_data = commit.get('commit', {})
                author_info = commit_data.get('author', {})
                formatted_commit = {
                    'sha': commit.get('sha'),
                    'message': commit_data.get('message'),
                    'author': author_info.get('name') if author_info else None,
                    'author_email': author_info.get('email') if author_info else None,
                    'author_date': author_info.get('date') if author_info else None,
                    'committer': commit_data.get('committer', {}).get('name') if commit_data.get('committer') else None,
                    'committer_date': commit_data.get('committer', {}).get('date') if commit_data.get('committer') else None,
                    'url': commit.get('html_url')
                }
                formatted_commits.append(formatted_commit)
            
            return formatted_commits
            
        except Exception as e:
            logger.error(f"Error getting pull request commits: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    def _normalize_username(self, username: str) -> str:
        """Normalize username for comparison (lowercase, remove special chars)."""
        if not username:
            return ""
        normalized = username.lower().strip()
        # Remove common special characters for comparison
        normalized = normalized.replace('.', '').replace('-', '').replace('_', '')
        return normalized
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity score between two strings (0.0 to 1.0)."""
        if not str1 or not str2:
            return 0.0
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _generate_username_variations(self, username: str, max_variations: int = 3) -> List[str]:
        """Generate username variations for approximate matching."""
        if not username:
            return []
        
        variations = []
        seen = set()
        
        # Add normalized version
        normalized = self._normalize_username(username)
        if normalized and normalized not in seen:
            variations.append(normalized)
            seen.add(normalized)
        
        # Add lowercase version
        lower = username.lower().strip()
        if lower != username and lower not in seen and len(variations) < max_variations:
            variations.append(lower)
            seen.add(lower)
        
        # Remove dots
        no_dots = username.replace('.', '')
        if no_dots != username and no_dots not in seen and len(variations) < max_variations:
            variations.append(no_dots)
            seen.add(no_dots)
        
        # Remove hyphens
        no_hyphens = username.replace('-', '')
        if no_hyphens != username and no_hyphens not in seen and len(variations) < max_variations:
            variations.append(no_hyphens)
            seen.add(no_hyphens)
        
        # First part before dot/hyphen
        if '.' in username:
            first_part = username.split('.')[0]
            if first_part not in seen and len(variations) < max_variations:
                variations.append(first_part)
                seen.add(first_part)
        
        if '-' in username:
            first_part = username.split('-')[0]
            if first_part not in seen and len(variations) < max_variations:
                variations.append(first_part)
                seen.add(first_part)
        
        return variations[:max_variations]
    
    @property
    def search_users_property(self):
        description = """
        Search GitHub users by username, name, or email. Returns a list of matching users with their GitHub usernames.
        This tool is useful for finding the correct GitHub username when you have a partial name, email, or slightly different username.
        
        IMPORTANT: If this tool returns zero results, always suggest the user confirm the correct GitHub username/handle. 
        The user may need to provide the exact username or check their GitHub profile.
        """
        return {
            "type": "custom",
            "name": "github_search_users",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (username, name, or email). For email search, use format: 'email@example.com in:email'"
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Number of results per page. Defaults to 10 if not specified (max 100)."
                    }
                },
                "required": ["query"]
            }
        }
    
    def search_users(self, query: str, per_page: int = 10) -> List[Dict]:
        """
        Search GitHub users by username, name, or email.
        Uses GET /users/{username} for exact username matches, then GET /users to list and filter users.
        
        Args:
            query: Search query (username, name, or email) - will try exact match first, then list and filter
            per_page: Number of results per page (default: 10, max: 100)
            
        Returns:
            List of user dictionaries with login, id, avatar_url, etc.
        """
        logger.info(f"Executing search_users with query: {query}")
        try:
            if not query:
                return []
            
            # Clean the query - remove whitespace
            query = query.strip()
            query_lower = query.lower()
            
            # Check if query looks like an exact GitHub username (no spaces, no @, no special chars except hyphens and underscores)
            # GitHub usernames can contain alphanumeric characters and hyphens, but cannot start/end with hyphen
            is_likely_username = (
                len(query) > 0 and 
                len(query) <= 39 and  # GitHub username max length
                not ' ' in query and 
                not '@' in query and
                re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9]|-(?![.-]))*[a-zA-Z0-9]$|^[a-zA-Z0-9]$', query)
            )
            
            # First, try to get the user directly if it looks like an exact username
            if is_likely_username:
                logger.info(f"Query looks like exact username, trying GET /users/{query}")
                user_result = self._make_request('GET', f'/users/{query}')
                
                if 'error' not in user_result and user_result.get('login'):
                    # Successfully found user by exact username
                    formatted_user = {
                        'login': user_result.get('login'),
                        'id': user_result.get('id'),
                        'avatar_url': user_result.get('avatar_url'),
                        'url': user_result.get('html_url'),
                        'type': user_result.get('type')
                    }
                    logger.info(f"Found user by exact username: {formatted_user['login']}")
                    return [formatted_user]
                # If exact match failed, continue to list and filter
            
            # Use GET /users to list users and filter client-side
            logger.info(f"Using GET /users to list and filter users for query: {query}")
            all_matches = []
            max_iterations = 10  # Limit to 10 iterations (up to 1000 users) for performance
            per_page_param = 100  # Maximum per page for GET /users
            since_id = None  # Track last user ID for pagination
            
            for iteration in range(max_iterations):
                params = {
                    'per_page': per_page_param
                }
                if since_id:
                    params['since'] = since_id
                
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
                
                # Track the last user ID for next iteration
                if users:
                    since_id = users[-1].get('id')
                
                # Score and filter users based on query
                for user in users:
                    login = user.get('login', '')
                    name = user.get('name', '')
                    email = user.get('email', '')
                    
                    if not login:
                        continue
                    
                    # Calculate match score
                    score = 0
                    login_lower = login.lower()
                    name_lower = name.lower() if name else ''
                    
                    # Exact match gets highest score
                    if login_lower == query_lower:
                        score = 1000
                    # Starts with query
                    elif login_lower.startswith(query_lower):
                        score = 500 - len(login)  # Shorter usernames rank higher
                    # Contains query
                    elif query_lower in login_lower:
                        score = 200 - len(login)
                    # Name matches
                    elif name and query_lower in name_lower:
                        score = 100
                    # Email matches (if available)
                    elif email and query_lower in email.lower():
                        score = 50
                    else:
                        continue  # Skip users that don't match at all
                    
                    formatted_user = {
                        'login': login,
                        'id': user.get('id'),
                        'avatar_url': user.get('avatar_url'),
                        'url': user.get('html_url'),
                        'type': user.get('type'),
                        '_score': score  # Store score for sorting
                    }
                    all_matches.append(formatted_user)
                
                # If we have enough high-quality matches, we can stop early
                if len(all_matches) >= per_page * 2:
                    # Check if we have enough high-scoring matches
                    high_score_matches = [m for m in all_matches if m.get('_score', 0) >= 200]
                    if len(high_score_matches) >= per_page:
                        break
            
            # Sort by score (descending) and remove score from output
            all_matches.sort(key=lambda x: x.get('_score', 0), reverse=True)
            formatted_users = []
            for user in all_matches[:per_page]:
                user.pop('_score', None)  # Remove internal score field
                formatted_users.append(user)
            
            # If no users found, return a clear message
            if len(formatted_users) == 0:
                return {
                    "error": f"No GitHub users found matching '{query}'",
                    "message": f"No GitHub users found whose username contains '{query}'. Please check the spelling or try a different search term.",
                    "users": []
                }
            
            return formatted_users
            
        except Exception as e:
            logger.error(f"Error searching users: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    def find_matching_username(self, given_identifier: str, owner: str = None, repo: str = None, max_variations: int = 3) -> Dict:
        """
        Find matching GitHub username by searching for usernames that contain the given identifier.
        
        Args:
            given_identifier: The name/username/email provided by the user
            owner: Repository owner (optional, for testing username validity)
            repo: Repository name (optional, for testing username validity)
            max_variations: Maximum number of variations to try (default: 3, not used in simplified version)
            
        Returns:
            Dictionary with 'match', 'confidence', 'action', and 'alternatives' keys
            - match: The matched username (or None)
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
        
        # Step 1: Try exact match (if we have repo context to test)
        # IMPORTANT: Disable approximate matching to prevent infinite recursion
        if owner and repo:
            test_result = self.list_commits(owner, repo, author=given_identifier, per_page=1, try_approximate_match=False)
            if isinstance(test_result, list) and len(test_result) > 0:
                return {
                    "match": given_identifier,
                    "confidence": 1.0,
                    "action": "proceed",
                    "alternatives": []
                }
        
        # Step 2: Search for usernames containing the identifier
        search_results = self.search_users(given_identifier, per_page=10)
        
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
            # Filter and score results
            scored_results = []
            given_lower = given_identifier.lower()
            
            for user in search_results:
                login = user.get('login', '')
                if not login:
                    continue
                
                login_lower = login.lower()
                
                # Calculate confidence based on how well it matches
                if login_lower == given_lower:
                    # Exact match
                    confidence = 1.0
                elif login_lower.startswith(given_lower) or login_lower.endswith(given_lower):
                    # Starts or ends with query - high confidence
                    confidence = 0.9
                elif given_lower in login_lower:
                    # Contains query - medium-high confidence
                    confidence = 0.8
                else:
                    # Shouldn't happen since search_users filters, but just in case
                    confidence = 0.5
                
                scored_results.append({
                    "login": login,
                    "confidence": confidence,
                    "user": user
                })
            
            if scored_results:
                # Sort by confidence (highest first)
                scored_results.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Verify users exist before proceeding
                verified_results = []
                for result in scored_results:
                    login = result["login"]
                    
                    # Verify user exists by trying to get user info or test with repo if available
                    verified = False
                    if owner and repo:
                        # Test if we can list commits with this author (verifies user exists)
                        test_result = self.list_commits(owner, repo, author=login, per_page=1, try_approximate_match=False)
                        # If we get a list (even empty) or no error, user exists
                        if isinstance(test_result, list):
                            verified = True
                        elif isinstance(test_result, dict) and 'error' not in test_result:
                            verified = True
                    else:
                        # If no repo context, try to get user info directly
                        try:
                            user_result = self._make_request('GET', f'/users/{login}')
                            if 'error' not in user_result:
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
                    logger.warning(f"High confidence match '{best_match['login']}' not verified")
                elif best_match['confidence'] >= 0.7:
                    action = "confirm"
                else:
                    action = "ask_user"
                
                # Get top alternatives for confirmation (prefer verified ones)
                alternatives = []
                for r in verified_results[:4]:  # Get more to filter
                    if r["login"] != best_match["login"]:
                        alternatives.append(r["login"])
                        if len(alternatives) >= 3:
                            break
                
                return {
                    "match": best_match["login"],
                    "confidence": best_match['confidence'],
                    "action": action,
                    "alternatives": alternatives,
                    "verified": best_match.get("verified", False)
                }
        
        # Step 3: No match found
        return {
            "match": None,
            "confidence": 0.0,
            "action": "ask_user",
            "alternatives": [],
            "message": f"No GitHub users found matching '{given_identifier}'. Please check the spelling or try a different identifier."
        }

