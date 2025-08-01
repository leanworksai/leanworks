import requests
import logging
import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class GitlabTool:
    def __init__(self, gitlab_auth):
        """
        Initialize GitlabTool with GitLab connection details.
        
        Args:
            gitlab_auth: Dictionary containing gitlab_url and gitlab_token
        """
        self.gitlab_url = gitlab_auth["gitlab_url"].rstrip('/')
        self.access_token = gitlab_auth["gitlab_token"]
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        logger.info(f"GitlabTool initialized with URL: {self.gitlab_url}")
        
    def _make_request(self, endpoint: str, params: Dict = None) -> List[Dict]:
        """Make a GET request to GitLab API and handle pagination."""
        url = f"{self.gitlab_url}/api/v4{endpoint}"
        all_results = []
        page = 1
        per_page = 100
        
        while True:
            request_params = {'page': page, 'per_page': per_page}
            if params:
                request_params.update(params)
                
            try:
                logger.debug(f"Making request to: {url} with params: {request_params}")
                response = requests.get(url, headers=self.headers, params=request_params)
                response.raise_for_status()
                
                data = response.json()
                if not data:  # Empty response means no more pages
                    break
                    
                all_results.extend(data)
                
                # Check if there are more pages
                if len(data) < per_page:
                    break
                    
                page += 1
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error making request to {url}: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response body: {e.response.text}")
                break
                
        return all_results
        
    def _make_single_request(self, endpoint: str) -> Optional[Dict]:
        """Make a GET request to GitLab API for a single item (no pagination)."""
        url = f"{self.gitlab_url}/api/v4{endpoint}"
        
        try:
            logger.debug(f"Making single request to: {url}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making single request to {url}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return None
        
    @property
    def list_gitlab_projects_property(self):
        description = """
        List GitLab projects accessible to the user. The response will be a list of dictionaries, each containing project details such as id, name, path, description, web_url, created_at, last_activity_at, and visibility.
        This tool should be called to retrieve project information when project details are needed to answer questions and might be complimentary to list_projects tool.
        You can filter projects by visibility (public, internal, private) or search by name.
        project_id can be used to link projects to issues, merge requests, and other GitLab resources.
        """
        return {
            "type": "custom",
            "name": "list_gitlab_projects",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Search term to filter projects by name or path"
                    },
                    "owned": {
                        "type": "boolean",
                        "description": "If true, only return projects owned by the authenticated user"
                    }
                }
            }
        }
    
    def list_gitlab_projects(self, search: str = None, owned: bool = False):
        logger.info(f"list_gitlab_projects called with parameters: search={search}, owned={owned}")
        try:
            params = {}
            if search:
                params['search'] = search
            if owned:
                params['owned'] = True
            
            logger.info(f"Querying GitLab projects with params: {params}")
            projects = self._make_request('/projects', params)
            logger.info(f"Retrieved {len(projects)} projects from GitLab")
            
            result = []
            for project in projects:
                project_dict = {
                    'id': project.get('id'),
                    'name': project.get('name'),
                    'path': project.get('path'),
                    'path_with_namespace': project.get('path_with_namespace'),
                    'description': project.get('description', ''),
                    'web_url': project.get('web_url'),
                    'created_at': project.get('created_at')
                }
                result.append(project_dict)
            
            logger.info(f"Returning {len(result)} formatted project records")
            return result
        except Exception as e:
            logger.error(f"Error in list_gitlab_projects: {str(e)}")
            return []
        
    @property
    def list_gitlab_issues_property(self):
        description = """
        List issues from GitLab projects. The response will be a list of dictionaries containing issue details such as id, iid, title, description, state, created_at, updated_at, author, assignee, labels, and milestone.
        This tool should be called to retrieve issue information when issue details are needed to answer questions and might be complimentary to list_tasks tool.
        You can filter issues by project, state (opened, closed), assignee, labels, or search by title/description.
        If project_id is not provided, issues from all accessible projects will be returned.
        Since this tool only provide basic issue information, you are recommended to call search_knowledge tool after if you want to dive deeper into a specific issue.
        project_id can be used to link the issues to projects.
        You might need to call list_gitlab_projects before or after to understand the relationship among projects and issues through project_id.
        """
        return {
            "type": "custom",
            "name": "list_gitlab_issues",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "GitLab project ID or path (e.g., 'group/project-name')"
                    },
                    "state": {
                        "type": "string",
                        "description": "Filter by issue state: 'opened', 'closed', or 'all'"
                    },
                    "assignee_username": {
                        "type": "string",
                        "description": "Filter by assignee username"
                    },
                    "labels": {
                        "type": "string",
                        "description": "Comma-separated list of label names to filter by"
                    },
                    "search": {
                        "type": "string",
                        "description": "Search term for issue title or description"
                    }
                }
            }
        }
    
    def list_gitlab_issues(self, project_id: str = None, state: str = "opened", 
                          assignee_username: str = None, labels: str = None, search: str = None):
        logger.info(f"list_gitlab_issues called with parameters: project_id={project_id}, state={state}, assignee_username={assignee_username}, labels={labels}, search={search}")
        
        try:
            params = {'state': state}
            if assignee_username:
                params['assignee_username'] = assignee_username
            if labels:
                params['labels'] = labels
            if search:
                params['search'] = search
                
            if project_id:
                # Get issues from specific project
                logger.info(f"Retrieving issues from specific project: {project_id}")
                # URL encode the project_id in case it contains special characters like '/'
                import urllib.parse
                encoded_project_id = urllib.parse.quote(project_id, safe='')
                endpoint = f'/projects/{encoded_project_id}/issues'
                logger.info(f"Querying project issues with params: {params}")
                issues = self._make_request(endpoint, params)
            else:
                # Get issues from all projects
                logger.info("Retrieving issues from all accessible projects")
                params['scope'] = 'all'
                logger.info(f"Querying all issues with params: {params}")
                issues = self._make_request('/issues', params)
            
            logger.info(f"Retrieved {len(issues)} issues from GitLab")
            
            result = []
            for issue in issues:
                author_info = issue.get('author', {})
                assignee_info = issue.get('assignee', {})
                milestone_info = issue.get('milestone', {})
                
                issue_dict = {
                    'id': issue.get('id'),
                    'iid': issue.get('iid'),
                    'project_id': issue.get('project_id'),
                    'title': issue.get('title'),
                    'description': issue.get('description', ''),
                    'state': issue.get('state'),
                    'created_at': issue.get('created_at'),
                    'updated_at': issue.get('updated_at'),
                    'author': author_info.get('name', '') if author_info else '',
                    'assignee': assignee_info.get('name', '') if assignee_info else '',
                    'labels': issue.get('labels', []),
                    'milestone': milestone_info.get('title', '') if milestone_info else '',
                    'web_url': issue.get('web_url')
                }
                result.append(issue_dict)
            
            logger.info(f"Returning {len(result)} formatted issue records")
            return result
        except Exception as e:
            logger.error(f"Error in list_gitlab_issues: {str(e)}")
            return []

    @property
    def find_gitlab_user_by_email_property(self):
        description = """
        Find GitLab user information by email address. The response will be a dictionary containing user details such as id, username, name, email, and profile information.
        This tool should be called when you need to find a GitLab username or user details using an email address.
        The search will return the user if the email matches exactly with a GitLab user account.
        """
        return {
            "type": "custom",
            "name": "find_gitlab_user_by_email",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "Email address to search for in GitLab users"
                    }
                },
                "required": ["email"]
            }
        }

    @property
    def list_gitlab_project_members_property(self):
        description = """
        List members of a specific GitLab project. The response will be a list of dictionaries containing member details such as id, username, name, access_level, and expires_at.
        This tool should be called when you need to see who has access to a project and their permission levels.
        Access levels: Guest=10, Reporter=20, Developer=30, Maintainer=40, Owner=50.
        You need to provide the project ID (numeric) or project path (group/project-name format).
        """
        return {
            "type": "custom",
            "name": "list_gitlab_project_members",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "GitLab project ID (numeric) or project path (e.g., 'group/project-name')"
                    }
                },
                "required": ["project_id"]
            }
        }
    
    def find_gitlab_user_by_email(self, email: str):
        logger.info(f"find_gitlab_user_by_email called with email: {email}")
        
        try:
            # Search for users by email
            logger.info(f"Searching for GitLab user with email: {email}")
            users = self._make_request('/users', {'search': email})
            
            # Filter to find exact email match by getting detailed user info
            matching_users = []
            for user in users:
                user_id = user.get('id')
                if user_id:
                    # Get detailed user information
                    user_info = self._make_single_request(f'/users/{user_id}')
                    if user_info and user_info.get('email', '').lower() == email.lower():
                        user_dict = {
                            'id': user_info.get('id'),
                            'username': user_info.get('username'),
                            'name': user_info.get('name'),
                            'email': user_info.get('email'),
                            'state': user_info.get('state', ''),
                            'avatar_url': user_info.get('avatar_url', ''),
                            'web_url': user_info.get('web_url', ''),
                            'created_at': user_info.get('created_at', ''),
                            'is_admin': user_info.get('is_admin', False),
                            'bio': user_info.get('bio', ''),
                            'location': user_info.get('location', ''),
                            'public_email': user_info.get('public_email', ''),
                            'organization': user_info.get('organization', '')
                        }
                        matching_users.append(user_dict)
            
            logger.info(f"Found {len(matching_users)} users matching email: {email}")
            
            if len(matching_users) == 1:
                logger.info(f"Returning single user match: {matching_users[0]['username']}")
                return matching_users[0]
            elif len(matching_users) > 1:
                logger.warning(f"Multiple users found with email {email}, returning all matches")
                return matching_users
            else:
                logger.info(f"No user found with email: {email}")
                return None
                
        except Exception as e:
            logger.error(f"Error in find_gitlab_user_by_email: {str(e)}")
            return None

    def list_gitlab_project_members(self, project_id: str):
        logger.info(f"list_gitlab_project_members called with project_id: {project_id}")
        
        try:
            # URL encode the project_id in case it contains special characters like '/'
            import urllib.parse
            encoded_project_id = urllib.parse.quote(project_id, safe='')
            
            logger.info(f"Retrieving members for project: {project_id}")
            members = self._make_request(f'/projects/{encoded_project_id}/members')
            logger.info(f"Retrieved {len(members)} members from GitLab project")
            
            # Access level mapping for better readability
            access_level_names = {
                10: "Guest",
                20: "Reporter", 
                30: "Developer",
                40: "Maintainer",
                50: "Owner"
            }
            
            result = []
            for member in members:
                access_level = member.get('access_level', 0)
                member_dict = {
                    'id': member.get('id'),
                    'username': member.get('username'),
                    'name': member.get('name'),
                    'email': member.get('email', ''),
                    'state': member.get('state', ''),
                    'access_level': access_level,
                    'access_level_name': access_level_names.get(access_level, 'Unknown'),
                    'expires_at': member.get('expires_at'),
                    'web_url': member.get('web_url', '')
                }
                result.append(member_dict)
            
            logger.info(f"Returning {len(result)} formatted member records")
            return result
            
        except Exception as e:
            logger.error(f"Error in list_gitlab_project_members: {str(e)}")
            return []

    @property
    def get_gitlab_project_detail_property(self):
        description = """
        Get detailed information about a specific GitLab project. The response will be a dictionary containing comprehensive project details such as id, name, description, statistics, permissions, repository information, CI/CD settings, and more.
        This tool should be called when you need detailed information about a specific project, such as its README content, file tree, branch information, or project settings.
        You need to provide either the project ID (numeric) or the project path (group/project-name format).
        This provides much more detailed information than list_gitlab_projects, including repository statistics, permissions, and configuration details.
        """
        return {
            "type": "custom",
            "name": "get_gitlab_project_detail",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "GitLab project ID (numeric) or project path (e.g., 'group/project-name')"
                    },
                    "include_statistics": {
                        "type": "boolean",
                        "description": "Include project statistics like commit count, repository size, etc.",
                        "default": True
                    }
                },
                "required": ["project_id"]
            }
        }
    
    def get_gitlab_project_detail(self, project_id: str, include_statistics: bool = True):
        logger.info(f"get_gitlab_project_detail called with project_id: {project_id}, include_statistics: {include_statistics}")
        
        try:
            # URL encode the project_id in case it contains special characters like '/'
            import urllib.parse
            encoded_project_id = urllib.parse.quote(project_id, safe='')
            
            # Build endpoint with optional statistics
            endpoint = f'/projects/{encoded_project_id}'
            if include_statistics:
                endpoint += '?statistics=true'
            
            logger.info(f"Retrieving detailed information for project: {project_id}")
            project_detail = self._make_single_request(endpoint)
            
            if not project_detail:
                logger.warning(f"Project {project_id} not found or not accessible")
                return None
            
            # Extract essential project information
            result = {
                'id': project_detail.get('id'),
                'name': project_detail.get('name'),
                'path': project_detail.get('path'),
                'path_with_namespace': project_detail.get('path_with_namespace'),
                'description': project_detail.get('description', ''),
                'web_url': project_detail.get('web_url'),
                'ssh_url_to_repo': project_detail.get('ssh_url_to_repo'),
                'http_url_to_repo': project_detail.get('http_url_to_repo'),
                'readme_url': project_detail.get('readme_url'),
                'created_at': project_detail.get('created_at'),
                'last_activity_at': project_detail.get('last_activity_at'),
                'visibility': project_detail.get('visibility'),
                'default_branch': project_detail.get('default_branch', ''),
                'topics': project_detail.get('topics', []),
                'star_count': project_detail.get('star_count', 0),
                'forks_count': project_detail.get('forks_count', 0),
                'open_issues_count': project_detail.get('open_issues_count', 0),
                'issues_enabled': project_detail.get('issues_enabled', False),
                'merge_requests_enabled': project_detail.get('merge_requests_enabled', False),
                'wiki_enabled': project_detail.get('wiki_enabled', False),
                'archived': project_detail.get('archived', False),
                'empty_repo': project_detail.get('empty_repo', True)
            }
            
            # Include statistics if requested and available
            if include_statistics and 'statistics' in project_detail:
                stats = project_detail['statistics']
                result['statistics'] = {
                    'commit_count': stats.get('commit_count', 0),
                    'storage_size': stats.get('storage_size', 0),
                    'repository_size': stats.get('repository_size', 0),
                    'wiki_size': stats.get('wiki_size', 0),
                    'lfs_objects_size': stats.get('lfs_objects_size', 0),
                    'job_artifacts_size': stats.get('job_artifacts_size', 0),
                    'packages_size': stats.get('packages_size', 0)
                }
            
            # Include namespace information if available
            if 'namespace' in project_detail:
                namespace = project_detail['namespace']
                result['namespace'] = {
                    'id': namespace.get('id'),
                    'name': namespace.get('name'),
                    'path': namespace.get('path'),
                    'kind': namespace.get('kind'),
                    'full_path': namespace.get('full_path')
                }
            
            # Include owner information if available
            if 'owner' in project_detail:
                owner = project_detail['owner']
                result['owner'] = {
                    'id': owner.get('id'),
                    'username': owner.get('username'),
                    'name': owner.get('name')
                }
            
            logger.info(f"Successfully retrieved detailed information for project: {project_detail.get('name')}")
            return result
            
        except Exception as e:
            logger.error(f"Error in get_gitlab_project_detail: {str(e)}")
            return None