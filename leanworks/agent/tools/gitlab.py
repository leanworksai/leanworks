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
            'PRIVATE-TOKEN': self.access_token,  # GitLab uses PRIVATE-TOKEN, not Bearer
            'Content-Type': 'application/json',
            'User-Agent': 'leanworks-agent'
        }
        # Rate limiting
        self.rate_limit_delay = 0.5  # 500ms delay between requests
        self.max_retries = 3
        logger.info(f"GitlabTool initialized with URL: {self.gitlab_url}")
        
    def _make_request(self, endpoint: str, params: Dict = None) -> List[Dict]:
        """Make a GET request to GitLab API and handle pagination with improved error handling."""
        url = f"{self.gitlab_url}/api/v4{endpoint}"
        all_results = []
        page = 1
        per_page = 100
        
        while True:
            request_params = {'page': page, 'per_page': per_page}
            if params:
                request_params.update(params)
                
            success, data, status_code = self._make_request_with_retry(url, request_params)
            
            if not success:
                logger.error(f"Failed to fetch data from {url}: HTTP {status_code}")
                break
                
            if not data:  # Empty response means no more pages
                break
                
            all_results.extend(data)
            
            # Check if there are more pages
            if len(data) < per_page:
                break
                
            page += 1
                
        return all_results
        
    def _make_single_request(self, endpoint: str) -> Optional[Dict]:
        """Make a GET request to GitLab API for a single item with improved error handling."""
        url = f"{self.gitlab_url}/api/v4{endpoint}"
        
        success, data, status_code = self._make_request_with_retry(url)
        
        if success:
            return data
        else:
            logger.error(f"Failed to fetch data from {url}: HTTP {status_code}")
            return None
    
    def _make_request_with_retry(self, url: str, params: Dict = None) -> tuple:
        """
        Make an HTTP request with retry logic and better error handling.
        
        Args:
            url (str): The URL to request
            params (dict): Query parameters
            
        Returns:
            tuple: (success: bool, response_data: dict/list/None, status_code: int)
        """
        import time
        import random
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add rate limiting delay
                if attempt > 0:
                    delay = self.rate_limit_delay * (2 ** (attempt - 1))  # Exponential backoff
                    jitter = random.uniform(0.1, 0.3)  # Add jitter
                    time.sleep(delay + jitter)
                else:
                    time.sleep(self.rate_limit_delay)
                
                logger.debug(f"Making request to: {url} with params: {params} (attempt {attempt + 1})")
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 200:
                    return True, response.json(), response.status_code
                elif response.status_code == 429:  # Rate limit exceeded
                    if attempt < self.max_retries:
                        retry_after = response.headers.get('Retry-After', '60')
                        wait_time = int(retry_after) if retry_after.isdigit() else 60
                        logger.warning(f"Rate limit exceeded, waiting {wait_time} seconds before retry {attempt + 1}/{self.max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded, max retries reached for {url}")
                        return False, None, response.status_code
                elif response.status_code in [502, 503, 504]:  # Server errors
                    if attempt < self.max_retries:
                        logger.warning(f"Server error {response.status_code} for {url}, retrying {attempt + 1}/{self.max_retries}")
                        continue
                    else:
                        logger.error(f"Server error {response.status_code} for {url}, max retries reached")
                        return False, None, response.status_code
                elif response.status_code == 404:
                    logger.error(f"Resource not found (404) for {url}")
                    logger.error(f"Response body: {response.text}")
                    return False, None, response.status_code
                elif response.status_code == 401:
                    logger.error(f"Unauthorized (401) for {url} - check your GitLab token")
                    logger.error(f"Response body: {response.text}")
                    return False, None, response.status_code
                elif response.status_code == 403:
                    logger.error(f"Forbidden (403) for {url} - insufficient permissions")
                    logger.error(f"Response body: {response.text}")
                    return False, None, response.status_code
                else:
                    logger.error(f"HTTP {response.status_code} for {url}")
                    logger.error(f"Response body: {response.text}")
                    return False, None, response.status_code
                        
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    logger.warning(f"Request error for {url}: {str(e)}, retrying {attempt + 1}/{self.max_retries}")
                    continue
                else:
                    logger.error(f"Request error for {url}: {str(e)}, max retries reached")
                    return False, None, 0
            except Exception as e:
                logger.error(f"Unexpected error for {url}: {str(e)}")
                return False, None, 0
        
        return False, None, 0
        
    def _make_single_request_with_params(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a GET request to GitLab API for a single JSON object with query params."""
        url = f"{self.gitlab_url}/api/v4{endpoint}"
        success, data, status_code = self._make_request_with_retry(url, params)
        if success:
            return data
        logger.error(f"Failed to fetch data from {url}: HTTP {status_code}")
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
            params = {'membership': 'true'}  # Only projects where user is a member
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
            return {"error": f"list_gitlab_projects failed: {str(e)}"}
        
    @property
    def list_gitlab_issues_property(self):
        description = """
        List issues from GitLab projects or groups.

        Response fields:
          - total_issues: either the full list of issue dictionaries (each with id, iid, project_id, title, description, state, created_at, updated_at, author, assignee, labels, milestone, weight, web_url), or the string 'too large to display' if more than 30 issues match.
          - first_30_issues: the first 30 issues according to the requested ordering (or the full list if 30 or fewer).
          - total_issues_statistics: aggregated statistics (e.g., counts for all/opened/closed) computed with the same filters/scope.

        Use this to retrieve issue details along with statistics. You can filter by project id(s) or group id(s), state (opened, closed), assignee, labels, and search by title/description.
        If neither project_id nor group_id is provided, issues from all accessible projects will be returned. IDs can be single values or comma-separated lists to query multiple projects/groups.
        For deeper details on a specific issue, call get_issue_detail after locating it here.
        You might also call list_gitlab_projects or list_gitlab_groups to understand relationships.
        """
        return {
            "type": "custom",
            "name": "list_gitlab_issues",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "assignee_id": {
                        "type": "integer",
                        "description": "Return issues assigned to the given user id. Mutually exclusive with assignee_username. None returns unassigned issues. Any returns issues with an assignee."
                    },
                    "assignee_username": {
                        "type": "string",
                        "description": "Return issues assigned to the given username. Similar to assignee_id and mutually exclusive with assignee_id. In GitLab CE, the assignee_username array should only contain a single value."
                    },
                    "author_id": {
                        "type": "integer",
                        "description": "Return issues created by the given user id. Mutually exclusive with author_username. Combine with scope=all or scope=assigned_to_me."
                    },
                    "author_username": {
                        "type": "string",
                        "description": "Return issues created by the given username. Similar to author_id and mutually exclusive with author_id."
                    },
                    "created_after": {
                        "type": "string",
                        "description": "Return issues created on or after the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)."
                    },
                    "created_before": {
                        "type": "string",
                        "description": "Return issues created on or before the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)."
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Return issues that have no due date, are overdue, or whose due date is this week, this month, or between two weeks ago and next month. Accepts: 0 (no due date), any, today, tomorrow, overdue, week, month, next_month_and_previous_two_weeks."
                    },
                    "epic_id": {
                        "type": "integer",
                        "description": "Return issues associated with the given epic ID. None returns issues that are not associated with an epic. Any returns issues that are associated with an epic. Premium and Ultimate only."
                    },
                    "health_status": {
                        "type": "string",
                        "description": "Return issues with the specified health_status. None returns issues with no health status assigned, and Any returns issues with a health status assigned. Ultimate only."
                    },
                    "in": {
                        "type": "string",
                        "description": "Modify the scope of the search attribute. title, description, or a string joining them with comma. Default is title,description."
                    },
                    "issue_type": {
                        "type": "string",
                        "description": "Filter to a given type of issue. One of issue, incident, test_case or task."
                    },
                    "labels": {
                        "type": "string",
                        "description": "Comma-separated list of case-sensitive label names to filter by (e.g., 'bug,urgent')"
                    },
                    "milestone": {
                        "type": "string",
                        "description": "Return issues for a specific milestone. None returns issues that do not belong to a milestone. Any returns issues that belong to a milestone."
                    },
                    "not": {
                        "type": "object",
                        "description": "Return issues that do not match the parameters supplied. Accepts: assignee_id, assignee_username, author_id, author_username, confidential, created_after, created_before, due_date, epic_id, health_status, iids, issue_type, iteration_id, iteration_title, labels, milestone, my_reaction_emoji, non_archived, not, order_by, scope, search, state, updated_after, updated_before, weight."
                    },
                    "order_by": {
                        "type": "string",
                        "description": "Return issues ordered by created_at, updated_at, priority, due_date, relative_position, label_priority, milestone_due, popularity, weight fields. Default is created_at."
                    },
                    "project_id": {
                        "type": "string",
                        "description": "GitLab project ID (not the project name) or comma-separated list of project IDs."
                    },
                    "group_id": {
                        "type": "string",
                        "description": "GitLab group ID (not the group name) or comma-separated list of group IDs."
                    },
                    "scope": {
                        "type": "string",
                        "description": "Return issues for given scope: created_by_me, assigned_to_me or all. Defaults to created_by_me."
                    },
                    "search": {
                        "type": "string",
                        "description": "Search term for issue title or description"
                    },
                    "sort": {
                        "type": "string",
                        "description": "Return issues sorted in asc or desc order. Default is desc."
                    },
                    "state": {
                        "type": "string",
                        "description": "Filter by issue state: 'opened' (default), 'closed', or 'all'",
                        "default": "opened"
                    },
                    "updated_after": {
                        "type": "string",
                        "description": "Return issues updated on or after the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)."
                    },
                    "updated_before": {
                        "type": "string",
                        "description": "Return issues updated on or before the given time. Expected in ISO 8601 format (2019-03-15T08:00:00Z)."
                    },
                    "weight": {
                        "type": "integer",
                        "description": "Return issues with the specified weight. None returns issues with no weight assigned. Any returns issues with a weight assigned."
                    },
                    "with_labels_details": {
                        "type": "boolean",
                        "description": "If true, response returns more details for each label in labels field: :name, :color, :description, :description_html, :text_color. Default is false."
                    }
                }
            }
        }
    
    def _fetch_issues_parallel(self, ids: list, params: dict, resource_type: str) -> list:
        """
        Fetch issues from multiple projects or groups in parallel.
        
        Args:
            ids: List of project or group IDs
            params: Query parameters for the API request
            resource_type: Either 'projects' or 'groups'
            
        Returns:
            List of all issues from all resources
        """
        import threading
        import urllib.parse
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        all_issues = []
        results_lock = threading.Lock()
        
        def fetch_single_resource(resource_id: str) -> tuple:
            """Fetch issues from a single project or group."""
            try:
                logger.debug(f"Fetching issues from {resource_type[:-1]}: {resource_id}")
                encoded_id = urllib.parse.quote(resource_id, safe='')
                endpoint = f'/{resource_type}/{encoded_id}/issues'
                
                # Add small delay to respect rate limiting
                time.sleep(0.1)
                
                issues = self._make_request(endpoint, params)
                logger.debug(f"Retrieved {len(issues)} issues from {resource_type[:-1]} {resource_id}")
                return resource_id, issues, None
                
            except Exception as e:
                logger.error(f"Error fetching issues from {resource_type[:-1]} {resource_id}: {str(e)}")
                return resource_id, [], str(e)
        
        # Determine optimal number of threads (max 5 to respect GitLab rate limits)
        max_workers = min(len(ids), 5)
        logger.info(f"Fetching issues from {len(ids)} {resource_type} using {max_workers} parallel workers")
        
        # Track success/failure counts
        success_count = 0
        failure_count = 0
        
        # Use ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_id = {executor.submit(fetch_single_resource, resource_id): resource_id 
                           for resource_id in ids}
            
            # Collect results as they complete
            for future in as_completed(future_to_id):
                resource_id = future_to_id[future]
                try:
                    returned_id, issues, error = future.result()
                    
                    if error:
                        logger.warning(f"Failed to fetch issues from {resource_type[:-1]} {returned_id}: {error}")
                        failure_count += 1
                    else:
                        with results_lock:
                            all_issues.extend(issues)
                        success_count += 1
                            
                except Exception as e:
                    logger.error(f"Unexpected error processing {resource_type[:-1]} {resource_id}: {str(e)}")
                    failure_count += 1
        
        logger.info(f"Parallel fetch completed: {success_count} successful, {failure_count} failed. Total issues: {len(all_issues)}")
        return all_issues
    
    def list_gitlab_issues(self, project_id: str = None, group_id: str = None, state: str = "opened", 
                          assignee_username: str = None, labels: str = None, search: str = None,
                          assignee_id: int = None, author_id: int = None, author_username: str = None,
                          created_after: str = None, created_before: str = None,
                          due_date: str = None, epic_id: int = None, health_status: str = None,
                          in_scope: str = None, issue_type: str = None,
                          milestone: str = None, not_params: dict = None,
                          order_by: str = None, scope: str = None, sort: str = None,
                          updated_after: str = None, updated_before: str = None, weight: int = None,
                          with_labels_details: bool = None):
        logger.info(f"list_gitlab_issues called with parameters: project_id={project_id}, group_id={group_id}, state={state}, assignee_username={assignee_username}, labels={labels}, search={search}, assignee_id={assignee_id}, author_id={author_id}, author_username={author_username}, created_after={created_after}, created_before={created_before}, due_date={due_date}, epic_id={epic_id}, health_status={health_status}, in_scope={in_scope}, issue_type={issue_type}, milestone={milestone}, not_params={not_params}, order_by={order_by}, scope={scope}, sort={sort}, updated_after={updated_after}, updated_before={updated_before}, weight={weight}, with_labels_details={with_labels_details}")
        
        try:
            # Validate that both project_id and group_id are not provided simultaneously
            if project_id and group_id:
                logger.warning("Both project_id and group_id provided. Using project_id and ignoring group_id.")
                group_id = None
            
            # Validate state parameter
            valid_states = ['opened', 'closed', 'all']
            if state not in valid_states:
                logger.warning(f"Invalid state '{state}', defaulting to 'opened'. Valid states: {valid_states}")
                state = 'opened'
            
            params = {'state': state}
            
            # Add all the new parameters to the request
            if assignee_id is not None:
                params['assignee_id'] = assignee_id
            if assignee_username:
                params['assignee_username'] = assignee_username
            if author_id is not None:
                params['author_id'] = author_id
            if author_username:
                params['author_username'] = author_username
            if created_after:
                params['created_after'] = created_after
            if created_before:
                params['created_before'] = created_before
            if due_date:
                params['due_date'] = due_date
            if epic_id is not None:
                params['epic_id'] = epic_id
            if health_status:
                params['health_status'] = health_status
            if in_scope:
                params['in'] = in_scope
            if issue_type:
                params['issue_type'] = issue_type
            if labels:
                # Validate labels format (should be comma-separated)
                if isinstance(labels, str):
                    params['labels'] = labels
                else:
                    logger.warning(f"Labels should be a comma-separated string, got: {type(labels)}")
            if milestone:
                params['milestone'] = milestone
            if not_params:
                params['not'] = not_params
            if order_by:
                params['order_by'] = order_by
            if scope:
                params['scope'] = scope
            if search:
                params['search'] = search
            if sort:
                params['sort'] = sort
            if updated_after:
                params['updated_after'] = updated_after
            if updated_before:
                params['updated_before'] = updated_before
            if weight is not None:
                params['weight'] = weight
            if with_labels_details is not None:
                params['with_labels_details'] = with_labels_details
                
            all_issues = []
            import urllib.parse
            
            if project_id:
                # Parse project_id to handle multiple projects (comma-separated)
                project_ids = [pid.strip() for pid in project_id.split(',') if pid.strip()]
                logger.info(f"Parsed project IDs: {project_ids}")
                
                # Optimize for single project (avoid threading overhead)
                if len(project_ids) == 1:
                    encoded_project_id = urllib.parse.quote(project_ids[0], safe='')
                    endpoint = f'/projects/{encoded_project_id}/issues'
                    logger.info(f"Single project optimization: querying {project_ids[0]}")
                    all_issues = self._make_request(endpoint, params)
                else:
                    # Get issues from multiple projects in parallel
                    all_issues = self._fetch_issues_parallel(project_ids, params, 'projects')
                    
            elif group_id:
                # Parse group_id to handle multiple groups (comma-separated)
                group_ids = [gid.strip() for gid in group_id.split(',') if gid.strip()]
                logger.info(f"Parsed group IDs: {group_ids}")
                
                # Optimize for single group (avoid threading overhead)
                if len(group_ids) == 1:
                    encoded_group_id = urllib.parse.quote(group_ids[0], safe='')
                    endpoint = f'/groups/{encoded_group_id}/issues'
                    logger.info(f"Single group optimization: querying {group_ids[0]}")
                    all_issues = self._make_request(endpoint, params)
                else:
                    # Get issues from multiple groups in parallel
                    all_issues = self._fetch_issues_parallel(group_ids, params, 'groups')
                    
            else:
                # Get issues from all projects
                logger.info("Retrieving issues from all accessible projects")
                params['scope'] = 'all'
                logger.info(f"Querying all issues with params: {params}")
                all_issues = self._make_request('/issues', params)
            
            # Deduplicate issues based on issue ID (in case same issue appears in multiple groups/projects)
            seen_issue_ids = set()
            unique_issues = []
            duplicates_skipped = 0
            
            for issue in all_issues:
                issue_id = issue.get('id')
                if issue_id and issue_id not in seen_issue_ids:
                    seen_issue_ids.add(issue_id)
                    unique_issues.append(issue)
                elif issue_id:
                    duplicates_skipped += 1
                    logger.debug(f"Skipping duplicate issue ID: {issue_id}")
            
            logger.info(f"Retrieved {len(all_issues)} total issues, deduplicated to {len(unique_issues)} unique issues ({duplicates_skipped} duplicates skipped)")
            
            result = []
            for issue in unique_issues:
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
                    'weight': issue.get('weight'),
                    'web_url': issue.get('web_url')
                }
                result.append(issue_dict)
            
            logger.info(f"Returning {len(result)} formatted issue records")

            # Also compute statistics using the same filters so callers receive both issues and counts together
            try:
                statistics = self.get_issues_statistics(
                    project_id=project_id,
                    group_id=group_id,
                    labels=labels,
                    milestone=milestone,
                    scope=scope,
                    author_id=author_id,
                    author_username=author_username,
                    assignee_id=assignee_id,
                    assignee_username=assignee_username,
                    my_reaction_emoji=None,
                    iids=None,
                    search=search,
                    in_scope=in_scope,
                    created_after=created_after,
                    created_before=created_before,
                    updated_after=updated_after,
                    updated_before=updated_before,
                    confidential=None,
                    state=state,
                )
            except Exception as stats_error:
                logger.error(f"Failed to fetch issues statistics alongside list: {str(stats_error)}")
                statistics = {"error": f"issues statistics failed: {str(stats_error)}"}

            if len(result) > 15:
                total_issues_value = 'too large to display'
            else:
                total_issues_value = result

            return {"issues": total_issues_value, "issues_statistics": statistics}
        except Exception as e:
            logger.error(f"Error in list_gitlab_issues: {str(e)}")
            return {"error": f"list_gitlab_issues failed: {str(e)}"}

    @property
    def get_issue_detail_property(self):
        description = """
        Get detailed information about a specific GitLab issue. The response will be a dictionary containing comprehensive issue details such as id, iid, title, description, state, author, assignee, labels, milestone, weight, comments_count, time tracking, and more.
        This tool should be called when you need detailed information about a specific issue, including its full description, comments count, time estimates, and other metadata.
        You need to provide the global issue ID (the 'id' field from list_gitlab_issues, not the 'iid').
        This provides much more detailed information than list_gitlab_issues, including time tracking data, detailed assignee information, and issue metadata.
        """
        return {
            "type": "custom",
            "name": "get_issue_detail",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "issue_id": {
                        "type": "integer",
                        "description": "The global GitLab issue ID (the 'id' field from issue lists, not the 'iid')"
                    }
                },
                "required": ["issue_id"]
            }
        }
    
    def get_issue_detail(self, issue_id: int):
        logger.info(f"get_issue_detail called with issue_id: {issue_id}")
        
        try:
            logger.info(f"Retrieving detailed information for issue ID: {issue_id}")
            endpoint = f'/issues/{issue_id}'
            issue_detail = self._make_single_request(endpoint)
            
            if not issue_detail:
                logger.warning(f"Issue {issue_id} not found or not accessible")
                return {"error": f"Issue {issue_id} not found or not accessible"}
            
            # Extract comprehensive issue information
            author_info = issue_detail.get('author', {})
            assignee_info = issue_detail.get('assignee', {})
            milestone_info = issue_detail.get('milestone', {})
            time_stats = issue_detail.get('time_stats', {})
            
            result = {
                'id': issue_detail.get('id'),
                'iid': issue_detail.get('iid'),
                'project_id': issue_detail.get('project_id'),
                'title': issue_detail.get('title'),
                'description': issue_detail.get('description', ''),
                'state': issue_detail.get('state'),
                'created_at': issue_detail.get('created_at'),
                'updated_at': issue_detail.get('updated_at'),
                'closed_at': issue_detail.get('closed_at'),
                'closed_by': issue_detail.get('closed_by'),
                'labels': issue_detail.get('labels', []),
                'milestone': milestone_info.get('title', '') if milestone_info else '',
                'weight': issue_detail.get('weight'),
                'assignees': [assignee.get('name', '') for assignee in issue_detail.get('assignees', [])],
                'author': {
                    'id': author_info.get('id'),
                    'name': author_info.get('name', ''),
                    'username': author_info.get('username', ''),
                    'email': author_info.get('email', ''),
                    'web_url': author_info.get('web_url', '')
                } if author_info else {},
                'assignee': {
                    'id': assignee_info.get('id'),
                    'name': assignee_info.get('name', ''),
                    'username': assignee_info.get('username', ''),
                    'email': assignee_info.get('email', ''),
                    'web_url': assignee_info.get('web_url', '')
                } if assignee_info else {},
                'user_notes_count': issue_detail.get('user_notes_count', 0),
                'merge_requests_count': issue_detail.get('merge_requests_count', 0),
                'upvotes': issue_detail.get('upvotes', 0),
                'downvotes': issue_detail.get('downvotes', 0),
                'due_date': issue_detail.get('due_date'),
                'confidential': issue_detail.get('confidential', False),
                'discussion_locked': issue_detail.get('discussion_locked', False),
                'issue_type': issue_detail.get('issue_type', 'issue'),
                'web_url': issue_detail.get('web_url'),
                'time_stats': {
                    'time_estimate': time_stats.get('time_estimate', 0),
                    'total_time_spent': time_stats.get('total_time_spent', 0),
                    'human_time_estimate': time_stats.get('human_time_estimate'),
                    'human_total_time_spent': time_stats.get('human_total_time_spent')
                } if time_stats else {},
                'task_completion_status': issue_detail.get('task_completion_status', {}),
                'has_tasks': issue_detail.get('has_tasks', False),
                'task_status': issue_detail.get('task_status', ''),
                'blocking_issues_count': issue_detail.get('blocking_issues_count', 0),
                'blocked_by_issues_count': issue_detail.get('blocked_by_issues_count', 0)
            }
            
            # Include milestone details if available
            if milestone_info:
                result['milestone_detail'] = {
                    'id': milestone_info.get('id'),
                    'title': milestone_info.get('title', ''),
                    'description': milestone_info.get('description', ''),
                    'state': milestone_info.get('state', ''),
                    'created_at': milestone_info.get('created_at'),
                    'updated_at': milestone_info.get('updated_at'),
                    'due_date': milestone_info.get('due_date'),
                    'start_date': milestone_info.get('start_date'),
                    'web_url': milestone_info.get('web_url', '')
                }
            
            # Include references information if available
            references = issue_detail.get('references', {})
            if references:
                result['references'] = {
                    'short': references.get('short', ''),
                    'relative': references.get('relative', ''),
                    'full': references.get('full', '')
                }
            
            logger.info(f"Successfully retrieved detailed information for issue: {issue_detail.get('title')}")
            return result
            
        except Exception as e:
            logger.error(f"Error in get_issue_detail: {str(e)}")
            return {"error": f"get_issue_detail failed: {str(e)}"}

    def get_issues_statistics(
        self,
        project_id: str = None,
        group_id: str = None,
        labels: str = None,
        milestone: str = None,
        scope: str = None,
        author_id: int = None,
        author_username: str = None,
        assignee_id: int = None,
        assignee_username: str = None,
        my_reaction_emoji: str = None,
        iids: List[int] = None,
        search: str = None,
        in_scope: str = None,
        created_after: str = None,
        created_before: str = None,
        updated_after: str = None,
        updated_before: str = None,
        confidential: bool = None,
        state: str = None,
    ) -> str:
        logger.info(
            "get_issues_statistics called with params: project_id=%s, group_id=%s, labels=%s, milestone=%s, scope=%s, author_id=%s, author_username=%s, assignee_id=%s, assignee_username=%s, my_reaction_emoji=%s, iids=%s, search=%s, in_scope=%s, created_after=%s, created_before=%s, updated_after=%s, updated_before=%s, confidential=%s, state=%s",
            project_id, group_id, labels, milestone, scope, author_id, author_username, assignee_id, assignee_username, my_reaction_emoji, iids, search, in_scope, created_after, created_before, updated_after, updated_before, confidential, state,
        )

        try:
            # If both provided, prefer project scope
            if project_id and group_id:
                logger.warning("Both project_id and group_id provided. Using project_id and ignoring group_id.")
                group_id = None

            params: Dict[str, Any] = {}
            if labels:
                params["labels"] = labels
            if milestone:
                params["milestone"] = milestone
            if scope:
                params["scope"] = scope
            if author_id is not None:
                params["author_id"] = author_id
            if author_username:
                params["author_username"] = author_username
            if assignee_id is not None:
                params["assignee_id"] = assignee_id
            if assignee_username:
                params["assignee_username"] = assignee_username
            if my_reaction_emoji:
                params["my_reaction_emoji"] = my_reaction_emoji
            if iids:
                # GitLab expects repeated iids[]=42&iids[]=43
                # requests encodes list properly if key is 'iids[]'
                params["iids[]"] = iids
            if search:
                params["search"] = search
            if in_scope:
                params["in"] = in_scope
            if created_after:
                params["created_after"] = created_after
            if created_before:
                params["created_before"] = created_before
            if updated_after:
                params["updated_after"] = updated_after
            if updated_before:
                params["updated_before"] = updated_before
            if confidential is not None:
                params["confidential"] = confidential
            if state:
                params["state"] = state

            def merge_counts(acc: Dict[str, int], counts: Dict[str, int]) -> Dict[str, int]:
                for key in ("all", "opened", "closed"):
                    acc[key] = acc.get(key, 0) + int(counts.get(key, 0))
                return acc

            import urllib.parse

            # Determine scope and fetch
            detail_breakdown = []
            aggregated_counts: Dict[str, int] = {}

            # Build a clear, human-readable description of filters
            filter_parts = []
            if state:
                filter_parts.append(f"state={state}")
            if labels:
                filter_parts.append(f"labels={labels}")
            if milestone:
                filter_parts.append(f"milestone={milestone}")
            if scope:
                filter_parts.append(f"scope={scope}")
            if author_id is not None:
                filter_parts.append(f"author_id={author_id}")
            if author_username:
                filter_parts.append(f"author_username={author_username}")
            if assignee_id is not None:
                filter_parts.append(f"assignee_id={assignee_id}")
            if assignee_username:
                filter_parts.append(f"assignee_username={assignee_username}")
            if my_reaction_emoji:
                filter_parts.append(f"my_reaction_emoji={my_reaction_emoji}")
            if iids:
                filter_parts.append(f"iids={','.join(str(x) for x in iids)}")
            if search:
                filter_parts.append(f"search={search}")
            if in_scope:
                filter_parts.append(f"in={in_scope}")
            if created_after:
                filter_parts.append(f"created_after={created_after}")
            if created_before:
                filter_parts.append(f"created_before={created_before}")
            if updated_after:
                filter_parts.append(f"updated_after={updated_after}")
            if updated_before:
                filter_parts.append(f"updated_before={updated_before}")
            if confidential is not None:
                filter_parts.append(f"confidential={confidential}")
            filters_desc = ", ".join(filter_parts) if filter_parts else "no additional filters"

            if project_id:
                project_ids = [pid.strip() for pid in project_id.split(',') if pid.strip()]
                logger.info(f"Fetching issues statistics for projects: {project_ids}")
                for pid in project_ids:
                    encoded = urllib.parse.quote(pid, safe='')
                    data = self._make_single_request_with_params(f"/projects/{encoded}/issues_statistics", params)
                    if not data:
                        logger.warning(f"Failed to fetch statistics for project {pid}")
                        continue
                    counts = (data.get("statistics") or {}).get("counts") or {}
                    aggregated_counts = merge_counts(aggregated_counts, counts)
                    detail_breakdown.append({"project_id": pid, "counts": counts})
                # Compose human-readable string
                total_all = int(aggregated_counts.get("all", 0))
                total_open = int(aggregated_counts.get("opened", 0))
                total_closed = int(aggregated_counts.get("closed", 0))
                parts = [
                    f"GitLab issues statistics for project(s) {', '.join(project_ids)} with {filters_desc} — All: {total_all}, Open: {total_open}, Closed: {total_closed}"
                ]
                if detail_breakdown:
                    parts.append("Breakdown:")
                    for entry in detail_breakdown:
                        c = entry.get("counts", {}) or {}
                        parts.append(
                            f"- Project {entry.get('project_id')}: All {int(c.get('all', 0))}, Open {int(c.get('opened', 0))}, Closed {int(c.get('closed', 0))}"
                        )
                return "\n".join(parts)

            if group_id:
                group_ids = [gid.strip() for gid in group_id.split(',') if gid.strip()]
                logger.info(f"Fetching issues statistics for groups: {group_ids}")
                for gid in group_ids:
                    encoded = urllib.parse.quote(gid, safe='')
                    data = self._make_single_request_with_params(f"/groups/{encoded}/issues_statistics", params)
                    if not data:
                        logger.warning(f"Failed to fetch statistics for group {gid}")
                        continue
                    counts = (data.get("statistics") or {}).get("counts") or {}
                    aggregated_counts = merge_counts(aggregated_counts, counts)
                    detail_breakdown.append({"group_id": gid, "counts": counts})
                total_all = int(aggregated_counts.get("all", 0))
                total_open = int(aggregated_counts.get("opened", 0))
                total_closed = int(aggregated_counts.get("closed", 0))
                parts = [
                    f"GitLab issues statistics for group(s) {', '.join(group_ids)} with {filters_desc} — All: {total_all}, Open: {total_open}, Closed: {total_closed}"
                ]
                if detail_breakdown:
                    parts.append("Breakdown:")
                    for entry in detail_breakdown:
                        c = entry.get("counts", {}) or {}
                        parts.append(
                            f"- Group {entry.get('group_id')}: All {int(c.get('all', 0))}, Open {int(c.get('opened', 0))}, Closed {int(c.get('closed', 0))}"
                        )
                return "\n".join(parts)

            # Global scope
            logger.info("Fetching global issues statistics")
            data = self._make_single_request_with_params("/issues_statistics", params)
            if not data:
                return "Error: issues_statistics request returned no data"
            counts = (data.get("statistics") or {}).get("counts") or {}
            total_all = int(counts.get("all", 0))
            total_open = int(counts.get("opened", 0))
            total_closed = int(counts.get("closed", 0))
            return (
                f"GitLab issues statistics (all accessible projects) with {filters_desc} — "
                f"All: {total_all}, Open: {total_open}, Closed: {total_closed}"
            )

        except Exception as e:
            logger.error(f"Error in get_issues_statistics: {str(e)}")
            return f"Error: get_issues_statistics failed: {str(e)}"

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
                        "description": "GitLab project ID (not the project name) or comma-separated list of project IDs."
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
                return {"error": f"No user found with email: {email}"}
                
        except Exception as e:
            logger.error(f"Error in find_gitlab_user_by_email: {str(e)}")
            return {"error": f"find_gitlab_user_by_email failed: {str(e)}"}

    def verify_project_access(self, project_id: str):
        """
        Verify if a project exists and is accessible to debug 404 errors.
        
        Args:
            project_id: GitLab project ID (numeric) or project path
            
        Returns:
            Dict with project info if accessible, None if not found/accessible
        """
        logger.info(f"verify_project_access called with project_id: {project_id}")
        
        try:
            import urllib.parse
            encoded_project_id = urllib.parse.quote(project_id, safe='')
            endpoint = f'/projects/{encoded_project_id}'
            
            logger.info(f"Verifying access to project: {project_id}")
            logger.info(f"Encoded project ID: {project_id} -> {encoded_project_id}")
            logger.info(f"Verification URL: {self.gitlab_url}/api/v4{endpoint}")
            
            project_info = self._make_single_request(endpoint)
            
            if project_info:
                logger.info(f"Project accessible: {project_info.get('name')} (ID: {project_info.get('id')})")
                return {
                    'id': project_info.get('id'),
                    'name': project_info.get('name'),
                    'path_with_namespace': project_info.get('path_with_namespace'),
                    'visibility': project_info.get('visibility'),
                    'issues_enabled': project_info.get('issues_enabled', False)
                }
            else:
                logger.warning(f"Project {project_id} not found or not accessible")
                return {"error": f"Project {project_id} not found or not accessible"}
                
        except Exception as e:
            logger.error(f"Error verifying project access: {str(e)}")
            return {"error": f"verify_project_access failed: {str(e)}"}

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
            return {"error": f"list_gitlab_project_members failed: {str(e)}"}

    @property
    def list_gitlab_milestones_property(self):
        description = """
        List milestones for a GitLab project. Returns a list with fields like id, iid, project_id, title, description,
        due_date, start_date, state, updated_at, created_at, and expired. Accepts optional filters such as iids, state,
        title, search, include_ancestors, updated_before, and updated_after. The project can be specified by numeric ID
        or by URL-encoded path (group/subgroup/project).
        """
        return {
            "type": "custom",
            "name": "list_gitlab_milestones",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "GitLab project ID or URL-encoded path (e.g., group/project)"
                    },
                    "iids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Return only milestones having the given iid(s)"
                    },
                    "state": {
                        "type": "string",
                        "description": "Filter by milestone state: active or closed"
                    },
                    "title": {
                        "type": "string",
                        "description": "Return only milestones with the given title"
                    },
                    "search": {
                        "type": "string",
                        "description": "Return milestones where title or description matches the string"
                    },
                    "include_ancestors": {
                        "type": "boolean",
                        "description": "Include milestones from parent groups"
                    },
                    "updated_before": {
                        "type": "string",
                        "description": "ISO 8601 datetime. Return milestones updated before this time"
                    },
                    "updated_after": {
                        "type": "string",
                        "description": "ISO 8601 datetime. Return milestones updated after this time"
                    }
                },
                "required": ["project_id"]
            }
        }

    def list_gitlab_milestones(
        self,
        project_id: str,
        iids: List[int] = None,
        state: str = None,
        title: str = None,
        search: str = None,
        include_ancestors: bool = None,
        updated_before: str = None,
        updated_after: str = None,
    ) -> List[Dict[str, Any]]:
        logger.info(
            "list_gitlab_milestones called with params: project_id=%s, iids=%s, state=%s, title=%s, search=%s, include_ancestors=%s, updated_before=%s, updated_after=%s",
            project_id, iids, state, title, search, include_ancestors, updated_before, updated_after,
        )
        try:
            import urllib.parse
            encoded_project_id = urllib.parse.quote(project_id, safe='')

            params: Dict[str, Any] = {}
            if iids:
                params["iids[]"] = iids
            if state:
                # GitLab expects 'active' or 'closed'
                if state not in ("active", "closed"):
                    logger.warning("Invalid state '%s' for milestones. Expected 'active' or 'closed'", state)
                else:
                    params["state"] = state
            if title:
                params["title"] = title
            if search:
                params["search"] = search
            if include_ancestors is not None:
                params["include_ancestors"] = include_ancestors
            if updated_before:
                params["updated_before"] = updated_before
            if updated_after:
                params["updated_after"] = updated_after

            logger.info(f"Querying milestones for project: {project_id} with params: {params}")
            milestones = self._make_request(f"/projects/{encoded_project_id}/milestones", params)
            logger.info(f"Retrieved {len(milestones)} milestones from GitLab")

            result: List[Dict[str, Any]] = []
            for m in milestones:
                result.append({
                    "id": m.get("id"),
                    "iid": m.get("iid"),
                    "project_id": m.get("project_id"),
                    "title": m.get("title"),
                    "description": m.get("description", ""),
                    "due_date": m.get("due_date"),
                    "start_date": m.get("start_date"),
                    "state": m.get("state"),
                    "updated_at": m.get("updated_at"),
                    "created_at": m.get("created_at"),
                    "expired": m.get("expired", False),
                    "web_url": m.get("web_url", "")
                })

            logger.info(f"Returning {len(result)} formatted milestone records")
            return result
        except Exception as e:
            logger.error(f"Error in list_gitlab_milestones: {str(e)}")
            return {"error": f"list_gitlab_milestones failed: {str(e)}"}

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
                        "description": "GitLab project ID (not the project name) or comma-separated list of project IDs."
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
                return {"error": f"Project {project_id} not found or not accessible"}
            
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
            return {"error": f"get_gitlab_project_detail failed: {str(e)}"}

    @property
    def list_gitlab_groups_property(self):
        description = """
        List GitLab groups accessible to the user. The response will be a list of dictionaries, each containing group details such as id, name, path, description, web_url, visibility, and member count.
        This tool should be called to retrieve group information when group details are needed to answer questions.
        You can filter groups by visibility (public, internal, private) or search by name.
        group_id can be used to link groups to projects, subgroups, and other GitLab resources.
        Groups in GitLab are used to organize projects and manage permissions at a higher level.
        """
        return {
            "type": "custom",
            "name": "list_gitlab_groups",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Search term to filter groups by name or path"
                    },
                    "owned": {
                        "type": "boolean",
                        "description": "If true, only return groups owned by the authenticated user"
                    },
                    "top_level_only": {
                        "type": "boolean",
                        "description": "If true, only return top-level groups (no subgroups)"
                    }
                }
            }
        }
    
    def list_gitlab_groups(self, search: str = None, owned: bool = False, top_level_only: bool = False):
        logger.info(f"list_gitlab_groups called with parameters: search={search}, owned={owned}, top_level_only={top_level_only}")
        try:
            params = {}
            if search:
                params['search'] = search
            if owned:
                params['owned'] = True
            if top_level_only:
                params['top_level_only'] = True
            
            logger.info(f"Querying GitLab groups with params: {params}")
            groups = self._make_request('/groups', params)
            logger.info(f"Retrieved {len(groups)} groups from GitLab")
            
            result = []
            for group in groups:
                group_dict = {
                    'id': group.get('id'),
                    'name': group.get('name'),
                    'path': group.get('path'),
                    'full_path': group.get('full_path'),
                    'description': group.get('description', ''),
                    'web_url': group.get('web_url'),
                    'visibility': group.get('visibility'),
                    'created_at': group.get('created_at'),
                    'parent_id': group.get('parent_id'),
                    'projects_count': len(group.get('projects', [])),
                    'subgroups_count': len(group.get('shared_projects', [])),
                    'avatar_url': group.get('avatar_url', '')
                }
                result.append(group_dict)
            
            logger.info(f"Returning {len(result)} formatted group records")
            return result
        except Exception as e:
            logger.error(f"Error in list_gitlab_groups: {str(e)}")
            return {"error": f"list_gitlab_groups failed: {str(e)}"}

    @property
    def get_gitlab_group_detail_property(self):
        description = """
        Get detailed information about a specific GitLab group. The response will be a dictionary containing comprehensive group details such as id, name, description, members, projects, subgroups, permissions, and settings.
        This tool should be called when you need detailed information about a specific group, such as its members, projects, subgroups, or group settings.
        You need to provide the group ID (numeric) or the group path (group-name or parent-group/subgroup format).
        This provides much more detailed information than list_gitlab_groups, including member information, project lists, and group configuration details.
        """
        return {
            "type": "custom",
            "name": "get_gitlab_group_detail",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "group_id": {
                        "type": "string",
                        "description": "GitLab group ID (not the group name) or comma-separated list of group IDs."
                    },
                    "include_projects": {
                        "type": "boolean",
                        "description": "Include list of projects in the group",
                        "default": True
                    },
                    "include_subgroups": {
                        "type": "boolean", 
                        "description": "Include list of subgroups",
                        "default": True
                    }
                },
                "required": ["group_id"]
            }
        }
    
    def get_gitlab_group_detail(self, group_id: str, include_projects: bool = True, include_subgroups: bool = True):
        logger.info(f"get_gitlab_group_detail called with group_id: {group_id}, include_projects: {include_projects}, include_subgroups: {include_subgroups}")
        
        try:
            # URL encode the group_id in case it contains special characters like '/'
            import urllib.parse
            encoded_group_id = urllib.parse.quote(group_id, safe='')
            
            logger.info(f"Retrieving detailed information for group: {group_id}")
            group_detail = self._make_single_request(f'/groups/{encoded_group_id}')
            
            if not group_detail:
                logger.warning(f"Group {group_id} not found or not accessible")
                return {"error": f"Group {group_id} not found or not accessible"}
            
            # Extract essential group information
            result = {
                'id': group_detail.get('id'),
                'name': group_detail.get('name'),
                'path': group_detail.get('path'),
                'full_path': group_detail.get('full_path'),
                'description': group_detail.get('description', ''),
                'web_url': group_detail.get('web_url'),
                'visibility': group_detail.get('visibility'),
                'created_at': group_detail.get('created_at'),
                'parent_id': group_detail.get('parent_id'),
                'avatar_url': group_detail.get('avatar_url', ''),
                'shared_runners_minutes_limit': group_detail.get('shared_runners_minutes_limit'),
                'extra_shared_runners_minutes_limit': group_detail.get('extra_shared_runners_minutes_limit'),
                'prevent_forking_outside_group': group_detail.get('prevent_forking_outside_group', False),
                'membership_lock': group_detail.get('membership_lock', False),
                'share_with_group_lock': group_detail.get('share_with_group_lock', False),
                'require_two_factor_authentication': group_detail.get('require_two_factor_authentication', False),
                'two_factor_grace_period': group_detail.get('two_factor_grace_period', 0),
                'auto_devops_enabled': group_detail.get('auto_devops_enabled'),
                'emails_disabled': group_detail.get('emails_disabled', False),
                'mentions_disabled': group_detail.get('mentions_disabled', False),
                'lfs_enabled': group_detail.get('lfs_enabled', True),
                'default_branch_protection': group_detail.get('default_branch_protection', 0),
                'request_access_enabled': group_detail.get('request_access_enabled', True)
            }
            
            # Get group members
            try:
                logger.info(f"Retrieving members for group: {group_id}")
                members = self._make_request(f'/groups/{encoded_group_id}/members')
                
                # Access level mapping for better readability
                access_level_names = {
                    10: "Guest",
                    20: "Reporter", 
                    30: "Developer",
                    40: "Maintainer",
                    50: "Owner"
                }
                
                member_list = []
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
                    member_list.append(member_dict)
                
                result['members'] = member_list
                result['members_count'] = len(member_list)
                logger.info(f"Retrieved {len(member_list)} members for group")
                
            except Exception as e:
                logger.error(f"Error retrieving group members: {str(e)}")
                result['members_error'] = f"group members failed: {str(e)}"
            
            # Get group projects if requested
            if include_projects:
                try:
                    logger.info(f"Retrieving projects for group: {group_id}")
                    projects = self._make_request(f'/groups/{encoded_group_id}/projects')
                    
                    project_list = []
                    for project in projects:
                        project_dict = {
                            'id': project.get('id'),
                            'name': project.get('name'),
                            'path': project.get('path'),
                            'path_with_namespace': project.get('path_with_namespace'),
                            'description': project.get('description', ''),
                            'web_url': project.get('web_url'),
                            'visibility': project.get('visibility'),
                            'created_at': project.get('created_at'),
                            'last_activity_at': project.get('last_activity_at'),
                            'archived': project.get('archived', False)
                        }
                        project_list.append(project_dict)
                    
                    result['projects'] = project_list
                    result['projects_count'] = len(project_list)
                    logger.info(f"Retrieved {len(project_list)} projects for group")
                    
                except Exception as e:
                    logger.error(f"Error retrieving group projects: {str(e)}")
                    result['projects_error'] = f"group projects failed: {str(e)}"
            
            # Get subgroups if requested
            if include_subgroups:
                try:
                    logger.info(f"Retrieving subgroups for group: {group_id}")
                    subgroups = self._make_request(f'/groups/{encoded_group_id}/subgroups')
                    
                    subgroup_list = []
                    for subgroup in subgroups:
                        subgroup_dict = {
                            'id': subgroup.get('id'),
                            'name': subgroup.get('name'),
                            'path': subgroup.get('path'),
                            'full_path': subgroup.get('full_path'),
                            'description': subgroup.get('description', ''),
                            'web_url': subgroup.get('web_url'),
                            'visibility': subgroup.get('visibility'),
                            'created_at': subgroup.get('created_at'),
                            'parent_id': subgroup.get('parent_id')
                        }
                        subgroup_list.append(subgroup_dict)
                    
                    result['subgroups'] = subgroup_list
                    result['subgroups_count'] = len(subgroup_list)
                    logger.info(f"Retrieved {len(subgroup_list)} subgroups for group")
                    
                except Exception as e:
                    logger.error(f"Error retrieving subgroups: {str(e)}")
                    result['subgroups_error'] = f"group subgroups failed: {str(e)}"
            
            logger.info(f"Successfully retrieved detailed information for group: {group_detail.get('name')}")
            return result
            
        except Exception as e:
            logger.error(f"Error in get_gitlab_group_detail: {str(e)}")
            return {"error": f"get_gitlab_group_detail failed: {str(e)}"}