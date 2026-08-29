import logging
from typing import List, Dict, Optional
import requests
import json
import time
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class WorkdayTool:
    """
    Workday integration tool for employee operations.
    Uses OAuth 2.0 Client Credentials flow for authentication.
    """

    def __init__(self, client_id: str = None, client_secret: str = None,
                 tenant_id: str = None, base_url: str = None):
        """
        Initialize WorkdayTool with OAuth 2.0 credentials.

        Args:
            client_id: Workday OAuth client ID
            client_secret: Workday OAuth client secret
            tenant_id: Workday tenant ID
            base_url: Workday base URL (e.g., https://wd2-impl-services1.workday.com)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.base_url = base_url.rstrip('/') if base_url else None

        # Token management
        self._access_token = None
        self._token_expires_at = None

        # API version (can be made configurable if needed)
        self.api_version = "v42.0"

        logger.info(f"WorkdayTool initialized: tenant={tenant_id}, base_url={base_url}")

    def _get_access_token(self) -> str:
        """
        Get a fresh access token using OAuth 2.0 Client Credentials flow.

        Returns:
            Access token string

        Raises:
            Exception: If token request fails
        """
        if not all([self.client_id, self.client_secret, self.tenant_id, self.base_url]):
            raise ValueError("Workday credentials not configured (client_id, client_secret, tenant_id, base_url required)")

        token_url = f"{self.base_url}/ccx/oauth2/{self.tenant_id}/token"

        data = {
            'grant_type': 'client_credentials'
        }

        auth = (self.client_id, self.client_secret)

        try:
            response = requests.post(token_url, data=data, auth=auth, timeout=30)
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data['access_token']

            # Calculate expiration time (default to 1 hour if not provided)
            expires_in = token_data.get('expires_in', 3600)
            self._token_expires_at = time.time() + expires_in - 60  # Refresh 1 minute early

            logger.debug("Successfully obtained Workday access token")
            return self._access_token

        except requests.HTTPError as e:
            logger.error(
                "Workday token request failed (status=%s)",
                e.response.status_code,
            )
            raise Exception(f"Failed to get Workday access token: {e.response.status_code}")
        except Exception as e:
            logger.error(
                "Error getting Workday access token (error_type=%s)",
                type(e).__name__,
            )
            raise

    def _ensure_valid_token(self) -> str:
        """
        Ensure we have a valid access token, refreshing if necessary.

        Returns:
            Valid access token
        """
        if not self._access_token or (self._token_expires_at and time.time() >= self._token_expires_at):
            return self._get_access_token()
        return self._access_token

    def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        Make an authenticated request to the Workday API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data

        Returns:
            Response data as dictionary or error dictionary
        """
        try:
            access_token = self._ensure_valid_token()
            url = f"{self.base_url}{endpoint}"

            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }

            logger.debug(f"Making {method} request to {url}")

            if data:
                data = json.dumps(data)

            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                timeout=60
            )

            response.raise_for_status()

            if response.content:
                return response.json()
            return {}

        except requests.HTTPError as e:
            error_msg = f"Workday API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                if 'errors' in error_data:
                    error_msg = error_data['errors'][0].get('message', error_msg)
                else:
                    error_msg = error_data.get('message', error_msg)
            except:
                error_msg = e.response.text or error_msg
            logger.error(
                "Workday API request failed (status=%s)",
                getattr(e.response, "status_code", None),
            )
            return {"error": error_msg}
        except Exception as e:
            logger.error(f"Error making Workday API request: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def get_employee_property(self):
        description = """
        Get detailed information about a specific Workday employee by their worker ID.
        Returns comprehensive employee data including personal information, employment details, and organizational data.
        """
        return {
            "type": "custom",
            "name": "workday_get_employee",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "worker_id": {
                        "type": "string",
                        "description": "Workday Worker ID (e.g., '123456' or 'EMP001')"
                    }
                },
                "required": ["worker_id"]
            }
        }

    def get_employee(self, worker_id: str) -> Dict:
        """
        Get detailed information about a specific employee.

        Args:
            worker_id: Workday Worker ID

        Returns:
            Employee details dictionary
        """
        logger.info(f"Executing get_employee for worker_id: {worker_id}")

        endpoint = f"/ccx/service/{self.tenant_id}/Human_Resources/{self.api_version}/Workers/{worker_id}"

        result = self._make_request('GET', endpoint)

        if 'error' in result:
            return result

        # Workday returns data in a specific structure
        # Parse and format the response
        try:
            if 'data' in result:
                worker_data = result['data']
            else:
                worker_data = result

            # Extract relevant employee information
            formatted_employee = self._format_employee_data(worker_data)
            return formatted_employee

        except Exception as e:
            logger.error(f"Error parsing employee data: {str(e)}")
            return {"error": f"Failed to parse employee data: {str(e)}"}

    @property
    def list_employees_property(self):
        description = """
        List Workday employees with optional filters. Returns a list of employees with basic information.
        Supports filtering by department, job title, manager, or other criteria.
        """
        return {
            "type": "custom",
            "name": "workday_list_employees",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of employees to return. Defaults to 50 if not specified (max: 200)."
                    },
                    "department": {
                        "type": "string",
                        "description": "Filter by department name (optional)"
                    },
                    "manager_id": {
                        "type": "string",
                        "description": "Filter by manager's worker ID (optional)"
                    },
                    "active_only": {
                        "type": "boolean",
                        "description": "Include only active employees. Defaults to true."
                    }
                },
                "required": []
            }
        }

    def list_employees(self, limit: int = 50, department: str = None,
                      manager_id: str = None, active_only: bool = True) -> List[Dict]:
        """
        List employees with optional filters.

        Args:
            limit: Maximum number of employees to return
            department: Filter by department name
            manager_id: Filter by manager's worker ID
            active_only: Include only active employees

        Returns:
            List of employee dictionaries
        """
        logger.info(f"Executing list_employees with limit: {limit}, department: {department}, manager_id: {manager_id}")

        endpoint = f"/ccx/service/{self.tenant_id}/Human_Resources/{self.api_version}/Workers"

        # Build query parameters
        params = {
            'limit': min(limit, 200)  # Workday API limit
        }

        # Add filters if provided
        if department:
            params['department'] = department
        if manager_id:
            params['manager'] = manager_id
        if active_only:
            params['status'] = 'Active'

        result = self._make_request('GET', endpoint, params=params)

        if 'error' in result:
            return result

        try:
            # Parse Workday response format
            if 'data' in result and isinstance(result['data'], list):
                employees_data = result['data']
            elif isinstance(result, list):
                employees_data = result
            else:
                # Try to extract workers from various response formats
                employees_data = result.get('workers', result.get('data', []))

            formatted_employees = []
            for employee_data in employees_data[:limit]:
                formatted_employee = self._format_employee_data(employee_data)
                formatted_employees.append(formatted_employee)

            return formatted_employees

        except Exception as e:
            logger.error(f"Error parsing employees list: {str(e)}")
            return {"error": f"Failed to parse employees data: {str(e)}"}

    @property
    def search_employees_property(self):
        description = """
        Search Workday employees by name, email, or other criteria. Returns matching employees with their basic information.
        Useful for finding employees when you have partial information like name or email.
        """
        return {
            "type": "custom",
            "name": "workday_search_employees",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (name, email, or other employee identifier)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Defaults to 20 if not specified (max: 50)."
                    }
                },
                "required": ["query"]
            }
        }

    def search_employees(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search employees by query string.

        Args:
            query: Search query (name, email, etc.)
            limit: Maximum number of results to return

        Returns:
            List of matching employee dictionaries
        """
        logger.info(
            "Executing search_employees (query_chars=%d, limit=%d)",
            len(query), limit,
        )

        endpoint = f"/ccx/service/{self.tenant_id}/Human_Resources/{self.api_version}/Workers"

        # Build search parameters
        params = {
            'search': query,
            'limit': min(limit, 50)  # Reasonable limit for search
        }

        result = self._make_request('GET', endpoint, params=params)

        if 'error' in result:
            return result

        try:
            # Parse search results
            if 'data' in result and isinstance(result['data'], list):
                employees_data = result['data']
            elif isinstance(result, list):
                employees_data = result
            else:
                employees_data = result.get('workers', result.get('data', []))

            formatted_employees = []
            for employee_data in employees_data[:limit]:
                formatted_employee = self._format_employee_data(employee_data)
                formatted_employees.append(formatted_employee)

            return formatted_employees

        except Exception as e:
            logger.error(f"Error parsing search results: {str(e)}")
            return {"error": f"Failed to parse search results: {str(e)}"}

    def _format_employee_data(self, employee_data: Dict) -> Dict:
        """
        Format raw Workday employee data into a consistent structure.

        Args:
            employee_data: Raw employee data from Workday API

        Returns:
            Formatted employee dictionary
        """
        try:
            # Extract basic employee information
            # Note: This structure may need adjustment based on actual Workday API response format

            formatted = {
                'id': employee_data.get('id') or employee_data.get('worker_id'),
                'worker_id': employee_data.get('worker_id') or employee_data.get('id'),
                'name': self._extract_employee_name(employee_data),
                'email': self._extract_employee_email(employee_data),
                'job_title': self._extract_job_title(employee_data),
                'department': self._extract_department(employee_data),
                'manager': self._extract_manager(employee_data),
                'status': employee_data.get('status') or employee_data.get('employment_status'),
                'hire_date': employee_data.get('hire_date') or employee_data.get('original_hire_date'),
                'location': self._extract_location(employee_data)
            }

            # Remove None values
            formatted = {k: v for k, v in formatted.items() if v is not None}

            return formatted

        except Exception as e:
            logger.error(f"Error formatting employee data: {str(e)}")
            return {"error": f"Failed to format employee data: {str(e)}"}

    def _extract_employee_name(self, data: Dict) -> Optional[str]:
        """Extract employee name from various possible locations in the data."""
        # Try different possible paths for employee name
        name_paths = [
            ['name'],
            ['personal_data', 'name'],
            ['worker_data', 'personal_data', 'name'],
            ['full_name']
        ]

        for path in name_paths:
            value = self._get_nested_value(data, path)
            if value:
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict):
                    # Handle structured name data
                    first = value.get('first_name') or value.get('given_name')
                    last = value.get('last_name') or value.get('family_name')
                    if first and last:
                        return f"{first} {last}"
                    elif first:
                        return first
                    elif last:
                        return last

        return None

    def _extract_employee_email(self, data: Dict) -> Optional[str]:
        """Extract employee email from various possible locations."""
        email_paths = [
            ['email'],
            ['personal_data', 'email'],
            ['contact_data', 'email'],
            ['work_email']
        ]

        for path in email_paths:
            value = self._get_nested_value(data, path)
            if value and isinstance(value, str):
                return value

        return None

    def _extract_job_title(self, data: Dict) -> Optional[str]:
        """Extract job title from employment data."""
        title_paths = [
            ['job_title'],
            ['position_title'],
            ['employment_data', 'job_title'],
            ['position_data', 'title']
        ]

        for path in title_paths:
            value = self._get_nested_value(data, path)
            if value and isinstance(value, str):
                return value

        return None

    def _extract_department(self, data: Dict) -> Optional[str]:
        """Extract department from organizational data."""
        dept_paths = [
            ['department'],
            ['org_data', 'department'],
            ['organization_data', 'department_name']
        ]

        for path in dept_paths:
            value = self._get_nested_value(data, path)
            if value and isinstance(value, str):
                return value

        return None

    def _extract_manager(self, data: Dict) -> Optional[str]:
        """Extract manager name."""
        manager_paths = [
            ['manager'],
            ['manager_name'],
            ['supervisor'],
            ['org_data', 'manager']
        ]

        for path in manager_paths:
            value = self._get_nested_value(data, path)
            if value:
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict):
                    return value.get('name')

        return None

    def _extract_location(self, data: Dict) -> Optional[str]:
        """Extract work location."""
        location_paths = [
            ['location'],
            ['work_location'],
            ['office_location'],
            ['employment_data', 'location']
        ]

        for path in location_paths:
            value = self._get_nested_value(data, path)
            if value and isinstance(value, str):
                return value

        return None

    def _get_nested_value(self, data: Dict, path: List[str]) -> any:
        """Get nested value from dictionary using path list."""
        current = data
        try:
            for key in path:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return None
