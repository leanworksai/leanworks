import logging
from typing import List, Dict, Optional
import requests
import json

logger = logging.getLogger(__name__)

class NotionTool:
    def __init__(self, integration_token: str = None):
        """
        Initialize NotionTool with Notion API credentials.
        
        Args:
            integration_token: Notion integration token (internal integration)
        """
        self.integration_token = integration_token
        self.base_url = "https://api.notion.com/v1"
        self.api_version = "2022-06-28"
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Make an HTTP request to the Notion API.
        
        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            endpoint: API endpoint (relative to base_url)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response data as dictionary or error dictionary
        """
        if not self.integration_token:
            return {"error": "Notion credentials not configured"}
        
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            headers = kwargs.pop('headers', {})
            headers.setdefault('Authorization', f'Bearer {self.integration_token}')
            headers.setdefault('Notion-Version', self.api_version)
            headers.setdefault('Content-Type', 'application/json')
            
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                **kwargs
            )
            
            if response.status_code >= 400:
                error_msg = f"Notion API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('message', error_msg)
                    if 'code' in error_data:
                        error_msg = f"{error_data['code']}: {error_msg}"
                except:
                    error_msg = response.text or error_msg
                logger.error("Notion API request failed (status=%s)", response.status_code)
                return {"error": error_msg}
            
            if response.content:
                return response.json()
            return {}
            
        except Exception as e:
            logger.error(f"Error making Notion API request: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def search_pages_property(self):
        description = """
        Search for pages in Notion workspace. Returns a list of pages matching the query.
        This tool searches across all pages the integration has access to.
        """
        return {
            "type": "custom",
            "name": "notion_search_pages",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string (optional, searches all pages if not provided)"
                    },
                    "filter": {
                        "type": "string",
                        "description": "Filter type: 'page' or 'database'. Optional, searches both if not specified."
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort direction: 'ascending' or 'descending'. Defaults to 'descending'."
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of results per page. Defaults to 100 if not specified (max 100)."
                    }
                },
                "required": []
            }
        }
    
    def search_pages(self, query: str = None, filter: str = None, sort: str = "descending", page_size: int = 100) -> List[Dict]:
        """
        Search for pages in Notion workspace.
        
        Args:
            query: Search query string (optional)
            filter: Filter type ('page' or 'database') (optional)
            sort: Sort direction ('ascending' or 'descending') (default: 'descending')
            page_size: Number of results per page (default: 100, max: 100)
            
        Returns:
            List of page dictionaries
        """
        logger.info(
            "Executing Notion search_pages (query_chars=%d, has_filter=%s)",
            len(query), bool(filter),
        )
        try:
            payload = {
                "page_size": min(page_size, 100)
            }
            
            if query:
                payload["query"] = query
            
            if filter:
                if filter.lower() == "page":
                    payload["filter"] = {"property": "object", "value": "page"}
                elif filter.lower() == "database":
                    payload["filter"] = {"property": "object", "value": "database"}
            
            if sort:
                payload["sort"] = {
                    "direction": sort.lower(),
                    "timestamp": "last_edited_time"
                }
            
            result = self._make_request('POST', '/search', json=payload)
            
            if 'error' in result:
                return result
            
            pages = result.get('results', [])
            formatted_pages = []
            
            for page in pages:
                formatted_page = {
                    'id': page.get('id'),
                    'object': page.get('object'),
                    'created_time': page.get('created_time'),
                    'last_edited_time': page.get('last_edited_time'),
                    'url': page.get('url'),
                }
                
                # Extract title from properties
                if 'properties' in page:
                    props = page['properties']
                    # Title is usually in a property with type 'title'
                    for prop_name, prop_data in props.items():
                        if isinstance(prop_data, dict) and prop_data.get('type') == 'title':
                            title_parts = prop_data.get('title', [])
                            if title_parts:
                                formatted_page['title'] = ''.join([part.get('plain_text', '') for part in title_parts])
                                break
                
                # Fallback to extracting from title property directly
                if 'title' not in formatted_page:
                    title_parts = page.get('title', [])
                    if title_parts:
                        formatted_page['title'] = ''.join([part.get('plain_text', '') for part in title_parts])
                
                formatted_pages.append(formatted_page)
            
            return formatted_pages
            
        except Exception as e:
            logger.error(f"Error searching pages: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def get_page_property(self):
        description = """
        Get detailed information about a specific Notion page by its ID. Returns complete page content including properties and blocks.
        """
        return {
            "type": "custom",
            "name": "notion_get_page",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID (UUID format)"
                    }
                },
                "required": ["page_id"]
            }
        }
    
    def get_page(self, page_id: str) -> Dict:
        """
        Get detailed information about a specific page.
        
        Args:
            page_id: Notion page ID (UUID)
            
        Returns:
            Complete page details including properties and content blocks
        """
        logger.info(f"Executing get_page for page_id: {page_id}")
        try:
            # Get page properties
            result = self._make_request('GET', f'/pages/{page_id}')
            
            if 'error' in result:
                return result
            
            # Get page content blocks
            blocks_result = self._make_request('GET', f'/blocks/{page_id}/children')
            blocks = []
            if 'error' not in blocks_result:
                blocks = blocks_result.get('results', [])
            
            formatted_page = {
                'id': result.get('id'),
                'object': result.get('object'),
                'created_time': result.get('created_time'),
                'last_edited_time': result.get('last_edited_time'),
                'url': result.get('url'),
                'properties': result.get('properties', {}),
                'blocks': blocks
            }
            
            # Extract title
            props = result.get('properties', {})
            for prop_name, prop_data in props.items():
                if isinstance(prop_data, dict) and prop_data.get('type') == 'title':
                    title_parts = prop_data.get('title', [])
                    if title_parts:
                        formatted_page['title'] = ''.join([part.get('plain_text', '') for part in title_parts])
                        break
            
            return formatted_page
            
        except Exception as e:
            logger.error(f"Error getting page: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def create_page_property(self):
        description = """
        Create a new page in Notion. Can create a page in a parent page or database.
        """
        return {
            "type": "custom",
            "name": "notion_create_page",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "parent_id": {
                        "type": "string",
                        "description": "Parent page ID or database ID where to create the page"
                    },
                    "parent_type": {
                        "type": "string",
                        "description": "Parent type: 'page_id' or 'database_id'. Defaults to 'page_id'."
                    },
                    "title": {
                        "type": "string",
                        "description": "Page title"
                    },
                    "properties": {
                        "type": "string",
                        "description": "JSON string of page properties (optional, for database pages)"
                    }
                },
                "required": ["parent_id", "title"]
            }
        }
    
    def create_page(self, parent_id: str, title: str, parent_type: str = "page_id", properties: str = None) -> Dict:
        """
        Create a new page in Notion.
        
        Args:
            parent_id: Parent page ID or database ID
            title: Page title
            parent_type: Parent type ('page_id' or 'database_id') (default: 'page_id')
            properties: JSON string of page properties (optional, for database pages)
            
        Returns:
            Created page details
        """
        logger.info(f"Executing create_page with parent_id: {parent_id}, title: {title}")
        try:
            payload = {
                "parent": {
                    parent_type: parent_id
                },
                "properties": {
                    "title": {
                        "title": [
                            {
                                "text": {
                                    "content": title
                                }
                            }
                        ]
                    }
                }
            }
            
            # If properties provided, merge them (for database pages)
            if properties:
                try:
                    props_dict = json.loads(properties)
                    payload["properties"].update(props_dict)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in properties: {properties}")
            
            result = self._make_request('POST', '/pages', json=payload)
            
            if 'error' in result:
                return result
            
            formatted_page = {
                'id': result.get('id'),
                'url': result.get('url'),
                'created_time': result.get('created_time'),
                'last_edited_time': result.get('last_edited_time'),
                'properties': result.get('properties', {})
            }
            
            # Extract title
            props = result.get('properties', {})
            for prop_name, prop_data in props.items():
                if isinstance(prop_data, dict) and prop_data.get('type') == 'title':
                    title_parts = prop_data.get('title', [])
                    if title_parts:
                        formatted_page['title'] = ''.join([part.get('plain_text', '') for part in title_parts])
                        break
            
            return formatted_page
            
        except Exception as e:
            logger.error(f"Error creating page: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def update_page_property(self):
        description = """
        Update an existing Notion page. Can update page properties and content.
        """
        return {
            "type": "custom",
            "name": "notion_update_page",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID (UUID format)"
                    },
                    "properties": {
                        "type": "string",
                        "description": "JSON string of properties to update"
                    },
                    "archived": {
                        "type": "boolean",
                        "description": "Whether to archive the page. Defaults to false."
                    }
                },
                "required": ["page_id"]
            }
        }
    
    def update_page(self, page_id: str, properties: str = None, archived: bool = False) -> Dict:
        """
        Update an existing page.
        
        Args:
            page_id: Notion page ID (UUID)
            properties: JSON string of properties to update (optional)
            archived: Whether to archive the page (default: False)
            
        Returns:
            Updated page details
        """
        logger.info(f"Executing update_page for page_id: {page_id}")
        try:
            payload = {}
            
            if archived:
                payload["archived"] = True
            elif properties:
                try:
                    props_dict = json.loads(properties)
                    payload["properties"] = props_dict
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in properties: {properties}")
                    return {"error": "Invalid JSON format in properties"}
            
            if not payload:
                return {"error": "No properties or archived flag provided"}
            
            result = self._make_request('PATCH', f'/pages/{page_id}', json=payload)
            
            if 'error' in result:
                return result
            
            formatted_page = {
                'id': result.get('id'),
                'url': result.get('url'),
                'last_edited_time': result.get('last_edited_time'),
                'properties': result.get('properties', {}),
                'archived': result.get('archived', False)
            }
            
            return formatted_page
            
        except Exception as e:
            logger.error(f"Error updating page: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def archive_page_property(self):
        description = """
        Archive (delete) a Notion page. Archived pages can be restored later.
        """
        return {
            "type": "custom",
            "name": "notion_archive_page",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID (UUID format)"
                    }
                },
                "required": ["page_id"]
            }
        }
    
    def archive_page(self, page_id: str) -> Dict:
        """
        Archive a page.
        
        Args:
            page_id: Notion page ID (UUID)
            
        Returns:
            Archived page details
        """
        logger.info(f"Executing archive_page for page_id: {page_id}")
        return self.update_page(page_id, archived=True)
    
    @property
    def query_database_property(self):
        description = """
        Query a Notion database to retrieve entries. Supports filtering, sorting, and pagination.
        """
        return {
            "type": "custom",
            "name": "notion_query_database",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "Notion database ID (UUID format)"
                    },
                    "filter": {
                        "type": "string",
                        "description": "JSON string of filter conditions (optional)"
                    },
                    "sorts": {
                        "type": "string",
                        "description": "JSON string of sort conditions (optional)"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of results per page. Defaults to 100 if not specified (max 100)."
                    },
                    "start_cursor": {
                        "type": "string",
                        "description": "Pagination cursor from previous query (optional)"
                    }
                },
                "required": ["database_id"]
            }
        }
    
    def query_database(self, database_id: str, filter: str = None, sorts: str = None, 
                      page_size: int = 100, start_cursor: str = None) -> Dict:
        """
        Query a database to retrieve entries.
        
        Args:
            database_id: Notion database ID (UUID)
            filter: JSON string of filter conditions (optional)
            sorts: JSON string of sort conditions (optional)
            page_size: Number of results per page (default: 100, max: 100)
            start_cursor: Pagination cursor (optional)
            
        Returns:
            Dictionary with results and pagination info
        """
        logger.info(f"Executing query_database for database_id: {database_id}")
        try:
            payload = {
                "page_size": min(page_size, 100)
            }
            
            if filter:
                try:
                    filter_dict = json.loads(filter)
                    payload["filter"] = filter_dict
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in filter: {filter}")
            
            if sorts:
                try:
                    sorts_list = json.loads(sorts)
                    payload["sorts"] = sorts_list
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in sorts: {sorts}")
            
            if start_cursor:
                payload["start_cursor"] = start_cursor
            
            result = self._make_request('POST', f'/databases/{database_id}/query', json=payload)
            
            if 'error' in result:
                return result
            
            formatted_result = {
                'results': result.get('results', []),
                'has_more': result.get('has_more', False),
                'next_cursor': result.get('next_cursor'),
                'object': result.get('object')
            }
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def get_database_property(self):
        description = """
        Get detailed information about a Notion database including its schema and properties.
        """
        return {
            "type": "custom",
            "name": "notion_get_database",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "Notion database ID (UUID format)"
                    }
                },
                "required": ["database_id"]
            }
        }
    
    def get_database(self, database_id: str) -> Dict:
        """
        Get detailed information about a database.
        
        Args:
            database_id: Notion database ID (UUID)
            
        Returns:
            Complete database details including schema
        """
        logger.info(f"Executing get_database for database_id: {database_id}")
        try:
            result = self._make_request('GET', f'/databases/{database_id}')
            
            if 'error' in result:
                return result
            
            formatted_database = {
                'id': result.get('id'),
                'object': result.get('object'),
                'created_time': result.get('created_time'),
                'last_edited_time': result.get('last_edited_time'),
                'title': result.get('title', []),
                'properties': result.get('properties', {}),
                'url': result.get('url')
            }
            
            # Extract title text
            if formatted_database['title']:
                formatted_database['title_text'] = ''.join([part.get('plain_text', '') for part in formatted_database['title']])
            
            return formatted_database
            
        except Exception as e:
            logger.error(f"Error getting database: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def create_database_entry_property(self):
        description = """
        Create a new entry (page) in a Notion database.
        """
        return {
            "type": "custom",
            "name": "notion_create_database_entry",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "database_id": {
                        "type": "string",
                        "description": "Notion database ID (UUID format)"
                    },
                    "properties": {
                        "type": "string",
                        "description": "JSON string of entry properties matching the database schema"
                    }
                },
                "required": ["database_id", "properties"]
            }
        }
    
    def create_database_entry(self, database_id: str, properties: str) -> Dict:
        """
        Create a new entry in a database.
        
        Args:
            database_id: Notion database ID (UUID)
            properties: JSON string of entry properties
            
        Returns:
            Created entry details
        """
        logger.info(f"Executing create_database_entry for database_id: {database_id}")
        try:
            try:
                props_dict = json.loads(properties)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format in properties"}
            
            payload = {
                "parent": {
                    "database_id": database_id
                },
                "properties": props_dict
            }
            
            result = self._make_request('POST', '/pages', json=payload)
            
            if 'error' in result:
                return result
            
            formatted_entry = {
                'id': result.get('id'),
                'url': result.get('url'),
                'created_time': result.get('created_time'),
                'last_edited_time': result.get('last_edited_time'),
                'properties': result.get('properties', {})
            }
            
            return formatted_entry
            
        except Exception as e:
            logger.error(f"Error creating database entry: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def update_database_entry_property(self):
        description = """
        Update an existing entry in a Notion database.
        """
        return {
            "type": "custom",
            "name": "notion_update_database_entry",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID (UUID format) - the entry to update"
                    },
                    "properties": {
                        "type": "string",
                        "description": "JSON string of properties to update"
                    }
                },
                "required": ["page_id", "properties"]
            }
        }
    
    def update_database_entry(self, page_id: str, properties: str) -> Dict:
        """
        Update an existing database entry.
        
        Args:
            page_id: Notion page ID (UUID) - the entry to update
            properties: JSON string of properties to update
            
        Returns:
            Updated entry details
        """
        logger.info(f"Executing update_database_entry for page_id: {page_id}")
        try:
            try:
                props_dict = json.loads(properties)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format in properties"}
            
            payload = {
                "properties": props_dict
            }
            
            result = self._make_request('PATCH', f'/pages/{page_id}', json=payload)
            
            if 'error' in result:
                return result
            
            formatted_entry = {
                'id': result.get('id'),
                'url': result.get('url'),
                'last_edited_time': result.get('last_edited_time'),
                'properties': result.get('properties', {})
            }
            
            return formatted_entry
            
        except Exception as e:
            logger.error(f"Error updating database entry: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
