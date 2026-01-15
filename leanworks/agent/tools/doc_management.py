import json
import logging
import secrets
import tempfile
import os
from typing import Dict, List, Any, Optional, Union
import re
import markdown

from .base_api_client import BaseAPIClient
from .postgres import AI_AGENT_ID

logger = logging.getLogger(__name__)


class DocManagementTool(BaseAPIClient):
    """
    Document management tool for creating and updating documents via leanworks-hub API.
    
    Documents are worked with in markdown format using Anthropic's text editor tool,
    then converted to TipTap JSON format when saving via the API. The AI agent uses the text
    editor tool directly for formatting, eliminating the need for specific formatting
    helper functions.
    """
    
    def __init__(self, postgres_client_wrapper, user_id: Optional[str] = None):
        """
        Initialize DocManagementTool with API access.
        
        Args:
            postgres_client_wrapper: An object with attributes `org_slug` (organization name)
            user_id: Optional user ID used for attribution (falls back to AI_AGENT_ID if None)
        """
        # Get org_slug from wrapper (use client_name as fallback)
        org_slug = getattr(postgres_client_wrapper, 'org_slug', None)
        if not org_slug:
            # Fallback: construct org_slug from client_name if available
            client_name = getattr(postgres_client_wrapper, 'client_name', 'unknown')
            org_slug = f"{client_name}.ai" if client_name != 'unknown' else 'leanworks.ai'
            logger.warning(f"org_slug not provided in wrapper, using fallback: {org_slug}")
        
        # Initialize BaseAPIClient
        super().__init__(org_slug, user_id)
        
        self.postgres_client_wrapper = postgres_client_wrapper
        self.user_id = user_id or AI_AGENT_ID
        
        # Track temporary markdown files
        self._temp_files: Dict[str, str] = {}  # docId -> file_path
    
    # ============================================================================
    # Core Document Operations (via API)
    # ============================================================================
    
    @property
    def create_doc_property(self):
        description = f"""
        Create a new document in the docs table for org `{self.org_slug}`.
        
        This tool creates documents that are owned by the user who created them. Documents will have owner_email set to the user's email address.
        
        Parameters:
        - title (required): Document title
        - content (required): Document content in markdown format
        - projectId (optional): Associated project ID
        - teamId (optional): Associated team ID
        - tags (optional): Array of tag strings
        - visibility (optional): 'all_members' or 'specific_members' (default: 'all_members')
        - visibleToMembers (optional): Array of email addresses
        - metadata (optional): JSON object for additional metadata
        
        Content Format:
        - Input: Markdown format (required)
        - Output: Markdown format (returned to agent)
        
        Returns:
        - Success: Dictionary with doc id and created fields
        - Error: Dictionary with error message
        """
        return {
            "type": "custom",
            "name": "create_doc",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Document title (required)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Document content in markdown format (required). Will be converted to TipTap JSON format for storage."
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Associated project ID"
                    },
                    "teamId": {
                        "type": "string",
                        "description": "Associated team ID"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of tag strings"
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["all_members", "specific_members"],
                        "description": "Document visibility (default: 'all_members')"
                    },
                    "visibleToMembers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of email addresses"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "JSON object for additional metadata"
                    }
                },
                "required": ["title", "content"]
            }
        }
    
    def create_doc(
        self,
        title: str,
        content: str,
        projectId: Optional[str] = None,
        teamId: Optional[str] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "all_members",
        visibleToMembers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new document via API.
        
        Args:
            title: Document title (required)
            content: Document content (markdown or HTML) (required)
            projectId: Associated project ID
            teamId: Associated team ID
            tags: Array of tag strings
            visibility: Document visibility
            visibleToMembers: Array of email addresses
            metadata: JSON object for additional metadata
            
        Returns:
            Dictionary with doc id and created fields, or error dictionary
        """
        try:
            if not title or not content:
                return {"error": "title and content are required"}
            
            # Normalize content to markdown first (handles HTML/TipTap JSON input)
            # Then convert to TipTap JSON format for storage
            markdown_content = self._normalize_content_to_markdown(content)
            tiptap_json_content = self.markdown_to_tiptap_json(markdown_content)
            
            # Validate visibility
            valid_visibility = ['all_members', 'specific_members']
            doc_visibility = visibility if visibility in valid_visibility else 'all_members'
            
            # Validate visibleToMembers for specific_members visibility
            visible_to_members_array = []
            if doc_visibility == 'specific_members':
                if not visibleToMembers or not isinstance(visibleToMembers, list) or len(visibleToMembers) == 0:
                    return {"error": "visibleToMembers must be a non-empty array when visibility is specific_members"}
                visible_to_members_array = [email.lower() for email in visibleToMembers]
            
            # Prepare request body for API
            request_body = {
                "title": title,
                "content": tiptap_json_content,  # Send TipTap JSON to API
                "projectId": projectId,
                "teamId": teamId,
                "tags": tags or [],
                "visibility": doc_visibility,
                "visibleToMembers": visible_to_members_array
            }
            
            if metadata:
                request_body["metadata"] = metadata
            
            # Call API to create document
            result = self._make_request('POST', '/api/docs', json=request_body)
            
            logger.info(f"Document created via API: id={result.get('id')}, title={title}")
            
            # Return markdown content to agent (not TipTap JSON)
            return {
                "id": result.get('id'),
                "title": title,
                "content": content,  # Return original markdown, not TipTap JSON
                "ownerEmail": result.get('ownerEmail'),
                "projectId": projectId,
                "teamId": teamId,
                "tags": tags or [],
                "visibility": doc_visibility,
                "visibleToMembers": visible_to_members_array
            }
        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def update_doc_property(self):
        description = f"""
        Update an existing document in the docs table for org `{self.org_slug}`.
        
        Parameters:
        - docId (required): Document ID to update
        - title (optional): Update title
        - content (optional): Update content in markdown format
        - projectId (optional): Update project association
        - teamId (optional): Update team association
        - tags (optional): Update tags array
        - visibility (optional): Update visibility
        - visibleToMembers (optional): Update visible members
        - metadata (optional): Update metadata
        
        Content Format:
        - Input: Markdown format
        - Output: Success status (content is converted to TipTap JSON format for storage)
        
        Returns:
        - Success: Dictionary with success: true
        - Error: Dictionary with error message
        """
        return {
            "type": "custom",
            "name": "update_doc",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "docId": {
                        "type": "string",
                        "description": "Document ID to update (required)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Update title"
                    },
                    "content": {
                        "type": "string",
                        "description": "Update content in markdown format. Will be converted to TipTap JSON format for storage."
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Update project association"
                    },
                    "teamId": {
                        "type": "string",
                        "description": "Update team association"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Update tags array"
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["all_members", "specific_members"],
                        "description": "Update visibility"
                    },
                    "visibleToMembers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Update visible members"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Update metadata"
                    }
                },
                "required": ["docId"]
            }
        }
    
    @property
    def get_doc_property(self):
        description = f"""
        Get one or more documents by their IDs from the docs table for org `{self.org_slug}`.
        
        This tool retrieves full document content and all properties for the specified document IDs.
        Use this to read document content when you need the full text, not just a preview.
        
        Parameters:
        - docIds (required): Array of document IDs to retrieve. Can be a single document ID or multiple IDs.
        
        Returns:
        - Success: List of document dictionaries with all fields including: id, title, content (markdown format), owner_email, project_id, team_id, tags, visibility, visible_to_members, created_at, updated_at, metadata
        - Error: Dictionary with error message
        
        Note: Content is returned as markdown format (converted from TipTap JSON stored in database). Documents are stored in TipTap JSON format internally, but are automatically converted to markdown for the agent.
        
        Example Use Cases:
        - Read a specific document's full content
        - Get multiple documents at once
        - Retrieve document details and content for analysis
        """
        return {
            "type": "custom",
            "name": "get_doc",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "docIds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of document IDs to retrieve (required). Can contain one or more document IDs."
                    }
                },
                "required": ["docIds"]
            }
        }
    
    def get_doc(
        self,
        docIds: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get one or more documents by their IDs via API.
        
        Args:
            docIds: List of document IDs to retrieve
            
        Returns:
            List of document dictionaries with full content, or error dictionary
        """
        try:
            if not docIds:
                return {"error": "docIds is required and must be a non-empty array"}
            
            if not isinstance(docIds, list):
                return {"error": "docIds must be an array"}
            
            # Limit the number of documents to prevent excessive queries
            if len(docIds) > 50:
                return {"error": f"Too many document IDs requested. Maximum is 50, got {len(docIds)}"}
            
            # Fetch documents one by one (API doesn't support batch GET)
            docs = []
            found_ids = set()
            
            for doc_id in docIds:
                try:
                    # Call API to get document
                    doc = self._make_request('GET', f'/api/docs/{doc_id}')
                    
                    if doc:
                        # Convert content from TipTap JSON (or legacy HTML) to markdown
                        if 'content' in doc and doc['content']:
                            content = doc['content']
                            doc['content'] = self._convert_content_to_markdown(content)
                        
                        # Normalize field names (API returns camelCase, but we want consistency)
                        if 'ownerEmail' in doc:
                            doc['owner_email'] = doc.pop('ownerEmail')
                        if 'projectId' in doc:
                            doc['project_id'] = doc.pop('projectId')
                        if 'teamId' in doc:
                            doc['team_id'] = doc.pop('teamId')
                        if 'visibleToMembers' in doc:
                            doc['visible_to_members'] = doc.pop('visibleToMembers')
                        if 'createdAt' in doc:
                            doc['created_at'] = doc.pop('createdAt')
                        if 'updatedAt' in doc:
                            doc['updated_at'] = doc.pop('updatedAt')
                        
                        found_ids.add(doc['id'])
                        docs.append(doc)
                except Exception as e:
                    logger.warning(f"Error fetching document {doc_id}: {str(e)}")
                    # Continue with other documents
                    continue
            
            # Check if any requested documents were not found
            missing_ids = set(docIds) - found_ids
            if missing_ids:
                logger.warning(f"Some document IDs were not found: {missing_ids}")
            
            logger.info(f"Retrieved {len(docs)} documents out of {len(docIds)} requested")
            return docs
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    @property
    def get_doc_markdown_path_property(self):
        description = f"""
        Get or create a temporary markdown file path for a document in org `{self.org_slug}`.
        
        This allows the AI agent to use the text editor tool to view and edit document content
        in markdown format. The file is temporary and will be cleaned up after operations.
        
        Parameters:
        - docId (required): Document ID
        
        Returns:
        - Success: Path to temporary markdown file
        - Error: Dictionary with error message
        """
        return {
            "type": "custom",
            "name": "get_doc_markdown_path",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "docId": {
                        "type": "string",
                        "description": "Document ID (required)"
                    }
                },
                "required": ["docId"]
            }
        }
    
    @property
    def create_doc_from_markdown_file_property(self):
        description = f"""
        Create a new document from a markdown file in org `{self.org_slug}`.
        
        This tool reads a markdown file (typically created/edited with the text editor tool),
        converts it to TipTap JSON format, and saves it to the database. The temporary file is cleaned up
        after the operation. Content is stored as TipTap JSON internally but returned as markdown to the agent.
        
        Parameters:
        - file_path (required): Path to markdown file
        - title (required): Document title
        - projectId (optional): Associated project ID
        - teamId (optional): Associated team ID
        - tags (optional): Array of tag strings
        - visibility (optional): 'all_members' or 'specific_members' (default: 'all_members')
        - visibleToMembers (optional): Array of email addresses
        - metadata (optional): JSON object for additional metadata
        
        Returns:
        - Success: Dictionary with doc id and created fields
        - Error: Dictionary with error message
        """
        return {
            "type": "custom",
            "name": "create_doc_from_markdown_file",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to markdown file (required)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Document title (required)"
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Associated project ID"
                    },
                    "teamId": {
                        "type": "string",
                        "description": "Associated team ID"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of tag strings"
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["all_members", "specific_members"],
                        "description": "Document visibility (default: 'all_members')"
                    },
                    "visibleToMembers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of email addresses"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "JSON object for additional metadata"
                    }
                },
                "required": ["file_path", "title"]
            }
        }
    
    @property
    def update_doc_from_markdown_file_property(self):
        description = f"""
        Update an existing document from a markdown file in org `{self.org_slug}`.
        
        This tool reads a markdown file (typically created/edited with the text editor tool),
        converts it to TipTap JSON format, and updates the document in the database. The temporary file is
        cleaned up after the operation. Content is stored as TipTap JSON internally but returned as markdown to the agent.
        
        Parameters:
        - docId (required): Document ID to update
        - file_path (required): Path to markdown file
        - title (optional): Update title
        - projectId (optional): Update project association
        - teamId (optional): Update team association
        - tags (optional): Update tags array
        - visibility (optional): Update visibility
        - visibleToMembers (optional): Update visible members
        - metadata (optional): Update metadata
        
        Returns:
        - Success: Dictionary with success: true
        - Error: Dictionary with error message
        """
        return {
            "type": "custom",
            "name": "update_doc_from_markdown_file",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "docId": {
                        "type": "string",
                        "description": "Document ID to update (required)"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to markdown file (required)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Update title"
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Update project association"
                    },
                    "teamId": {
                        "type": "string",
                        "description": "Update team association"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Update tags array"
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["all_members", "specific_members"],
                        "description": "Update visibility"
                    },
                    "visibleToMembers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Update visible members"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Update metadata"
                    }
                },
                "required": ["docId", "file_path"]
            }
        }
    
    def update_doc(
        self,
        docId: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        projectId: Optional[str] = None,
        teamId: Optional[str] = None,
        tags: Optional[List[str]] = None,
        visibility: Optional[str] = None,
        visibleToMembers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update an existing document via API.
        
        Args:
            docId: Document ID to update (required)
            title: Update title
            content: Update content (markdown or HTML)
            projectId: Update project association
            teamId: Update team association
            tags: Update tags array
            visibility: Update visibility
            visibleToMembers: Update visible members
            metadata: Update metadata
            
        Returns:
            Dictionary with success status, or error dictionary
        """
        try:
            if not docId:
                return {"error": "docId is required"}
            
            # Build update payload
            updates = {}
            
            if title is not None:
                updates['title'] = title
            
            if content is not None:
                # Normalize content to markdown first (handles HTML/TipTap JSON input)
                # Then convert to TipTap JSON if content field
                markdown_content = self._normalize_content_to_markdown(content)
                updates['content'] = self.markdown_to_tiptap_json(markdown_content)
            
            if projectId is not None:
                updates['projectId'] = projectId
            
            if teamId is not None:
                updates['teamId'] = teamId
            
            if tags is not None:
                updates['tags'] = tags
            
            if visibility is not None:
                valid_visibility = ['all_members', 'specific_members']
                doc_visibility = visibility if visibility in valid_visibility else 'all_members'
                updates['visibility'] = doc_visibility
                
                # Handle visibleToMembers
                if doc_visibility == 'specific_members':
                    if not visibleToMembers or not isinstance(visibleToMembers, list) or len(visibleToMembers) == 0:
                        return {"error": "visibleToMembers must be a non-empty array when visibility is specific_members"}
                    updates['visibleToMembers'] = [email.lower() for email in visibleToMembers]
                else:
                    updates['visibleToMembers'] = []
            elif visibleToMembers is not None:
                updates['visibleToMembers'] = [email.lower() for email in visibleToMembers] if isinstance(visibleToMembers, list) else []
            
            if metadata is not None:
                updates['metadata'] = metadata
            
            if len(updates) == 0:
                return {"error": "No fields to update"}
            
            # Call API to update document
            result = self._make_request('PATCH', f'/api/docs/{docId}', json=updates)
            
            logger.info(f"Document updated via API: id={docId}")
            return result if result else {"success": True}
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    # ============================================================================
    # Content Conversion Functions
    # ============================================================================
    
    def markdown_to_html(self, markdown_text: str) -> str:
        """
        Convert markdown to TipTap-compatible HTML using the markdown library.
        
        Supports:
        - Headers (h1-h6)
        - Bold, italic, strikethrough
        - Links and images
        - Lists (ordered, unordered, nested)
        - Tables
        - Code blocks and inline code
        - Blockquotes
        - Horizontal rules
        - Line breaks
        
        Args:
            markdown_text: Markdown content to convert
            
        Returns:
            HTML content compatible with TipTap editor
        """
        if not markdown_text:
            return markdown_text
        
        # Configure markdown with extensions
        md = markdown.Markdown(
            extensions=[
                'tables',      # Table support
                'fenced_code', # Code blocks with ```
                'nl2br',       # Line breaks (two spaces + newline)
                'sane_lists',  # Better list handling
            ],
            output_format='html'
        )
        
        # Convert markdown to HTML
        html = md.convert(markdown_text)
        
        # Post-process to add custom styling for links (preserve original behavior)
        # Add class="text-primary underline" to links if not already present
        def add_link_class(match):
            href = match.group(1)
            attrs = match.group(2)
            # Check if class already exists
            if 'class=' in attrs:
                # Append to existing class
                attrs = re.sub(
                    r'class="([^"]*)"',
                    r'class="\1 text-primary underline"',
                    attrs
                )
            else:
                # Add new class attribute
                attrs = f' class="text-primary underline"{attrs}'
            return f'<a href="{href}"{attrs}>'
        
        html = re.sub(r'<a href="([^"]+)"([^>]*)>', add_link_class, html)
        
        # Clean up any extra whitespace
        html = html.strip()
        
        return html
    
    def html_to_markdown(self, html: str) -> str:
        """Convert HTML to markdown for text editor tool."""
        markdown = html
        
        # Remove HTML tags and convert to markdown
        # Headers
        markdown = re.sub(r'<h1>(.*?)</h1>', r'# \1', markdown, flags=re.IGNORECASE | re.DOTALL)
        markdown = re.sub(r'<h2>(.*?)</h2>', r'## \1', markdown, flags=re.IGNORECASE | re.DOTALL)
        markdown = re.sub(r'<h3>(.*?)</h3>', r'### \1', markdown, flags=re.IGNORECASE | re.DOTALL)
        
        # Bold
        markdown = re.sub(r'<strong>(.*?)</strong>', r'**\1**', markdown, flags=re.IGNORECASE | re.DOTALL)
        markdown = re.sub(r'<b>(.*?)</b>', r'**\1**', markdown, flags=re.IGNORECASE | re.DOTALL)
        
        # Italic
        markdown = re.sub(r'<em>(.*?)</em>', r'*\1*', markdown, flags=re.IGNORECASE | re.DOTALL)
        markdown = re.sub(r'<i>(.*?)</i>', r'*\1*', markdown, flags=re.IGNORECASE | re.DOTALL)
        
        # Links
        markdown = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', markdown, flags=re.IGNORECASE | re.DOTALL)
        
        # Lists
        markdown = re.sub(r'<li>(.*?)</li>', r'- \1', markdown, flags=re.IGNORECASE | re.DOTALL)
        markdown = re.sub(r'<ul[^>]*>|</ul>|<ol[^>]*>|</ol>', '', markdown, flags=re.IGNORECASE)
        
        # Code
        markdown = re.sub(r'<code>(.*?)</code>', r'`\1`', markdown, flags=re.IGNORECASE | re.DOTALL)
        markdown = re.sub(r'<pre[^>]*><code>(.*?)</code></pre>', r'```\n\1\n```', markdown, flags=re.IGNORECASE | re.DOTALL)
        
        # Paragraphs
        markdown = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', markdown, flags=re.IGNORECASE | re.DOTALL)
        
        # Blockquotes
        markdown = re.sub(r'<blockquote[^>]*>(.*?)</blockquote>', r'> \1', markdown, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove remaining HTML tags
        markdown = re.sub(r'<[^>]+>', '', markdown)
        
        # Decode HTML entities
        markdown = markdown.replace('&amp;', '&')
        markdown = markdown.replace('&lt;', '<')
        markdown = markdown.replace('&gt;', '>')
        markdown = markdown.replace('&quot;', '"')
        markdown = markdown.replace('&#39;', "'")
        
        # Clean up extra whitespace
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        markdown = markdown.strip()
        
        return markdown
    
    def tiptap_json_to_markdown(self, tiptap_json: Union[str, dict]) -> str:
        """
        Convert TipTap JSON format to markdown.
        
        TipTap JSON structure:
        {
            "type": "doc",
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Hello"}]},
                {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Title"}]}
            ]
        }
        
        Args:
            tiptap_json: TipTap JSON as string or dict
            
        Returns:
            Markdown string
        """
        if not tiptap_json:
            return ""
        
        # Parse JSON string if needed
        if isinstance(tiptap_json, str):
            try:
                doc = json.loads(tiptap_json)
            except json.JSONDecodeError:
                # If not valid JSON, might be HTML or plain text - return as is
                return tiptap_json
        else:
            doc = tiptap_json
        
        # Validate TipTap JSON structure
        if not isinstance(doc, dict) or doc.get("type") != "doc":
            # Not TipTap JSON format, return as string
            if isinstance(tiptap_json, str):
                return tiptap_json
            return json.dumps(tiptap_json)
        
        content = doc.get("content", [])
        if not content:
            return ""
        
        markdown_lines = []
        
        def process_node(node: dict, list_context: Optional[dict] = None) -> str:
            """Process a TipTap node and return markdown representation."""
            node_type = node.get("type", "")
            attrs = node.get("attrs", {})
            node_content = node.get("content", [])
            
            if node_type == "text":
                text = node.get("text", "")
                marks = node.get("marks", [])
                
                # Apply marks (bold, italic, etc.)
                for mark in marks:
                    mark_type = mark.get("type", "")
                    if mark_type == "bold":
                        text = f"**{text}**"
                    elif mark_type == "italic":
                        text = f"*{text}*"
                    elif mark_type == "strike":
                        text = f"~~{text}~~"
                    elif mark_type == "code":
                        text = f"`{text}`"
                    elif mark_type == "link":
                        href = mark.get("attrs", {}).get("href", "")
                        text = f"[{text}]({href})"
                
                return text
            
            elif node_type == "paragraph":
                if not node_content:
                    return "\n"
                text = "".join(process_node(child) for child in node_content)
                return text + "\n"
            
            elif node_type == "heading":
                level = attrs.get("level", 1)
                prefix = "#" * level + " "
                text = "".join(process_node(child) for child in node_content)
                return prefix + text.strip() + "\n"
            
            elif node_type == "bulletList":
                items = []
                for child in node_content:
                    if child.get("type") == "listItem":
                        item_text = process_list_item(child)
                        items.append(f"- {item_text}")
                return "\n".join(items) + "\n"
            
            elif node_type == "orderedList":
                items = []
                for idx, child in enumerate(node_content, 1):
                    if child.get("type") == "listItem":
                        item_text = process_list_item(child)
                        items.append(f"{idx}. {item_text}")
                return "\n".join(items) + "\n"
            
            elif node_type == "listItem":
                # Handled by parent list
                return ""
            
            elif node_type == "codeBlock":
                language = attrs.get("language", "")
                code = "".join(process_node(child) for child in node_content)
                lang_prefix = language if language else ""
                return f"```{lang_prefix}\n{code}\n```\n"
            
            elif node_type == "blockquote":
                quote_text = "".join(process_node(child) for child in node_content)
                lines = quote_text.strip().split("\n")
                return "\n".join(f"> {line}" for line in lines if line.strip()) + "\n"
            
            elif node_type == "horizontalRule":
                return "---\n"
            
            elif node_type == "hardBreak":
                return "\n"
            
            elif node_type == "image":
                src = attrs.get("src", "")
                alt = attrs.get("alt", "")
                title = attrs.get("title", "")
                title_part = f' "{title}"' if title else ""
                return f"![{alt}]({src}{title_part})\n"
            
            elif node_type == "table":
                # Process table
                rows = []
                for child in node_content:
                    if child.get("type") == "tableRow":
                        cells = []
                        for cell_node in child.get("content", []):
                            if cell_node.get("type") in ["tableHeader", "tableCell"]:
                                cell_text = "".join(process_node(c) for c in cell_node.get("content", []))
                                cells.append(cell_text.strip())
                        if cells:
                            rows.append("| " + " | ".join(cells) + " |")
                
                if rows:
                    # Add separator row after header
                    if len(rows) > 0:
                        separator = "| " + " | ".join(["---"] * len(rows[0].split("|")[1:-1])) + " |"
                        return "\n".join([rows[0], separator] + rows[1:]) + "\n"
                return ""
            
            else:
                # Unknown node type - try to process content
                if node_content:
                    return "".join(process_node(child) for child in node_content)
                return ""
        
        def process_list_item(item_node: dict) -> str:
            """Process a list item node."""
            text_parts = []
            for child in item_node.get("content", []):
                if child.get("type") == "paragraph":
                    text_parts.append(process_node(child))
                elif child.get("type") in ["bulletList", "orderedList"]:
                    # Nested list
                    nested = process_node(child)
                    # Indent nested list items
                    nested_lines = nested.strip().split("\n")
                    text_parts.append("\n" + "\n".join("  " + line for line in nested_lines))
                else:
                    text_parts.append(process_node(child))
            return "".join(text_parts).strip()
        
        # Process all top-level content nodes
        for node in content:
            result = process_node(node)
            if result:
                markdown_lines.append(result)
        
        # Join and clean up
        markdown_text = "".join(markdown_lines)
        # Remove excessive blank lines
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
        return markdown_text.strip()
    
    def markdown_to_tiptap_json(self, markdown_text: str) -> str:
        """
        Convert markdown to TipTap JSON format.
        
        Args:
            markdown_text: Markdown content
            
        Returns:
            TipTap JSON as stringified JSON
        """
        if not markdown_text or not markdown_text.strip():
            # Return empty document
            empty_doc = {
                "type": "doc",
                "content": [{"type": "paragraph"}]
            }
            return json.dumps(empty_doc)
        
        # First convert markdown to HTML using existing method
        html = self.markdown_to_html(markdown_text)
        
        # Then convert HTML to TipTap JSON
        # Parse HTML and build TipTap JSON structure
        return self._html_to_tiptap_json(html)
    
    def _html_to_tiptap_json(self, html: str) -> str:
        """
        Convert HTML to TipTap JSON format.
        
        This is a helper method that parses HTML and builds TipTap JSON structure.
        
        Args:
            html: HTML content
            
        Returns:
            TipTap JSON as stringified JSON
        """
        if not html or not html.strip():
            empty_doc = {"type": "doc", "content": [{"type": "paragraph"}]}
            return json.dumps(empty_doc)
        
        # Use regex to parse HTML into TipTap JSON structure
        # This is a simplified parser - for production, consider using BeautifulSoup
        content = []
        
        # Split HTML into blocks (rough parsing)
        # Remove script and style tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.IGNORECASE | re.DOTALL)
        
        # Extract paragraphs, headers, lists, etc.
        # This is a simplified approach - for better results, use proper HTML parser
        
        # Split by block-level elements
        blocks = re.split(r'(<(?:h[1-6]|p|ul|ol|blockquote|pre|table|hr)[^>]*>.*?</(?:h[1-6]|p|ul|ol|blockquote|pre|table|hr)>)', 
                        html, flags=re.IGNORECASE | re.DOTALL)
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Parse different block types
            if re.match(r'<h([1-6])', block, re.IGNORECASE):
                match = re.match(r'<h([1-6])[^>]*>(.*?)</h[1-6]>', block, re.IGNORECASE | re.DOTALL)
                if match:
                    level = int(match.group(1))
                    text_content = self._extract_text_from_html(match.group(2))
                    content.append({
                        "type": "heading",
                        "attrs": {"level": level},
                        "content": self._text_to_tiptap_nodes(text_content)
                    })
            
            elif re.match(r'<p', block, re.IGNORECASE):
                match = re.match(r'<p[^>]*>(.*?)</p>', block, re.IGNORECASE | re.DOTALL)
                if match:
                    text_content = match.group(1)
                    nodes = self._html_inline_to_tiptap_nodes(text_content)
                    if nodes:
                        content.append({"type": "paragraph", "content": nodes})
                    else:
                        content.append({"type": "paragraph"})
            
            elif re.match(r'<ul', block, re.IGNORECASE):
                list_items = re.findall(r'<li[^>]*>(.*?)</li>', block, re.IGNORECASE | re.DOTALL)
                items = []
                for item_html in list_items:
                    item_nodes = self._html_inline_to_tiptap_nodes(item_html)
                    items.append({
                        "type": "listItem",
                        "content": [{"type": "paragraph", "content": item_nodes} if item_nodes else {"type": "paragraph"}]
                    })
                if items:
                    content.append({"type": "bulletList", "content": items})
            
            elif re.match(r'<ol', block, re.IGNORECASE):
                list_items = re.findall(r'<li[^>]*>(.*?)</li>', block, re.IGNORECASE | re.DOTALL)
                items = []
                for item_html in list_items:
                    item_nodes = self._html_inline_to_tiptap_nodes(item_html)
                    items.append({
                        "type": "listItem",
                        "content": [{"type": "paragraph", "content": item_nodes} if item_nodes else {"type": "paragraph"}]
                    })
                if items:
                    content.append({"type": "orderedList", "content": items})
            
            elif re.match(r'<blockquote', block, re.IGNORECASE):
                match = re.match(r'<blockquote[^>]*>(.*?)</blockquote>', block, re.IGNORECASE | re.DOTALL)
                if match:
                    quote_text = self._extract_text_from_html(match.group(1))
                    quote_nodes = self._text_to_tiptap_nodes(quote_text)
                    content.append({
                        "type": "blockquote",
                        "content": [{"type": "paragraph", "content": quote_nodes} if quote_nodes else {"type": "paragraph"}]
                    })
            
            elif re.match(r'<pre', block, re.IGNORECASE):
                match = re.match(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>', block, re.IGNORECASE | re.DOTALL)
                if match:
                    code_text = self._extract_text_from_html(match.group(1))
                    content.append({
                        "type": "codeBlock",
                        "attrs": {},
                        "content": [{"type": "text", "text": code_text}]
                    })
            
            elif re.match(r'<hr', block, re.IGNORECASE):
                content.append({"type": "horizontalRule"})
        
        # If no content was parsed, create a paragraph with the text
        if not content:
            text_content = self._extract_text_from_html(html)
            if text_content.strip():
                content.append({
                    "type": "paragraph",
                    "content": self._text_to_tiptap_nodes(text_content)
                })
            else:
                content.append({"type": "paragraph"})
        
        doc = {"type": "doc", "content": content}
        return json.dumps(doc)
    
    def _extract_text_from_html(self, html: str) -> str:
        """Extract plain text from HTML, preserving basic structure."""
        # Remove HTML tags but keep text
        text = re.sub(r'<[^>]+>', '', html)
        # Decode HTML entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        text = text.replace('&nbsp;', ' ')
        return text.strip()
    
    def _html_inline_to_tiptap_nodes(self, html: str) -> List[dict]:
        """Convert inline HTML to TipTap text nodes with marks."""
        if not html:
            return []
        
        nodes = []
        # Extract text and inline formatting
        # This is simplified - handles bold, italic, links, code
        
        # Process links first (they can contain other formatting)
        parts = re.split(r'(<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>)', html, flags=re.IGNORECASE | re.DOTALL)
        
        for i, part in enumerate(parts):
            if i % 4 == 0:  # Regular text
                if part:
                    # Process inline formatting in text
                    nodes.extend(self._text_with_marks_to_nodes(part))
            elif i % 4 == 1:  # Full link match
                href = parts[i+1] if i+1 < len(parts) else ""
                link_text = parts[i+2] if i+2 < len(parts) else ""
                # Process link text for formatting
                link_nodes = self._text_with_marks_to_nodes(link_text, marks=[{"type": "link", "attrs": {"href": href}}])
                nodes.extend(link_nodes)
        
        # If no links found, process as regular text
        if len(nodes) == 0:
            nodes = self._text_with_marks_to_nodes(html)
        
        return nodes if nodes else [{"type": "text", "text": self._extract_text_from_html(html)}]
    
    def _text_with_marks_to_nodes(self, text: str, marks: Optional[List[dict]] = None) -> List[dict]:
        """Convert text with markdown-style formatting to TipTap nodes."""
        if not text:
            return []
        
        # Simple approach: extract text and apply marks
        # For more complex formatting, parse markdown syntax
        text = self._extract_text_from_html(text)
        
        # Check for markdown-style formatting
        # Bold: **text** or __text__
        # Italic: *text* or _text_
        # Code: `text`
        # Strikethrough: ~~text~~
        
        nodes = []
        current_pos = 0
        
        # Process bold first (to avoid conflicts with italic)
        while True:
            bold_match = re.search(r'\*\*(.+?)\*\*|__(.+?)__', text[current_pos:])
            if not bold_match:
                break
            
            # Add text before bold
            if bold_match.start() > 0:
                before_text = text[current_pos:current_pos + bold_match.start()]
                if before_text:
                    nodes.append({"type": "text", "text": before_text})
            
            # Add bold text
            bold_text = bold_match.group(1) or bold_match.group(2)
            node_marks = (marks or []) + [{"type": "bold"}]
            nodes.append({"type": "text", "text": bold_text, "marks": node_marks})
            
            current_pos += bold_match.end()
        
        # Add remaining text
        if current_pos < len(text):
            remaining = text[current_pos:]
            if remaining:
                if marks:
                    nodes.append({"type": "text", "text": remaining, "marks": marks})
                else:
                    nodes.append({"type": "text", "text": remaining})
        
        # If no formatting found, return simple text node
        if not nodes:
            if marks:
                return [{"type": "text", "text": text, "marks": marks}]
            return [{"type": "text", "text": text}]
        
        return nodes
    
    def _text_to_tiptap_nodes(self, text: str) -> List[dict]:
        """Convert plain text to TipTap text nodes."""
        if not text:
            return []
        return [{"type": "text", "text": text}]
    
    def _convert_content_to_markdown(self, content: str) -> str:
        """
        Convert content from database (TipTap JSON or legacy HTML) to markdown.
        Handles backward compatibility with HTML content.
        
        Args:
            content: Content from database (TipTap JSON string or HTML string)
            
        Returns:
            Markdown string
        """
        if not content:
            return ""
        
        # Check if content is TipTap JSON format
        if isinstance(content, str):
            # Try to detect TipTap JSON
            content_stripped = content.strip()
            if content_stripped.startswith('{') and '"type":"doc"' in content_stripped.replace(' ', ''):
                try:
                    # It's TipTap JSON
                    return self.tiptap_json_to_markdown(content)
                except Exception as e:
                    logger.warning(f"Failed to parse TipTap JSON, treating as HTML: {str(e)}")
                    # Fall through to HTML conversion
        
        # Check if it's already a dict (TipTap JSON object)
        if isinstance(content, dict):
            return self.tiptap_json_to_markdown(content)
        
        # Otherwise, treat as HTML (legacy format) and convert to markdown
        return self.html_to_markdown(content)
    
    def _is_tiptap_json(self, content: str) -> bool:
        """
        Check if content is TipTap JSON format.
        
        Args:
            content: Content string to check
            
        Returns:
            True if content appears to be TipTap JSON
        """
        if not content or not isinstance(content, str):
            return False
        
        content_stripped = content.strip()
        # Check for TipTap JSON signature
        return (content_stripped.startswith('{') and 
                ('"type":"doc"' in content_stripped.replace(' ', '') or 
                 '"type": "doc"' in content_stripped))
    
    def _normalize_content_to_markdown(self, content: str) -> str:
        """
        Normalize content to markdown format.
        Handles TipTap JSON, HTML, or markdown input.
        
        Args:
            content: Content in any format (TipTap JSON, HTML, or markdown)
            
        Returns:
            Markdown string
        """
        if not content:
            return ""
        
        # Check if it's TipTap JSON
        if self._is_tiptap_json(content):
            return self.tiptap_json_to_markdown(content)
        
        # Check if it's HTML (contains HTML tags)
        if isinstance(content, str) and '<' in content and '>' in content:
            # Check for common HTML patterns
            html_patterns = ['<p', '<div', '<h1', '<h2', '<h3', '<ul', '<ol', '<li', '<br', '<strong', '<em', '<a href']
            if any(pattern in content for pattern in html_patterns):
                return self.html_to_markdown(content)
        
        # Otherwise, assume it's already markdown
        return content
    
    # ============================================================================
    # Temporary File Management
    # ============================================================================
    
    def get_doc_markdown_path(self, docId: str) -> str:
        """
        Get or create temporary markdown file path for a document.
        
        Args:
            docId: Document ID
            
        Returns:
            Path to temporary markdown file
        """
        if docId in self._temp_files:
            return self._temp_files[docId]
        
        # Create new temporary file
        file_path = self.create_temp_markdown_file(docId)
        return file_path
    
    def create_temp_markdown_file(self, docId: str, content: str = None) -> str:
        """
        Create a temporary markdown file for a document.
        
        Args:
            docId: Document ID
            content: Optional initial content (if None, will fetch from DB)
            
        Returns:
            Path to temporary markdown file
        """
        # Create temporary file
        fd, file_path = tempfile.mkstemp(suffix='.md', prefix=f'doc_{docId}_', text=True)
        
        try:
            if content is None:
                # Try to fetch from API and convert to markdown
                try:
                    doc = self._make_request('GET', f'/api/docs/{docId}')
                    if doc and 'content' in doc:
                        db_content = doc['content']
                        # Convert from TipTap JSON (or legacy HTML) to markdown
                        content = self._convert_content_to_markdown(db_content)
                    else:
                        content = ""
                except Exception as e:
                    logger.warning(f"Error fetching document {docId} for markdown file: {str(e)}")
                    content = ""
            
            # Write content to file
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Track the file
            self._temp_files[docId] = file_path
            
            logger.info(f"Created temporary markdown file: {file_path} for doc {docId}")
            return file_path
        except Exception as e:
            os.close(fd)
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error creating temporary markdown file: {str(e)}")
            raise
    
    def cleanup_temp_file(self, file_path: str):
        """
        Clean up a temporary file.
        
        Args:
            file_path: Path to temporary file to clean up
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            
            # Remove from tracking
            docId_to_remove = None
            for doc_id, path in self._temp_files.items():
                if path == file_path:
                    docId_to_remove = doc_id
                    break
            if docId_to_remove:
                del self._temp_files[docId_to_remove]
        except Exception as e:
            logger.warning(f"Error cleaning up temporary file {file_path}: {str(e)}")
    
    def create_doc_from_markdown_file(
        self,
        file_path: str,
        title: str,
        projectId: Optional[str] = None,
        teamId: Optional[str] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "all_members",
        visibleToMembers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a document from a markdown file.
        
        Args:
            file_path: Path to markdown file
            title: Document title
            **kwargs: Additional arguments passed to create_doc
            
        Returns:
            Dictionary with doc id and created fields, or error dictionary
        """
        try:
            # Read markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Create document with markdown content (will be converted to HTML)
            result = self.create_doc(
                title=title,
                content=markdown_content,
                projectId=projectId,
                teamId=teamId,
                tags=tags,
                visibility=visibility,
                visibleToMembers=visibleToMembers,
                metadata=metadata,
                **kwargs
            )
            
            # Clean up temporary file if it was tracked
            self.cleanup_temp_file(file_path)
            
            return result
        except Exception as e:
            logger.error(f"Error creating doc from markdown file: {str(e)}")
            return {"error": f"Error reading markdown file: {str(e)}"}
    
    def update_doc_from_markdown_file(
        self,
        docId: str,
        file_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update a document from a markdown file.
        
        Args:
            docId: Document ID to update
            file_path: Path to markdown file
            **kwargs: Additional arguments passed to update_doc
            
        Returns:
            Dictionary with success status, or error dictionary
        """
        try:
            # Read markdown file
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Update document with markdown content (will be converted to HTML)
            result = self.update_doc(
                docId=docId,
                content=markdown_content,
                **kwargs
            )
            
            # Clean up temporary file if it was tracked
            self.cleanup_temp_file(file_path)
            
            return result
        except Exception as e:
            logger.error(f"Error updating doc from markdown file: {str(e)}")
            return {"error": f"Error reading markdown file: {str(e)}"}
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not isinstance(text, str):
            text = str(text)
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))
    
    def _sanitize_html(self, html: str) -> str:
        """
        Sanitize HTML content by removing dangerous elements and attributes.
        
        Args:
            html: HTML content to sanitize
            
        Returns:
            Sanitized HTML content
        """
        if not html:
            return html
        
        sanitized = html
        
        # Remove script tags and their content
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove iframe, object, and embed tags (potentially dangerous)
        sanitized = re.sub(r'<iframe[^>]*>.*?</iframe>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'<object[^>]*>.*?</object>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'<embed[^>]*>.*?</embed>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove event handler attributes (onclick, onerror, onload, etc.)
        event_handlers = [
            'onclick', 'onerror', 'onload', 'onmouseover', 'onmouseout',
            'onfocus', 'onblur', 'onchange', 'onsubmit', 'onreset',
            'onselect', 'onunload', 'onabort', 'onkeydown', 'onkeypress',
            'onkeyup', 'ondblclick', 'onmousedown', 'onmouseup', 'onmousemove'
        ]
        for handler in event_handlers:
            sanitized = re.sub(
                rf'\s+{handler}\s*=\s*["\'][^"\']*["\']',
                '',
                sanitized,
                flags=re.IGNORECASE
            )
            sanitized = re.sub(
                rf'\s+{handler}\s*=\s*[^\s>]+',
                '',
                sanitized,
                flags=re.IGNORECASE
            )
        
        # Remove javascript: protocol from href and src attributes
        sanitized = re.sub(
            r'(href|src)\s*=\s*["\']javascript:[^"\']*["\']',
            r'\1="#"',
            sanitized,
            flags=re.IGNORECASE
        )
        sanitized = re.sub(
            r'(href|src)\s*=\s*javascript:[^\s>]+',
            r'\1="#"',
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Log if any sanitization occurred
        if sanitized != html:
            logger.warning("HTML sanitization removed potentially dangerous content")
        
        return sanitized
    
    def _convert_to_html_if_needed(self, content: str) -> str:
        """
        Convert content to HTML. Defaults to treating content as markdown.
        Only accepts HTML if explicitly wrapped in <html>...</html> tags.
        
        Args:
            content: Content string (markdown preferred, or HTML wrapped in <html>...</html>)
            
        Returns:
            HTML content
        """
        if not content or not content.strip():
            return content
        
        # Check if content is explicitly wrapped in <html>...</html> tags
        html_wrapper_pattern = re.compile(
            r'^\s*<html[^>]*>(.*?)</html>\s*$',
            re.IGNORECASE | re.DOTALL
        )
        
        match = html_wrapper_pattern.match(content)
        if match:
            # Extract content between tags
            html_content = match.group(1).strip()
            logger.info("HTML content detected (wrapped in <html> tags), sanitizing...")
            
            # Sanitize HTML before returning
            sanitized_html = self._sanitize_html(html_content)
            return sanitized_html
        else:
            # Default to markdown
            logger.debug("Treating content as markdown (no <html> wrapper detected)")
            return self.markdown_to_html(content)
    
    @property
    def list_docs_property(self):
        description = f"""
        List documents from the docs table for org `{self.org_slug}`.
        
        This tool queries the docs table to retrieve documents with optional filtering.
        Use this to find documents by project, team, owner, tags, or other criteria.
        
        Parameters:
        - projectId (optional): Filter documents by project ID
        - teamId (optional): Filter documents by team ID
        - ownerEmail (optional): Filter documents by owner email
        - tags (optional): Filter documents that contain any of these tags (array of strings)
        - visibility (optional): Filter by visibility ('all_members' or 'specific_members')
        - searchTitle (optional): Search for documents with title containing this text (case-insensitive)
        - limit (optional): Maximum number of documents to return (default: 50, max: 200)
        - orderBy (optional): Order results by 'created_at' or 'updated_at' (default: 'created_at')
        - orderDirection (optional): 'asc' or 'desc' (default: 'desc' for newest first)
        
        Returns:
        - Success: List of document dictionaries with fields: id, title, content_preview (first 200 chars), owner_email, project_id, team_id, tags, visibility, created_at, updated_at
        - Note: Full content is not included - only a preview. Use get_doc_markdown_path or query_postgres to get full content if needed.
        - Error: Dictionary with error message
        
        Example Use Cases:
        - List all documents in a project
        - Find documents by specific tags
        - List pinned documents
        - Search for documents by title
        - Get recent documents
        """
        return {
            "type": "custom",
            "name": "list_docs",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "projectId": {
                        "type": "string",
                        "description": "Filter documents by project ID"
                    },
                    "teamId": {
                        "type": "string",
                        "description": "Filter documents by team ID"
                    },
                    "ownerEmail": {
                        "type": "string",
                        "description": "Filter documents by owner email"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter documents that contain any of these tags"
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["all_members", "specific_members"],
                        "description": "Filter by visibility"
                    },
                    "searchTitle": {
                        "type": "string",
                        "description": "Search for documents with title containing this text (case-insensitive)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to return (default: 50, max: 200)",
                        "minimum": 1,
                        "maximum": 200
                    },
                    "orderBy": {
                        "type": "string",
                        "enum": ["created_at", "updated_at"],
                        "description": "Order results by field (default: 'created_at')"
                    },
                    "orderDirection": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Order direction (default: 'desc' for newest first)"
                    }
                }
            }
        }
    
    def list_docs(
        self,
        projectId: Optional[str] = None,
        teamId: Optional[str] = None,
        ownerEmail: Optional[str] = None,
        tags: Optional[List[str]] = None,
        visibility: Optional[str] = None,
        searchTitle: Optional[str] = None,
        limit: int = 50,
        orderBy: str = "created_at",
        orderDirection: str = "desc",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        List documents via API with optional filtering.
        
        Note: The API endpoint returns all documents with visibility filtering.
        Client-side filtering for projectId, teamId, ownerEmail, tags, and searchTitle
        is applied after fetching from the API.
        
        Args:
            projectId: Filter by project ID
            teamId: Filter by team ID
            ownerEmail: Filter by owner email
            tags: Filter by tags (documents containing any of these tags)
            visibility: Filter by visibility
            searchTitle: Search title text (case-insensitive)
            limit: Maximum number of documents (default: 50, max: 200)
            orderBy: Order by field ('created_at' or 'updated_at')
            orderDirection: Order direction ('asc' or 'desc')
            
        Returns:
            List of document dictionaries, or error dictionary
        """
        try:
            # Validate limit
            if limit < 1:
                limit = 50
            if limit > 200:
                limit = 200
            
            # Call API to get all documents (API handles visibility filtering)
            all_docs = self._make_request('GET', '/api/docs')
            
            if not isinstance(all_docs, list):
                return {"error": "Unexpected response from API"}
            
            # Apply client-side filtering
            filtered_docs = []
            for doc in all_docs:
                # Normalize field names
                doc_project_id = doc.get('projectId') or doc.get('project_id')
                doc_team_id = doc.get('teamId') or doc.get('team_id')
                doc_owner_email = doc.get('ownerEmail') or doc.get('owner_email', '').lower()
                doc_tags = doc.get('tags', [])
                doc_visibility = doc.get('visibility')
                doc_title = doc.get('title', '').lower()
                
                # Apply filters
                if projectId and doc_project_id != projectId:
                    continue
                if teamId and doc_team_id != teamId:
                    continue
                if ownerEmail and doc_owner_email != ownerEmail.lower():
                    continue
                if visibility and doc_visibility != visibility:
                    continue
                if searchTitle and searchTitle.lower() not in doc_title:
                    continue
                if tags:
                    # Check if any tag matches
                    doc_tags_lower = [t.lower() if isinstance(t, str) else str(t).lower() for t in doc_tags]
                    tags_lower = [t.lower() for t in tags]
                    if not any(tag in doc_tags_lower for tag in tags_lower):
                        continue
                
                # Create content preview from full content
                content = doc.get('content', '')
                if content:
                    # Convert TipTap JSON to markdown first
                    content_md = self._convert_content_to_markdown(content)
                    # Strip markdown formatting for preview
                    content_preview = re.sub(r'[#*_`\[\]()]', '', content_md)
                    content_preview = ' '.join(content_preview.split())
                    if len(content_preview) > 200:
                        content_preview = content_preview[:200] + '...'
                else:
                    content_preview = ''
                
                # Normalize field names for consistency
                result_doc = {
                    'id': doc.get('id'),
                    'title': doc.get('title'),
                    'content_preview': content_preview,
                    'owner_email': doc_owner_email,
                    'project_id': doc_project_id,
                    'team_id': doc_team_id,
                    'tags': doc_tags,
                    'visibility': doc_visibility,
                    'visible_to_members': doc.get('visibleToMembers') or doc.get('visible_to_members', []),
                    'created_at': doc.get('createdAt') or doc.get('created_at'),
                    'updated_at': doc.get('updatedAt') or doc.get('updated_at')
                }
                
                filtered_docs.append(result_doc)
            
            # Apply sorting
            valid_order_by = ['created_at', 'updated_at']
            if orderBy not in valid_order_by:
                orderBy = 'created_at'
            
            valid_directions = ['asc', 'desc']
            if orderDirection not in valid_directions:
                orderDirection = 'desc'
            
            # Sort documents
            reverse = (orderDirection == 'desc')
            filtered_docs.sort(
                key=lambda x: x.get(orderBy, ''),
                reverse=reverse
            )
            
            # Apply limit
            filtered_docs = filtered_docs[:limit]
            
            logger.info(f"Listed {len(filtered_docs)} documents with filters: projectId={projectId}, teamId={teamId}, ownerEmail={ownerEmail}, tags={tags}, searchTitle={searchTitle}")
            return filtered_docs
                
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

