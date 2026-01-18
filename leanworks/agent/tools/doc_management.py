import json
import logging
import secrets
import tempfile
import os
from typing import Dict, List, Any, Optional, Union
import re
import markdown
import anthropic
from datetime import datetime, timezone

from .base_api_client import BaseAPIClient
from .postgres import AI_AGENT_ID

logger = logging.getLogger(__name__)


class DocManagementTool(BaseAPIClient):
    """
    Document management tool for creating and updating documents via leanworks-hub API.
    
    Documents are worked with in HTML format using Anthropic's text editor tool,
    then converted directly to TipTap JSON format when saving via the API. The AI agent uses the text
    editor tool directly for formatting, eliminating the need for specific formatting
    helper functions. This avoids structure loss from multiple format conversions.
    """
    
    def __init__(
        self, 
        postgres_client_wrapper, 
        user_id: Optional[str] = None,
        # Optional dependencies for workflow features
        rag_storage_tool=None,
        search_tool=None,
        bash_tool=None,
        text_editor_tool=None,
        model_client: Optional[anthropic.Anthropic] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DocManagementTool with API access.
        
        Args:
            postgres_client_wrapper: An object with attributes `org_slug` (organization name)
            user_id: Optional user ID used for attribution (falls back to AI_AGENT_ID if None)
            rag_storage_tool: Optional RAGStorageTool instance for storing large docs
            search_tool: Optional SearchTool instance for RAG retrieval
            bash_tool: Optional bash tool function for file operations
            text_editor_tool: Optional text editor tool function for file editing
            model_client: Optional Anthropic client for token counting API
            config: Optional workflow configuration (uses defaults from DOC_WORKFLOW_CONFIG if not provided)
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
        self._workflow_temp_files: List[str] = []  # List of temp file paths for workflow operations
        self._selected_text_positions: Dict[str, Dict[str, int]] = {}  # docId -> {"html_from": int, "html_to": int}
        
        # Store workflow dependencies
        self.rag_storage = rag_storage_tool
        self.search_tool = search_tool
        self.bash_tool = bash_tool
        self.text_editor = text_editor_tool
        self.model_client = model_client
        self.org_slug = org_slug
        
        # Load workflow configuration
        if config is None:
            from leanworks.setting import DOC_WORKFLOW_CONFIG
            self.config = DOC_WORKFLOW_CONFIG.copy()
        else:
            # Default configuration
            default_config = {
                "max_context_tokens": 30000,
                "context_sandwich_tokens": 300,
                "large_doc_threshold": 50000,
                "max_heading_depth": 3,
                "bridge_sentences": (1, 3),
                "chunk_by_headings": True,
                "heading_chunk_overlap": 75,
                "paragraph_chunk_overlap": 128,
                "run_continuity_pass": True,
                "run_formatting_pass": True,
                "run_compression_pass_threshold": 50000,
                "enable_impact_map": True,
                "require_user_confirmation": False,
                "enable_post_update_validation": True,
            }
            default_config.update(config)
            self.config = default_config
    
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
                        "description": "Document content in HTML format (required). Will be converted to TipTap JSON format for storage."
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
            content: Document content in HTML format (required)
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
            
            # Normalize content to HTML (handles TipTap JSON, HTML, or markdown input)
            html_content = self._normalize_content_to_html(content)
            
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
                "content": html_content,  # API converts HTML to TipTap JSON
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
            
            # Return HTML content to agent (not TipTap JSON)
            return {
                "id": result.get('id'),
                "title": title,
                "content": html_content,  # Return HTML, not TipTap JSON
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
        - content (optional): Update content in HTML format
        - projectId (optional): Update project association
        - teamId (optional): Update team association
        - tags (optional): Update tags array
        - visibility (optional): Update visibility
        - visibleToMembers (optional): Update visible members
        - metadata (optional): Update metadata
        
        Content Format:
        - Input: HTML format
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
                        "description": "Update content in HTML format. Will be converted to TipTap JSON format for storage."
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
        - Success: List of document dictionaries with all fields including: id, title, content (HTML format), owner_email, project_id, team_id, tags, visibility, visible_to_members, created_at, updated_at, metadata
        - Error: Dictionary with error message
        
        Note: Content is returned as HTML format (converted from TipTap JSON stored in database). Documents are stored in TipTap JSON format internally, but are automatically converted to HTML for the agent.
        
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
                        # Convert content from TipTap JSON (or legacy HTML) to HTML
                        if 'content' in doc and doc['content']:
                            content = doc['content']
                            doc['content'] = self._convert_content_to_html(content)
                        
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
    def get_doc_html_path_property(self):
        description = f"""
        Get or create a temporary HTML file path for a document in org `{self.org_slug}`.
        
        This allows the AI agent to use the text editor tool to view and edit document content
        in HTML format. The file is temporary and will be cleaned up after operations.
        
        Parameters:
        - docId (required): Document ID
        
        Returns:
        - Success: Path to temporary HTML file
        - Error: Dictionary with error message
        """
        return {
            "type": "custom",
            "name": "get_doc_html_path",
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
    def create_doc_from_html_file_property(self):
        description = f"""
        Create a new document from an HTML file in org `{self.org_slug}`.
        
        This tool reads an HTML file (typically created/edited with the text editor tool),
        converts it directly to TipTap JSON format, and saves it to the database. The temporary file is cleaned up
        after the operation. Content is stored as TipTap JSON internally but returned as HTML to the agent.
        
        Parameters:
        - file_path (required): Path to HTML file
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
            "name": "create_doc_from_html_file",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to HTML file (required)"
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
    def update_doc_from_html_file_property(self):
        description = f"""
        Update an existing document from an HTML file in org `{self.org_slug}`.
        
        This tool reads an HTML file (typically created/edited with the text editor tool),
        converts it directly to TipTap JSON format, and updates the document in the database. The temporary file is
        cleaned up after the operation. Content is stored as TipTap JSON internally but returned as HTML to the agent.
        
        Parameters:
        - docId (required): Document ID to update
        - file_path (required): Path to HTML file
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
            "name": "update_doc_from_html_file",
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
                        "description": "Path to HTML file (required)"
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
            content: Update content in HTML format
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
                # Normalize content to HTML (handles TipTap JSON, HTML, or markdown input)
                html_content = self._normalize_content_to_html(content)
                updates['content'] = html_content
            
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
    
    def tiptap_json_to_html(self, tiptap_json: Union[str, dict]) -> str:
        """
        Convert TipTap JSON format to HTML.
        
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
            HTML string
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
        
        html_parts = []
        
        def process_node(node: dict, list_context: Optional[dict] = None) -> str:
            """Process a TipTap node and return HTML representation."""
            node_type = node.get("type", "")
            attrs = node.get("attrs", {})
            node_content = node.get("content", [])
            
            if node_type == "text":
                text = self._escape_html(node.get("text", ""))
                marks = node.get("marks", [])
                
                # Apply marks (bold, italic, etc.) as HTML tags
                for mark in marks:
                    mark_type = mark.get("type", "")
                    if mark_type == "bold":
                        text = f"<strong>{text}</strong>"
                    elif mark_type == "italic":
                        text = f"<em>{text}</em>"
                    elif mark_type == "strike":
                        text = f"<s>{text}</s>"
                    elif mark_type == "code":
                        text = f"<code>{text}</code>"
                    elif mark_type == "link":
                        href = mark.get("attrs", {}).get("href", "")
                        href_escaped = self._escape_html(href)
                        text = f'<a href="{href_escaped}" class="text-primary underline">{text}</a>'
                
                return text
            
            elif node_type == "paragraph":
                if not node_content:
                    return "<p></p>"
                inner_html = "".join(process_node(child) for child in node_content)
                return f"<p>{inner_html}</p>"
            
            elif node_type == "heading":
                level = attrs.get("level", 1)
                inner_html = "".join(process_node(child) for child in node_content)
                return f"<h{level}>{inner_html}</h{level}>"
            
            elif node_type == "bulletList":
                items = []
                for child in node_content:
                    if child.get("type") == "listItem":
                        item_html = process_list_item(child)
                        items.append(f"<li>{item_html}</li>")
                return f"<ul>{''.join(items)}</ul>"
            
            elif node_type == "orderedList":
                items = []
                for child in node_content:
                    if child.get("type") == "listItem":
                        item_html = process_list_item(child)
                        items.append(f"<li>{item_html}</li>")
                return f"<ol>{''.join(items)}</ol>"
            
            elif node_type == "listItem":
                # Handled by parent list
                return ""
            
            elif node_type == "codeBlock":
                language = attrs.get("language", "")
                code = "".join(process_node(child) for child in node_content)
                code_escaped = self._escape_html(code)
                lang_attr = f' class="language-{language}"' if language else ""
                return f"<pre><code{lang_attr}>{code_escaped}</code></pre>"
            
            elif node_type == "blockquote":
                quote_html = "".join(process_node(child) for child in node_content)
                return f"<blockquote>{quote_html}</blockquote>"
            
            elif node_type == "horizontalRule":
                return "<hr>"
            
            elif node_type == "hardBreak":
                return "<br>"
            
            elif node_type == "image":
                src = attrs.get("src", "")
                alt = attrs.get("alt", "")
                title = attrs.get("title", "")
                src_escaped = self._escape_html(src)
                alt_escaped = self._escape_html(alt)
                title_attr = f' title="{self._escape_html(title)}"' if title else ""
                return f'<img src="{src_escaped}" alt="{alt_escaped}"{title_attr}>'
            
            elif node_type == "table":
                # Process table
                rows = []
                for child in node_content:
                    if child.get("type") == "tableRow":
                        cells = []
                        for cell_node in child.get("content", []):
                            if cell_node.get("type") == "tableHeader":
                                cell_html = "".join(process_node(c) for c in cell_node.get("content", []))
                                cells.append(f"<th>{cell_html}</th>")
                            elif cell_node.get("type") == "tableCell":
                                cell_html = "".join(process_node(c) for c in cell_node.get("content", []))
                                cells.append(f"<td>{cell_html}</td>")
                        if cells:
                            rows.append(f"<tr>{''.join(cells)}</tr>")
                
                if rows:
                    return f"<table>{''.join(rows)}</table>"
                return ""
            
            else:
                # Unknown node type - try to process content
                if node_content:
                    return "".join(process_node(child) for child in node_content)
                return ""
        
        def process_list_item(item_node: dict) -> str:
            """Process a list item node."""
            html_parts = []
            for child in item_node.get("content", []):
                if child.get("type") == "paragraph":
                    html_parts.append(process_node(child))
                elif child.get("type") in ["bulletList", "orderedList"]:
                    # Nested list
                    html_parts.append(process_node(child))
                else:
                    html_parts.append(process_node(child))
            return "".join(html_parts)
        
        # Process all top-level content nodes
        for node in content:
            result = process_node(node)
            if result:
                html_parts.append(result)
        
        # Join HTML parts
        html = "".join(html_parts)
        return html
    
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
        
        nodes = [node for node in nodes if node.get("text", "").strip() != ""]
        return nodes
    
    def _text_with_marks_to_nodes(self, text: str, marks: Optional[List[dict]] = None) -> List[dict]:
        """Convert text with markdown-style formatting to TipTap nodes."""
        if not text:
            return []
        
        # Simple approach: extract text and apply marks
        # For more complex formatting, parse markdown syntax
        text = self._extract_text_from_html(text)
        if not text.strip():
            return []
        
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
        if not text or not text.strip():
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
    
    def _convert_content_to_html(self, content: str) -> str:
        """
        Convert content from database (TipTap JSON or legacy HTML) to HTML.
        Handles backward compatibility with HTML content.
        
        Args:
            content: Content from database (TipTap JSON string or HTML string)
            
        Returns:
            HTML string
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
                    return self.tiptap_json_to_html(content)
                except Exception as e:
                    logger.warning(f"Failed to parse TipTap JSON, treating as HTML: {str(e)}")
                    # Fall through to HTML sanitization
        
        # Check if it's already a dict (TipTap JSON object)
        if isinstance(content, dict):
            return self.tiptap_json_to_html(content)
        
        # Otherwise, treat as HTML (legacy format) and sanitize
        return self._sanitize_html(content)
    
    def _normalize_content_to_html(self, content: str) -> str:
        """
        Normalize content to HTML format.
        Handles TipTap JSON, HTML, or markdown input.
        
        Args:
            content: Content in any format (TipTap JSON, HTML, or markdown)
            
        Returns:
            HTML string
        """
        if not content:
            return ""
        
        # Check if it's TipTap JSON
        if self._is_tiptap_json(content):
            return self.tiptap_json_to_html(content)
        
        # Check if it's HTML (contains HTML tags)
        if isinstance(content, str) and '<' in content and '>' in content:
            # Check for common HTML patterns
            html_patterns = ['<p', '<div', '<h1', '<h2', '<h3', '<ul', '<ol', '<li', '<br', '<strong', '<em', '<a href']
            if any(pattern in content for pattern in html_patterns):
                return self._sanitize_html(content)
        
        # Otherwise, assume it's markdown and convert to HTML
        return self.markdown_to_html(content)
    
    # ============================================================================
    # Temporary File Management
    # ============================================================================
    
    def get_doc_html_path(self, docId: str) -> str:
        """
        Get or create temporary HTML file path for a document.
        
        Args:
            docId: Document ID
            
        Returns:
            Path to temporary HTML file
        """
        if docId in self._temp_files:
            return self._temp_files[docId]
        
        # Create new temporary file
        file_path = self.create_temp_html_file(docId)
        return file_path
    
    def create_temp_html_file(self, docId: str, content: str = None) -> str:
        """
        Create a temporary HTML file for a document.
        
        Args:
            docId: Document ID
            content: Optional initial content (if None, will fetch from DB)
            
        Returns:
            Path to temporary HTML file
        """
        existing_path = self._temp_files.get(docId)
        if existing_path and os.path.exists(existing_path):
            if content is not None:
                try:
                    with open(existing_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"Reused temporary HTML file (updated): {existing_path} for doc {docId}")
                except Exception as e:
                    logger.warning(f"Failed to update cached temp file {existing_path} for doc {docId}: {str(e)}")
            else:
                logger.info(f"Reused temporary HTML file: {existing_path} for doc {docId}")
            return existing_path
        if existing_path and not os.path.exists(existing_path):
            del self._temp_files[docId]

        # Create temporary file
        fd, file_path = tempfile.mkstemp(suffix='.html', prefix=f'doc_{docId}_', text=True)
        
        try:
            if content is None:
                # Try to fetch from API and convert to HTML
                try:
                    doc = self._make_request('GET', f'/api/docs/{docId}')
                    if doc and 'content' in doc:
                        db_content = doc['content']
                        # Convert from TipTap JSON (or legacy HTML) to HTML
                        content = self._convert_content_to_html(db_content)
                    else:
                        content = ""
                except Exception as e:
                    logger.warning(f"Error fetching document {docId} for HTML file: {str(e)}")
                    content = ""
            
            # Write content to file
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Track the file
            self._temp_files[docId] = file_path
            
            logger.info(f"Created temporary HTML file: {file_path} for doc {docId}")
            return file_path
        except Exception as e:
            os.close(fd)
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error creating temporary HTML file: {str(e)}")
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

    def set_selected_text_positions(self, doc_id: str, html_from: Optional[int], html_to: Optional[int]) -> None:
        """Store selected text HTML positions for a document."""
        if not doc_id:
            return
        if html_from is None or html_to is None:
            return
        try:
            self._selected_text_positions[doc_id] = {
                "html_from": int(html_from),
                "html_to": int(html_to)
            }
        except Exception:
            return

    def get_selected_text_positions(self, doc_id: str) -> Optional[Dict[str, int]]:
        """Get stored selected text HTML positions for a document."""
        if not doc_id:
            return None
        return self._selected_text_positions.get(doc_id)
    
    @property
    def extract_text_at_html_positions_property(self):
        """Property definition for extract_text_at_html_positions tool."""
        description = """
        Extract text from a document at specific HTML character positions.
        
        This tool is useful when working with selected text that has HTML positions
        (converted from ProseMirror positions). The extracted text can be used
        as a precise search target for editing operations.
        
        When the user provides selected text context (cited_context with selectedText),
        HTML positions (htmlFrom, htmlTo) are automatically converted from ProseMirror
        positions and made available. Use this tool to extract the exact text at those
        positions for accurate editing.
        
        Parameters:
        - docId (required): Document ID
        - html_from (required): HTML start position (character offset)
        - html_to (required): HTML end position (character offset)
        - context_before (optional): Characters of context to include before html_from (default: 0)
        - context_after (optional): Characters of context to include after html_to (default: 0)
        
        Returns:
        - Success: Dictionary with extracted_html, extracted_text (plain text), and metadata
        - Error: Dictionary with error message
        """
        return {
            "type": "custom",
            "name": "extract_text_at_html_positions",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "docId": {
                        "type": "string",
                        "description": "Document ID (required)"
                    },
                    "html_from": {
                        "type": "integer",
                        "description": "HTML start position (character offset, required)"
                    },
                    "html_to": {
                        "type": "integer",
                        "description": "HTML end position (character offset, required)"
                    },
                    "context_before": {
                        "type": "integer",
                        "description": "Optional: character count of context to include before html_from"
                    },
                    "context_after": {
                        "type": "integer",
                        "description": "Optional: character count of context to include after html_to"
                    }
                },
                "required": ["docId", "html_from", "html_to"]
            }
        }
    
    def extract_text_at_html_positions(
        self,
        docId: str,
        html_from: int,
        html_to: int,
        context_before: int = 0,
        context_after: int = 0
    ) -> Dict[str, Any]:
        """
        Extract text from a document at specific HTML character positions.
        
        This is useful when working with selected text that has HTML positions
        (converted from ProseMirror positions). The extracted text can be used
        as a precise search target for editing operations.
        
        Args:
            docId: Document ID
            html_from: HTML start position (character offset)
            html_to: HTML end position (character offset)
            context_before: Optional character count of context to include before html_from
            context_after: Optional character count of context to include after html_to
            
        Returns:
            Dictionary with extracted text and metadata, or error dictionary
        """
        try:
            html_content = None
            cached_path = self._temp_files.get(docId)
            if cached_path and os.path.exists(cached_path):
                try:
                    with open(cached_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    logger.info(f"Using cached temp file for HTML extraction: {cached_path}")
                except Exception as e:
                    logger.warning(f"Failed to read cached temp file {cached_path}: {str(e)}")
                    html_content = None

            if html_content is None:
                # Get document content
                doc_result = self.get_doc([docId])
                if "error" in doc_result:
                    return doc_result
                
                if not isinstance(doc_result, list) or not doc_result:
                    return {"error": f"Document not found for id: {docId}"}

                content = doc_result[0].get("content", "")
                
                # Convert to HTML if needed
                html_content = self._convert_content_to_html(content)
            
            # Extract text at positions
            if html_from < 0 or html_to > len(html_content) or html_from >= html_to:
                return {
                    "error": f"Invalid HTML positions: from={html_from}, to={html_to}, content_length={len(html_content)}"
                }
            
            extracted_html = html_content[html_from:html_to]
            context_before = max(0, int(context_before))
            context_after = max(0, int(context_after))
            context_from = max(0, html_from - context_before)
            context_to = min(len(html_content), html_to + context_after)
            context_html = html_content[context_from:context_to]
            
            # Remove HTML tags to get plain text
            import re
            import html as html_lib
            plain_text = re.sub(r'<[^>]+>', '', extracted_html)
            context_text = re.sub(r'<[^>]+>', '', context_html)
            # Decode HTML entities
            plain_text = html_lib.unescape(plain_text)
            context_text = html_lib.unescape(context_text)
            
            return {
                "success": True,
                "docId": docId,
                "html_from": html_from,
                "html_to": html_to,
                "context_from": context_from,
                "context_to": context_to,
                "extracted_html": extracted_html,
                "extracted_text": plain_text.strip(),
                "context_html": context_html,
                "context_text": context_text.strip(),
                "length": len(plain_text)
            }
            
        except Exception as e:
            logger.error(f"Error extracting text at HTML positions: {str(e)}")
            return {"error": str(e)}
    
    def create_doc_from_html_file(
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
        Create a document from an HTML file.
        
        Args:
            file_path: Path to HTML file
            title: Document title
            **kwargs: Additional arguments passed to create_doc
            
        Returns:
            Dictionary with doc id and created fields, or error dictionary
        """
        try:
            # Read HTML file
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Sanitize HTML content
            html_content = self._sanitize_html(html_content)
            
            # Create document with HTML content (will be converted directly to TipTap JSON)
            result = self.create_doc(
                title=title,
                content=html_content,
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
            logger.error(f"Error creating doc from HTML file: {str(e)}")
            return {"error": f"Error reading HTML file: {str(e)}"}
    
    def update_doc_from_html_file(
        self,
        docId: str,
        file_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update a document from an HTML file.
        
        Args:
            docId: Document ID to update
            file_path: Path to HTML file
            **kwargs: Additional arguments passed to update_doc
            
        Returns:
            Dictionary with success status, or error dictionary
        """
        try:
            # Read HTML file
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Sanitize HTML content
            html_content = self._sanitize_html(html_content)
            
            # Update document with HTML content (will be converted directly to TipTap JSON)
            result = self.update_doc(
                docId=docId,
                content=html_content,
                **kwargs
            )
            
            # Clean up temporary file if it was tracked
            self.cleanup_temp_file(file_path)
            
            return result
        except Exception as e:
            logger.error(f"Error updating doc from HTML file: {str(e)}")
            return {"error": f"Error reading HTML file: {str(e)}"}
    
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
        - Note: Full content is not included - only a preview. Use get_doc_html_path or query_postgres to get full content if needed.
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
                    # Convert TipTap JSON to HTML first
                    content_html = self._convert_content_to_html(content)
                    # Strip HTML tags for preview
                    content_preview = re.sub(r'<[^>]+>', '', content_html)
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
    
    # ============================================================================
    # Document Workflow Tools (merged from DocumentWorkflowOrchestrator)
    # ============================================================================
    
    # Token Estimation Utilities
    def estimate_tokens(self, text: str, use_api: bool = True) -> int:
        """
        Estimate token count for text using Claude's token counting API.
        Falls back to character-based approximation if API is unavailable.
        
        Reference: https://platform.claude.com/docs/en/build-with-claude/token-counting
        
        Args:
            text: Text to estimate tokens for
            use_api: Whether to use the API (default True). Set False for fallback.
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Try using Claude's token counting API for accurate counts
        if use_api and self.model_client:
            try:
                response = self.model_client.messages.count_tokens(
                    model="claude-sonnet-4-5",
                    messages=[{
                        "role": "user",
                        "content": text
                    }]
                )
                return response.input_tokens
            except Exception as e:
                logger.warning(f"Token counting API failed, using fallback: {str(e)}")
                # Fall through to character-based estimation
        
        # Fallback: character-based approximation (~4 characters per token)
        return len(text) // 4
    
    def fits_in_context(self, text: str, max_tokens: Optional[int] = None) -> bool:
        """
        Check if text fits within context window.
        
        Args:
            text: Text to check
            max_tokens: Maximum tokens (uses config default if not provided)
            
        Returns:
            True if text fits in context window
        """
        if max_tokens is None:
            max_tokens = self.config.get("max_context_tokens", 30000)
        
        estimated = self.estimate_tokens(text)
        return estimated < max_tokens
    
    def extract_last_n_tokens(self, text: str, n_tokens: int) -> str:
        """
        Extract approximately the last N tokens from text.
        
        Args:
            text: Source text
            n_tokens: Number of tokens to extract
            
        Returns:
            Last ~N tokens of text
        """
        if not text:
            return ""
        
        # Convert tokens to approximate character count
        n_chars = n_tokens * 4
        
        if len(text) <= n_chars:
            return text
        
        # Extract from the end, try to break at paragraph boundary
        excerpt = text[-n_chars:]
        
        # Try to start at a paragraph break
        paragraph_break = excerpt.find('\n\n')
        if paragraph_break > 0 and paragraph_break < len(excerpt) // 2:
            excerpt = excerpt[paragraph_break + 2:]
        
        return excerpt
    
    def extract_first_n_tokens(self, text: str, n_tokens: int) -> str:
        """
        Extract approximately the first N tokens from text.
        
        Args:
            text: Source text
            n_tokens: Number of tokens to extract
            
        Returns:
            First ~N tokens of text
        """
        if not text:
            return ""
        
        # Convert tokens to approximate character count
        n_chars = n_tokens * 4
        
        if len(text) <= n_chars:
            return text
        
        # Extract from the beginning, try to break at paragraph boundary
        excerpt = text[:n_chars]
        
        # Try to end at a paragraph break
        paragraph_break = excerpt.rfind('\n\n')
        if paragraph_break > len(excerpt) // 2:
            excerpt = excerpt[:paragraph_break]
        
        return excerpt
    
    # Tool Property Definitions (for Claude API)
    @property
    def create_doc_with_workflow_property(self):
        """Property definition for create_doc_with_workflow tool."""
        return {
            "type": "custom",
            "name": "create_doc_with_workflow",
            "description": f"""Create a new document using TOC-first workflow for org `{self.org_slug}`.
            
This initiates an intelligent document creation workflow:
1. Analyzes requirements (clear vs exploratory)
2. Generates Table of Contents with Document Contract
3. Guides section-by-section drafting with context sandwiches
4. Runs quality passes (continuity, formatting, compression)

Use this for complex documents that need structured, token-safe creation.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Document title"},
                    "requirements": {"type": "string", "description": "User requirements/description for the document"},
                    "projectId": {"type": "string", "description": "Optional project ID"},
                    "teamId": {"type": "string", "description": "Optional team ID"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
                    "visibility": {"type": "string", "enum": ["all_members", "specific_members"], "description": "Document visibility"},
                    "visibleToMembers": {"type": "array", "items": {"type": "string"}, "description": "Optional list of email addresses"},
                    "metadata": {"type": "object", "description": "Optional metadata"}
                },
                "required": ["title", "requirements"]
            }
        }
    
    @property
    def update_doc_with_workflow_property(self):
        """Property definition for update_doc_with_workflow tool."""
        return {
            "type": "custom",
            "name": "update_doc_with_workflow",
            "description": f"""Update an existing document using intelligent workflow for org `{self.org_slug}`.
            
Automatically detects the best update strategy based on doc size and request type:
- Direct update (< 30K tokens)
- Targeted edit (specific location in large doc)
- Broad update (general changes in large doc)
- RAG fallback (unknown location in large doc)

Returns strategy recommendation and next steps.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "docId": {"type": "string", "description": "Document ID to update"},
                    "update_request": {"type": "string", "description": "Description of what to update"}
                },
                "required": ["docId", "update_request"]
            }
        }
    
    @property
    def generate_toc_property(self):
        """Property definition for generate_toc tool."""
        return {
            "type": "custom",
            "name": "generate_toc",
            "description": f"""Generate Table of Contents with document contract for org `{self.org_slug}`.
            
Analyzes requirements to determine if they're clear or exploratory, then creates:
- Document Contract (purpose, audience, scope, non-goals, evidence rule)
- TOC structure with max 3 heading levels
- Section descriptions

Returns TOC template for user confirmation before drafting.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Document title"},
                    "requirements": {"type": "string", "description": "User requirements/description"},
                    "max_depth": {"type": "integer", "description": "Maximum heading depth (default: 3)"}
                },
                "required": ["title", "requirements"]
            }
        }
    
    @property
    def create_toc_file_property(self):
        """Property definition for create_toc_file tool."""
        return {
            "type": "custom",
            "name": "create_toc_file",
            "description": f"""Create a temporary markdown file with TOC structure for org `{self.org_slug}`.
            
Converts TOC dictionary to markdown format and saves to temp file for editing.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "toc_structure": {"type": "object", "description": "TOC dictionary structure"}
                },
                "required": ["toc_structure"]
            }
        }
    
    @property
    def prepare_section_context_property(self):
        """Property definition for prepare_section_context tool."""
        return {
            "type": "custom",
            "name": "prepare_section_context",
            "description": f"""Prepare context sandwich for drafting a section for org `{self.org_slug}`.
            
Creates drafting context including:
- Last ~300 tokens from previous section (context above)
- Current section info (heading, description, outline)
- Next section heading (context below)
- Drafting prompt with required output format

This ensures fluent transitions between sections.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "section_info": {"type": "object", "description": "Section information (heading, description, level)"},
                    "previous_content": {"type": "string", "description": "Previously drafted content"},
                    "next_section_heading": {"type": "string", "description": "Heading of next section (optional)"}
                },
                "required": ["section_info", "previous_content"]
            }
        }
    
    @property
    def upsert_section_to_file_property(self):
        """Property definition for upsert_section_to_file tool."""
        return {
            "type": "custom",
            "name": "upsert_section_to_file",
            "description": f"""Upsert section content to document file for org `{self.org_slug}`.
            
Appends drafted section content to the working document file.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to document file"},
                    "section_content": {"type": "string", "description": "Section content to add"},
                    "section_id": {"type": "string", "description": "Section identifier (e.g., '2.3')"}
                },
                "required": ["file_path", "section_content", "section_id"]
            }
        }
    
    @property
    def draft_document_iteratively_property(self):
        """Property definition for draft_document_iteratively tool."""
        return {
            "type": "custom",
            "name": "draft_document_iteratively",
            "description": f"""Get instructions for drafting document section by section for org `{self.org_slug}`.
            
Returns section list and instructions for iterative drafting workflow.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "toc": {"type": "object", "description": "Table of Contents structure"},
                    "output_file": {"type": "string", "description": "Path to output file"}
                },
                "required": ["toc", "output_file"]
            }
        }
    
    @property
    def run_quality_passes_property(self):
        """Property definition for run_quality_passes tool."""
        return {
            "type": "custom",
            "name": "run_quality_passes",
            "description": f"""Run quality validation passes on document content for org `{self.org_slug}`.
            
Runs enabled passes:
- Continuity: term consistency, broken references, unclear pronouns
- Formatting: heading levels, list consistency, duplicate content
- Compression: suggestions for large docs (> 50K tokens)

Returns quality report with issues and suggestions.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Document content to validate"},
                    "passes": {"type": "array", "items": {"type": "string"}, "description": "Optional list of passes to run"}
                },
                "required": ["content"]
            }
        }
    
    @property
    def edit_doc_section_property(self):
        """Property definition for edit_doc_section tool (consolidated)."""
        return {
            "type": "custom",
            "name": "edit_doc_section",
            "description": f"""Edit a specific section in a document (end-to-end workflow) for org `{self.org_slug}`.
            
This consolidated tool handles the complete targeted edit workflow:
1. Exports doc to temp file
2. Searches for target area (exact → fuzzy → RAG fallback)
3. Applies diff-first edit (OLD_BLOCK → NEW_BLOCK)
4. Merges changes back to document

Use this instead of calling export → search → edit → merge separately.
Notes:
- Reuses cached temp files when available to avoid re-fetching the document.
- search_target should come from cited_context.selectedText.text when provided.
- old_block must match the document HTML substring; if you have htmlFrom/htmlTo, this tool will extract the exact selection and surrounding context automatically.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "docId": {"type": "string", "description": "Document ID"},
                    "search_target": {"type": "string", "description": "Text to search for to locate the section"},
                    "old_block": {"type": "string", "description": "Exact text to replace"},
                    "new_block": {"type": "string", "description": "Replacement text"},
                    "context_lines": {"type": "integer", "description": "Lines of context to show (default: 10)"},
                    "html_from": {"type": "integer", "description": "Optional: HTML start position (character offset)"},
                    "html_to": {"type": "integer", "description": "Optional: HTML end position (character offset)"},
                    "context_before": {"type": "integer", "description": "Optional: characters of context before html_from"},
                    "context_after": {"type": "integer", "description": "Optional: characters of context after html_to"}
                },
                "required": ["docId", "search_target", "old_block", "new_block"]
            }
        }
    
    @property
    def search_large_doc_property(self):
        """Property definition for search_large_doc tool (consolidated)."""
        return {
            "type": "custom",
            "name": "search_large_doc",
            "description": f"""Search in a large document using RAG (auto-chunks if needed) for org `{self.org_slug}`.
            
This consolidated tool:
1. Checks if document needs chunking
2. Chunks document by headings with overlap if needed
3. Provides retrieval instructions for RAG search

Use this instead of calling chunk → retrieve separately.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "docId": {"type": "string", "description": "Document ID"},
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of chunks to retrieve (default: 5)"}
                },
                "required": ["docId", "query"]
            }
        }
    
    @property
    def finalize_doc_update_property(self):
        """Property definition for finalize_doc_update tool (consolidated)."""
        return {
            "type": "custom",
            "name": "finalize_doc_update",
            "description": f"""Finalize document update with validation and change log for org `{self.org_slug}`.
            
This consolidated tool:
1. Validates the update (continuity, references, contradictions)
2. Creates change log entry with timestamp
3. Returns consolidated validation report

Use this instead of calling validate → change_log separately.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "docId": {"type": "string", "description": "Document ID"},
                    "original_content": {"type": "string", "description": "Original content before update"},
                    "updated_content": {"type": "string", "description": "Updated content"},
                    "change_description": {"type": "string", "description": "Description of what changed"},
                    "changed_sections": {"type": "array", "items": {"type": "string"}, "description": "Optional list of section names that changed"},
                    "source": {"type": "string", "description": "Optional source of the update"}
                },
                "required": ["docId", "original_content", "updated_content", "change_description"]
            }
        }
    
    @property
    def generate_impact_map_property(self):
        """Property definition for generate_impact_map tool."""
        return {
            "type": "custom",
            "name": "generate_impact_map",
            "description": f"""Generate impact map for broad document updates for org `{self.org_slug}`.
            
Returns template and instructions for agent to analyze which sections need updates based on new information.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "doc_structure": {"type": "string", "description": "Document structure (headings, sections)"},
                    "update_inputs": {"type": "string", "description": "New information/requirements"}
                },
                "required": ["doc_structure", "update_inputs"]
            }
        }
    
    @property
    def update_section_with_rag_property(self):
        """Property definition for update_section_with_rag tool."""
        return {
            "type": "custom",
            "name": "update_section_with_rag",
            "description": f"""Update a specific section using RAG retrieval for org `{self.org_slug}`.
            
Returns instructions for retrieving section content via search_documents and incorporating updates.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document ID"},
                    "section_id": {"type": "string", "description": "Section identifier"},
                    "section_heading": {"type": "string", "description": "Section heading to search for"},
                    "update_info": {"type": "string", "description": "New information to incorporate"}
                },
                "required": ["doc_id", "section_id", "section_heading", "update_info"]
            }
        }

    def create_doc_with_workflow(
        self,
        title: str,
        requirements: str,
        projectId: Optional[str] = None,
        teamId: Optional[str] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "all_members",
        visibleToMembers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new document using TOC-first workflow.
        
        This method orchestrates the complete document creation process:
        1. Generate TOC with document contract
        2. Draft sections iteratively with context sandwiches
        3. Run quality passes
        4. Create final document
        
        Args:
            title: Document title
            requirements: User requirements/description for the document
            projectId: Optional project association
            teamId: Optional team association
            tags: Optional tags
            visibility: Document visibility
            visibleToMembers: Optional list of emails for specific visibility
            metadata: Optional metadata
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with created document info or error
        """
        try:
            logger.info(f"Starting TOC-first document creation: {title}")
            
            # This is a placeholder that will return instructions for the agent
            # The actual workflow will be driven by agent interactions
            return {
                "workflow_initiated": True,
                "title": title,
                "next_step": "generate_toc",
                "instructions": """Document creation workflow initiated.

Next steps:
1. Analyze the requirements to determine if they provide clear structure or just a topic
2. Generate a Table of Contents including:
   - Document Contract (purpose, audience, scope, non-goals, evidence rule)
   - Major sections (H1) with subsections (H2, optionally H3)
   - Max 3 heading levels
3. Show the TOC for confirmation before drafting content
4. Once confirmed, draft sections iteratively with context sandwiches
5. Run quality passes after all sections are complete
6. Create the final document

Please proceed with generating the TOC based on these requirements.""",
                "requirements": requirements
            }
            
        except Exception as e:
            logger.error(f"Error in create_doc_with_workflow: {str(e)}")
            return {"error": str(e)}
    
    def update_doc_with_workflow(
        self,
        docId: str,
        update_request: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update an existing document using intelligent workflow.
        
        This method:
        1. Loads the document and checks size
        2. Determines update strategy (direct/targeted/broad/RAG)
        3. Executes appropriate update workflow
        4. Runs post-update validation
        
        Args:
            docId: Document ID to update
            update_request: Description of what to update
            **kwargs: Additional update parameters
            
        Returns:
            Dictionary with update status or error
        """
        try:
            logger.info(f"Starting intelligent document update: {docId}")
            
            # Load document
            doc_result = self.get_doc([docId])
            if "error" in doc_result:
                return doc_result
            
            if not isinstance(doc_result, list) or not doc_result:
                return {"error": f"Document not found for id: {docId}"}

            content = doc_result[0].get("content", "")
            
            # Detect strategy
            strategy = self._detect_update_strategy(content, update_request)
            
            return {
                "workflow_initiated": True,
                "docId": docId,
                "strategy": strategy["strategy"],
                "doc_size_tokens": strategy["doc_size_tokens"],
                "fits_in_context": strategy["fits_in_context"],
                "has_specific_target": strategy["has_specific_target"],
                "instructions": self._get_update_instructions(strategy),
                "current_content": content if strategy["fits_in_context"] else None,
                "update_request": update_request
            }
            
        except Exception as e:
            logger.error(f"Error in update_doc_with_workflow: {str(e)}")
            return {"error": str(e)}
    
    # ============================================================================
    # TOC Generation and Document Structure
    # ============================================================================
    
    def generate_toc(
        self,
        title: str,
        requirements: str,
        max_depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate Table of Contents with document contract.
        
        This is an agent-assisted method that returns structure for the agent
        to populate with actual content based on requirements analysis.
        
        Args:
            title: Document title
            requirements: User requirements/description
            max_depth: Maximum heading depth (default from config)
            
        Returns:
            TOC structure template and instructions
        """
        if max_depth is None:
            max_depth = self.config["max_heading_depth"]
        
        # Analyze if requirements are clear or exploratory
        is_clear_structure = self._analyze_requirements_clarity(requirements)
        
        toc_template = {
            "title": title,
            "requirements_clarity": "clear" if is_clear_structure else "exploratory",
            "contract": {
                "purpose": "[What this document achieves]",
                "audience": "[Who reads this document]",
                "scope": "[What's covered]",
                "non_goals": "[What's explicitly excluded]",
                "evidence_rule": "Claims must be sourced or marked with TODO/ASSUMPTION tags"
            },
            "sections": [],
            "max_depth": max_depth,
            "instructions": None
        }
        
        if is_clear_structure:
            toc_template["instructions"] = """Requirements are clear. Generate TOC directly:
1. Fill in the Document Contract (purpose, audience, scope, non-goals)
2. Create major sections (H1) based on requirements
3. Add subsections (H2, optionally H3) as needed
4. Keep to max {} heading levels
5. Provide 1-2 sentence description for each section
6. Present TOC for user confirmation before drafting""".format(max_depth)
        else:
            toc_template["instructions"] = """Requirements are exploratory. Use discovery approach:
1. Generate a "Discovery TOC" with 3-5 key questions to explore
2. Present these to user for confirmation
3. Once confirmed, expand into full TOC with:
   - Document Contract
   - Major sections (H1) addressing each question
   - Subsections (H2, H3) for details
4. Keep to max {} heading levels""".format(max_depth)
        
        return toc_template
    
    def _analyze_requirements_clarity(self, requirements: str) -> bool:
        """
        Analyze if requirements provide clear structure or are exploratory.
        
        Args:
            requirements: User requirements text
            
        Returns:
            True if requirements are clear/structured, False if exploratory
        """
        # Indicators of clear structure
        clear_indicators = [
            r'\d+\.',  # Numbered lists
            r'-\s+\w',  # Bullet points
            r'section',
            r'chapter',
            r'include',
            r'cover',
            r'must have',
            r'should have',
            r'outline',
            r'structure',
        ]
        
        # Count clear indicators
        clear_count = 0
        for pattern in clear_indicators:
            if re.search(pattern, requirements, re.IGNORECASE):
                clear_count += 1
        
        # If requirements are detailed (>200 words) and have multiple indicators
        word_count = len(requirements.split())
        
        if word_count > 200 and clear_count >= 3:
            return True
        elif clear_count >= 5:
            return True
        
        return False
    
    def create_toc_file(self, toc_structure: Dict[str, Any]) -> str:
        """
        Create a temporary markdown file with TOC structure.
        
        Args:
            toc_structure: TOC dictionary structure
            
        Returns:
            Path to temporary TOC file
        """
        # Generate markdown TOC content
        md_content = self._toc_to_markdown(toc_structure)
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.md',
            prefix='toc_',
            delete=False
        )
        temp_file.write(md_content)
        temp_file.close()
        
        self._workflow_temp_files.append(temp_file.name)
        logger.info(f"Created TOC file: {temp_file.name}")
        
        return temp_file.name
    
    def _toc_to_markdown(self, toc: Dict[str, Any]) -> str:
        """
        Convert TOC structure to markdown format.
        
        Args:
            toc: TOC dictionary
            
        Returns:
            Markdown formatted TOC
        """
        lines = []
        
        # Title
        lines.append(f"# {toc['title']}\n")
        
        # Document Contract
        lines.append("## Document Contract\n")
        contract = toc.get('contract', {})
        lines.append(f"- **Purpose**: {contract.get('purpose', 'TBD')}")
        lines.append(f"- **Audience**: {contract.get('audience', 'TBD')}")
        lines.append(f"- **Scope**: {contract.get('scope', 'TBD')}")
        lines.append(f"- **Non-goals**: {contract.get('non_goals', 'TBD')}")
        lines.append(f"- **Evidence Rule**: {contract.get('evidence_rule', 'TBD')}\n")
        
        # Table of Contents
        lines.append("## Table of Contents\n")
        
        sections = toc.get('sections', [])
        if sections:
            for section in sections:
                self._format_section_toc(section, lines, level=1)
        else:
            lines.append("_Sections to be defined_\n")
        
        return '\n'.join(lines)
    
    def _format_section_toc(
        self,
        section: Dict[str, Any],
        lines: List[str],
        level: int
    ):
        """
        Format a section for TOC display (recursive for subsections).
        
        Args:
            section: Section dictionary
            lines: List to append formatted lines to
            level: Current heading level
        """
        indent = "  " * (level - 1)
        heading = section.get('heading', 'Untitled')
        description = section.get('description', '')
        
        lines.append(f"{indent}- **{heading}**")
        if description:
            lines.append(f"{indent}  _{description}_")
        
        # Recursively format subsections
        subsections = section.get('subsections', [])
        for subsection in subsections:
            self._format_section_toc(subsection, lines, level + 1)
    
    # ============================================================================
    # Section Drafting with Context Sandwiches
    # ============================================================================
    
    def prepare_section_context(
        self,
        section_info: Dict[str, Any],
        previous_content: str,
        next_section_heading: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare context sandwich for drafting a section.
        
        Args:
            section_info: Section information (heading, description, level)
            previous_content: Previously drafted content
            next_section_heading: Heading of next section (if any)
            
        Returns:
            Context package for agent
        """
        context_tokens = self.config["context_sandwich_tokens"]
        
        # Extract context above (last ~300 tokens from previous content)
        context_above = self.extract_last_n_tokens(previous_content, context_tokens)
        
        # Prepare drafting prompt
        prompt_parts = []
        
        if context_above:
            prompt_parts.append("### Context from Previous Section\n")
            prompt_parts.append(context_above)
            prompt_parts.append("\n")
        
        prompt_parts.append(f"### Section to Draft\n")
        prompt_parts.append(f"**Heading**: {section_info.get('heading', 'Untitled')}\n")
        
        if section_info.get('description'):
            prompt_parts.append(f"**Description**: {section_info['description']}\n")
        
        if section_info.get('outline'):
            prompt_parts.append(f"**Outline**: {section_info['outline']}\n")
        
        if next_section_heading:
            prompt_parts.append(f"\n**Next Section**: {next_section_heading}\n")
        
        prompt_parts.append("\n### Required Output Format\n")
        prompt_parts.append("1. **Bridge-in** (1-3 sentences connecting from previous section)\n")
        prompt_parts.append("2. **Section content** (follow the outline and description)\n")
        prompt_parts.append("3. **Bridge-out** (1-3 sentences leading to next section)\n")
        prompt_parts.append("4. **Change log**: Brief note on what was added\n")
        prompt_parts.append("\n**Evidence Rule**: Use TODO or ASSUMPTION tags for unsourced claims\n")
        
        return {
            "section_info": section_info,
            "context_above": context_above,
            "next_section_heading": next_section_heading,
            "drafting_prompt": ''.join(prompt_parts)
        }
    
    def upsert_section_to_file(
        self,
        file_path: str,
        section_content: str,
        section_id: str
    ) -> Dict[str, Any]:
        """
        Upsert section content to a document file.
        
        Args:
            file_path: Path to document file
            section_content: Content to add/update
            section_id: Section identifier for tracking
            
        Returns:
            Status dictionary
        """
        try:
            # Read current content
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    current_content = f.read()
            else:
                current_content = ""
            
            # Append new section
            updated_content = current_content + "\n\n" + section_content
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"Upserted section {section_id} to {file_path}")
            
            return {
                "success": True,
                "section_id": section_id,
                "file_path": file_path,
                "current_size_tokens": self.estimate_tokens(updated_content)
            }
            
        except Exception as e:
            logger.error(f"Error upserting section: {str(e)}")
            return {"error": str(e)}
    
    def get_section_list_from_toc(self, toc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract flat list of sections from TOC structure.
        
        Args:
            toc: TOC dictionary with nested sections
            
        Returns:
            Flat list of sections in order
        """
        sections = []
        
        def traverse(section_list, parent_id=""):
            for idx, section in enumerate(section_list):
                section_id = f"{parent_id}.{idx + 1}" if parent_id else str(idx + 1)
                
                section_item = {
                    "id": section_id,
                    "heading": section.get("heading", ""),
                    "description": section.get("description", ""),
                    "outline": section.get("outline", ""),
                    "level": len(section_id.split('.'))
                }
                sections.append(section_item)
                
                # Recursively traverse subsections
                if "subsections" in section:
                    traverse(section["subsections"], section_id)
        
        traverse(toc.get("sections", []))
        return sections
    
    def draft_document_iteratively(
        self,
        toc: Dict[str, Any],
        output_file: str
    ) -> Dict[str, Any]:
        """
        Draft document section by section with context sandwiches.
        
        This returns instructions for the agent to follow, as the actual
        drafting requires agent interaction.
        
        Args:
            toc: Table of Contents structure
            output_file: Path to output file for iterative writing
            
        Returns:
            Instructions and section list for agent
        """
        sections = self.get_section_list_from_toc(toc)
        
        return {
            "workflow": "iterative_drafting",
            "output_file": output_file,
            "total_sections": len(sections),
            "sections": sections,
            "instructions": """Draft each section iteratively:

For each section:
1. Call prepare_section_context() to get context sandwich
2. Draft the section content with:
   - Bridge-in (1-3 sentences from previous section)
   - Main content (follow outline and description)
   - Bridge-out (1-3 sentences to next section)
   - Change log entry
3. Call upsert_section_to_file() to append to document
4. Move to next section

After all sections are drafted:
- Run quality passes (continuity, formatting, compression if needed)
- Create final document via create_doc()"""
        }
    
    # ============================================================================
    # Quality Passes
    # ============================================================================
    
    def run_quality_passes(
        self,
        content: str,
        passes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run quality validation passes on document content.
        
        Args:
            content: Document content to validate
            passes: List of passes to run (default: all enabled in config)
            
        Returns:
            Quality report with issues and suggestions
        """
        if passes is None:
            passes = []
            if self.config["run_continuity_pass"]:
                passes.append("continuity")
            if self.config["run_formatting_pass"]:
                passes.append("formatting")
            
            # Only run compression if doc is large
            doc_size = self.estimate_tokens(content)
            if doc_size > self.config["run_compression_pass_threshold"]:
                passes.append("compression")
        
        results = {
            "passes_run": passes,
            "total_issues": 0,
            "issues_by_pass": {},
            "suggestions": []
        }
        
        if "continuity" in passes:
            continuity_result = self._run_continuity_pass(content)
            results["issues_by_pass"]["continuity"] = continuity_result
            results["total_issues"] += len(continuity_result["issues"])
            results["suggestions"].extend(continuity_result.get("suggestions", []))
        
        if "formatting" in passes:
            formatting_result = self._run_formatting_pass(content)
            results["issues_by_pass"]["formatting"] = formatting_result
            results["total_issues"] += len(formatting_result["issues"])
            results["suggestions"].extend(formatting_result.get("suggestions", []))
        
        if "compression" in passes:
            compression_result = self._run_compression_pass(content)
            results["issues_by_pass"]["compression"] = compression_result
            results["suggestions"].extend(compression_result.get("suggestions", []))
        
        return results
    
    def _run_continuity_pass(self, content: str) -> Dict[str, Any]:
        """
        Check document continuity: term consistency, references, transitions.
        
        Args:
            content: Document content
            
        Returns:
            Continuity check results
        """
        issues = []
        suggestions = []
        
        # Check for common term inconsistencies
        term_variations = self._find_term_variations(content)
        if term_variations:
            issues.append({
                "type": "term_inconsistency",
                "count": len(term_variations),
                "details": term_variations
            })
            suggestions.append("Standardize terminology across document")
        
        # Check for broken internal references
        broken_refs = self._find_broken_references(content)
        if broken_refs:
            issues.append({
                "type": "broken_reference",
                "count": len(broken_refs),
                "details": broken_refs
            })
            suggestions.append("Fix or remove broken section references")
        
        # Check for unclear pronouns
        unclear_pronouns = self._find_unclear_pronouns(content)
        if unclear_pronouns:
            issues.append({
                "type": "unclear_pronoun",
                "count": len(unclear_pronouns),
                "details": unclear_pronouns
            })
            suggestions.append("Replace ambiguous pronouns with specific nouns")
        
        return {
            "pass": "continuity",
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _run_formatting_pass(self, content: str) -> Dict[str, Any]:
        """
        Check document formatting: heading levels, lists, tables.
        
        Args:
            content: Document content
            
        Returns:
            Formatting check results
        """
        issues = []
        suggestions = []
        
        # Check heading levels (should not skip levels)
        heading_issues = self._check_heading_levels(content)
        if heading_issues:
            issues.append({
                "type": "heading_level_skip",
                "count": len(heading_issues),
                "details": heading_issues
            })
            suggestions.append("Fix heading level hierarchy (H1 → H2 → H3, no skipping)")
        
        # Check for inconsistent list styles
        list_inconsistencies = self._check_list_consistency(content)
        if list_inconsistencies:
            issues.append({
                "type": "list_inconsistency",
                "count": len(list_inconsistencies),
                "details": list_inconsistencies
            })
            suggestions.append("Standardize list formatting (bullets vs numbered)")
        
        # Check for duplicated content
        duplicates = self._find_duplicate_content(content)
        if duplicates:
            issues.append({
                "type": "duplicate_content",
                "count": len(duplicates),
                "details": duplicates
            })
            suggestions.append("Remove or consolidate duplicate sections")
        
        return {
            "pass": "formatting",
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _run_compression_pass(self, content: str) -> Dict[str, Any]:
        """
        Suggest compression opportunities for large documents.
        
        Args:
            content: Document content
            
        Returns:
            Compression suggestions
        """
        suggestions = []
        
        doc_size = self.estimate_tokens(content)
        
        # Find repetitive paragraphs that could be bullets
        repetitive_sections = self._find_repetitive_sections(content)
        if repetitive_sections:
            suggestions.append({
                "type": "convert_to_bullets",
                "count": len(repetitive_sections),
                "details": repetitive_sections,
                "description": "Replace repetitive paragraphs with bullet points"
            })
        
        # Suggest moving detailed content to appendices
        detailed_sections = self._find_overly_detailed_sections(content)
        if detailed_sections:
            suggestions.append({
                "type": "move_to_appendix",
                "count": len(detailed_sections),
                "details": detailed_sections,
                "description": "Move deep technical details to appendices"
            })
        
        suggestions.append({
            "type": "overall_size",
            "current_tokens": doc_size,
            "threshold": self.config["run_compression_pass_threshold"],
            "description": f"Document is {doc_size} tokens (threshold: {self.config['run_compression_pass_threshold']})"
        })
        
        return {
            "pass": "compression",
            "issues": [],
            "suggestions": suggestions
        }
    
    def _find_term_variations(self, content: str) -> List[Dict[str, Any]]:
        """Find variations of the same term that should be standardized."""
        # Common patterns to check
        patterns = [
            (r'\bAPI\b', r'\bapi\b', r'\bApi\b'),
            (r'\bID\b', r'\bId\b', r'\bid\b'),
            (r'\bURL\b', r'\bUrl\b', r'\burl\b'),
            (r'\bJSON\b', r'\bJson\b', r'\bjson\b'),
        ]
        
        variations = []
        for pattern_group in patterns:
            found_variants = []
            for pattern in pattern_group:
                if re.search(pattern, content):
                    found_variants.append(pattern)
            
            if len(found_variants) > 1:
                variations.append({
                    "term": pattern_group[0],
                    "variants_found": found_variants,
                    "suggestion": f"Standardize to {pattern_group[0]}"
                })
        
        return variations
    
    def _find_broken_references(self, content: str) -> List[str]:
        """Find references to sections that don't exist."""
        # Find section references like "Section X" or "Chapter Y"
        ref_pattern = r'(?:Section|Chapter|Part)\s+(\d+(?:\.\d+)*)'
        references = re.findall(ref_pattern, content, re.IGNORECASE)
        
        # Find actual sections
        heading_pattern = r'^#{1,6}\s+(.+)$'
        headings = re.findall(heading_pattern, content, re.MULTILINE)
        
        broken = []
        for ref in set(references):
            # Check if this section number exists in headings
            found = False
            for heading in headings:
                if ref in heading:
                    found = True
                    break
            if not found:
                broken.append(f"Section {ref}")
        
        return broken
    
    def _find_unclear_pronouns(self, content: str) -> List[str]:
        """Find potentially unclear pronoun usage."""
        # Look for sentences starting with pronouns that might be ambiguous
        pronoun_starts = re.findall(
            r'(?:^|\.\s+)((?:This|That|It|These|Those)\s+[^.]+\.)',
            content,
            re.MULTILINE
        )
        
        # Return first few examples (not all)
        return pronoun_starts[:5] if len(pronoun_starts) > 5 else pronoun_starts
    
    def _check_heading_levels(self, content: str) -> List[str]:
        """Check for skipped heading levels."""
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        headings = re.findall(heading_pattern, content, re.MULTILINE)
        
        issues = []
        prev_level = 0
        
        for hash_marks, heading_text in headings:
            current_level = len(hash_marks)
            
            if current_level > prev_level + 1:
                issues.append(f"Skipped level before: {hash_marks} {heading_text}")
            
            prev_level = current_level
        
        return issues
    
    def _check_list_consistency(self, content: str) -> List[str]:
        """Check for inconsistent list formatting."""
        # This is a simplified check - could be more sophisticated
        bullet_lists = len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE))
        numbered_lists = len(re.findall(r'^\s*\d+\.\s', content, re.MULTILINE))
        
        issues = []
        if bullet_lists > 0 and numbered_lists > 0:
            issues.append(f"Mixed list styles: {bullet_lists} bullet lists, {numbered_lists} numbered lists")
        
        return issues
    
    def _find_duplicate_content(self, content: str) -> List[str]:
        """Find potentially duplicate paragraphs."""
        paragraphs = content.split('\n\n')
        
        seen = {}
        duplicates = []
        
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:  # Skip short paragraphs
                continue
            
            # Use first 100 chars as key
            key = para[:100]
            if key in seen:
                duplicates.append(para[:100] + "...")
            else:
                seen[key] = True
        
        return duplicates
    
    def _find_repetitive_sections(self, content: str) -> List[str]:
        """Find sections with repetitive structure that could be compressed."""
        # Look for paragraphs starting with similar patterns
        paragraphs = content.split('\n\n')
        
        repetitive = []
        pattern_counts = {}
        
        for para in paragraphs:
            if len(para) < 50:
                continue
            
            # Extract pattern (first few words)
            words = para.split()[:5]
            pattern = ' '.join(words)
            
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        for pattern, count in pattern_counts.items():
            if count >= 3:
                repetitive.append(f"Pattern '{pattern}...' appears {count} times")
        
        return repetitive
    
    def _find_overly_detailed_sections(self, content: str) -> List[str]:
        """Find sections that are very long and might benefit from appendices."""
        # Split by main headings
        sections = re.split(r'^##\s+(.+)$', content, flags=re.MULTILINE)
        
        detailed = []
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                heading = sections[i]
                section_content = sections[i + 1]
                
                tokens = self.estimate_tokens(section_content)
                if tokens > 2000:  # Very large section
                    detailed.append(f"Section '{heading}' is {tokens} tokens")
        
        return detailed
    
    # ============================================================================
    # Consolidated Workflows (Deterministic Sequences)
    # ============================================================================
    
    def edit_doc_section(
        self,
        docId: str,
        search_target: str,
        old_block: str,
        new_block: str,
        context_lines: int = 10,
        html_from: Optional[int] = None,
        html_to: Optional[int] = None,
        context_before: int = 0,
        context_after: int = 0
    ) -> Dict[str, Any]:
        """
        Edit a specific section in a document (end-to-end workflow).
        
        This consolidated method handles the complete targeted edit workflow:
        1. Export doc to temp file
        2. Search for target area (exact → fuzzy → RAG)
        3. Apply diff-first edit
        4. Merge changes back to document
        
        Args:
            docId: Document ID
            search_target: Text to search for (helps locate the section)
            old_block: Exact text to replace
            new_block: Replacement text
            context_lines: Lines of context to show (default: 10)
            
        Returns:
            Dictionary with edit status and details
        """
        try:
            logger.info(f"Starting consolidated edit workflow for doc {docId}")
            temp_file = None
            tokens_before = None
            preferred_offset = html_from

            if html_from is not None and html_to is not None:
                extraction = self.extract_text_at_html_positions(
                    docId=docId,
                    html_from=html_from,
                    html_to=html_to,
                    context_before=context_before,
                    context_after=context_after
                )
                if extraction.get("success"):
                    extracted_text = extraction.get("extracted_text") or ""
                    context_text = extraction.get("context_text") or ""
                    if extracted_text:
                        old_block = extracted_text
                        search_target = context_text or extracted_text
                        logger.info(
                            f"Derived edit blocks from HTML positions for doc {docId}: "
                            f"context_from={extraction.get('context_from')}, context_to={extraction.get('context_to')}"
                        )
                else:
                    logger.warning(
                        f"Failed to extract text at HTML positions for doc {docId}: {extraction.get('error')}"
                    )

            # Step 1: Reuse existing temp file when available
            cached_path = self._temp_files.get(docId)
            if cached_path:
                if os.path.exists(cached_path):
                    temp_file = cached_path
                    try:
                        with open(temp_file, 'r', encoding='utf-8') as f:
                            tokens_before = self.estimate_tokens(f.read())
                    except Exception as e:
                        logger.warning(f"Error estimating tokens from cached file {temp_file}: {str(e)}")
                    logger.info(f"Using cached temp file for doc {docId}: {temp_file}")
                else:
                    logger.warning(f"Cached temp file missing for doc {docId}: {cached_path}")
                    del self._temp_files[docId]

            # Step 2: Load document and export to temp file if no cache is available
            if not temp_file:
                doc_result = self.get_doc([docId])
                if "error" in doc_result:
                    return doc_result

                if not isinstance(doc_result, list) or not doc_result:
                    return {"error": f"Document not found for id: {docId}"}

                content = doc_result[0].get("content", "")
                export_result = self._export_doc_to_temp_file_internal(docId, content)
                if "error" in export_result:
                    return export_result

                temp_file = export_result["file_path"]
                tokens_before = export_result.get("size_tokens")
            
            try:
                # Step 3: Search for target area
                search_result = self._search_in_doc_internal(
                    temp_file, 
                    search_target, 
                    context_lines
                )
                
                if not search_result.get("found"):
                    return {
                        "error": f"Could not locate target area: {search_target}",
                        "suggestion": search_result.get("suggestion", "Try RAG search or provide more specific target")
                    }
                
                # Step 4: Apply diff edit
                edit_result = self._apply_diff_edit_internal(
                    temp_file,
                    old_block,
                    new_block,
                    preferred_offset=preferred_offset
                )
                if "error" in edit_result:
                    return edit_result
                
                # Step 5: Merge back to document
                merge_result = self._merge_temp_file_to_doc_internal(docId, temp_file)
                if "error" in merge_result:
                    return merge_result
                
                logger.info(f"Successfully completed edit workflow for doc {docId}")
                
                return {
                    "success": True,
                    "docId": docId,
                    "search_result": {
                        "found": True,
                        "match_type": search_result.get("match_type"),
                        "line_number": search_result.get("line_number")
                    },
                    "edit_applied": True,
                    "tokens_before": tokens_before,
                    "tokens_after": merge_result.get("updated_tokens"),
                    "message": "Document section updated successfully"
                }
                
            finally:
                # Cleanup temp file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except Exception as e:
            logger.error(f"Error in edit_doc_section: {str(e)}")
            return {"error": str(e)}
    
    def search_large_doc(
        self,
        docId: str,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search in a large document using RAG (auto-chunks if needed).
        
        This consolidated method:
        1. Checks if document needs chunking
        2. Chunks document by headings (with overlap) if needed
        3. Retrieves relevant chunks via RAG
        
        Args:
            docId: Document ID
            query: Search query
            top_k: Number of chunks to retrieve
            
        Returns:
            Retrieved chunks with context
        """
        try:
            logger.info(f"Searching large doc {docId} with query: {query}")
            
            # Load document to check if chunking needed
            doc_result = self.get_doc(docId)
            if "error" in doc_result:
                return doc_result
            
            content = doc_result.get("content", "")
            doc_size = self.estimate_tokens(content)
            
            # If doc is small enough, just return relevant excerpt
            if doc_size < self.config["max_context_tokens"]:
                return {
                    "success": True,
                    "docId": docId,
                    "needs_chunking": False,
                    "doc_size_tokens": doc_size,
                    "message": "Document is small enough to process directly",
                    "content": content
                }
            
            # Document is large, chunk and search
            # Step 1: Chunk document
            chunk_result = self._chunk_document_for_rag_internal(docId, content)
            if "error" in chunk_result:
                return chunk_result
            
            # Step 2: Retrieve chunks
            retrieval_instructions = {
                "workflow": "rag_chunk_retrieval",
                "docId": docId,
                "query": query,
                "rag_document_id": chunk_result["rag_document_id"],
                "chunk_count": chunk_result["chunk_count"],
                "instructions": f"""Use search_documents to retrieve relevant chunks:

1. Query: "{query}"
2. Namespace: {self.org_slug}_tool_responses
3. Filter by metadata: doc_id = {docId}
4. Request top {top_k} chunks
5. Chunks include neighbor context for continuity

The document has been chunked into {chunk_result['chunk_count']} chunks by {chunk_result['chunk_method']}.""",
                "namespace": f"{self.org_slug}_tool_responses",
                "top_k": top_k
            }
            
            return {
                "success": True,
                "docId": docId,
                "needs_chunking": True,
                "doc_size_tokens": doc_size,
                "chunked": True,
                "chunk_info": chunk_result,
                "retrieval_instructions": retrieval_instructions
            }
            
        except Exception as e:
            logger.error(f"Error in search_large_doc: {str(e)}")
            return {"error": str(e)}
    
    def finalize_doc_update(
        self,
        docId: str,
        original_content: str,
        updated_content: str,
        change_description: str,
        changed_sections: Optional[List[str]] = None,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Finalize document update with validation and change log.
        
        This consolidated method:
        1. Validates the update (continuity, references, contradictions)
        2. Creates change log entry
        3. Returns consolidated report
        
        Args:
            docId: Document ID
            original_content: Original content before update
            updated_content: Updated content
            change_description: Description of what changed
            changed_sections: Optional list of section names that changed
            source: Optional source of the update (e.g., "API v2.0 spec")
            
        Returns:
            Validation report with change log entry
        """
        try:
            logger.info(f"Finalizing update for doc {docId}")
            
            # Step 1: Validate update
            if self.config["enable_post_update_validation"]:
                validation = self.validate_document_update(
                    docId,
                    original_content,
                    updated_content,
                    change_description
                )
            else:
                validation = {
                    "valid": True,
                    "checks": {},
                    "issues": [],
                    "suggestions": []
                }
            
            # Step 2: Create change log entry
            change_entry = {
                "changed_sections": changed_sections or ["Content updated"],
                "reason": change_description,
                "source": source or "Manual update",
                "issues": "None detected" if validation["valid"] else f"{len(validation['issues'])} issues found"
            }
            
            change_log = self.update_change_log(docId, change_entry)
            
            # Step 3: Return consolidated result
            return {
                "success": True,
                "docId": docId,
                "validation": validation,
                "change_log_entry": change_log.get("log_entry"),
                "summary": {
                    "valid": validation["valid"],
                    "issues_count": len(validation.get("issues", [])),
                    "suggestions_count": len(validation.get("suggestions", [])),
                    "change_logged": change_log.get("success", False)
                },
                "instructions": change_log.get("instructions") if not validation["valid"] else None
            }
            
        except Exception as e:
            logger.error(f"Error in finalize_doc_update: {str(e)}")
            return {"error": str(e)}
    
    # ============================================================================
    # Internal Methods (used by consolidated workflows)
    # ============================================================================
    
    def _export_doc_to_temp_file_internal(
        self,
        doc_id: str,
        content: str
    ) -> Dict[str, Any]:
        """
        Export document content to temporary file for editing.
        
        Args:
            doc_id: Document ID
            content: Document content
            
        Returns:
            Dictionary with file path and status
        """
        try:
            # Create temp file (HTML format)
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.html',
                prefix=f'doc_{doc_id}_',
                delete=False
            )
            temp_file.write(content)
            temp_file.close()
            
            self._workflow_temp_files.append(temp_file.name)
            logger.info(f"Exported doc {doc_id} to {temp_file.name}")
            
            return {
                "success": True,
                "file_path": temp_file.name,
                "doc_id": doc_id,
                "size_tokens": self.estimate_tokens(content)
            }
            
        except Exception as e:
            logger.error(f"Error exporting doc to temp file: {str(e)}")
            return {"error": str(e)}
    
    def _search_in_doc_internal(
        self,
        file_path: str,
        search_target: str,
        context_lines: int = 10
    ) -> Dict[str, Any]:
        """
        Search for target text in document and extract context window.
        
        Args:
            file_path: Path to document file
            search_target: Text to search for
            context_lines: Number of lines before/after to include
            
        Returns:
            Search results with context window
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Try exact match first
            matches = []
            for i, line in enumerate(lines):
                if search_target in line:
                    matches.append({
                        "line_number": i + 1,
                        "match_type": "exact",
                        "line": line.strip()
                    })
            
            # If no exact match, try fuzzy match
            if not matches:
                matches = self._fuzzy_search(lines, search_target)
            
            if not matches:
                return {
                    "found": False,
                    "message": "Target not found with exact or fuzzy search",
                    "suggestion": "Use RAG fallback for semantic search"
                }
            
            # Extract context window for first match
            match = matches[0]
            line_idx = match["line_number"] - 1
            
            start_idx = max(0, line_idx - context_lines)
            end_idx = min(len(lines), line_idx + context_lines + 1)
            
            context_window = ''.join(lines[start_idx:end_idx])
            
            return {
                "found": True,
                "match_type": match["match_type"],
                "line_number": match["line_number"],
                "total_matches": len(matches),
                "context_window": context_window,
                "context_range": {
                    "start_line": start_idx + 1,
                    "end_line": end_idx
                }
            }
            
        except Exception as e:
            logger.error(f"Error searching in doc: {str(e)}")
            return {"error": str(e)}
    
    def _fuzzy_search(
        self,
        lines: List[str],
        search_target: str
    ) -> List[Dict[str, Any]]:
        """
        Perform fuzzy search for target text.
        
        Args:
            lines: Document lines
            search_target: Text to search for
            
        Returns:
            List of fuzzy matches
        """
        matches = []
        search_words = set(search_target.lower().split())
        
        for i, line in enumerate(lines):
            line_words = set(line.lower().split())
            
            # Calculate overlap
            overlap = len(search_words & line_words)
            if overlap >= len(search_words) * 0.6:  # 60% match threshold
                matches.append({
                    "line_number": i + 1,
                    "match_type": "fuzzy",
                    "line": line.strip(),
                    "confidence": overlap / len(search_words)
                })
        
        # Sort by confidence
        matches.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return matches[:5]  # Return top 5 matches
    
    def _apply_diff_edit_internal(
        self,
        file_path: str,
        old_block: str,
        new_block: str,
        preferred_offset: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Apply diff-first edit to document file.
        
        Args:
            file_path: Path to document file
            old_block: Text to replace (must match exactly)
            new_block: Replacement text
            
        Returns:
            Status of edit operation
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if old_block exists
            if old_block not in content:
                return {
                    "error": "Old block not found in document",
                    "suggestion": "Verify the exact text to replace"
                }
            
            # Handle multiple occurrences with preferred offset if provided
            occurrences = []
            start = 0
            while True:
                idx = content.find(old_block, start)
                if idx == -1:
                    break
                occurrences.append(idx)
                start = idx + len(old_block)
            
            if not occurrences:
                return {
                    "error": "Old block not found in document",
                    "suggestion": "Verify the exact text to replace"
                }
            
            if len(occurrences) == 1:
                target_idx = occurrences[0]
            elif preferred_offset is not None:
                target_idx = min(occurrences, key=lambda x: abs(x - preferred_offset))
            else:
                return {
                    "error": f"Old block appears {len(occurrences)} times",
                    "suggestion": "Provide more context to make the block unique"
                }
            
            updated_content = content[:target_idx] + new_block + content[target_idx + len(old_block):]
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"Applied diff edit to {file_path}")
            
            return {
                "success": True,
                "file_path": file_path,
                "old_size_tokens": self.estimate_tokens(content),
                "new_size_tokens": self.estimate_tokens(updated_content)
            }
            
        except Exception as e:
            logger.error(f"Error applying diff edit: {str(e)}")
            return {"error": str(e)}
    
    def _merge_temp_file_to_doc_internal(
        self,
        doc_id: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Merge edited temp file back to document.
        
        Args:
            doc_id: Document ID
            file_path: Path to edited temp file
            
        Returns:
            Status of merge operation
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                updated_content = f.read()
            
            # Sanitize HTML content before updating
            updated_content = self._sanitize_html(updated_content)

            if not updated_content or not updated_content.strip():
                logger.error(f"Refusing to update doc {doc_id} with empty content from {file_path}")
                return {"error": "Updated content is empty; aborting update"}
            
            # Update document via doc_tool (now accepts HTML)
            result = self.update_doc(
                docId=doc_id,
                content=updated_content
            )
            
            if "error" in result:
                return result
            
            logger.info(f"Merged temp file to doc {doc_id}")
            
            return {
                "success": True,
                "doc_id": doc_id,
                "updated_tokens": self.estimate_tokens(updated_content)
            }
            
        except Exception as e:
            logger.error(f"Error merging temp file to doc: {str(e)}")
            return {"error": str(e)}
    
    # ============================================================================
    # Broad Update Workflow
    # ============================================================================
    
    def generate_impact_map(
        self,
        doc_structure: str,
        update_inputs: str
    ) -> Dict[str, Any]:
        """
        Generate impact map for broad document updates.
        
        This returns a template for the agent to populate based on analysis.
        
        Args:
            doc_structure: Document structure (headings, sections)
            update_inputs: New information/requirements
            
        Returns:
            Impact map template for agent
        """
        return {
            "workflow": "impact_map_generation",
            "instructions": """Analyze the document structure and update inputs to determine:

1. Which sections are impacted by the new information?
2. What is the reasoning for each impact?
3. What information is missing or unclear?
4. Are there new sections needed?

Output format:
{
    "impacted_sections": ["Section ID 1", "Section ID 2", ...],
    "reasoning": "Explanation of why these sections need updates",
    "missing_info": ["What info is needed", "What needs clarification"],
    "new_sections": ["New section to add", ...]
}""",
            "doc_structure": doc_structure,
            "update_inputs": update_inputs
        }
    
    def update_section_with_rag(
        self,
        doc_id: str,
        section_id: str,
        section_heading: str,
        update_info: str
    ) -> Dict[str, Any]:
        """
        Update a specific section using RAG retrieval.
        
        Args:
            doc_id: Document ID
            section_id: Section identifier
            section_heading: Section heading to search for
            update_info: New information to incorporate
            
        Returns:
            Instructions for section update
        """
        return {
            "workflow": "rag_section_update",
            "section_id": section_id,
            "section_heading": section_heading,
            "instructions": f"""Update section '{section_heading}':

1. Use search_documents to retrieve the section content
2. Review the current section content
3. Incorporate the new information: {update_info}
4. Maintain the section structure and style
5. Add bridge sentences if needed to maintain flow
6. Mark any assumptions or TODOs
7. Provide the updated section content for merging""",
            "doc_id": doc_id,
            "update_info": update_info
        }
    
    # ============================================================================
    # RAG Fallback for Large Documents
    # ============================================================================
    
    def _chunk_document_for_rag_internal(
        self,
        doc_id: str,
        content: str,
        chunk_by_headings: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Chunk document and store in RAG for semantic search.
        
        Args:
            doc_id: Document ID
            content: Document content
            chunk_by_headings: Whether to chunk by headings (default from config)
            
        Returns:
            Chunking status and document_id for retrieval
        """
        if chunk_by_headings is None:
            chunk_by_headings = self.config["chunk_by_headings"]
        
        try:
            if chunk_by_headings:
                chunks = self._chunk_by_headings(content)
            else:
                chunks = self._chunk_by_paragraphs(content)
            
            # Store in RAG using existing RAGStorageTool
            rag_doc_id = self.rag_storage.store_tool_response(
                content=content,
                tool_name="doc_management",
                tool_input={"doc_id": doc_id, "action": "chunk_for_editing"},
                metadata={
                    "doc_id": doc_id,
                    "chunk_count": len(chunks),
                    "chunk_method": "headings" if chunk_by_headings else "paragraphs"
                }
            )
            
            logger.info(f"Chunked doc {doc_id} into {len(chunks)} chunks, stored as {rag_doc_id}")
            
            return {
                "success": True,
                "doc_id": doc_id,
                "rag_document_id": rag_doc_id,
                "chunk_count": len(chunks),
                "chunk_method": "headings" if chunk_by_headings else "paragraphs"
            }
            
        except Exception as e:
            logger.error(f"Error chunking document for RAG: {str(e)}")
            return {"error": str(e)}
    
    def _chunk_by_headings(self, content: str) -> List[Dict[str, Any]]:
        """
        Chunk document by heading boundaries, preserving structure.
        
        Args:
            content: Document content
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split by headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_chunk = []
        current_heading = None
        current_level = 0
        heading_ancestry = []
        
        for line in lines:
            heading_match = re.match(heading_pattern, line)
            
            if heading_match:
                # Save previous chunk if exists
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "heading": current_heading,
                        "level": current_level,
                        "heading_ancestry": ' > '.join(heading_ancestry),
                        "tokens": self.estimate_tokens(chunk_text)
                    })
                
                # Start new chunk
                level = len(heading_match.group(1))
                heading = heading_match.group(2)
                
                # Update ancestry
                if level == 1:
                    heading_ancestry = [heading]
                elif level <= len(heading_ancestry):
                    heading_ancestry = heading_ancestry[:level-1] + [heading]
                else:
                    heading_ancestry.append(heading)
                
                current_chunk = [line]
                current_heading = heading
                current_level = level
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "heading": current_heading,
                "level": current_level,
                "heading_ancestry": ' > '.join(heading_ancestry),
                "tokens": self.estimate_tokens(chunk_text)
            })
        
        # Add overlap between chunks
        chunks = self._add_chunk_overlap(
            chunks,
            self.config["heading_chunk_overlap"]
        )
        
        return chunks
    
    def _chunk_by_paragraphs(self, content: str) -> List[Dict[str, Any]]:
        """
        Chunk document by paragraphs with overlap.
        
        Args:
            content: Document content
            
        Returns:
            List of chunks with metadata
        """
        paragraphs = content.split('\n\n')
        chunks = []
        
        chunk_size_tokens = 512  # Target chunk size
        overlap_tokens = self.config["paragraph_chunk_overlap"]
        
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)
            
            if current_tokens + para_tokens > chunk_size_tokens and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "tokens": current_tokens,
                    "paragraph_count": len(current_chunk)
                })
                
                # Start new chunk with overlap
                # Keep last paragraph(s) for overlap
                overlap_content = []
                overlap_token_count = 0
                for prev_para in reversed(current_chunk):
                    para_tok = self.estimate_tokens(prev_para)
                    if overlap_token_count + para_tok <= overlap_tokens:
                        overlap_content.insert(0, prev_para)
                        overlap_token_count += para_tok
                    else:
                        break
                
                current_chunk = overlap_content + [para]
                current_tokens = overlap_token_count + para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "tokens": current_tokens,
                "paragraph_count": len(current_chunk)
            })
        
        return chunks
    
    def _add_chunk_overlap(
        self,
        chunks: List[Dict[str, Any]],
        overlap_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Add overlap between heading-based chunks.
        
        Args:
            chunks: List of chunks
            overlap_tokens: Number of tokens to overlap
            
        Returns:
            Chunks with overlap added
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            
            # Add content from previous chunk (end)
            if i > 0:
                prev_text = chunks[i-1]["text"]
                prev_excerpt = self.extract_last_n_tokens(prev_text, overlap_tokens)
                chunk_text = prev_excerpt + "\n\n" + chunk_text
            
            # Add content from next chunk (beginning)
            if i < len(chunks) - 1:
                next_text = chunks[i+1]["text"]
                next_excerpt = self.extract_first_n_tokens(next_text, overlap_tokens)
                chunk_text = chunk_text + "\n\n" + next_excerpt
            
            overlapped_chunk = chunk.copy()
            overlapped_chunk["text"] = chunk_text
            overlapped_chunk["tokens"] = self.estimate_tokens(chunk_text)
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def retrieve_chunks_for_update(
        self,
        doc_id: str,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks from RAG for document update.
        
        Args:
            doc_id: Document ID
            query: Search query
            top_k: Number of chunks to retrieve
            
        Returns:
            Retrieved chunks with context
        """
        return {
            "workflow": "rag_chunk_retrieval",
            "doc_id": doc_id,
            "query": query,
            "instructions": f"""Retrieve relevant chunks for updating:

1. Use search_documents with query: "{query}"
2. Specify namespace: {self.org_slug}_tool_responses
3. Filter by metadata: doc_id = {doc_id}
4. Request top {top_k} chunks
5. Review chunks and their neighbor context
6. Identify which chunks need updating
7. Apply edits to relevant chunks
8. Merge updated chunks back to document

The chunks include overlap with neighbors for continuity.""",
            "namespace": f"{self.org_slug}_tool_responses",
            "top_k": top_k
        }
    
    # ============================================================================
    # Post-Update Validation
    # ============================================================================
    
    def validate_document_update(
        self,
        doc_id: str,
        original_content: str,
        updated_content: str,
        change_description: str
    ) -> Dict[str, Any]:
        """
        Validate document after update.
        
        Args:
            doc_id: Document ID
            original_content: Original content before update
            updated_content: Updated content
            change_description: Description of changes made
            
        Returns:
            Validation report
        """
        validation_report = {
            "doc_id": doc_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "change_description": change_description,
            "checks": {},
            "issues": [],
            "suggestions": [],
            "valid": True
        }
        
        # Run continuity checks on updated content
        if self.config["enable_post_update_validation"]:
            continuity = self._run_continuity_pass(updated_content)
            validation_report["checks"]["continuity"] = continuity
            
            if continuity["issues"]:
                validation_report["issues"].extend([
                    f"Continuity: {issue['type']}" for issue in continuity["issues"]
                ])
                validation_report["valid"] = False
            
            validation_report["suggestions"].extend(continuity["suggestions"])
            
            # Check for contradictions introduced
            contradictions = self._check_for_contradictions(
                original_content,
                updated_content
            )
            if contradictions:
                validation_report["checks"]["contradictions"] = contradictions
                validation_report["issues"].append(f"Found {len(contradictions)} potential contradictions")
                validation_report["suggestions"].append("Review and resolve contradictions")
        
        return validation_report
    
    def _check_for_contradictions(
        self,
        original: str,
        updated: str
    ) -> List[str]:
        """
        Check for potential contradictions between original and updated content.
        
        This is a basic implementation that could be enhanced with LLM-based checking.
        
        Args:
            original: Original content
            updated: Updated content
            
        Returns:
            List of potential contradictions
        """
        contradictions = []
        
        # Look for statements that were reversed
        # e.g., "X is true" changed to "X is false"
        # This is a simplified heuristic
        
        original_sentences = re.split(r'[.!?]+', original)
        updated_sentences = re.split(r'[.!?]+', updated)
        
        # Look for negation flips
        negation_words = ['not', 'no', 'never', 'none', 'cannot', 'shouldn\'t', 'won\'t']
        
        for orig_sent in original_sentences:
            orig_sent = orig_sent.strip()
            if len(orig_sent) < 20:
                continue
            
            # Extract key terms
            orig_words = set(orig_sent.lower().split())
            has_negation_orig = any(neg in orig_words for neg in negation_words)
            
            for upd_sent in updated_sentences:
                upd_sent = upd_sent.strip()
                if len(upd_sent) < 20:
                    continue
                
                upd_words = set(upd_sent.lower().split())
                has_negation_upd = any(neg in upd_words for neg in negation_words)
                
                # Check for similar sentences with flipped negation
                overlap = len(orig_words & upd_words)
                if overlap > len(orig_words) * 0.5:  # 50% word overlap
                    if has_negation_orig != has_negation_upd:
                        contradictions.append(f"Original: {orig_sent[:100]}... | Updated: {upd_sent[:100]}...")
        
        return contradictions[:5]  # Return first 5 potential contradictions
    
    def update_change_log(
        self,
        doc_id: str,
        change_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add entry to document change log.
        
        Args:
            doc_id: Document ID
            change_entry: Change log entry with timestamp, changes, reason, source
            
        Returns:
            Status of change log update
        """
        timestamp = change_entry.get("timestamp") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        changed_sections = change_entry.get("changed_sections", [])
        reason = change_entry.get("reason", "")
        source = change_entry.get("source", "")
        issues = change_entry.get("issues", "None detected")
        
        # Format change log entry
        log_entry = f"""
### {timestamp}
- **Changed**: {', '.join(changed_sections) if changed_sections else 'Multiple sections'}
- **Reason**: {reason}
- **Source**: {source}
- **Issues**: {issues}
"""
        
        return {
            "success": True,
            "doc_id": doc_id,
            "log_entry": log_entry,
            "instructions": """Append this change log entry to the document:

1. Check if document has a '## Change Log' section
2. If not, add one at the end of the document
3. Append the new entry under the Change Log heading
4. Update the document via update_doc()"""
        }
    
    # ============================================================================
    # Internal Helper Methods
    # ============================================================================
    
    def _detect_update_strategy(
        self,
        content: str,
        update_request: str
    ) -> Dict[str, Any]:
        """
        Detect appropriate update strategy based on document size and request.
        
        Args:
            content: Current document content
            update_request: User's update request
            
        Returns:
            Strategy information dictionary
        """
        doc_size_tokens = self.estimate_tokens(content)
        fits_in_context = doc_size_tokens < self.config["max_context_tokens"]
        
        # Check if request has specific target (section name, quote, line number)
        has_specific_target = self._has_specific_target(update_request)
        
        if fits_in_context:
            strategy = "direct"
        elif has_specific_target:
            strategy = "targeted"
        else:
            strategy = "broad"
        
        return {
            "strategy": strategy,
            "fits_in_context": fits_in_context,
            "doc_size_tokens": doc_size_tokens,
            "has_specific_target": has_specific_target
        }
    
    def _has_specific_target(self, update_request: str) -> bool:
        """
        Check if update request specifies a specific target location.
        
        Args:
            update_request: User's update request
            
        Returns:
            True if request has specific target indicators
        """
        # Look for section references
        section_patterns = [
            r'section\s+\d+',
            r'heading\s+"[^"]+"',
            r'paragraph\s+about',
            r'line\s+\d+',
            r'in\s+the\s+.+\s+section',
        ]
        
        for pattern in section_patterns:
            if re.search(pattern, update_request, re.IGNORECASE):
                return True
        
        # Look for quoted text (likely a specific target)
        if '"' in update_request or "'" in update_request:
            return True
        
        return False
    
    def _get_update_instructions(self, strategy: Dict[str, Any]) -> str:
        """
        Generate instructions for the agent based on update strategy.
        
        Args:
            strategy: Strategy information from _detect_update_strategy
            
        Returns:
            Instructions string for the agent
        """
        strategy_type = strategy["strategy"]
        
        if strategy_type == "direct":
            return """The document fits in context. You can update it directly:
1. Review the current content
2. Apply the requested updates
3. Use update_doc() to save changes
4. Run post-update validation if enabled"""
        
        elif strategy_type == "targeted":
            return """The document is large but you have a specific target. Use targeted edit workflow:
1. Export the document to a temp file
2. Search for the target area (exact match, then fuzzy if needed)
3. Extract a local window (context before + target + context after)
4. Apply diff-first edit (OLD_BLOCK → NEW_BLOCK)
5. Merge changes back using update_doc()
6. Run post-update validation"""
        
        else:  # broad
            return """The document is large and requires broad updates. Use structure-first workflow:
1. Generate an impact map (which sections need updates)
2. Optionally confirm with user
3. For each impacted section:
   - Retrieve section using RAG search
   - Update with new information
   - Upsert back to document
4. Run post-update validation
5. Update change log"""
    
    # ============================================================================
    # Cleanup
    # ============================================================================
    
    def cleanup_temp_files(self):
        """Clean up any temporary files created during workflows."""
        for temp_file in self._workflow_temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Removed temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        self._workflow_temp_files.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup_temp_files()
