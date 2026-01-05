import json
import logging
import secrets
import tempfile
import os
from typing import Dict, List, Any, Optional, Union
from psycopg2.extras import RealDictCursor
import re

from .postgres import PostgresTool, AI_AGENT_ID

logger = logging.getLogger(__name__)


class DocManagementTool:
    """
    Document management tool for creating and updating documents.
    
    Documents are worked with in markdown format using Anthropic's text editor tool,
    then converted to HTML when saving to the database. The AI agent uses the text
    editor tool directly for formatting, eliminating the need for specific formatting
    helper functions.
    """
    
    def __init__(self, postgres_client_wrapper, user_id: Optional[str] = None):
        """
        Initialize DocManagementTool with database access.
        
        Args:
            postgres_client_wrapper: An object with attributes `org_slug` (organization name)
            user_id: Optional user ID (not used for attribution - all creations use AI_AGENT_ID)
        """
        self.postgres_client_wrapper = postgres_client_wrapper
        self.user_id = user_id
        
        # Get org_slug from wrapper (use client_name as fallback)
        self.org_slug = getattr(self.postgres_client_wrapper, 'org_slug', None)
        if not self.org_slug:
            # Fallback: construct org_slug from client_name if available
            client_name = getattr(self.postgres_client_wrapper, 'client_name', 'unknown')
            self.org_slug = f"{client_name}.ai" if client_name != 'unknown' else 'leanworks.ai'
            logger.warning(f"org_slug not provided in wrapper, using fallback: {self.org_slug}")
        
        # Get database name from org_slug
        self.database_name = PostgresTool._get_database_name(self.org_slug)
        
        # Get credential path
        credential_path = getattr(self.postgres_client_wrapper, 'credential_path', 'gcp_credential.json')
        
        # Reuse PostgresTool's connection pool mechanism
        self.pool = PostgresTool._get_connection_pool(self.database_name, credential_path)
        
        # Track temporary markdown files
        self._temp_files: Dict[str, str] = {}  # docId -> file_path
    
    # ============================================================================
    # Core Document Operations (moved from PostgresTool)
    # ============================================================================
    
    @property
    def create_doc_property(self):
        description = f"""
        Create a new document in the docs table for org `{self.org_slug}`.
        
        This tool creates documents that are owned by the AI agent 'lean'. All documents created through this tool will have owner_email set to 'lean@leanworks.ai'.
        
        Parameters:
        - title (required): Document title
        - content (required): Document content (markdown or HTML/rich text)
        - projectId (optional): Associated project ID
        - teamId (optional): Associated team ID
        - tags (optional): Array of tag strings
        - isPinned (optional): Boolean (default: false)
        - visibility (optional): 'all_members' or 'specific_members' (default: 'all_members')
        - visibleToMembers (optional): Array of email addresses
        - metadata (optional): JSON object for additional metadata
        
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
                        "description": "Document content (markdown or HTML/rich text) (required)"
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
                    "isPinned": {
                        "type": "boolean",
                        "description": "Whether document is pinned (default: false)"
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
        isPinned: bool = False,
        visibility: str = "all_members",
        visibleToMembers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new document in the docs table.
        
        Args:
            title: Document title (required)
            content: Document content (markdown or HTML) (required)
            projectId: Associated project ID
            teamId: Associated team ID
            tags: Array of tag strings
            isPinned: Whether document is pinned
            visibility: Document visibility
            visibleToMembers: Array of email addresses
            metadata: JSON object for additional metadata
            
        Returns:
            Dictionary with doc id and created fields, or error dictionary
        """
        conn = None
        try:
            if not title or not content:
                return {"error": "title and content are required"}
            
            # Detect if content is markdown and convert to HTML
            html_content = self._convert_to_html_if_needed(content)
            
            # Validate visibility
            valid_visibility = ['all_members', 'specific_members']
            doc_visibility = visibility if visibility in valid_visibility else 'all_members'
            
            # Validate visibleToMembers for specific_members visibility
            visible_to_members_array = []
            if doc_visibility == 'specific_members':
                if not visibleToMembers or not isinstance(visibleToMembers, list) or len(visibleToMembers) == 0:
                    return {"error": "visibleToMembers must be a non-empty array when visibility is specific_members"}
                visible_to_members_array = [email.lower() for email in visibleToMembers]
            
            # Generate doc ID
            doc_id = secrets.token_hex(16)
            
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                # Try to insert with metadata column first
                try:
                    cursor.execute("""
                        INSERT INTO docs (id, title, content, owner_email, project_id, team_id, tags, metadata, is_pinned, visibility, visible_to_members, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """, (
                        doc_id,
                        title,
                        html_content,
                        AI_AGENT_ID,  # Always use AI agent ID
                        projectId,
                        teamId,
                        json.dumps(tags) if tags else '[]',
                        json.dumps(metadata) if metadata else '{}',
                        isPinned,
                        doc_visibility,
                        json.dumps(visible_to_members_array)
                    ))
                except Exception as e:
                    # If metadata column doesn't exist, insert without it
                    if 'column "metadata" does not exist' in str(e).lower() or 'metadata' in str(e).lower():
                        cursor.execute("""
                            INSERT INTO docs (id, title, content, owner_email, project_id, team_id, tags, is_pinned, visibility, visible_to_members, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        """, (
                            doc_id,
                            title,
                            html_content,
                            AI_AGENT_ID,  # Always use AI agent ID
                            projectId,
                            teamId,
                            json.dumps(tags) if tags else '[]',
                            isPinned,
                            doc_visibility,
                            json.dumps(visible_to_members_array)
                        ))
                    else:
                        raise
                
                conn.commit()
            
            logger.info(f"Document created: id={doc_id}, title={title}, owner_email={AI_AGENT_ID}")
            return {
                "id": doc_id,
                "title": title,
                "content": html_content,
                "ownerEmail": AI_AGENT_ID,
                "projectId": projectId,
                "teamId": teamId,
                "tags": tags or [],
                "isPinned": isPinned,
                "visibility": doc_visibility,
                "visibleToMembers": visible_to_members_array
            }
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error creating document: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @property
    def update_doc_property(self):
        description = f"""
        Update an existing document in the docs table for org `{self.org_slug}`.
        
        Parameters:
        - docId (required): Document ID to update
        - title (optional): Update title
        - content (optional): Update content (markdown or HTML)
        - projectId (optional): Update project association
        - teamId (optional): Update team association
        - tags (optional): Update tags array
        - isPinned (optional): Update pinned status
        - visibility (optional): Update visibility
        - visibleToMembers (optional): Update visible members
        - metadata (optional): Update metadata
        
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
                        "description": "Update content (markdown or HTML)"
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
                    "isPinned": {
                        "type": "boolean",
                        "description": "Update pinned status"
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
        - Success: List of document dictionaries with all fields including: id, title, content (full HTML), owner_email, project_id, team_id, tags, is_pinned, visibility, visible_to_members, created_at, updated_at, metadata
        - Error: Dictionary with error message
        
        Note: Content is returned as HTML. Use get_doc_markdown_path if you need markdown format for editing.
        
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
        Get one or more documents by their IDs.
        
        Args:
            docIds: List of document IDs to retrieve
            
        Returns:
            List of document dictionaries with full content, or error dictionary
        """
        conn = None
        try:
            if not docIds:
                return {"error": "docIds is required and must be a non-empty array"}
            
            if not isinstance(docIds, list):
                return {"error": "docIds must be an array"}
            
            # Limit the number of documents to prevent excessive queries
            if len(docIds) > 50:
                return {"error": f"Too many document IDs requested. Maximum is 50, got {len(docIds)}"}
            
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                # Build query with IN clause for multiple IDs
                placeholders = ','.join(['%s'] * len(docIds))
                query = f"""
                    SELECT id, title, content, owner_email, project_id, team_id, tags, 
                           is_pinned, visibility, visible_to_members, created_at, updated_at, metadata
                    FROM docs 
                    WHERE id IN ({placeholders})
                """
                cursor.execute(query, tuple(docIds))
                rows = cursor.fetchall()
                
                # Convert to list of dicts
                docs = []
                found_ids = set()
                for row in rows:
                    doc = dict(row)
                    found_ids.add(doc['id'])
                    
                    # Parse JSONB fields if they're strings
                    if 'tags' in doc:
                        tags_value = doc['tags']
                        if isinstance(tags_value, str):
                            try:
                                tags_value = json.loads(tags_value)
                            except:
                                tags_value = []
                        elif not isinstance(tags_value, list):
                            tags_value = []
                        doc['tags'] = tags_value
                    
                    if 'visible_to_members' in doc:
                        visible_to_members = doc['visible_to_members']
                        if isinstance(visible_to_members, str):
                            try:
                                visible_to_members = json.loads(visible_to_members)
                            except:
                                visible_to_members = []
                        elif not isinstance(visible_to_members, list):
                            visible_to_members = []
                        doc['visible_to_members'] = visible_to_members
                    
                    if 'metadata' in doc:
                        metadata_value = doc['metadata']
                        if isinstance(metadata_value, str):
                            try:
                                metadata_value = json.loads(metadata_value)
                            except:
                                metadata_value = {}
                        elif not isinstance(metadata_value, dict):
                            metadata_value = {}
                        doc['metadata'] = metadata_value
                    
                    docs.append(doc)
                
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
        finally:
            if conn:
                self.pool.putconn(conn)
    
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
        converts it to HTML, and saves it to the database. The temporary file is cleaned up
        after the operation.
        
        Parameters:
        - file_path (required): Path to markdown file
        - title (required): Document title
        - projectId (optional): Associated project ID
        - teamId (optional): Associated team ID
        - tags (optional): Array of tag strings
        - isPinned (optional): Boolean (default: false)
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
                    "isPinned": {
                        "type": "boolean",
                        "description": "Whether document is pinned (default: false)"
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
        converts it to HTML, and updates the document in the database. The temporary file is
        cleaned up after the operation.
        
        Parameters:
        - docId (required): Document ID to update
        - file_path (required): Path to markdown file
        - title (optional): Update title
        - projectId (optional): Update project association
        - teamId (optional): Update team association
        - tags (optional): Update tags array
        - isPinned (optional): Update pinned status
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
                    "isPinned": {
                        "type": "boolean",
                        "description": "Update pinned status"
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
        isPinned: Optional[bool] = None,
        visibility: Optional[str] = None,
        visibleToMembers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update an existing document.
        
        Args:
            docId: Document ID to update (required)
            title: Update title
            content: Update content (markdown or HTML)
            projectId: Update project association
            teamId: Update team association
            tags: Update tags array
            isPinned: Update pinned status
            visibility: Update visibility
            visibleToMembers: Update visible members
            metadata: Update metadata
            
        Returns:
            Dictionary with success status, or error dictionary
        """
        conn = None
        try:
            if not docId:
                return {"error": "docId is required"}
            
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                # Check if doc exists
                cursor.execute(
                    "SELECT id FROM docs WHERE id = %s",
                    (docId,)
                )
                doc_check = cursor.fetchone()
                
                if not doc_check:
                    return {"error": "Document not found"}
                
                # Build dynamic UPDATE query
                set_clauses = []
                values = []
                
                field_map = {
                    'title': 'title',
                    'content': 'content',
                    'projectId': 'project_id',
                    'teamId': 'team_id',
                    'isPinned': 'is_pinned',
                }
                
                # Handle visibility separately
                if visibility is not None:
                    valid_visibility = ['all_members', 'specific_members']
                    doc_visibility = visibility if visibility in valid_visibility else 'all_members'
                    set_clauses.append("visibility = %s")
                    values.append(doc_visibility)
                    
                    # Handle visibleToMembers
                    if doc_visibility == 'specific_members':
                        if not visibleToMembers or not isinstance(visibleToMembers, list) or len(visibleToMembers) == 0:
                            return {"error": "visibleToMembers must be a non-empty array when visibility is specific_members"}
                        visible_to_members_array = [email.lower() for email in visibleToMembers]
                        set_clauses.append("visible_to_members = %s::jsonb")
                        values.append(json.dumps(visible_to_members_array))
                    else:
                        set_clauses.append("visible_to_members = %s::jsonb")
                        values.append(json.dumps([]))
                
                # Handle regular fields
                updates_dict = {
                    'title': title,
                    'content': content,
                    'projectId': projectId,
                    'teamId': teamId,
                    'isPinned': isPinned,
                }
                for key, db_field in field_map.items():
                    value = updates_dict.get(key)
                    if value is not None:
                        # Convert markdown to HTML if content field
                        if key == 'content':
                            value = self._convert_to_html_if_needed(value)
                        set_clauses.append(f"{db_field} = %s")
                        values.append(value)
                
                # Handle tags
                if tags is not None:
                    set_clauses.append("tags = %s::jsonb")
                    values.append(json.dumps(tags))
                
                # Handle metadata
                if metadata is not None:
                    # Try to update metadata column, but handle if it doesn't exist
                    try:
                        set_clauses.append("metadata = %s::jsonb")
                        values.append(json.dumps(metadata))
                    except Exception:
                        # Metadata column might not exist, skip it
                        pass
                
                if len(set_clauses) == 0:
                    return {"error": "No fields to update"}
                
                # Add updated_at
                set_clauses.append("updated_at = NOW()")
                
                # Add docId to values
                values.append(docId)
                
                # Execute UPDATE
                query = f"UPDATE docs SET {', '.join(set_clauses)} WHERE id = %s"
                cursor.execute(query, values)
                conn.commit()
            
            logger.info(f"Document updated: id={docId}")
            return {"success": True}
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error updating document: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
        finally:
            if conn:
                self.pool.putconn(conn)
    
    # ============================================================================
    # Content Conversion Functions
    # ============================================================================
    
    def markdown_to_html(self, markdown: str) -> str:
        """Convert markdown to TipTap-compatible HTML."""
        html = markdown
        
        # Headers
        html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        
        # Bold (must come before italic to avoid conflicts)
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'__(.*?)__', r'<strong>\1</strong>', html)
        
        # Italic (process after bold to avoid conflicts)
        html = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', html)
        html = re.sub(r'(?<!_)_(?!_)([^_]+?)(?<!_)_(?!_)', r'<em>\1</em>', html)
        
        # Links
        html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2" class="text-primary underline">\1</a>', html)
        
        # Tables (process before lists to avoid conflicts)
        # Markdown table format: | col1 | col2 | col3 |
        #                        |------|------|------|
        #                        | val1 | val2 | val3 |
        table_pattern = r'(\|.+\|(?:\n\|[-\s|:]+\|)?(?:\n\|.+\|)+)'
        def convert_table(match):
            table_text = match.group(1)
            lines = [l.strip() for l in table_text.split('\n') if l.strip()]
            if len(lines) < 2:
                return table_text
            
            # First line is header, second is separator, rest are rows
            header_line = lines[0]
            rows = lines[2:] if len(lines) > 2 else []
            
            # Parse header
            headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
            
            # Build HTML table
            table_html = '<table><thead><tr>'
            for header in headers:
                escaped_header = self._escape_html(header)
                table_html += f'<th>{escaped_header}</th>'
            table_html += '</tr></thead>'
            
            if rows:
                table_html += '<tbody>'
                for row_line in rows:
                    cells = [cell.strip() for cell in row_line.split('|')[1:-1]]
                    table_html += '<tr>'
                    for cell in cells:
                        escaped_cell = self._escape_html(cell)
                        table_html += f'<td>{escaped_cell}</td>'
                    table_html += '</tr>'
                table_html += '</tbody>'
            
            table_html += '</table>'
            return table_html
        
        html = re.sub(table_pattern, convert_table, html, flags=re.MULTILINE)
        
        # Lists (simple implementation)
        lines = html.split('\n')
        in_list = False
        list_type = None
        result_lines = []
        
        for line in lines:
            # Skip lines that are already part of HTML tables
            if line.strip().startswith('<table') or line.strip().startswith('</table') or \
               line.strip().startswith('<thead') or line.strip().startswith('</thead') or \
               line.strip().startswith('<tbody') or line.strip().startswith('</tbody') or \
               line.strip().startswith('<tr') or line.strip().startswith('</tr') or \
               line.strip().startswith('<th') or line.strip().startswith('</th') or \
               line.strip().startswith('<td') or line.strip().startswith('</td'):
                if in_list:
                    result_lines.append(f'</{list_type}>')
                    in_list = False
                    list_type = None
                result_lines.append(line)
                continue
            
            # Ordered list
            if re.match(r'^\d+\.\s+(.*)', line):
                if not in_list or list_type != 'ol':
                    if in_list:
                        result_lines.append(f'</{list_type}>')
                    result_lines.append('<ol>')
                    in_list = True
                    list_type = 'ol'
                match = re.match(r'^\d+\.\s+(.*)', line)
                result_lines.append(f'<li>{self._escape_html(match.group(1))}</li>')
            # Unordered list
            elif re.match(r'^[-*]\s+(.*)', line):
                if not in_list or list_type != 'ul':
                    if in_list:
                        result_lines.append(f'</{list_type}>')
                    result_lines.append('<ul>')
                    in_list = True
                    list_type = 'ul'
                match = re.match(r'^[-*]\s+(.*)', line)
                result_lines.append(f'<li>{self._escape_html(match.group(1))}</li>')
            else:
                if in_list:
                    result_lines.append(f'</{list_type}>')
                    in_list = False
                    list_type = None
                if line.strip():
                    # Convert plain text lines to paragraphs
                    escaped_line = self._escape_html(line)
                    escaped_line = escaped_line.replace('\n', '<br>')
                    result_lines.append(f'<p style="white-space: pre-wrap;">{escaped_line}</p>')
        
        if in_list:
            result_lines.append(f'</{list_type}>')
        
        html = '\n'.join(result_lines)
        
        # Code blocks (simple)
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
        
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
                # Try to fetch from database and convert HTML to markdown
                conn = None
                try:
                    conn = self.pool.getconn()
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT content FROM docs WHERE id = %s", (docId,))
                        result = cursor.fetchone()
                        if result:
                            html_content = result.get('content', '')
                            content = self.html_to_markdown(html_content)
                        else:
                            content = ""
                except Exception as e:
                    logger.warning(f"Error fetching document {docId} for markdown file: {str(e)}")
                    content = ""
                finally:
                    if conn:
                        self.pool.putconn(conn)
            
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
        isPinned: bool = False,
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
                isPinned=isPinned,
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
    
    def _convert_to_html_if_needed(self, content: str) -> str:
        """
        Detect if content is markdown and convert to HTML if needed.
        
        Args:
            content: Content string (markdown or HTML)
            
        Returns:
            HTML content
        """
        if not content:
            return content
        
        # Simple heuristic: if content looks like HTML (contains HTML tags), treat as HTML
        # Otherwise, treat as markdown
        if re.search(r'<[a-z][\s\S]*>', content, re.IGNORECASE):
            # Looks like HTML, return as-is
            return content
        else:
            # Looks like markdown, convert to HTML
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
        - isPinned (optional): Filter by pinned status (true/false)
        - visibility (optional): Filter by visibility ('all_members' or 'specific_members')
        - searchTitle (optional): Search for documents with title containing this text (case-insensitive)
        - limit (optional): Maximum number of documents to return (default: 50, max: 200)
        - orderBy (optional): Order results by 'created_at' or 'updated_at' (default: 'created_at')
        - orderDirection (optional): 'asc' or 'desc' (default: 'desc' for newest first)
        
        Returns:
        - Success: List of document dictionaries with fields: id, title, content_preview (first 200 chars), owner_email, project_id, team_id, tags, is_pinned, visibility, created_at, updated_at
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
                    "isPinned": {
                        "type": "boolean",
                        "description": "Filter by pinned status"
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
        isPinned: Optional[bool] = None,
        visibility: Optional[str] = None,
        searchTitle: Optional[str] = None,
        limit: int = 50,
        orderBy: str = "created_at",
        orderDirection: str = "desc",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        List documents from the docs table with optional filtering.
        
        Args:
            projectId: Filter by project ID
            teamId: Filter by team ID
            ownerEmail: Filter by owner email
            tags: Filter by tags (documents containing any of these tags)
            isPinned: Filter by pinned status
            visibility: Filter by visibility
            searchTitle: Search title text (case-insensitive)
            limit: Maximum number of documents (default: 50, max: 200)
            orderBy: Order by field ('created_at' or 'updated_at')
            orderDirection: Order direction ('asc' or 'desc')
            
        Returns:
            List of document dictionaries, or error dictionary
        """
        conn = None
        try:
            # Validate limit
            if limit < 1:
                limit = 50
            if limit > 200:
                limit = 200
            
            # Validate orderBy
            valid_order_by = ['created_at', 'updated_at']
            if orderBy not in valid_order_by:
                orderBy = 'created_at'
            
            # Validate orderDirection
            valid_directions = ['asc', 'desc']
            if orderDirection not in valid_directions:
                orderDirection = 'desc'
            
            conn = self.pool.getconn()
            with conn.cursor() as cursor:
                # Build query - include content preview (first 200 characters), not full content
                # Use LEFT() function to get first 200 characters of content
                query_parts = ["SELECT id, title, LEFT(content, 200) as content_preview, owner_email, project_id, team_id, tags, is_pinned, visibility, visible_to_members, created_at, updated_at FROM docs WHERE 1=1"]
                params = []
                
                # Add filters
                if projectId:
                    query_parts.append("AND project_id = %s")
                    params.append(projectId)
                
                if teamId:
                    query_parts.append("AND team_id = %s")
                    params.append(teamId)
                
                if ownerEmail:
                    query_parts.append("AND owner_email = %s")
                    params.append(ownerEmail.lower())
                
                if isPinned is not None:
                    query_parts.append("AND is_pinned = %s")
                    params.append(isPinned)
                
                if visibility:
                    query_parts.append("AND visibility = %s")
                    params.append(visibility)
                
                if searchTitle:
                    query_parts.append("AND LOWER(title) LIKE %s")
                    params.append(f"%{searchTitle.lower()}%")
                
                if tags:
                    # Filter documents that contain any of the specified tags
                    # Use JSONB containment operator @>
                    tag_conditions = []
                    for tag in tags:
                        tag_conditions.append("tags @> %s::jsonb")
                        params.append(json.dumps([tag]))
                    if tag_conditions:
                        query_parts.append(f"AND ({' OR '.join(tag_conditions)})")
                
                # Add ordering
                query_parts.append(f"ORDER BY {orderBy} {orderDirection.upper()}")
                
                # Add limit
                query_parts.append("LIMIT %s")
                params.append(limit)
                
                query = " ".join(query_parts)
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to list of dicts
                docs = []
                for row in rows:
                    doc = dict(row)
                    
                    # Clean up content preview - strip HTML tags and truncate if needed
                    if 'content_preview' in doc and doc['content_preview']:
                        content_preview = doc['content_preview']
                        # Strip HTML tags for a cleaner preview
                        content_preview = re.sub(r'<[^>]+>', '', content_preview)
                        # Remove extra whitespace
                        content_preview = ' '.join(content_preview.split())
                        # Truncate to 200 characters if longer
                        if len(content_preview) > 200:
                            content_preview = content_preview[:200] + '...'
                        doc['content_preview'] = content_preview
                    else:
                        doc['content_preview'] = ''
                    
                    # Parse JSONB fields if they're strings
                    if 'tags' in doc:
                        tags_value = doc['tags']
                        if isinstance(tags_value, str):
                            try:
                                tags_value = json.loads(tags_value)
                            except:
                                tags_value = []
                        elif not isinstance(tags_value, list):
                            tags_value = []
                        doc['tags'] = tags_value
                    
                    if 'visible_to_members' in doc:
                        visible_to_members = doc['visible_to_members']
                        if isinstance(visible_to_members, str):
                            try:
                                visible_to_members = json.loads(visible_to_members)
                            except:
                                visible_to_members = []
                        elif not isinstance(visible_to_members, list):
                            visible_to_members = []
                        doc['visible_to_members'] = visible_to_members
                    
                    docs.append(doc)
                
                logger.info(f"Listed {len(docs)} documents with filters: projectId={projectId}, teamId={teamId}, ownerEmail={ownerEmail}, tags={tags}, isPinned={isPinned}, searchTitle={searchTitle}")
                return docs
                
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
        finally:
            if conn:
                self.pool.putconn(conn)

