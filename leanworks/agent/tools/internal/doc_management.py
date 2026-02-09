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

from leanworks.agent.tools.base_api_client import BaseAPIClient

# Default AI agent ID for attribution when user_id is not provided
AI_AGENT_ID = "leanworks-ai-agent"

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
        config: Optional[Dict[str, Any]] = None,
        memory_manager=None,  # NEW: For working context registration
        working_context=None,  # NEW: For accessing cited documents
        tool_use_ref=None  # Reference to ToolUse instance for workspace access
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
            working_context: Optional WorkingContext instance for accessing cited documents
            tool_use_ref: Optional reference to ToolUse instance for workspace directory access
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

        # Store memory manager and working context reference
        self.memory_manager = memory_manager
        # Use provided working_context, or fall back to memory_manager's working_context
        self.working_context = working_context or (memory_manager.working_context if memory_manager else None)
        
        # Reference to ToolUse instance for workspace access (used by get_doc file downloads)
        self._tool_use_ref = tool_use_ref

    # ============================================================================
    # Working Context Query Tool
    # ============================================================================

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
        - content (optional): Document content in HTML format. Required if file_path not provided.
        - file_path (optional): Path to HTML file to read content from. Required if content not provided.
        - projectId (optional): Associated project ID
        - tags (optional): Array of tag strings
        - visibility (optional): 'all_members' or 'specific_members' (default: 'all_members')
        - visibleToMembers (optional): Array of email addresses
        - metadata (optional): JSON object for additional metadata

        Content Format:
        - Input: HTML format (from content parameter or file_path)
        - Output: HTML format (returned to agent)
        - Storage: Converted to TipTap JSON format internally

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
                        "description": "Document content in HTML format. Required if file_path not provided. Will be converted to TipTap JSON format for storage."
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to HTML file to read content from. Required if content not provided."
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Associated project ID"
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
                "required": ["title"]
            }
        }
    
    def create_doc(
        self,
        title: str,
        content: Optional[str] = None,
        file_path: Optional[str] = None,
        projectId: Optional[str] = None,
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
            content: Document content in HTML format (required if file_path not provided)
            file_path: Path to HTML file to read content from (alternative to content parameter)
            projectId: Associated project ID
            tags: Array of tag strings
            visibility: Document visibility
            visibleToMembers: Array of email addresses
            metadata: JSON object for additional metadata

        Returns:
            Dictionary with doc id and created fields, or error dictionary
        """
        try:
            # Handle file_path parameter - read content from file if provided
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    logger.debug(f"Read content from file: {file_path}")
                except Exception as e:
                    return {"error": f"Failed to read file {file_path}: {str(e)}"}
            elif not content:
                return {"error": "Either content or file_path must be provided"}

            if not title:
                return {"error": "title is required"}

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
                "tags": tags or [],
                "visibility": doc_visibility,
                "visibleToMembers": visible_to_members_array
            }
            
            if metadata:
                request_body["metadata"] = metadata
            
            # Call API to create document
            result = self._make_request('POST', '/api/docs', json=request_body)
            
            logger.debug(f"Document created via API: id={result.get('id')}, title={title}")
            
            # Return HTML content to agent (not TipTap JSON)
            return {
                "id": result.get('id'),
                "title": title,
                "content": html_content,  # Return HTML, not TipTap JSON
                "ownerEmail": result.get('ownerEmail'),
                "projectId": projectId,
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
        - file_path (optional): Path to HTML file to read content from (alternative to content parameter)
        - projectId (optional): Update project association
        - tags (optional): Update tags array
        - visibility (optional): Update visibility
        - visibleToMembers (optional): Update visible members
        - metadata (optional): Update metadata

        Content Format:
        - Input: HTML format (or HTML file via file_path)
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
                    "file_path": {
                        "type": "string",
                        "description": "Path to HTML file to read content from (alternative to content parameter)"
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Update project association"
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

        This is the single tool for loading any document. Behavior depends on document type:

        IMPORTANT: This tool can only be called after get_create_doc_instruction, get_understand_doc_instruction, or get_update_doc_instruction tools.

        **RICH TEXT DOCUMENTS** (docType='rich_text'):
        - Returns full HTML content directly in the `content` field.
        - For large documents, content may be saved to a file with a path returned instead.

        **UPLOADED FILES** (docType='xlsx', 'csv', 'pdf', 'pptx', 'docx'):
        - The full binary file is **automatically downloaded** to /workspace/.
        - Response includes `file_path` (local path to the downloaded file), `original_name`, `file_size`, and `mime_type`.
        - The `content` field is removed (it contained unusable plain-text extraction).
        - Use bash/python tools to process the downloaded file:
          - **Excel (.xlsx, .xls)**: `pandas.read_excel(file_path)` or openpyxl
          - **CSV (.csv)**: `pandas.read_csv(file_path)` or Python csv module
          - **PDF (.pdf)**: pdfplumber, PyPDF2, or pdftotext
          - **PowerPoint (.pptx)**: python-pptx for slide manipulation
          - **Word (.docx)**: python-docx for text extraction

        Parameters:
        - docIds (required): Array of document IDs to retrieve. Can be a single document ID or multiple IDs.

        Returns:
        - For rich_text: List of document dicts with `content` (HTML), id, title, owner_email, project_id, tags, etc.
        - For uploaded files: List of document dicts with `file_path`, `original_name`, `file_size`, `mime_type`, plus metadata fields.
        - Error: Dictionary with error message.

        Example Use Cases:
        - Read a rich text document's full content for editing
        - Load an Excel file for data analysis (file auto-downloaded, use pandas to read)
        - Get multiple documents at once (mixed types supported)
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

    @property
    def summarize_doc_property(self):
        """Property definition for summarize_doc tool."""
        return {
            "type": "custom",
            "name": "summarize_doc",
            "description": f"""
Summarize a large document by processing it chunk-by-chunk for org `{self.org_slug}`.

WHEN TO USE: Use this tool when get_doc returns a large document (file path instead of content)
and you need to create a summary of the entire document.

WHAT IT DOES:
- Splits large document into manageable chunks
- Returns chunks for you to process and synthesize into a complete summary

PARAMETERS:
- docId (required): The document ID to summarize
- file_path (required): The file path returned by get_doc for large documents

NOTE: For small documents (content returned directly), summarize the content directly.
For large documents (file path returned), use this tool.
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "docId": {
                        "type": "string",
                        "description": "Document ID to summarize"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "File path returned by get_doc (for large documents)"
                    }
                },
                "required": ["docId", "file_path"]
            }
        }

    def _get_workspace_paths(self):
        """
        Get the host-side workspace directory and the container-visible path.
        
        Returns:
            tuple: (host_dir, container_dir) or (None, None) if unavailable
        """
        # Try to get workspace from ToolUse reference (Docker/K8s session)
        if self._tool_use_ref:
            session = getattr(self._tool_use_ref, '_bash_session', None)
            if session:
                return session.session_temp_dir, session.workspace_path
        
        # Fallback: use /workspace if it exists (running inside container)
        if os.path.exists('/workspace'):
            return '/workspace', '/workspace'
        
        # Last resort: use a temp directory
        import tempfile
        fallback_dir = os.path.join(tempfile.gettempdir(), 'leanworks_workspace')
        os.makedirs(fallback_dir, exist_ok=True)
        logger.warning(f"No workspace session available, using fallback: {fallback_dir}")
        return fallback_dir, fallback_dir

    def _download_file_to_workspace(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download an uploaded document's full file to the workspace directory.
        
        Args:
            doc: Document metadata dict (must have 'id' and 'storagePath')
            
        Returns:
            Dict with file_path, original_name, size, mime_type on success,
            or dict with 'error' key on failure
        """
        doc_id = doc.get('id')
        storage_path = doc.get('storagePath')
        
        if not storage_path:
            return {"error": f"No storage path found for document {doc_id}"}
        
        # Get original filename from fileMetadata
        file_metadata = doc.get('fileMetadata', {}) or {}
        original_name = file_metadata.get('originalName', f"document_{doc_id}")
        mime_type = doc.get('mimeType', 'application/octet-stream')
        
        try:
            # Download file content from leanworks-hub API
            response = self._make_request(
                'GET',
                f'/api/docs/{doc_id}/content',
                raw=True  # Request raw response, not JSON
            )
            
            if isinstance(response, dict) and 'error' in response:
                return response
            
            # Get workspace directory
            host_dir, container_dir = self._get_workspace_paths()
            
            # Sanitize filename to avoid path traversal
            safe_filename = os.path.basename(original_name)
            host_path = os.path.join(host_dir, safe_filename)
            container_path = os.path.join(container_dir, safe_filename)
            
            # Write file content
            with open(host_path, 'wb') as f:
                f.write(response)
            
            file_size = len(response)
            logger.info(f"Downloaded file for doc {doc_id} to {container_path} ({file_size} bytes)")
            
            return {
                "file_path": container_path,
                "original_name": original_name,
                "size": file_size,
                "mime_type": mime_type,
            }
        except Exception as e:
            logger.error(f"Error downloading file for doc {doc_id}: {str(e)}")
            return {"error": f"Failed to download file: {str(e)}"}

    def get_doc(
        self,
        docIds: List[str],
        format: str = 'html',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get one or more documents by their IDs via API.
        
        For rich_text documents: returns full HTML content directly.
        For uploaded files (xlsx, csv, pdf, pptx, docx): automatically downloads the full
        binary file to /workspace/ and returns metadata with the file path.

        Args:
            docIds: List of document IDs to retrieve
            format: Content format ('html' or 'json', default: 'html') — only applies to rich_text docs

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
                    doc = self._make_request('GET', f'/api/docs/{doc_id}', params={'format': format})
                    
                    if doc:
                        # Normalize field names (API returns camelCase, but we want consistency)
                        if 'ownerEmail' in doc:
                            doc['owner_email'] = doc.pop('ownerEmail')
                        if 'projectId' in doc:
                            doc['project_id'] = doc.pop('projectId')
                        if 'visibleToMembers' in doc:
                            doc['visible_to_members'] = doc.pop('visibleToMembers')
                        if 'createdAt' in doc:
                            doc['created_at'] = doc.pop('createdAt')
                        if 'updatedAt' in doc:
                            doc['updated_at'] = doc.pop('updatedAt')
                        
                        doc_type = doc.get('docType', 'rich_text')
                        
                        if doc_type == 'rich_text':
                            # Rich text: convert TipTap JSON to HTML (existing behavior)
                            if 'content' in doc and doc['content']:
                                content = doc['content']
                                if format == 'html':
                                    doc['content'] = self._convert_content_to_html(content)
                                # If format == 'json', keep raw content as-is
                        else:
                            # Uploaded file (xlsx, csv, pdf, pptx, docx):
                            # Automatically download the full binary file to /workspace/
                            file_result = self._download_file_to_workspace(doc)
                            
                            if 'error' in file_result:
                                # Download failed — keep the doc metadata but note the error
                                doc['file_download_error'] = file_result['error']
                                logger.warning(f"Failed to download file for doc {doc_id}: {file_result['error']}")
                            else:
                                # Rebuild doc with only essential fields, dropping bulky
                                # content, fileMetadata.previewData, storagePath, etc.
                                # This keeps the response small so file_path is directly
                                # visible to the agent without large response indirection.
                                doc = {
                                    'id': doc.get('id'),
                                    'title': doc.get('title'),
                                    'docType': doc_type,
                                    'file_path': file_result['file_path'],
                                    'original_name': file_result['original_name'],
                                    'file_size': file_result['size'],
                                    'mime_type': file_result.get('mime_type', 'application/octet-stream'),
                                    'owner_email': doc.get('owner_email'),
                                    'project_id': doc.get('project_id'),
                                    'created_at': doc.get('created_at'),
                                    'updated_at': doc.get('updated_at'),
                                    'file_info': (
                                        f"Full file downloaded to {file_result['file_path']} "
                                        f"({file_result['size']} bytes). "
                                        f"Use bash/python tools to process it."
                                    ),
                                }
                        
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
            
            logger.debug(f"Retrieved {len(docs)} documents out of {len(docIds)} requested")
            return docs
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
    
    # get_doc_full_file has been removed — file downloads are now handled
    # automatically by get_doc for non-rich_text documents.

    @property
    def upload_doc_property(self):
        """Property definition for upload_doc tool."""
        return {
            "type": "custom",
            "name": "upload_doc",
            "description": f"""
Upload a file to docs. Supported types: Excel (.xlsx, .xls), CSV, PDF, PowerPoint (.pptx, .ppt), Word (.docx, .doc).

The document is stored and processed asynchronously. After upload, use get_doc to access content once processing completes.

PARAMETERS:
- file_path (required): Path to the file (use /workspace/filename.ext for files created in bash session)
- title (optional): Document title; if omitted, the filename is used
- projectId (optional): Associate with a project ID

RETURNS:
- id: Document ID
- title: Document title
- docType: Detected type (e.g. xlsx, csv, pdf, pptx)
- processingStatus: Initial status (e.g. uploading); processing continues asynchronously
- createdAt: Creation timestamp
- message: Success message
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to file (e.g. /workspace/filename.xlsx for files created in bash). Resolved to session workspace when under /workspace/."
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional document title; defaults to filename"
                    },
                    "projectId": {
                        "type": "string",
                        "description": "Optional project ID to associate with the document"
                    }
                },
                "required": ["file_path"]
            }
        }

    # Extensions allowed by leanworks-hub POST /api/docs/upload (file-validation)
    _UPLOAD_ALLOWED_EXTENSIONS = frozenset({".pdf", ".xlsx", ".xls", ".csv", ".pptx", ".ppt", ".docx", ".doc"})

    def upload_doc(
        self,
        file_path: str,
        title: Optional[str] = None,
        projectId: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Upload a file (Excel, CSV, PDF, PPTX, or DOCX) to docs via leanworks-hub.

        Args:
            file_path: Local path to the file to upload
            title: Optional document title; defaults to filename
            projectId: Optional project ID

        Returns:
            Dict with id, title, docType, processingStatus, createdAt, message; or error dict
        """
        try:
            if not file_path or not file_path.strip():
                return {"error": "file_path is required"}
            file_path = file_path.strip()

            # Resolve /workspace/ paths (container) to host session directory
            if file_path.startswith("/workspace/") or file_path == "/workspace":
                base = os.path.normpath(self._get_workspace_dir())
                rel = file_path[len("/workspace"):].lstrip("/")
                resolved = os.path.normpath(os.path.join(base, rel)) if rel else base
                try:
                    if os.path.commonpath([resolved, base]) != base:
                        return {"error": "Invalid path: outside workspace"}
                except ValueError:
                    return {"error": "Invalid path"}
                file_path = resolved

            if not os.path.isfile(file_path):
                return {"error": f"File not found: {file_path}"}
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self._UPLOAD_ALLOWED_EXTENSIONS:
                return {
                    "error": (
                        f"Unsupported file type '{ext}'. "
                        f"Allowed: .pdf, .xlsx, .xls, .csv, .pptx, .ppt, .docx, .doc"
                    )
                }
            extra_data = {}
            if title is not None and str(title).strip():
                extra_data["title"] = str(title).strip()
            if projectId is not None and str(projectId).strip():
                extra_data["projectId"] = str(projectId).strip()

            result = self._make_upload_request(
                "/api/docs/upload",
                file_path,
                extra_data=extra_data if extra_data else None,
            )
            if result is None:
                return {"error": "Upload succeeded but no response body returned"}
            return {
                "id": result.get("id"),
                "title": result.get("title"),
                "docType": result.get("docType"),
                "processingStatus": result.get("processingStatus"),
                "createdAt": result.get("createdAt"),
                "message": result.get("message", "Document uploaded successfully. Processing will begin shortly."),
            }
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            error_msg = str(e).split("\n")[0] if "\n" in str(e) else str(e)
            return {"error": error_msg}

    def summarize_doc(
        self,
        docId: str,
        file_path: str,
        **kwargs
    ) -> str:
        """
        Summarize a large document by returning chunks for processing.

        Args:
            docId: Document ID
            file_path: Path to document file (from get_doc response)

        Returns:
            Formatted chunks for LLM to summarize
        """
        try:
            logger.info(f"Summarizing large document: {docId} from {file_path}")

            # Read file content
            if not os.path.exists(file_path):
                return f"Error: File not found at {file_path}"

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split into chunks (smart paragraph boundaries)
            chunks = self._split_content_into_chunks(content, chunk_size=4000)

            # Format chunks for LLM processing
            formatted_chunks = []
            for i, chunk in enumerate(chunks, 1):
                formatted_chunks.append(f"""
=== CHUNK {i} OF {len(chunks)} ===
{chunk}
====================
""")

            result = f"""Document split into {len(chunks)} chunks for summarization.

{chr(10).join(formatted_chunks)}

INSTRUCTIONS FOR SUMMARIZATION:
1. Read and summarize each chunk above
2. Combine the chunk summaries into a coherent overall summary
3. Identify key themes, main points, and important details
4. Structure: Overview + Key Sections + Conclusion

Document ID: {docId}
"""

            logger.info(f"Returning {len(chunks)} chunks for document {docId}")
            return result

        except Exception as e:
            logger.error(f"Error summarizing document {docId}: {e}")
            return f"Error: {str(e)}"

    def _split_content_into_chunks(self, content: str, chunk_size: int) -> List[str]:
        """
        Split content into chunks with smart paragraph boundaries.

        Args:
            content: Text content to split
            chunk_size: Target size per chunk in characters

        Returns:
            List of text chunks
        """
        # Try to split on paragraph boundaries
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            # If adding this paragraph exceeds chunk size and we have content, start new chunk
            if current_size + para_size > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Add remaining content
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks if chunks else [content]  # Fallback to full content if no splits




    def update_doc(
        self,
        docId: str,
        title: Optional[str] = None,
            content: Optional[str] = None,
            file_path: Optional[str] = None,
            projectId: Optional[str] = None,
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
            content: Update content in HTML format (required if file_path not provided)
            file_path: Path to HTML file to read content from (alternative to content parameter)
            projectId: Update project association
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

            # Handle file_path parameter - read content from file if provided
            if file_path:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    logger.debug(f"Read content from file: {file_path}")
                except Exception as e:
                    return {"error": f"Failed to read file {file_path}: {str(e)}"}
            elif not content and not any([title, projectId, tags, visibility, visibleToMembers, metadata]):
                return {"error": "Either content, file_path, or other update fields must be provided"}

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
            
            logger.debug(f"Document updated via API: id={docId}")
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
        Web app sends both ProseMirror positions (from, to) and HTML positions
        (htmlFrom, htmlTo). Use this tool to extract the exact text at those HTML
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
                    logger.debug(f"Using cached temp file for HTML extraction: {cached_path}")
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
    
    @property
    def list_docs_property(self):
        description = f"""
        List documents from the docs table for org `{self.org_slug}`.
        
        This tool queries the docs table to retrieve documents with optional filtering.
        Use this to find documents by project, owner, tags, or other criteria.
        
        Parameters:
        - projectId (optional): Filter documents by project ID
        - ownerEmail (optional): Filter documents by owner email
        - tags (optional): Filter documents that contain any of these tags (array of strings)
        - visibility (optional): Filter by visibility ('all_members' or 'specific_members')
        - searchTitle (optional): Search for documents with title containing this text (case-insensitive)
        - limit (optional): Maximum number of documents to return (default: 50, max: 200)
        - orderBy (optional): Order results by 'created_at' or 'updated_at' (default: 'created_at')
        - orderDirection (optional): 'asc' or 'desc' (default: 'desc' for newest first)
        
        Returns:
        - Success: List of document dictionaries with fields: id, title, content_preview (first 200 chars), owner_email, project_id, tags, visibility, created_at, updated_at
        - Note: Full content is not included - only a preview. Use get_doc to get full content if needed.
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
        Client-side filtering for projectId, ownerEmail, tags, and searchTitle
        is applied after fetching from the API.
        
        Args:
            projectId: Filter by project ID
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
                doc_owner_email = doc.get('ownerEmail') or doc.get('owner_email', '').lower()
                doc_tags = doc.get('tags', [])
                doc_visibility = doc.get('visibility')
                doc_title = doc.get('title', '').lower()
                
                # Apply filters
                if projectId and doc_project_id != projectId:
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
            
            logger.debug(f"Listed {len(filtered_docs)} documents with filters: projectId={projectId}, ownerEmail={ownerEmail}, tags={tags}, searchTitle={searchTitle}")
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
    def get_create_doc_instruction_property(self):
        """Property definition for get_create_doc_instruction tool."""
        return {
            "type": "custom",
            "name": "get_create_doc_instruction",
            "description": f"""
WHEN TO USE: When we need to create a new document.

WHAT IT RETURNS:
Detailed workflow instructions for creating a new document.
""",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    
    @property
    def get_update_doc_instruction_property(self):
        """Property definition for get_update_doc_instruction tool."""
        return {
            "type": "custom",
            "name": "get_update_doc_instruction",
            "description": f"""

WHEN TO USE: When we need to update, edit, or modify an existing document (especially with selected text).

WHAT IT RETURNS:
Detailed workflow instructions for updating/modifying an existing document.
""",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

    @property
    def get_understand_doc_instruction_property(self):
        """Property definition for get_understand_doc_instruction tool."""
        return {
            "type": "custom",
            "name": "get_understand_doc_instruction",
            "description": f"""
WHEN TO USE: When we need to read, view, understand, analyze, or review an existing document WITHOUT making changes.

WHAT IT RETURNS:
Detailed workflow instructions for reading and understanding document content (both small and large documents).
""",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
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
                    "next_section_heading": {"type": "string", "description": "Heading of next section (optional)"},
                    "section_number": {
                        "type": "integer",
                        "description": "Current section number (1-indexed, e.g., 2 for second section)"
                    },
                    "total_sections": {
                        "type": "integer",
                        "description": "Total number of sections in document"
                    }
                },
                "required": ["section_info", "previous_content"]
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
    
    def get_create_doc_instruction(
        self
    ) -> Dict[str, Any]:
        """
        Get instructions for creating a new document using TOC-first workflow.
        
        This method orchestrates the complete document creation process:
        1. Generate TOC with document contract
        2. Draft sections iteratively with context sandwiches
        3. Run quality passes
        4. Create final document
        
        Args:
            title: Document title
            requirements: User requirements/description for the document
            
        Returns:
            Dictionary with document creation instructions
        """
        try:
            logger.debug("Starting TOC-first document creation")
            
            # This is a placeholder that will return instructions for the agent
            # The actual workflow will be driven by agent interactions
            return """Document creation workflow initiated.

UPLOADED FILES (Excel, CSV, PDF, PowerPoint, Word):
- If the user wants to add a file (e.g. .xlsx, .csv, .pdf, .pptx, .docx) to docs, use upload_doc(file_path=...) instead of this workflow.
- create_doc is for rich-text/HTML documents; upload_doc is for binary files. After upload_doc, the document is processed asynchronously; use get_doc to access content once ready.

TO CREATE file formats from scratch (before upload_doc):
- Excel (.xlsx): Use pandas (DataFrame.to_excel) or openpyxl for detailed formatting, or xlsxwriter
- CSV: Use pandas (DataFrame.to_csv) or Python csv module, or bash commands
- PDF: Use reportlab to generate PDFs from scratch
- PowerPoint (.pptx): Use python-pptx to create presentations (slides, shapes, text, images)
- Word (.docx): Use python-docx to create Word documents

When the user asked for the file, after creating it in /workspace/, call upload_doc(file_path='/workspace/filename.ext', title='...') so it appears in the user's doc list. Do not upload intermediate or temp files.

RICH-TEXT DOCUMENT WORKFLOW:
1. Generate Table of Contents (use generate_toc tool)
   - Include Document Contract (purpose, audience, scope, non-goals, evidence rule)
   - Major sections (H1) with subsections (H2, optionally H3)
   - Max 3 heading levels
2. Show TOC for confirmation before drafting
3. Draft sections iteratively with context awareness (bridge-in, main content, bridge-out)
4. Run quality validation (use run_quality_passes tool)
5. Create final document (use create_doc with file_path parameter)

STANDARDS:
- Evidence: Never invent facts. Use TODO/ASSUMPTION tags. Cite sources.
- Quality: Max 3 heading levels, consistent terminology, bridge sentences, valid references

For tool syntax and file operations, see <core_tools_reference> and <workspace_reference> in system prompt.

Please proceed with generating the TOC based on these requirements."""
            
        except Exception as e:
            logger.error(f"Error in get_create_doc_instruction: {str(e)}")
            return f"Error getting create instructions: {str(e)}"
    
    def get_update_doc_instruction(
        self
    ) -> str:
        """
        Get instructions for updating an existing document using intelligent workflow.

        Returns:
            Instructions for document update workflow
        """
        try:
            logger.debug("Starting document update instructions")

            return """Document update workflow initiated.

UPLOADED FILES (docType xlsx, csv, pdf, pptx, docx):
- update_doc is for rich_text documents only. It cannot replace or edit the binary content of an uploaded file.
- If get_doc returns a document with docType other than rich_text (e.g. xlsx, csv, pdf, pptx, docx): do not use update_doc to change its content. get_doc automatically downloads the full file to /workspace/. Use bash/python tools to process it. To "replace" the file, the user would need to upload a new document via upload_doc (creating a new doc).

TO MODIFY uploaded files:
1. Use get_doc(docIds=[docId]) — for uploaded files, the full file is automatically downloaded to /workspace/
2. Modify using appropriate tools:
   - Excel (.xlsx): pandas (read_excel, to_excel) or openpyxl for cell-level edits
   - CSV: pandas (read_csv, to_csv), Python csv module, or bash (awk, sed)
   - PDF: Extract text with pdfplumber/PyPDF2; for layout changes, may need to recreate with reportlab
   - PowerPoint (.pptx): python-pptx to read/modify slides, shapes, text
   - Word (.docx): python-docx to read/modify paragraphs, tables, styles
3. Save changes to the same or new file path
4. When the user asked for the file, after creating it in /workspace/, call upload_doc(file_path='/workspace/filename.ext', title='...') so it appears in the user's doc list. Do not upload intermediate or temp files.

STEP 1: Load Document
- Call get_doc([docId]) to fetch content and check docType

STEP 2: Choose Workflow Based on Response

WORKFLOW A: RICH_TEXT, SMALL (content returned directly)
- Edit content string in memory
- Apply all requested changes
- Preserve HTML structure
- Save: update_doc(docId, content=modified_html)

WORKFLOW B: RICH_TEXT, LARGE (file path returned)
- Locate content to edit:
  * EXACT TEXT: Use grep to find line numbers (see <core_tools_reference>)
  * SECTION DESCRIBED: Use search_tool_response_in_vectorstore, then grep
- Edit file using text_editor (see <core_tools_reference>)
- For multiple edits, repeat for each change
- Save: update_doc(docId, file_path="/workspace/doc.html")

KEY DIFFERENCE: Small docs use content parameter, large docs use file_path parameter. Uploaded files (xlsx, csv, pdf, etc.) are not editable via update_doc.

Please proceed with this workflow."""

        except Exception as e:
            logger.error(f"Error in get_update_doc_instruction: {str(e)}")
            return f"Error getting update instructions: {str(e)}"

    def get_understand_doc_instruction(self) -> str:
        """
        Get instructions for reading/understanding an existing document.
        Returns instructions with two distinct tracks: Summarization and Q&A.
        """
        try:
            logger.debug("Starting document understand instructions")

            # Check for cited documents in working context
            cited_docs = self.get_cited_documents()

            if cited_docs:
                # Build instructions with specific cited documents
                doc_list = []
                for doc in cited_docs:
                    doc_id = doc.get('doc_id')
                    title = doc.get('title', 'Unknown')
                    if doc.get('has_selected_text'):
                        doc_list.append(f"- {doc_id} ({title}) - HAS SELECTED TEXT")
                    else:
                        doc_list.append(f"- {doc_id} ({title})")

                cited_docs_text = "\n".join(doc_list)

                return f"""Document understanding workflow initiated.

AVAILABLE CITED DOCUMENTS:
{cited_docs_text}

Choose your workflow based on the user's request:

═══════════════════════════════════════════════════════════════
TRACK 1: SUMMARIZATION WORKFLOW
═══════════════════════════════════════════════════════════════
Use when user wants: summary, overview, key points, what the document is about

STEP 1: Load Document
- Call get_doc([{cited_docs[0]['doc_id'] if cited_docs else 'docId'}])

STEP 2: Check Response Type

FOR SMALL DOCUMENTS (content returned directly):
- Content is available in full
- Summarize the content directly
- Identify key points and themes

FOR LARGE DOCUMENTS (file path returned):
- Call summarize_doc(docId='...', file_path='...')
- Process the chunks returned by the tool
- Synthesize chunk summaries into complete overview
- Focus on main themes and key points

═══════════════════════════════════════════════════════════════
TRACK 2: Q&A / SPECIFIC QUESTIONS WORKFLOW
═══════════════════════════════════════════════════════════════
Use when user asks: specific questions, "what does it say about X", find information

STEP 1: Load Document
- Call get_doc([{cited_docs[0]['doc_id'] if cited_docs else 'docId'}])

STEP 2: Check Response Type

FOR SMALL DOCUMENTS (content returned directly):
- Content is available in full
- Answer question directly from content

FOR LARGE DOCUMENTS (file path + RAG info returned):
- Document is being indexed for semantic search
- Wait for "RAG indexing complete" notification
- Use search_tool_response_in_vectorstore(query='...', document_id='...')
- Answer based on relevant sections retrieved

FALLBACK (if RAG unavailable):
- Use grep to search file: grep -i "keyword" file_path
- Use text_editor to read specific sections

═══════════════════════════════════════════════════════════════

Please proceed with the appropriate workflow for the user's request."""
            else:
                # No cited documents - provide generic instructions
                return """Document understanding workflow initiated.

Choose your workflow based on the user's request:

═══════════════════════════════════════════════════════════════
TRACK 1: SUMMARIZATION WORKFLOW
═══════════════════════════════════════════════════════════════
Use when user wants: summary, overview, key points

STEP 1: Load Document
- Call get_doc([docId])

STEP 2: Summarize Based on Size
- SMALL DOC (content returned): Summarize directly
- LARGE DOC (file path returned): Call summarize_doc(docId='...', file_path='...')

═══════════════════════════════════════════════════════════════
TRACK 2: Q&A / SPECIFIC QUESTIONS WORKFLOW
═══════════════════════════════════════════════════════════════
Use when user asks: specific questions about document content

STEP 1: Load Document
- Call get_doc([docId])

STEP 2: Answer Based on Size
- SMALL DOC (content returned): Answer directly
- LARGE DOC (file path returned): Wait for RAG, then use search_tool_response_in_vectorstore

═══════════════════════════════════════════════════════════════

Please proceed with the appropriate workflow."""

        except Exception as e:
            logger.error(f"Error in get_understand_doc_instruction: {str(e)}")
            return f"Error getting understand instructions: {str(e)}"

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
        next_section_heading: Optional[str] = None,
        section_number: Optional[int] = None,  # NEW
        total_sections: Optional[int] = None   # NEW
    ) -> Dict[str, Any]:
        """
        Prepare context sandwich for drafting a section.

        Args:
            section_info: Section information (heading, description, level)
            previous_content: Previously drafted content
            next_section_heading: Heading of next section (if any)
            section_number: Current section number (1-indexed, e.g., 2 for second section)
            total_sections: Total number of sections in document

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
        
        result = {
            "section_info": section_info,
            "context_above": context_above,
            "next_section_heading": next_section_heading,
            "drafting_prompt": ''.join(prompt_parts)
        }

        # Add progress metadata for streaming display
        if section_number and total_sections:
            result["_progress"] = {
                "current": section_number,
                "total": total_sections,
                "heading": section_info.get('heading', 'Untitled'),
                "stage": "preparing"
            }

        return result
    
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
            output_file: Path to output file for iterative writing (will be converted to /workspace/ if not already)
            
        Returns:
            Instructions and section list for agent
        """
        sections = self.get_section_list_from_toc(toc)

        # Ensure output_file is in Docker workspace
        if not output_file.startswith('/workspace/'):
            import os
            filename = os.path.basename(output_file)
            if not filename:
                filename = f"working_doc_{hash(str(toc)) % 10000}.html"
            output_file = f"/workspace/{filename}"
            logger.info(f"Converted output_file to Docker workspace path: {output_file}")

        return {
            "workflow": "iterative_drafting",
            "output_file": output_file,
            "total_sections": len(sections),
            "sections": sections,
            "instructions": f"""Draft each section iteratively with progress tracking:

For each section i (from 0 to {len(sections)-1}):
1. Call prepare_section_context() with progress info:
   prepare_section_context(
       section_info=sections[i],
       previous_content=<read from {output_file}>,
       next_section_heading=sections[i+1].heading if i+1 < len(sections) else None,
       section_number=i+1,           # IMPORTANT: pass current position
       total_sections={len(sections)} # IMPORTANT: pass total count
   )

2. Draft the section content with:
   - Bridge-in (1-3 sentences from previous section)
   - Main content (follow outline and description)
   - Bridge-out (1-3 sentences to next section)

3. Append to document using bash tool:
   bash(command=f'echo "{{{{section_content}}}}" >> "{output_file}"')

4. Move to next section

After all {len(sections)} sections are drafted:
- Run quality passes if needed
- Create final document: create_doc(title=..., file_path="{output_file}")
"""
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
    
    
    
    # ============================================================================
    # Internal Methods (used by consolidated workflows)
    # ============================================================================
    
    
    
    
    
    # ============================================================================
    # Broad Update Workflow
    # ============================================================================
    
    
    
    # ============================================================================
    # RAG Fallback for Large Documents
    # ============================================================================
    
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
    
    # ============================================================================
    # Workspace File Management
    # ============================================================================

    def _get_workspace_dir(self) -> str:
        """Get workspace directory (Docker-accessible temp dir)."""
        # Try to get from bash tool session
        if self.bash_tool:
            bash_instance = getattr(self.bash_tool, '__self__', None)
            if bash_instance and hasattr(bash_instance, '_bash_session'):
                session = bash_instance._bash_session
                if session and hasattr(session, 'session_temp_dir'):
                    return session.session_temp_dir

        # Fallback: use system temp (same pattern as conversation manager)
        import tempfile
        session_id = getattr(self, 'session_id', 'default')
        workspace_dir = os.path.join(tempfile.gettempdir(), f"session_{session_id}")
        os.makedirs(workspace_dir, exist_ok=True)
        return workspace_dir

    def cleanup_temp_files(self):
        """Clean up all tracked temporary files using bash."""
        all_files = []
        
        # Collect document temp files
        all_files.extend(self._temp_files.values())
        
        # Collect workflow temp files  
        all_files.extend(self._workflow_temp_files)
        
        # Cleanup using bash if available
        if self.bash_tool and all_files:
            files_str = ' '.join(f"'{f}'" for f in all_files)
            try:
                self.bash_tool(command=f"rm -f {files_str}")
                logger.debug(f"Cleaned up {len(all_files)} temp files via bash")
            except Exception as e:
                logger.warning(f"Error cleaning up temp files: {e}")
        
        # Clear tracking
        self._temp_files.clear()
        self._workflow_temp_files.clear()

    # ============================================================================
    # Cleanup
    # ============================================================================

    def get_cited_documents(self) -> List[Dict[str, Any]]:
        """
        Get list of cited documents from working context.

        Returns:
            List of cited document dictionaries with doc_id, title, and metadata
        """
        if not self.working_context:
            logger.debug("No working context available for cited documents")
            return []

        cited_docs = self.working_context.find_resources_by_metadata(
            resource_type="document_id",
            metadata_filters={"source": ["cited_context", "selected_text"]}
        )

        # Format for easier use
        result = []
        for doc in cited_docs:
            result.append({
                "doc_id": doc.get("metadata", {}).get("doc_id"),
                "title": doc.get("metadata", {}).get("title") or doc.get("path", ""),
                "source": doc.get("metadata", {}).get("source"),
                "has_selected_text": doc.get("metadata", {}).get("has_selected_text", False)
            })

        logger.debug(f"Found {len(result)} cited documents: {[d['doc_id'] for d in result if d['doc_id']]}")
        return result

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup_temp_files()
