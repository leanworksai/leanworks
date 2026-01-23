"""
Working Context Tool - Tool for querying operational state and resources.

This tool provides access to the WorkingContext that tracks active resources
(files, document IDs, storage references) during AI agent sessions.
"""
from typing import Dict, List, Any, Optional
from .base_api_client import BaseAPIClient
import logging

logger = logging.getLogger(__name__)


class WorkingContextTool(BaseAPIClient):
    """
    Tool for querying working context and operational state.

    Provides access to resources that have been loaded or created during
    the current session, separate from conversational memory.
    """

    def __init__(
        self,
        org_slug: str,
        working_context=None,
        user_id: Optional[str] = None
    ):
        """
        Initialize WorkingContextTool.

        Args:
            org_slug: Organization slug for API authentication
            working_context: WorkingContext instance to query
            user_id: Optional user ID for attribution
        """
        super().__init__(org_slug, user_id)
        self.working_context = working_context

    @property
    def query_working_context_property(self):
        """Query working context to discover available resources."""
        description = """
        Query working context to discover what resources have been loaded in this session.

        Working context tracks operational state separate from conversation memory:
        - Document, selected text, task and project references (persisted across sessions)
        - Large response file references including tool responses and workflow files (persisted across sessions)

        Use this tool to:
        - Check if a document has already been loaded without re-fetching
        - Find file paths for previously loaded or created content
        - Discover what resources are available in the current session
        - Avoid redundant operations by checking existing state

        Parameters:
        - resource_type (optional): Filter by type ('tool_response_file', 'document_id', 'temp_file', 'storage_ref', 'task_id', 'project_id')
        - doc_id (optional): Find resources related to a specific document ID
        - search_metadata (optional): Search metadata fields (dict of key-value pairs, e.g., {'tool': 'get_doc'})

        Returns:
        - List of resources with: resource_id, type, path, metadata (including doc_ids, tool, source), last_used timestamp
        - Empty list if no matching resources found

        Example queries:
        - query_working_context() → All active resources in session
        - query_working_context(resource_type='tool_response_file') → All loaded files
        - query_working_context(doc_id='57e6a9f7-...') → Resources for specific document
        - query_working_context(search_metadata={'tool': 'get_doc'}) → Files from get_doc calls
        - query_working_context(resource_type='document_id') → All cited/referenced documents
        """
        return {
            "type": "custom",
            "name": "query_working_context",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "resource_type": {
                        "type": "string",
                        "enum": ["tool_response_file", "document_id", "temp_file", "storage_ref", "task_id", "project_id", "resource_id"],
                        "description": "Filter by resource type"
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Find resources related to this document ID"
                    },
                    "search_metadata": {
                        "type": "object",
                        "description": "Search by metadata fields (e.g., {'tool': 'get_doc'}, {'source': 'cited_context'})"
                    }
                }
            }
        }

    def query_working_context(
        self,
        resource_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        search_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query working context for available resources"""
        if not self.working_context:
            return {
                "error": "Working context not available",
                "resources": []
            }

        # Handle doc_id query specially
        if doc_id:
            # Find all resources with this doc_id in metadata
            all_resources = self.working_context.list_resources()
            matching = []
            for resource in all_resources:
                metadata = resource.get('metadata', {})
                doc_ids = metadata.get('doc_ids', [])
                if doc_id in doc_ids:
                    matching.append(resource)

            if matching:
                # Sort by most recently used
                matching.sort(key=lambda x: x.get('last_used'), reverse=True)
                return {
                    "found": True,
                    "count": len(matching),
                    "resources": matching
                }
            else:
                return {
                    "found": False,
                    "message": f"No resources found for document ID: {doc_id}",
                    "resources": []
                }

        # Use working context query methods
        if resource_type or search_metadata:
            metadata_filters = search_metadata or {}
            resources = self.working_context.find_resources_by_metadata(
                resource_type=resource_type,
                metadata_filters=metadata_filters
            )
        else:
            # Get all resources
            resources = self.working_context.list_resources()

        # Sort by most recently used
        resources.sort(key=lambda x: x.get('last_used'), reverse=True)

        return {
            "found": len(resources) > 0,
            "count": len(resources),
            "resources": resources
        }