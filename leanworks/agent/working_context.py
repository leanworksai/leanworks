"""
WorkingContext manages operational state separate from conversational memory.

This class tracks active resources (files, IDs, storage references) during a session
to prevent important operational details from being lost during conversation summarization.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class WorkingContext:
    """
    Manages operational state separate from conversational memory.

    Tracks active resources (temp files, document IDs, storage references, etc.)
    during a session to prevent loss during summarization.
    """

    def __init__(self, default_ttl_hours: int = 24):
        """
        Initialize WorkingContext.

        Args:
            default_ttl_hours: Default time-to-live for resources in hours
        """
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.default_ttl_hours = default_ttl_hours

        # Resource type to TTL mapping
        self.ttl_config = {
            'temp_file': 12,      # Temp files last 12 hours
            'document_id': 48,    # Document refs last 48 hours
            'storage_ref': 24,    # Storage refs last 24 hours
            'resource_id': 48,    # Generic IDs last 48 hours
            'tool_state': 24      # Tool state last 24 hours
        }

    def register_resource(self,
                         resource_id: str,
                         resource_type: str,
                         path: str,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a resource for tracking.

        Args:
            resource_id: Unique identifier for the resource
            resource_type: Type of resource ('temp_file', 'document_id', etc.)
            path: Resource path or reference
            metadata: Additional metadata about the resource
        """
        if metadata is None:
            metadata = {}

        # Get TTL for this resource type
        ttl_hours = self.ttl_config.get(resource_type, self.default_ttl_hours)

        self.resources[resource_id] = {
            'type': resource_type,
            'path': path,
            'metadata': metadata,
            'created_at': datetime.now(),
            'last_used': datetime.now(),
            'ttl_hours': ttl_hours
        }

        logger.debug(f"Registered resource: {resource_id} ({resource_type}) - {path}")

    def get_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Get resource information by ID.

        Args:
            resource_id: Resource identifier

        Returns:
            Resource info dict or None if not found
        """
        resource = self.resources.get(resource_id)
        if resource:
            # Update last_used timestamp
            resource['last_used'] = datetime.now()
        return resource

    def touch_resource(self, resource_id: str) -> bool:
        """
        Update the last_used timestamp for a resource.

        Args:
            resource_id: Resource identifier

        Returns:
            True if resource exists, False otherwise
        """
        resource = self.resources.get(resource_id)
        if resource:
            resource['last_used'] = datetime.now()
            return True
        return False

    def get_active_resources(self, max_resources: int = 20) -> str:
        """
        Get formatted string of active resources for context injection.

        Args:
            max_resources: Maximum number of resources to include

        Returns:
            Formatted string of active resources or empty string
        """
        if not self.resources:
            return ""

        # Cleanup expired resources first
        self.cleanup_expired()

        if not self.resources:
            return ""

        # Sort by last_used (most recent first)
        sorted_resources = sorted(
            self.resources.items(),
            key=lambda x: x[1]['last_used'],
            reverse=True
        )

        # Take only the most recent resources
        active_resources = sorted_resources[:max_resources]

        # Format for context
        lines = ["Active Working Resources:"]
        for resource_id, resource_info in active_resources:
            resource_type = resource_info['type']
            path = resource_info['path']
            last_used = resource_info['last_used'].strftime('%H:%M')

            # Format based on type
            if resource_type == 'temp_file':
                lines.append(f"  - Temp file: {path} (id: {resource_id})")
            elif resource_type == 'document_id':
                lines.append(f"  - Document: {path} (id: {resource_id})")
            elif resource_type == 'storage_ref':
                lines.append(f"  - Storage: {path} (id: {resource_id})")
            elif resource_type == 'resource_id':
                lines.append(f"  - Resource: {path} (id: {resource_id})")
            elif resource_type == 'tool_state':
                lines.append(f"  - Tool state: {path} (id: {resource_id})")
            else:
                lines.append(f"  - {resource_type}: {path} (id: {resource_id})")

        return "\n".join(lines)

    def cleanup_expired(self) -> int:
        """
        Remove expired resources based on TTL.

        Returns:
            Number of resources removed
        """
        now = datetime.now()
        expired_ids = []

        for resource_id, resource_info in self.resources.items():
            ttl_hours = resource_info['ttl_hours']
            last_used = resource_info['last_used']

            if now - last_used > timedelta(hours=ttl_hours):
                expired_ids.append(resource_id)

        # Remove expired resources
        for resource_id in expired_ids:
            del self.resources[resource_id]

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired resources")

        return len(expired_ids)

    def clear(self) -> None:
        """
        Clear all resources. Called when session ends.
        """
        count = len(self.resources)
        self.resources.clear()
        logger.debug(f"Cleared {count} resources from working context")

    def get_resource_count(self) -> int:
        """
        Get the current number of tracked resources.

        Returns:
            Number of resources
        """
        return len(self.resources)

    def list_resources(self) -> List[Dict[str, Any]]:
        """
        Get a list of all resources with their info.

        Returns:
            List of resource dictionaries
        """
        resources_list = []
        for resource_id, resource_info in self.resources.items():
            resource_copy = resource_info.copy()
            resource_copy['resource_id'] = resource_id
            resources_list.append(resource_copy)

        return resources_list