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

    def __init__(self, default_ttl_hours: int = 24, resources_data: Dict[str, Dict[str, Any]] = None):
        """
        Initialize WorkingContext.

        Args:
            default_ttl_hours: Default time-to-live for resources in hours
            resources_data: Optional persisted resources data to restore
        """
        self.resources: Dict[str, Dict[str, Any]] = resources_data or {}
        self.default_ttl_hours = default_ttl_hours

        # Resource type to TTL mapping
        self.ttl_config = {
            'temp_file': 12,      # Temp files last 12 hours
            'document_id': 48,    # Document refs last 48 hours
            'project_id': 48,     # Project refs last 48 hours
            'task_id': 48,        # Task refs last 48 hours
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

    def find_resources_by_metadata(
        self,
        resource_type: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find resources matching type and metadata criteria.

        Args:
            resource_type: Filter by resource type (None = all types)
            metadata_filters: Dict of metadata key-value pairs to match

        Returns:
            List of matching resources with their IDs
        """
        matches = []

        for resource_id, resource_info in self.resources.items():
            # Check type filter
            if resource_type and resource_info.get('type') != resource_type:
                continue

            # Check metadata filters
            if metadata_filters:
                resource_metadata = resource_info.get('metadata', {})
                match = True

                for key, value in metadata_filters.items():
                    # Handle list values (e.g., doc_ids)
                    if isinstance(value, list):
                        metadata_value = resource_metadata.get(key, [])
                        if not any(v in metadata_value for v in value):
                            match = False
                            break
                    # Handle single values
                    elif resource_metadata.get(key) != value:
                        match = False
                        break

                if not match:
                    continue

            # Resource matches all criteria
            result = resource_info.copy()
            result['resource_id'] = resource_id
            matches.append(result)

        return matches

    def find_document_file(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Find workspace HTML file for a specific document ID.

        Args:
            doc_id: Document ID to search for

        Returns:
            Resource info dict with file path, or None if not found
        """
        resources = self.find_resources_by_metadata(
            resource_type='tool_response_file',
            metadata_filters={'doc_ids': [doc_id]}
        )

        # Return the most recently used file
        if resources:
            resources.sort(key=lambda x: x.get('last_used', datetime.min), reverse=True)
            return resources[0]

        return None

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
            
            # Handle both offset-naive and offset-aware datetimes
            # If last_used is a string (from deserialization), parse it
            if isinstance(last_used, str):
                try:
                    from datetime import datetime as dt
                    last_used = dt.fromisoformat(last_used.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    logger.warning(f"Could not parse datetime string for resource {resource_id}, skipping expiry check")
                    continue
            
            # Ensure both datetimes are offset-naive for comparison
            if hasattr(last_used, 'tzinfo') and last_used.tzinfo is not None:
                last_used = last_used.replace(tzinfo=None)

            if now - last_used > timedelta(hours=ttl_hours):
                expired_ids.append(resource_id)

        # Remove expired resources
        for resource_id in expired_ids:
            del self.resources[resource_id]

        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired resources")

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize working context for persistence.

        Returns:
            Dictionary representation of working context
        """
        return {
            "resources": self.resources,
            "default_ttl_hours": self.default_ttl_hours,
            "ttl_config": self.ttl_config
        }

    def to_dict_filtered(self) -> Dict[str, Any]:
        """
        Serialize only cited context and large response file references for persistence.

        Note: Only stores references (IDs, paths, metadata), NOT actual file contents or resource data.

        Returns:
            Dictionary representation containing only relevant resources for persistence
        """
        # Filter to only include cited context, large response files, and project/task references
        filtered_resources = {}

        for resource_id, resource_info in self.resources.items():
            resource_type = resource_info.get('type', '')
            # Include document_id (cited context), tool_response_file (large response files),
            # workflow_file (doc workflow files), project_id (cited projects), and task_id (cited tasks)
            # These store ONLY references (paths, IDs, metadata), NOT contents
            if resource_type in ['document_id', 'tool_response_file', 'workflow_file', 'project_id', 'task_id']:
                filtered_resources[resource_id] = resource_info

        return {
            "resources": filtered_resources,
            "default_ttl_hours": self.default_ttl_hours,
            "ttl_config": self.ttl_config
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkingContext':
        """
        Deserialize working context from persisted data.
        
        Args:
            data: Dictionary containing working context data
            
        Returns:
            WorkingContext instance with restored state
        """
        instance = cls(
            default_ttl_hours=data.get('default_ttl_hours', 24),
            resources_data=data.get('resources', {})
        )
        
        # Convert serialized datetime strings back to datetime objects
        for resource_id, resource_info in instance.resources.items():
            for datetime_field in ['created_at', 'last_used']:
                if datetime_field in resource_info and isinstance(resource_info[datetime_field], str):
                    try:
                        # Parse ISO format datetime string
                        resource_info[datetime_field] = datetime.fromisoformat(
                            resource_info[datetime_field].replace('Z', '+00:00')
                        )
                        # Convert to offset-naive for consistency
                        if hasattr(resource_info[datetime_field], 'tzinfo') and resource_info[datetime_field].tzinfo is not None:
                            resource_info[datetime_field] = resource_info[datetime_field].replace(tzinfo=None)
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Could not parse datetime for {resource_id}.{datetime_field}: {e}, using current time")
                        resource_info[datetime_field] = datetime.now()
        
        # Update TTL config if provided
        if 'ttl_config' in data:
            instance.ttl_config.update(data['ttl_config'])
            
        return instance

    def validate_resources(self) -> int:
        """
        Check if persisted resources still exist on disk.
        Validates temp files and removes ones that no longer exist.
        
        Returns:
            Number of removed invalid resources
        """
        import os
        removed_count = 0
        
        for resource_id, resource_info in list(self.resources.items()):
            resource_type = resource_info.get('type', '')
            path = resource_info.get('path', '')
            
            if resource_type == 'temp_file' and path:
                if not os.path.exists(path):
                    logger.warning(f"Temp file no longer exists, removing from working context: {path}")
                    del self.resources[resource_id]
                    removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"Validated working context - removed {removed_count} missing temp file resources")
        
        return removed_count