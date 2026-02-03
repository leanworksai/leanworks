import logging
from typing import List, Dict, Any, Optional
import io
from google.cloud import storage, bigquery
from datetime import datetime, timedelta
import json
import re
from leanworks.utils.env import get_storage_bucket, resolve_credential_path

logger = logging.getLogger(__name__)


class GCPTool:
    """
    Google Cloud Platform tool for accessing Cloud Storage and BigQuery services.

    Consolidates Cloud Storage operations (migrated from CloudStorageTool) and adds
    BigQuery read-only operations. Uses service account authentication for both services.
    """

    def __init__(self, storage_client, bigquery_client, org_slug: str,
                 bucket_name: Optional[str] = None, credential_path: Optional[str] = None):
        """
        Initialize GCPTool with Cloud Storage and BigQuery clients.

        Args:
            storage_client: Google Cloud Storage client instance
            bigquery_client: Google Cloud BigQuery client instance
            org_slug: Organization slug (e.g., 'leanworks.ai')
            bucket_name: GCS bucket name (default: environment-aware)
            credential_path: Path to GCP credential JSON file (default: environment-aware)
        """
        self.storage_client = storage_client
        self.bigquery_client = bigquery_client
        self.org_slug = org_slug
        self.bucket_name = bucket_name or get_storage_bucket()
        self.credential_path = credential_path or resolve_credential_path()

        # Image URL expiration time (default: 1 year)
        self.image_url_expiration_days = 365

    # ============================================================================
    # CLOUD STORAGE SECTION (Migrated from CloudStorageTool)
    # ============================================================================

    def _get_bucket(self):
        """Get or create the storage bucket."""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            # Check if bucket exists
            if not bucket.exists():
                logger.warning(f"Bucket {self.bucket_name} does not exist")
            return bucket
        except Exception as e:
            logger.error(f"Error accessing bucket {self.bucket_name}: {e}")
            raise

    def _generate_signed_url(self, storage_path: str, expiration_days: int = None) -> str:
        """
        Generate a signed URL for accessing a file in Cloud Storage.

        Args:
            storage_path: Path to the file in the bucket
            expiration_days: Number of days until URL expires (default: 1 year)

        Returns:
            Signed URL string
        """
        try:
            expiration_days = expiration_days or self.image_url_expiration_days
            bucket = self._get_bucket()
            blob = bucket.blob(storage_path)

            # Calculate expiration time
            expires_in = expiration_days * 24 * 60 * 60 * 1000  # Convert to milliseconds
            expires_at = datetime.utcnow() + timedelta(days=expiration_days)

            # Generate signed URL
            signed_url = blob.generate_signed_url(
                expiration=expires_at,
                method='GET'
            )

            return signed_url
        except Exception as e:
            logger.error(f"Error generating signed URL for {storage_path}: {e}")
            raise

    def _extract_storage_path(self, image_url_or_id: str, chat_id: str) -> Optional[str]:
        """
        Extract storage path from image URL or ID.

        Args:
            image_url_or_id: Either a signed URL or an image ID (UUID.jpg)
            chat_id: Chat ID for constructing path if image_id is provided

        Returns:
            Storage path or None if cannot be determined
        """
        # If it's already a storage path, return it
        if image_url_or_id.startswith('orgs/'):
            return image_url_or_id

        # If it's an imageId (UUID.jpg), construct the path
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jpg$', image_url_or_id, re.IGNORECASE):
            return f"orgs/{self.org_slug}/chat-images/{chat_id}/{image_url_or_id}"

        # Try to extract from signed URL
        try:
            from urllib.parse import urlparse
            url = urlparse(image_url_or_id)
            # Extract path from Google Cloud Storage signed URL
            # Format: https://storage.googleapis.com/bucket/path?signature=...
            if 'storage.googleapis.com' in url.netloc:
                path_match = url.path.split('/', 2)
                if len(path_match) >= 3:
                    return path_match[2]  # Path after /bucket/
        except Exception as e:
            logger.warning(f"Could not extract path from URL {image_url_or_id}: {e}")

        # Fallback: construct from chat_id
        if chat_id:
            return f"orgs/{self.org_slug}/chat-images/{chat_id}/{image_url_or_id}"

        return None

    @property
    def gcp_cloud_storage_get_image_url_property(self):
        description = f"""
        Get a signed URL for accessing an image from Google Cloud Storage for org `{self.org_slug}`.

        Images are stored in Google Cloud Storage at:
        orgs/{self.org_slug}/chat-images/{{chatId}}/{{imageId}}.jpg

        This tool generates signed URLs that provide temporary access to private images.
        Signed URLs expire after 1 year by default.

        Parameters:
        - imageId: The image ID (UUID.jpg format) or a signed URL that needs refreshing
        - chatId: The chat ID where the image was uploaded (required if imageId is provided)
        - expirationDays: Optional number of days until URL expires (default: 365)

        Examples:
        - Get URL for an image: {{"imageId": "123e4567-e89b-12d3-a456-426614174000.jpg", "chatId": "ai-assistant-user@example.com"}}
        - Refresh an expired URL: {{"imageId": "https://storage.googleapis.com/...", "chatId": "project-123"}}

        Returns:
        A dictionary with:
        - imageUrl: The signed URL for accessing the image
        - imageId: The image ID
        - expiresAt: ISO timestamp when the URL expires
        """
        return {
            "type": "custom",
            "name": "gcp_cloud_storage_get_image_url",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "imageId": {
                        "type": "string",
                        "description": "Image ID (UUID.jpg format) or existing signed URL that needs refreshing"
                    },
                    "chatId": {
                        "type": "string",
                        "description": "Chat ID where the image was uploaded (required if imageId is a UUID)"
                    },
                    "expirationDays": {
                        "type": "integer",
                        "description": "Number of days until URL expires (default: 365)",
                        "minimum": 1,
                        "maximum": 3650
                    }
                },
                "required": ["imageId", "chatId"]
            }
        }

    def gcp_cloud_storage_get_image_url(
        self,
        imageId: str,
        chatId: str,
        expirationDays: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get a signed URL for accessing an image.

        Args:
            imageId: Image ID (UUID.jpg) or existing signed URL
            chatId: Chat ID where image was uploaded
            expirationDays: Optional expiration days (default: 365)

        Returns:
            Dictionary with imageUrl, imageId, and expiresAt, or error dictionary
        """
        try:
            if not self.storage_client or not self.org_slug:
                return {"error": "Storage client or org_slug not initialized"}

            if not imageId or not chatId:
                return {"error": "imageId and chatId are required"}

            # Extract storage path
            storage_path = self._extract_storage_path(imageId, chatId)
            if not storage_path:
                return {"error": f"Could not determine storage path for imageId: {imageId}"}

            # Generate signed URL
            expiration_days = expirationDays or self.image_url_expiration_days
            signed_url = self._generate_signed_url(storage_path, expiration_days)

            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(days=expiration_days)

            # Extract image ID from path if needed
            actual_image_id = storage_path.split('/')[-1]

            logger.info(f"Generated signed URL for image: {actual_image_id} in chat: {chatId}")

            return {
                "imageUrl": signed_url,
                "imageId": actual_image_id,
                "expiresAt": expires_at.isoformat() + "Z"
            }

        except Exception as e:
            logger.error(f"Error getting image URL: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def gcp_cloud_storage_list_chat_images_property(self):
        description = f"""
        List all images in a chat from Google Cloud Storage for org `{self.org_slug}`.

        Lists all images stored for a specific chat ID and returns signed URLs for accessing them.
        Images are stored at: orgs/{self.org_slug}/chat-images/{{chatId}}/{{imageId}}.jpg

        Parameters:
        - chatId: The chat ID to list images for
        - limit: Maximum number of images to return (default: 50, max: 200)
        - expirationDays: Number of days until URLs expire (default: 365)

        Returns:
        A list of dictionaries, each containing:
        - imageUrl: Signed URL for accessing the image
        - imageId: The image ID (filename)
        - uploadedAt: ISO timestamp when the image was uploaded (if available)
        - expiresAt: ISO timestamp when the URL expires
        """
        return {
            "type": "custom",
            "name": "gcp_cloud_storage_list_chat_images",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "chatId": {
                        "type": "string",
                        "description": "Chat ID to list images for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of images to return (default: 50, max: 200)",
                        "minimum": 1,
                        "maximum": 200
                    },
                    "expirationDays": {
                        "type": "integer",
                        "description": "Number of days until URLs expire (default: 365)",
                        "minimum": 1,
                        "maximum": 3650
                    }
                },
                "required": ["chatId"]
            }
        }

    def gcp_cloud_storage_list_chat_images(
        self,
        chatId: str,
        limit: int = 50,
        expirationDays: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        List all images in a chat.

        Args:
            chatId: Chat ID to list images for
            limit: Maximum number of images (default: 50, max: 200)
            expirationDays: Optional expiration days (default: 365)

        Returns:
            List of image dictionaries, or error dictionary
        """
        try:
            if not self.storage_client or not self.org_slug:
                return {"error": "Storage client or org_slug not initialized"}

            if not chatId:
                return {"error": "chatId is required"}

            # Validate limit
            limit = min(max(1, limit), 200)

            # Build storage path prefix
            storage_prefix = f"orgs/{self.org_slug}/chat-images/{chatId}/"

            # List blobs in the prefix
            bucket = self._get_bucket()
            blobs = bucket.list_blobs(prefix=storage_prefix, max_results=limit)

            images = []
            expiration_days = expirationDays or self.image_url_expiration_days

            for blob in blobs:
                # Skip if not an image file
                if not blob.name.endswith('.jpg'):
                    continue

                # Generate signed URL
                try:
                    signed_url = self._generate_signed_url(blob.name, expiration_days)
                    expires_at = datetime.utcnow() + timedelta(days=expiration_days)

                    # Get upload time from blob metadata
                    uploaded_at = None
                    if blob.time_created:
                        uploaded_at = blob.time_created.isoformat() + "Z"

                    image_id = blob.name.split('/')[-1]

                    images.append({
                        "imageUrl": signed_url,
                        "imageId": image_id,
                        "uploadedAt": uploaded_at,
                        "expiresAt": expires_at.isoformat() + "Z"
                    })
                except Exception as e:
                    logger.warning(f"Error generating URL for {blob.name}: {e}")
                    continue

            logger.info(f"Listed {len(images)} images for chat: {chatId}")
            return images

        except Exception as e:
            logger.error(f"Error listing chat images: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    # ============================================================================
    # BIGQUERY SECTION (New functionality)
    # ============================================================================

    @property
    def gcp_bigquery_list_datasets_property(self):
        description = """
        List all BigQuery datasets in the current GCP project.

        Returns information about each dataset including ID, location, and creation time.
        Only datasets that the service account has access to will be listed.

        Parameters:
        - max_results: Maximum number of datasets to return (default: 50, max: 1000)

        Returns:
        A list of dictionaries, each containing:
        - dataset_id: The dataset ID
        - location: The geographic location of the dataset
        - created_at: ISO timestamp when the dataset was created
        - description: Dataset description (if available)
        """
        return {
            "type": "custom",
            "name": "gcp_bigquery_list_datasets",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of datasets to return (default: 50, max: 1000)",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 50
                    }
                }
            }
        }

    def gcp_bigquery_list_datasets(self, max_results: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """
        List BigQuery datasets in the project.

        Args:
            max_results: Maximum number of datasets to return

        Returns:
            List of dataset dictionaries, or error dictionary
        """
        try:
            if not self.bigquery_client:
                return {"error": "BigQuery client not initialized"}

            # Validate max_results
            max_results = min(max(1, max_results), 1000)

            # List datasets
            datasets = list(self.bigquery_client.list_datasets(max_results=max_results))

            result = []
            for dataset in datasets:
                dataset_ref = self.bigquery_client.dataset(dataset.dataset_id)
                full_dataset = self.bigquery_client.get_dataset(dataset_ref)

                result.append({
                    "dataset_id": dataset.dataset_id,
                    "location": full_dataset.location,
                    "created_at": full_dataset.created.isoformat() + "Z" if full_dataset.created else None,
                    "description": full_dataset.description
                })

            logger.info(f"Listed {len(result)} BigQuery datasets")
            return result

        except Exception as e:
            logger.error(f"Error listing BigQuery datasets: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def gcp_bigquery_list_tables_property(self):
        description = """
        List all tables in a BigQuery dataset.

        Returns information about each table including ID, type (table/view), row count, and size.
        The service account must have access to the dataset.

        Parameters:
        - dataset_id: The BigQuery dataset ID
        - max_results: Maximum number of tables to return (default: 100, max: 1000)

        Returns:
        A list of dictionaries, each containing:
        - table_id: The table ID
        - type: 'TABLE' or 'VIEW'
        - row_count: Number of rows in the table (for tables)
        - size_bytes: Size of the table in bytes
        - created_at: ISO timestamp when the table was created
        - modified_at: ISO timestamp when the table was last modified
        """
        return {
            "type": "custom",
            "name": "gcp_bigquery_list_tables",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "The BigQuery dataset ID"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of tables to return (default: 100, max: 1000)",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100
                    }
                },
                "required": ["dataset_id"]
            }
        }

    def gcp_bigquery_list_tables(self, dataset_id: str, max_results: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """
        List tables in a BigQuery dataset.

        Args:
            dataset_id: The BigQuery dataset ID
            max_results: Maximum number of tables to return

        Returns:
            List of table dictionaries, or error dictionary
        """
        try:
            if not self.bigquery_client:
                return {"error": "BigQuery client not initialized"}

            if not dataset_id:
                return {"error": "dataset_id is required"}

            # Validate max_results
            max_results = min(max(1, max_results), 1000)

            # Get dataset reference
            dataset_ref = self.bigquery_client.dataset(dataset_id)

            # List tables
            tables = list(self.bigquery_client.list_tables(dataset_ref, max_results=max_results))

            result = []
            for table in tables:
                table_ref = dataset_ref.table(table.table_id)
                full_table = self.bigquery_client.get_table(table_ref)

                result.append({
                    "table_id": table.table_id,
                    "type": full_table.table_type,
                    "row_count": full_table.num_rows,
                    "size_bytes": full_table.num_bytes,
                    "created_at": full_table.created.isoformat() + "Z" if full_table.created else None,
                    "modified_at": full_table.modified.isoformat() + "Z" if full_table.modified else None
                })

            logger.info(f"Listed {len(result)} tables in dataset: {dataset_id}")
            return result

        except Exception as e:
            logger.error(f"Error listing BigQuery tables: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def gcp_bigquery_get_table_schema_property(self):
        description = """
        Get the schema for a BigQuery table.

        Returns detailed column information including names, data types, modes (NULLABLE/REQUIRED/REPEATED),
        and descriptions. The service account must have access to the dataset.

        Parameters:
        - dataset_id: The BigQuery dataset ID
        - table_id: The BigQuery table ID

        Returns:
        A dictionary containing:
        - dataset_id: The dataset ID
        - table_id: The table ID
        - schema: List of column definitions with name, type, mode, and description
        """
        return {
            "type": "custom",
            "name": "gcp_bigquery_get_table_schema",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "The BigQuery dataset ID"
                    },
                    "table_id": {
                        "type": "string",
                        "description": "The BigQuery table ID"
                    }
                },
                "required": ["dataset_id", "table_id"]
            }
        }

    def gcp_bigquery_get_table_schema(self, dataset_id: str, table_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get schema for a BigQuery table.

        Args:
            dataset_id: The BigQuery dataset ID
            table_id: The BigQuery table ID

        Returns:
            Schema dictionary, or error dictionary
        """
        try:
            if not self.bigquery_client:
                return {"error": "BigQuery client not initialized"}

            if not dataset_id or not table_id:
                return {"error": "dataset_id and table_id are required"}

            # Get table reference
            table_ref = self.bigquery_client.dataset(dataset_id).table(table_id)
            table = self.bigquery_client.get_table(table_ref)

            # Extract schema information
            schema_fields = []
            for field in table.schema:
                schema_fields.append({
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description
                })

            result = {
                "dataset_id": dataset_id,
                "table_id": table_id,
                "schema": schema_fields
            }

            logger.info(f"Retrieved schema for table: {dataset_id}.{table_id} with {len(schema_fields)} columns")
            return result

        except Exception as e:
            logger.error(f"Error getting BigQuery table schema: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

    @property
    def gcp_bigquery_query_property(self):
        description = """
        Execute a BigQuery SQL query and return results.

        Executes a SQL query against BigQuery and returns the results as a list of dictionaries.
        Supports SELECT queries only (read-only operations). The service account must have
        appropriate permissions to query the datasets/tables referenced in the query.

        Parameters:
        - query: The SQL query to execute
        - max_results: Maximum number of rows to return (default: 100, max: 1000)
        - use_legacy_sql: Whether to use legacy SQL syntax (default: false, uses standard SQL)

        Returns:
        A dictionary containing:
        - results: List of row dictionaries
        - total_rows: Total number of rows returned
        - query_metadata: Query execution metadata (bytes processed, etc.)
        """
        return {
            "type": "custom",
            "name": "gcp_bigquery_query",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute (SELECT queries only)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of rows to return (default: 100, max: 1000)",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100
                    },
                    "use_legacy_sql": {
                        "type": "boolean",
                        "description": "Whether to use legacy SQL syntax (default: false)",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        }

    def gcp_bigquery_query(self, query: str, max_results: int = 100, use_legacy_sql: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Execute a BigQuery SQL query.

        Args:
            query: The SQL query to execute
            max_results: Maximum number of rows to return
            use_legacy_sql: Whether to use legacy SQL syntax

        Returns:
            Query results dictionary, or error dictionary
        """
        try:
            if not self.bigquery_client:
                return {"error": "BigQuery client not initialized"}

            if not query:
                return {"error": "query is required"}

            # Validate max_results
            max_results = min(max(1, max_results), 1000)

            # Validate query is SELECT-only (read-only)
            query_upper = query.strip().upper()
            if not query_upper.startswith('SELECT'):
                return {"error": "Only SELECT queries are allowed (read-only operations)"}

            # Execute query
            job_config = bigquery.QueryJobConfig(
                use_legacy_sql=use_legacy_sql,
                maximum_bytes_billed=100*1024*1024*1024  # 100GB limit
            )

            query_job = self.bigquery_client.query(
                query,
                job_config=job_config
            )

            # Wait for query to complete
            results = query_job.result()

            # Convert to list of dictionaries
            rows = []
            for row in results:
                rows.append(dict(row))
                if len(rows) >= max_results:
                    break

            # Get query metadata
            query_metadata = {
                "bytes_processed": query_job.total_bytes_processed,
                "bytes_billed": query_job.total_bytes_billed,
                "slot_millis": query_job.slot_millis,
                "cache_hit": query_job.cache_hit
            }

            result = {
                "results": rows,
                "total_rows": len(rows),
                "query_metadata": query_metadata
            }

            logger.info(f"Executed BigQuery query, returned {len(rows)} rows, processed {query_job.total_bytes_processed} bytes")
            return result

        except Exception as e:
            logger.error(f"Error executing BigQuery query: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
