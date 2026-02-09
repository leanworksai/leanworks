import logging
from typing import List, Dict, Any, Optional
from google.cloud import bigquery
from leanworks.utils.env import resolve_credential_path

logger = logging.getLogger(__name__)


class BigQueryTool:
    """
    Google Cloud BigQuery tool for accessing BigQuery datasets and tables.

    Provides read-only operations for BigQuery including listing datasets,
    listing tables, getting table schemas, and executing SELECT queries.
    Uses service account authentication.
    """

    def __init__(self, bigquery_client, credential_path: Optional[str] = None):
        """
        Initialize BigQueryTool with BigQuery client.

        Args:
            bigquery_client: Google Cloud BigQuery client instance
            credential_path: Path to GCP credential JSON file (default: environment-aware)
        """
        self.bigquery_client = bigquery_client
        self.credential_path = credential_path or resolve_credential_path()

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
        - rows_returned: Number of rows actually returned
        - total_rows_available: Total rows the query matched (may be larger than rows_returned)
        - truncated: Whether results were truncated due to max_results limit
        - max_results_limit: The effective max_results cap that was applied
        - query_metadata: Query execution metadata (bytes processed, etc.)

        IMPORTANT: If truncated is true, not all matching rows were returned. Consider using
        SQL aggregation (GROUP BY, COUNT, SUM, etc.) or filtering (WHERE) to reduce the result
        set, rather than trying to fetch all rows.
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
            
            # Capture total rows available before iterating
            total_rows_available = results.total_rows

            # Convert to list of dictionaries (capped by max_results)
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

            is_truncated = total_rows_available is not None and len(rows) < total_rows_available
            result = {
                "results": rows,
                "rows_returned": len(rows),
                "total_rows_available": total_rows_available,
                "truncated": is_truncated,
                "max_results_limit": max_results,
                "query_metadata": query_metadata
            }

            logger.info(f"Executed BigQuery query, returned {len(rows)}/{total_rows_available} rows (truncated={is_truncated}), processed {query_job.total_bytes_processed} bytes")
            return result

        except Exception as e:
            logger.error(f"Error executing BigQuery query: {str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}
