import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import json
import logging
import datetime
import re
from leanworks.setting import get_tables_and_schemas
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class FirestoreTool:
    # Class-level Firestore client initialization
    _firestore_initialized = False
    _firestore_lock = None
    _db = None
    
    @classmethod
    def _get_firestore_client(cls):
        """Initialize and return Firestore client."""
        if cls._firestore_lock is None:
            import threading
            cls._firestore_lock = threading.Lock()
        
        if not cls._firestore_initialized:
            with cls._firestore_lock:
                if not cls._firestore_initialized:
                    try:
                        if not firebase_admin._apps:
                            cred = credentials.Certificate("gcp_credential.json")
                            firebase_admin.initialize_app(cred)
                        # Use Firebase Admin SDK's firestore.client() with database_id parameter
                        # This is the recommended approach per Firebase documentation (firebase-admin 7.1.0+)
                        cls._db = firestore.client(database_id="leanworks-prod")
                        cls._firestore_initialized = True
                        logger.info("Firestore client initialized for FirestoreTool (database: leanworks-prod)")
                    except Exception as e:
                        logger.error(f"Failed to initialize Firestore: {e}")
                        raise
        return cls._db
    
    def __init__(self, firestore_client_wrapper):
        """
        Initialize FirestoreTool with a Firestore client wrapper.
        
        Args:
            firestore_client_wrapper: An object with attributes `domain` (client domain like 'leanworks.ai')
                                    and optionally `client_name` (for backward compatibility).
        """
        self.firestore_client_wrapper = firestore_client_wrapper
        self.db = self._get_firestore_client()
        
        # Get domain from wrapper (use client_name as fallback)
        self.domain = getattr(self.firestore_client_wrapper, 'domain', None)
        if not self.domain:
            # Fallback: construct domain from client_name if available
            client_name = getattr(self.firestore_client_wrapper, 'client_name', 'unknown')
            # Try to construct a reasonable domain (this is a fallback, should provide actual domain)
            self.domain = f"{client_name}.ai" if client_name != 'unknown' else 'leanworks.ai'
            logger.warning(f"Domain not provided in wrapper, using fallback: {self.domain}")
        
        # Load table schemas from settings for documentation purposes
        try:
            dataset_id = getattr(self.firestore_client_wrapper, 'client_name', self.domain.split('.')[0])
            self.tables_and_schemas = get_tables_and_schemas(dataset_id)
        except Exception as e:
            logger.warning(f"Failed to load schemas from settings: {str(e)}")
            self.tables_and_schemas = ""
    
    def _get_collection_path(self, collection_name: str) -> str:
        """
        Get Firestore collection path using domain-based structure.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            Full collection path: domains/{domain}/{collection_name}
        """
        return f"domains/{self.domain}/{collection_name}"
    
    
    def _parse_where_conditions(self, where_list: List[Dict[str, Any]]) -> List[Any]:
        """
        Parse where conditions into Firestore filters.
        
        Args:
            where_list: List of where condition dictionaries
        
        Returns:
            List of Firestore FieldFilter objects
        """
        filters = []
        
        for cond in where_list:
            if not isinstance(cond, dict):
                continue
            
            column = cond.get("column")
            op = cond.get("op", "=").upper()
            value = cond.get("value")
            
            if not column:
                continue
            
            # Map SQL operators to Firestore operators
            op_mapping = {
                "=": "==",
                "==": "==",
                "!=": "!=",
                ">": ">",
                ">=": ">=",
                "<": "<",
                "<=": "<=",
                "IN": "in",
                "NOT IN": "not-in",
                "ARRAY_CONTAINS": "array-contains",
                "ARRAY_CONTAINS_ANY": "array-contains-any"
            }
            
            firestore_op = op_mapping.get(op, "==")
            
            # Handle BETWEEN operator (need to split into two filters)
            if op == "BETWEEN" and isinstance(value, (list, tuple)) and len(value) == 2:
                filters.append(FieldFilter(column, ">=", value[0]))
                filters.append(FieldFilter(column, "<=", value[1]))
                continue
            
            # Handle LIKE operator (Firestore doesn't support LIKE, need to use >= and <)
            if op == "LIKE":
                # Convert SQL LIKE pattern to Firestore range query
                # For now, only support simple prefix matching
                if isinstance(value, str):
                    if value.startswith("%") and value.endswith("%"):
                        # Contains - Firestore doesn't support this easily, skip for now
                        logger.warning(f"LIKE operator with contains pattern not supported: {value}")
                        continue
                    elif value.endswith("%"):
                        # Prefix match
                        prefix = value.rstrip("%")
                        filters.append(FieldFilter(column, ">=", prefix))
                        filters.append(FieldFilter(column, "<", prefix + "\uf8ff"))
                        continue
                    else:
                        # Exact match
                        filters.append(FieldFilter(column, "==", value))
                        continue
            
            # Standard operator
            filters.append(FieldFilter(column, firestore_op, value))
        
        return filters
    
    def _compile_query_spec(self, spec: dict, collection_name: str) -> tuple:
        """
        Compile query spec into Firestore query parameters.
        
        Args:
            spec: Query specification dictionary
            collection_name: Name of the collection to query
        
        Returns:
            Tuple of (filters, order_by, limit)
        """
        if not isinstance(spec, dict):
            raise ValueError(f"spec must be an object, got {type(spec)}: {spec}")
        
        # Parse WHERE conditions
        where_list = spec.get("where") or spec.get("filters") or []
        filters = self._parse_where_conditions(where_list)
        
        # Parse ORDER BY
        order_by = []
        order_by_list = spec.get("order_by") or []
        for o in order_by_list:
            if isinstance(o, dict):
                field = o.get("expr") or o.get("field")
                direction = (o.get("dir") or o.get("direction") or "ASC").upper()
                if field:
                    # Firestore uses ASCENDING/DESCENDING
                    firestore_direction = firestore.Query.ASCENDING if direction == "ASC" else firestore.Query.DESCENDING
                    order_by.append((field, firestore_direction))
            elif isinstance(o, str):
                # Default to ascending
                order_by.append((o, firestore.Query.ASCENDING))
        
        # Parse LIMIT
        limit = spec.get("limit")
        if limit and not isinstance(limit, int):
            try:
                limit = int(limit)
            except (ValueError, TypeError):
                limit = None
        
        return filters, order_by, limit
    
    def _execute_firestore_query(self, collection_name: str, filters: List[Any], 
                                 order_by: List[tuple], limit: Optional[int]) -> List[Dict[str, Any]]:
        """
        Execute Firestore query and return results.
        
        Args:
            collection_name: Name of the collection
            filters: List of Firestore FieldFilter objects
            order_by: List of (field, direction) tuples
            limit: Maximum number of results
        
        Returns:
            List of documents as Firestore dictionaries
        """
        collection_path = self._get_collection_path(collection_name)
        query = self.db.collection(collection_path)
        
        # Apply filters
        for f in filters:
            query = query.where(filter=f)
        
        # Apply ordering
        for field, direction in order_by:
            query = query.order_by(field, direction=direction)
        
        # Apply limit
        if limit and limit > 0:
            query = query.limit(limit)
        
        # Execute query
        docs = query.stream()
        
        # Return Firestore documents as-is
        results = []
        for doc in docs:
            doc_dict = doc.to_dict()
            if doc_dict:
                # Add document ID if not present
                if "id" not in doc_dict:
                    doc_dict["id"] = doc.id
                results.append(doc_dict)
        
        return results
    
    @property
    def query_firestore_property(self):
        description = f"""
        Query Firestore collections in domain `{self.domain}`.
        
        This tool is strictly READ-ONLY. It queries Firestore collections and returns results in Firestore document format.
        
        Provide `spec`: a JSON QuerySpec that the tool uses to query Firestore.
        
        QuerySpec fields (JSON):
        - collection: Name of the collection to query (e.g., "projects", "tasks", "users")
        - where: [ {{"column": "status", "op": "==", "value": "completed"}} ]
        - order_by: [ {{"field": "createdAt", "direction": "DESC"}} ]
        - limit: 1000
        
        Available collections:
        - tasks: Task/action items for projects
        - updates: Work updates/progress reports from team members
        - update_summaries: Daily aggregated summaries of updates per project
        - users: User profile information
        - projects: Project configuration and metadata
        - integrations: External integration configurations (gitlab, jira, atlassian, etc.)
        - teams: Team information and membership (optional)
        - teamDetails: Detailed team information and settings (optional)
        
        Notes:
        - Most of the time, user won't directly give you any 'id' but rather a 'name'. You should try to get the mapping from name to id first (for example, project name to project id and user name to user id), and then filter the collection using the id.
        - When you filter by a 'name' field, you can use 'LIKE' operator with % wildcard for prefix matching.
        - If your response is empty, it means either you are filtering using a wrong value or the result is empty. In either case, you should try to query the first 5 rows to see if the result is empty. If it is not empty, then use those sample data to have a better understanding of the schema. After that, you can take another attempt to query with the correct filters.
        - Read-only: do not attempt any write operations. Only queries are allowed.
        
        Table schemas (for reference):
        {self.tables_and_schemas}
        """
        return {
            "type": "custom",
            "name": "query_firestore",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "object",
                        "description": "QuerySpec to compile into Firestore query (required).",
                    }
                },
                "required": ["spec"]
            }
        }
    
    def query_firestore(self, spec=None, **kwargs):
        """
        Query Firestore collections based on the provided spec.
        
        Args:
            spec: Query specification dictionary or JSON string
        
        Returns:
            List of documents in Firestore format, or error dictionary
        """
        try:
            # Handle case where spec might be passed in kwargs
            if spec is None and 'spec' in kwargs:
                spec = kwargs['spec']
            elif spec is None:
                raise ValueError("spec parameter is required")
            
            # Parse spec if it's a string (JSON)
            if isinstance(spec, str):
                try:
                    spec = json.loads(spec)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON spec: {spec}")
                    raise ValueError(f"spec must be valid JSON: {str(e)}")
            
            # Get collection name
            collection_name = spec.get("collection")
            if not collection_name:
                raise ValueError("spec.collection is required")
            
            # Map internal table names to Firestore collection names
            collection_mapping = {
                "project_config": "projects",
                "tasks": "tasks",
                "user_config": "users",
                "users": "users",
                "projects": "projects",
                "updates": "updates",
                "update_summaries": "update_summaries",
                "teams": "teams",
                "teamDetails": "teamDetails",
                "integrations": "integrations"
            }
            
            firestore_collection = collection_mapping.get(collection_name, collection_name)
            
            # Compile query spec
            filters, order_by, limit = self._compile_query_spec(spec, firestore_collection)
            
            # Execute query
            start_time = datetime.datetime.now()
            results = self._execute_firestore_query(firestore_collection, filters, order_by, limit)
            duration_ms = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
            
            logger.info(f"Firestore query completed in {duration_ms}ms, returned {len(results)} results from collection {firestore_collection}")
            
            return results
        
        except Exception as e:
            logger.error(f"Firestore tool failed: domain={self.domain}, error={str(e)}")
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            return {"error": error_msg}

