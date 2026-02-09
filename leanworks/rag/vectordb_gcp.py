"""
GCP Vector Search implementation (Vertex AI Vector Search).

Provides hybrid_search API for Leanworks integration.
"""

import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

try:
    from google.cloud import vectorsearch_v1beta
    from google.api_core import exceptions as gcp_exceptions
    from google.protobuf import field_mask_pb2, struct_pb2
    from google.protobuf.json_format import MessageToDict, ParseDict
    from google.oauth2 import service_account
except ImportError as e:
    import warnings

    warnings.warn(
        f"google-cloud-vectorsearch import failed: {e}. "
        "Install with: pip install google-cloud-vectorsearch"
    )
    vectorsearch_v1beta = None
    gcp_exceptions = None
    field_mask_pb2 = None
    MessageToDict = None
    ParseDict = None
    struct_pb2 = None
    service_account = None

logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_IMAGE_EMBEDDING_DIMENSION = 1408

# GCP Vector Search settings
GCP_VECTOR_SEARCH_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "leanworks-474204")
GCP_VECTOR_SEARCH_LOCATION = os.getenv("GCP_VECTOR_SEARCH_LOCATION", "us-central1")
GCP_VECTOR_SEARCH_BATCH_SIZE = int(os.getenv("GCP_VECTOR_SEARCH_BATCH_SIZE", "100"))
UPSERT_BATCH_SIZE = 100  # Batch size for upsert operations (e.g. in rag_storage)
GCP_VECTOR_SEARCH_REQUEST_TIMEOUT = int(os.getenv("GCP_VECTOR_SEARCH_REQUEST_TIMEOUT", "60"))

# Hybrid search weights (used only for RRF combine weights)
GCP_VECTOR_SEARCH_HYBRID_DENSE_WEIGHT = float(os.getenv("GCP_VECTOR_SEARCH_HYBRID_DENSE_WEIGHT", "0.6"))
GCP_VECTOR_SEARCH_HYBRID_TEXT_WEIGHT = float(os.getenv("GCP_VECTOR_SEARCH_HYBRID_TEXT_WEIGHT", "0.4"))

# Collection names
GCP_VECTOR_SEARCH_COLLECTION_TEXT = os.getenv("GCP_VECTOR_SEARCH_COLLECTION_TEXT", "leanworks-multimodal")
GCP_VECTOR_SEARCH_COLLECTION_IMAGE = os.getenv("GCP_VECTOR_SEARCH_COLLECTION_IMAGE", "leanworks-image")  # Reserved for future use - current architecture uses multimodal text collection
GCP_VECTOR_SEARCH_COLLECTION_CODES = os.getenv("GCP_VECTOR_SEARCH_COLLECTION_CODES", "leanworks-codes")
GCP_VECTOR_SEARCH_COLLECTION_TOOL_RESPONSES = os.getenv("GCP_VECTOR_SEARCH_COLLECTION_TOOL_RESPONSES", "leanworks-tool-responses")

# Error handling settings
MAX_RETRIES = 3
RETRY_DELAY = 2
EXPONENTIAL_BACKOFF_MULTIPLIER = 2


def retry_on_error(max_retries: int = MAX_RETRIES, delay: int = RETRY_DELAY):
    """Decorator for retrying operations on errors."""
    def decorator(func):
        def wrapper(instance, *args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(instance, *args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1 and instance._is_retryable_error(e):
                        wait_time = delay * (EXPONENTIAL_BACKOFF_MULTIPLIER ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed: {e}. Not retryable, failing immediately.")
                        else:
                            logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                        raise e
            raise last_error
        return wrapper
    return decorator


class GCPVectorSearchIndex:
    """
    GCP Vector Search implementation.

    Provides hybrid_search API for Leanworks integration.
    """

    backend = "gcp"

    def __init__(
        self,
        gcp_credential_path: Optional[str],
        embedding_model_client,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        """
        Initialize GCP Vector Search client.

        Args:
            gcp_credential_path: Path to GCP service account credentials JSON
            embedding_model_client: Embedding client for generating vectors
            chunk_size: Chunk size in tokens for local chunking
            chunk_overlap: Overlap in tokens between chunks
        """
        self.gcp_credential_path = gcp_credential_path
        self.embedding_model_client = embedding_model_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.project_id = self._resolve_project_id()
        self._tokenizer = tiktoken.get_encoding("o200k_base")
        self._unavailable_collections = set()

        self._init_gcp_clients()

        self.text_collection = None
        self.image_collection = None
        self.code_collection = None

        logger.info("GCPVectorSearchIndex initialized successfully")

    def _resolve_project_id(self) -> str:
        """Resolve the GCP project ID based on env override or credential file."""
        env_project_id = os.getenv("GCP_PROJECT_ID")
        if env_project_id:
            return env_project_id

        if self.gcp_credential_path and os.path.exists(self.gcp_credential_path):
            try:
                with open(self.gcp_credential_path, "r") as f_in:
                    creds = json.load(f_in)
                project_id = creds.get("project_id")
                if project_id:
                    return project_id
            except Exception as e:
                logger.warning(f"Failed to read project_id from credentials: {e}")

        return GCP_VECTOR_SEARCH_PROJECT_ID

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Non-retryable errors (fail immediately)
        non_retryable_conditions = [
            "not found" in error_str,
            "404" in error_str,
            "does not exist" in error_str,
        ]

        if any(non_retryable_conditions):
            return False

        # Retryable errors
        retryable_conditions = [
            "timeout" in error_str,
            "timeout" in error_type,
            "connection" in error_str,
            "temporary" in error_str,
            "service unavailable" in error_str,
            "internal error" in error_str,
            "resource exhausted" in error_str,
            "rate limit" in error_str,
            "quota exceeded" in error_str,
            "unavailable" in error_str,
            "deadline exceeded" in error_str,
        ]

        return any(retryable_conditions)

    def _is_not_found_error(self, error: Exception) -> bool:
        """Check if an error indicates a missing collection/resource."""
        error_str = str(error).lower()
        not_found_conditions = [
            "not found" in error_str,
            "404" in error_str,
            "does not exist" in error_str,
        ]
        return any(not_found_conditions)


    def _format_exception_details(self, exception: Exception) -> str:
        """Format exception details for better error reporting."""
        error_type = type(exception).__name__
        error_message = str(exception) if str(exception) else "No error message available"

        details = []
        for attr in ["status_code", "code", "details", "message"]:
            if hasattr(exception, attr):
                value = getattr(exception, attr)
                if value:
                    details.append(f"{attr}={value}")

        if details:
            return f"{error_type}: {error_message} ({', '.join(details)})"
        return f"{error_type}: {error_message}"

    def _struct_to_dict(self, struct_obj: Any) -> Dict[str, Any]:
        """Convert protobuf Struct to a Python dict."""
        if struct_obj is None:
            return {}
        if MessageToDict is None:
            return {}
        if hasattr(struct_obj, "items"):
            return dict(struct_obj)
        return MessageToDict(struct_obj, preserving_proto_field_name=True)

    def _dict_to_struct(self, data: Optional[Dict[str, Any]]) -> Any:
        """Convert a dict to protobuf Struct."""
        if not data:
            return struct_pb2.Struct() if struct_pb2 else None
        if ParseDict is None or struct_pb2 is None:
            return None
        return ParseDict(data, struct_pb2.Struct())

    def _normalize_data_object_id(self, raw_id: str) -> str:
        """Normalize an ID to RFC1035 format required by Vector Search."""
        if not raw_id:
            raw_id = str(uuid.uuid4())

        safe = re.sub(r"[^a-z0-9-]+", "-", raw_id.lower()).strip("-")
        if not safe or not safe[0].isalpha():
            safe = f"d{safe}"

        if len(safe) > 63:
            digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:10]
            keep = max(1, 63 - len(digest) - 1)
            safe = f"{safe[:keep].rstrip('-')}-{digest}"

        safe = safe.strip("-")
        if not safe:
            safe = f"d{hashlib.sha1(raw_id.encode('utf-8')).hexdigest()[:10]}"

        return safe[:63]

    def _matches_filter(self, data: Dict[str, Any], filter_expr: Optional[Dict[str, Any]]) -> bool:
        """Evaluate a simple filter expression against data."""
        if not filter_expr:
            return True

        if "$and" in filter_expr:
            return all(self._matches_filter(data, clause) for clause in filter_expr.get("$and", []))
        if "$or" in filter_expr:
            return any(self._matches_filter(data, clause) for clause in filter_expr.get("$or", []))

        for field, condition in filter_expr.items():
            if field in ("$and", "$or"):
                continue
            value = data.get(field)
            if isinstance(condition, dict):
                for op, expected in condition.items():
                    if op == "$eq" and value != expected:
                        return False
                    if op == "$ne" and value == expected:
                        return False
                    if op == "$in" and value not in expected:
                        return False
                    if op == "$nin" and value in expected:
                        return False
                    if op == "$gt" and (value is None or value <= expected):
                        return False
                    if op == "$gte" and (value is None or value < expected):
                        return False
                    if op == "$lt" and (value is None or value >= expected):
                        return False
                    if op == "$lte" and (value is None or value > expected):
                        return False
            else:
                if value != condition:
                    return False

        return True

    def _combine_filters(self, filters: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Combine multiple filter expressions with $and."""
        clauses = [f for f in filters if f]
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _flatten_logical_filter(self, filter_expr: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested $and/$or blocks where possible."""
        if not isinstance(filter_expr, dict):
            return filter_expr
        if "$and" in filter_expr:
            clauses = []
            for clause in filter_expr.get("$and", []):
                clause = self._flatten_logical_filter(clause)
                if isinstance(clause, dict) and "$and" in clause and len(clause) == 1:
                    clauses.extend(clause.get("$and", []))
                else:
                    clauses.append(clause)
            return {"$and": clauses}
        if "$or" in filter_expr:
            clauses = []
            for clause in filter_expr.get("$or", []):
                clause = self._flatten_logical_filter(clause)
                if isinstance(clause, dict) and "$or" in clause and len(clause) == 1:
                    clauses.extend(clause.get("$or", []))
                else:
                    clauses.append(clause)
            return {"$or": clauses}
        return filter_expr

    def _logical_depth(self, filter_expr: Dict[str, Any]) -> int:
        """Compute logical operator nesting depth."""
        if not isinstance(filter_expr, dict):
            return 0
        if "$and" in filter_expr:
            clauses = filter_expr.get("$and", [])
            return 1 + max((self._logical_depth(c) for c in clauses), default=0)
        if "$or" in filter_expr:
            clauses = filter_expr.get("$or", [])
            return 1 + max((self._logical_depth(c) for c in clauses), default=0)
        return 0

    def _collect_leaf_filters(self, filter_expr: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect non-logical filter clauses from a nested filter."""
        if not isinstance(filter_expr, dict):
            return []
        if "$and" in filter_expr or "$or" in filter_expr:
            op = "$and" if "$and" in filter_expr else "$or"
            leaves: List[Dict[str, Any]] = []
            for clause in filter_expr.get(op, []):
                leaves.extend(self._collect_leaf_filters(clause))
            return leaves
        return [filter_expr]

    def _normalize_filter_expression(self, filter_expr: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Normalize filter expression to comply with Vertex filter constraints."""
        if not filter_expr:
            return None
        normalized = self._flatten_logical_filter(filter_expr)
        depth = self._logical_depth(normalized)
        if depth <= 2:
            return normalized
        logger.warning(
            "Filter expression nesting depth %s exceeds Vertex limits; "
            "flattening to top-level logical clauses.",
            depth,
        )
        root_op = None
        if isinstance(normalized, dict):
            if "$and" in normalized:
                root_op = "$and"
            elif "$or" in normalized:
                root_op = "$or"
        leaf_filters = self._collect_leaf_filters(normalized)
        if not leaf_filters:
            return None
        return {root_op or "$and": leaf_filters}

    def _init_gcp_clients(self):
        """Initialize GCP Vector Search clients."""
        if vectorsearch_v1beta is None:
            raise ImportError(
                "google-cloud-vectorsearch not installed. "
                "Install with: pip install google-cloud-vectorsearch"
            )
        if service_account is None:
            raise ImportError("google-auth not installed. Install with: pip install google-auth")

        if self.gcp_credential_path and os.path.exists(self.gcp_credential_path):
            credentials = service_account.Credentials.from_service_account_file(
                self.gcp_credential_path
            )
        else:
            credentials = None

        self.vector_search_client = vectorsearch_v1beta.VectorSearchServiceClient(
            credentials=credentials
        )
        self.data_object_client = vectorsearch_v1beta.DataObjectServiceClient(
            credentials=credentials
        )
        self.data_object_search_client = vectorsearch_v1beta.DataObjectSearchServiceClient(
            credentials=credentials
        )

        logger.info("GCP Vector Search clients initialized")

    @retry_on_error()
    def ensure_hybrid_index(self, collection_name: str, create_if_missing: bool = True):
        """Ensure collection and associated index exist."""
        collection_path = self._get_collection_path(collection_name)

        try:
            collection = self.vector_search_client.get_collection(name=collection_path)
            logger.info(f"Found existing collection: {collection_name}")
            return collection
        except Exception as e:
            if not create_if_missing:
                raise e

            logger.info(f"Creating new collection: {collection_name}")
            return self._create_collection(collection_name)

    def _build_collection_schemas(self, collection_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build data and vector schemas for a collection."""
        data_schema = {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "document_id": {"type": "string"},
                "chunk_text": {"type": "string"},
                "org_slug": {"type": "string"},
                "type": {"type": "string"},
                "tool_name": {"type": "string"},
                "data_source": {"type": "string"},
                "vector_type": {"type": "string"},
                "timestamp": {"type": "number"},
                "metadata_json": {"type": "string"},
            },
        }

        # Multimodal text collection uses named text/image vector fields
        if collection_name == GCP_VECTOR_SEARCH_COLLECTION_TEXT:
            vector_schema = {
                "text_embedding": vectorsearch_v1beta.VectorField(
                    dense_vector=vectorsearch_v1beta.DenseVectorField(
                        dimensions=DEFAULT_EMBEDDING_DIMENSION
                    )
                ),
                "image_embedding": vectorsearch_v1beta.VectorField(
                    dense_vector=vectorsearch_v1beta.DenseVectorField(
                        dimensions=DEFAULT_IMAGE_EMBEDDING_DIMENSION
                    )
                ),
            }
        else:
            vector_dimensions = self._get_vector_dimensions_for_collection(collection_name)
            vector_schema = {
                "embedding": vectorsearch_v1beta.VectorField(
                    dense_vector=vectorsearch_v1beta.DenseVectorField(
                        dimensions=vector_dimensions
                    )
                )
            }

        return data_schema, vector_schema

    def _create_collection(self, collection_name: str):
        """Create a new collection with appropriate schema."""
        parent = f"projects/{self.project_id}/locations/{GCP_VECTOR_SEARCH_LOCATION}"
        data_schema, vector_schema = self._build_collection_schemas(collection_name)

        collection = vectorsearch_v1beta.Collection(
            display_name=collection_name,
            description=f"Vector search collection for {collection_name}",
            data_schema=data_schema,
            vector_schema=vector_schema,
        )

        request = vectorsearch_v1beta.CreateCollectionRequest(
            parent=parent,
            collection=collection,
            collection_id=collection_name,
        )

        operation = self.vector_search_client.create_collection(request=request)
        collection = operation.result()

        logger.info(f"Created collection: {collection_name}")
        return collection

    def _get_embeddings_batch(self, texts: List[str], task_type: str) -> List[Any]:
        """Get embeddings using available batch API on the embedding client."""
        if hasattr(self.embedding_model_client, "get_embeddings_batch_concurrent"):
            return self.embedding_model_client.get_embeddings_batch_concurrent(texts, task_type=task_type)
        if hasattr(self.embedding_model_client, "get_embeddings_batch"):
            return self.embedding_model_client.get_embeddings_batch(texts, task_type=task_type)
        return [self.embedding_model_client.get_embedding(text, task_type=task_type) for text in texts]

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text for chunking."""
        try:
            return self._tokenizer.encode(text)
        except Exception:
            return list(text)

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap based on token count."""
        if not text:
            return []

        if hasattr(self.embedding_model_client, "count_tokens"):
            try:
                token_count = self.embedding_model_client.count_tokens(text)
            except Exception:
                token_count = len(self._tokenize(text))
        else:
            token_count = len(self._tokenize(text))

        if token_count <= self.chunk_size:
            return [text]

        tokens = self._tokenize(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            try:
                chunk_text = self._tokenizer.decode(chunk_tokens)
            except Exception:
                chunk_text = text[start:end]
            chunks.append(chunk_text)

            if end == len(tokens):
                break
            start = max(0, end - self.chunk_overlap)

        return chunks

    def _prepare_chunks(self, documents) -> Tuple[List[str], List[Dict]]:
        """Prepare chunks and metadata from documents."""
        all_chunks: List[str] = []
        chunk_metadata_list: List[Dict[str, Any]] = []

        for doc in documents:
            if hasattr(doc, "page_content"):
                content = doc.page_content
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
            else:
                content = str(doc)
                metadata = {}

            document_id = metadata.get("id", str(uuid.uuid4()))
            chunks = self._chunk_text(content)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_number": i,
                    "chunk_text": chunk,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                })

                all_chunks.append(chunk)
                chunk_metadata_list.append(chunk_metadata)

        logger.info(f"Processing {len(all_chunks)} chunks for embedding generation")
        return all_chunks, chunk_metadata_list

    def _create_gcp_data_object(
        self,
        chunk_id: str,
        dense_embedding: List[float],
        metadata: Dict,
        org_slug: str,
    ) -> Dict[str, Any]:
        """Create GCP DataObject from chunk data."""
        data_source = metadata.get("data_source", "unknown")
        vector_type = metadata.get("vector_type", "text")

        allowed_fields = {
            "chunk_id",
            "document_id",
            "chunk_text",
            "org_slug",
            "type",
            "tool_name",
            "data_source",
            "vector_type",
            "timestamp",
        }

        data: Dict[str, Any] = {}
        for field in allowed_fields:
            if field in metadata:
                data[field] = metadata[field]

        data["chunk_id"] = chunk_id
        data["org_slug"] = org_slug
        data["data_source"] = data_source
        data["vector_type"] = vector_type
        data["type"] = metadata.get("type", "document")

        if "timestamp" in data and isinstance(data["timestamp"], str):
            try:
                data["timestamp"] = float(data["timestamp"])
            except (ValueError, TypeError):
                data.pop("timestamp", None)

        data["metadata_json"] = json.dumps(metadata or {}, ensure_ascii=True, default=str)

        data_object_id = self._normalize_data_object_id(chunk_id)
        collection_name = self._determine_collection_name_from_metadata(metadata)
        if collection_name == GCP_VECTOR_SEARCH_COLLECTION_TEXT:
            vectors = {
                "text_embedding": vectorsearch_v1beta.Vector(
                    dense=vectorsearch_v1beta.DenseVector(values=dense_embedding)
                )
            }
        else:
            vectors = {
                "embedding": vectorsearch_v1beta.Vector(
                    dense=vectorsearch_v1beta.DenseVector(values=dense_embedding)
                )
            }

        data_object = vectorsearch_v1beta.DataObject(
            data=data,
            vectors=vectors,
        )

        return {
            "id": data_object_id,
            "source_id": chunk_id,
            "data_object": data_object,
            "data": data,
        }

    def _create_vectors_concurrent(
        self,
        all_chunks: List[str],
        chunk_metadata_list: List[Dict],
        org_slug: str,
    ) -> List[Dict]:
        """Create GCP Data Objects from chunks (dense vectors only)."""
        logger.info(f"Generating dense embeddings for {len(all_chunks)} chunks...")

        dense_embeddings = self._get_embeddings_batch(
            all_chunks,
            task_type="RETRIEVAL_DOCUMENT",
        )

        data_objects = []
        for metadata, dense_embedding in zip(chunk_metadata_list, dense_embeddings):
            chunk_id = metadata["chunk_id"]
            if hasattr(dense_embedding, "tolist"):
                dense_embedding = dense_embedding.tolist()

            data_object = self._create_gcp_data_object(chunk_id, dense_embedding, metadata, org_slug)
            data_objects.append(data_object)

        return data_objects

    def upsert_documents_hybrid(self, documents, org_slug: str, retries: int = 3, delay: int = 2):
        """Upsert documents to GCP Vector Search with org filtering."""
        documents = [doc for doc in documents if doc is not None]
        if not documents:
            logger.warning("No documents to upsert")
            return None

        for attempt in range(retries):
            try:
                logger.info(
                    f"Preparing chunks for {len(documents)} documents (attempt {attempt + 1}/{retries})..."
                )
                all_chunks, chunk_metadata_list = self._prepare_chunks(documents)
                logger.info(f"Prepared {len(all_chunks)} chunks from {len(documents)} documents")

                data_objects = self._create_vectors_concurrent(
                    all_chunks, chunk_metadata_list, org_slug
                )

                collection_name = self._determine_collection_name(data_objects)
                self.ensure_hybrid_index(collection_name)
                self._upsert_data_objects(collection_name, data_objects)

                logger.info(f"Successfully upserted {len(data_objects)} data objects")
                return data_objects
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay} seconds..."
                    )
                    logger.error(e)
                    time.sleep(delay)
                else:
                    logger.error("Max retries reached, upsert operation failed.")
                    raise e

    def upsert_chunks_with_metadata(
        self,
        chunks: List[str],
        chunk_metadata_list: List[Dict[str, Any]],
        org_slug: str,
    ) -> List[Dict[str, Any]]:
        """Upsert pre-chunked content with metadata."""
        if not chunks or not chunk_metadata_list:
            logger.warning("No chunks provided for upsert")
            return []

        data_objects = self._create_vectors_concurrent(
            chunks,
            chunk_metadata_list,
            org_slug,
        )

        collection_name = self._determine_collection_name(data_objects)
        self.ensure_hybrid_index(collection_name)
        self._upsert_data_objects(collection_name, data_objects)
        return data_objects

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        collection_scope: str = "all",  # "all", "docs", "codes", "tool_responses"
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search.

        Returns list of dicts with id, metadata, combined_score.
        """
        query_embedding = self.embedding_model_client.get_embedding(
            query, task_type="RETRIEVAL_QUERY"
        )
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        filter_expr = self._build_query_restricts(namespace, filter)
        local_filter_expr = filter_expr
        if namespace and not namespace.endswith("_tool_responses"):
            local_filter_expr = self._combine_filters(
                [filter_expr, {"type": {"$ne": "tool_response"}}]
            )

        # Determine which collections to search based on collection_scope
        if collection_scope == "docs":
            # Docs scope includes text and images (multimodal collection)
            collections_to_search = [GCP_VECTOR_SEARCH_COLLECTION_TEXT]
        elif collection_scope == "codes":
            collections_to_search = [GCP_VECTOR_SEARCH_COLLECTION_CODES]
        elif collection_scope == "tool_responses":
            collections_to_search = [GCP_VECTOR_SEARCH_COLLECTION_TOOL_RESPONSES]
        elif collection_scope == "all":
            collections_to_search = [
                GCP_VECTOR_SEARCH_COLLECTION_TEXT,  # includes both text and images
                GCP_VECTOR_SEARCH_COLLECTION_CODES,
                GCP_VECTOR_SEARCH_COLLECTION_TOOL_RESPONSES,
            ]
        else:
            # Default to all collections
            collections_to_search = [
                GCP_VECTOR_SEARCH_COLLECTION_TEXT,
                GCP_VECTOR_SEARCH_COLLECTION_CODES,
                GCP_VECTOR_SEARCH_COLLECTION_TOOL_RESPONSES,
            ]

        output_fields = {
            "data_fields": [
                "chunk_id",
                "document_id",
                "chunk_text",
                "org_slug",
                "type",
                "data_source",
                "vector_type",
                "timestamp",
                "metadata_json",
            ]
        }

        search_requests = [
            {
                "vector_search": {
                    "vector": {"values": query_embedding},
                    "search_field": self._get_vector_search_field(
                        collections_to_search[0] if len(collections_to_search) == 1 else GCP_VECTOR_SEARCH_COLLECTION_TEXT
                    ),
                    "top_k": top_k * 2,
                    "filter": filter_expr,
                    "output_fields": output_fields,
                }
            },
            {
                "text_search": {
                    "search_text": query,
                    "data_field_names": ["chunk_text"],
                    "top_k": top_k * 2,
                    "output_fields": output_fields,
                }
            },
        ]

        if len(collections_to_search) == 1:
            # Single collection search
            collection_name = collections_to_search[0]
            if collection_name in self._unavailable_collections:
                return []
            try:
                response = self._execute_batch_search(
                    collection_name,
                    search_requests,
                    combine={
                        "ranker": {
                            "rrf": {
                                "weights": [alpha, 1.0 - alpha],
                            }
                        },
                        "top_k": top_k,
                        "output_fields": output_fields,
                    },
                )
                return self._format_combined_results(response, local_filter_expr)
            except Exception as e:
                if self._is_not_found_error(e):
                    self._unavailable_collections.add(collection_name)
                    logger.warning(f"Failed to search collection {collection_name}: {str(e)}")
                    return []
                raise
        else:
            # Multi-collection parallel search
            return self._search_multiple_collections(
                query_embedding, query, top_k, alpha, filter_expr, collections_to_search
            )

    def _search_multiple_collections(
        self,
        query_embedding: List[float],
        query: str,
        top_k: int,
        alpha: float,
        filter_expr: Optional[Dict],
        collection_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Search multiple collections in parallel and merge with RRF."""
        from concurrent.futures import ThreadPoolExecutor

        def search_collection(collection_name: str) -> List[Dict[str, Any]]:
            """Search a single collection and return formatted results."""
            if collection_name in self._unavailable_collections:
                return []
            try:
                vector_field = self._get_vector_search_field(collection_name)
                output_fields = {
                    "data_fields": [
                        "chunk_id",
                        "document_id",
                        "chunk_text",
                        "org_slug",
                        "type",
                        "data_source",
                        "vector_type",
                        "timestamp",
                        "metadata_json",
                    ]
                }

                search_requests = [
                    {
                        "vector_search": {
                            "vector": {"values": query_embedding},
                            "search_field": vector_field,
                            "top_k": top_k * 2,
                            "filter": filter_expr,
                            "output_fields": output_fields,
                        }
                    },
                    {
                        "text_search": {
                            "search_text": query,
                            "data_field_names": ["chunk_text"],
                            "top_k": top_k * 2,
                            "output_fields": output_fields,
                        }
                    },
                ]

                response = self._execute_batch_search(
                    collection_name,
                    search_requests,
                    combine={
                        "ranker": {
                            "rrf": {
                                "weights": [alpha, 1.0 - alpha],
                            }
                        },
                        "top_k": top_k,
                        "output_fields": output_fields,
                    },
                )

                results = self._format_combined_results(response, filter_expr)
                # Add collection metadata to results
                for result in results:
                    result["_collection"] = collection_name
                return results

            except Exception as e:
                if self._is_not_found_error(e):
                    if collection_name not in self._unavailable_collections:
                        logger.warning(f"Failed to search collection {collection_name}: {str(e)}")
                        self._unavailable_collections.add(collection_name)
                else:
                    logger.warning(f"Failed to search collection {collection_name}: {str(e)}")
                return []

        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=len(collection_names)) as executor:
            futures = {executor.submit(search_collection, col): col
                       for col in collection_names}

            all_results = []
            for future in futures:
                try:
                    results = future.result(timeout=30)  # 30 second timeout per collection
                    all_results.extend(results)
                except Exception as e:
                    collection_name = futures[future]
                    logger.error(f"Error getting results from collection {collection_name}: {str(e)}")

        # Apply RRF merging across all collections
        merged_results = self._merge_multi_collection_results(all_results, top_k)

        logger.info(f"Multi-collection search completed: {len(collection_names)} collections, {len(merged_results)} final results")
        return merged_results

    def _merge_multi_collection_results(
        self,
        all_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Merge results from multiple collections using RRF."""
        if not all_results:
            return []

        # Group by document ID and apply collection-level RRF
        doc_groups = {}
        for result in all_results:
            doc_id = result.get("id")
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(result)

        # For each document, keep the highest scoring instance across collections
        merged_by_doc = []
        for doc_id, results in doc_groups.items():
            # Sort by score within the document group and take the best
            best_result = max(results, key=lambda x: x.get("combined_score", 0))
            merged_by_doc.append(best_result)

        # Sort by combined_score and limit to top_k
        merged_by_doc.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        return merged_by_doc[:top_k]

    @retry_on_error()
    def _execute_batch_search(
        self,
        collection_name: str,
        search_requests: List[Dict],
        combine: Optional[Dict] = None,
    ) -> Dict:
        """Execute batch search request with retry logic."""
        collection_path = self._get_collection_path(collection_name)
        request = vectorsearch_v1beta.BatchSearchDataObjectsRequest(
            parent=collection_path,
            searches=search_requests,
            combine=combine,
        )

        response = self.data_object_search_client.batch_search_data_objects(
            request=request,
            timeout=GCP_VECTOR_SEARCH_REQUEST_TIMEOUT,
        )
        return response

    def _format_combined_results(
        self,
        gcp_response,
        filter_expr: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Format combined results with fallback scores."""
        if gcp_response is None:
            return []

        results_list = list(getattr(gcp_response, "results", []) or [])
        combined_results = results_list[0].results if results_list else []

        formatted: List[Dict[str, Any]] = []
        for rank, result in enumerate(combined_results):
            data_object = getattr(result, "data_object", None)
            if data_object is None:
                continue

            data = self._struct_to_dict(getattr(data_object, "data", None))
            if not self._matches_filter(data, filter_expr):
                continue

            result_id = (
                data.get("chunk_id")
                or getattr(data_object, "data_object_id", None)
                or (getattr(data_object, "name", "").split("/")[-1] if getattr(data_object, "name", "") else None)
            )
            if not result_id:
                continue

            metadata = dict(data)
            metadata_json = metadata.pop("metadata_json", None)
            if metadata_json:
                try:
                    extra_metadata = json.loads(metadata_json)
                    if isinstance(extra_metadata, dict):
                        for key, value in extra_metadata.items():
                            metadata.setdefault(key, value)
                except (TypeError, ValueError):
                    metadata["metadata_json"] = metadata_json

            if getattr(data_object, "data_object_id", None):
                metadata.setdefault("data_object_id", data_object.data_object_id)

            raw_score = getattr(result, "score", None)
            if raw_score is None and hasattr(result, "distance"):
                raw_score = getattr(result, "distance", None)

            if isinstance(raw_score, (int, float)):
                if hasattr(result, "distance"):
                    combined_score = 1.0 / (1.0 + float(raw_score))
                else:
                    combined_score = float(raw_score)
            else:
                combined_score = 1.0 / (rank + 1)

            formatted.append({
                "id": result_id,
                "metadata": metadata,
                "combined_score": combined_score,
            })

        return formatted

    def _build_query_restricts(
        self, namespace: str, filter_dict: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Build filter expression for query filtering."""
        filters: List[Dict[str, Any]] = []

        if namespace:
            if namespace.endswith("_tool_responses"):
                org_slug = namespace.replace("_tool_responses", "")
                filters.append({"org_slug": {"$eq": org_slug}})
                filters.append({"type": {"$eq": "tool_response"}})
            else:
                filters.append({"org_slug": {"$eq": namespace}})

        if filter_dict:
            filters.append(self._convert_metadata_filter_to_gcp(filter_dict))

        if not filters:
            return None
        combined = filters[0] if len(filters) == 1 else {"$and": filters}
        normalized = self._normalize_filter_expression(combined)
        logger.info("Vertex filter expression: %s", normalized)
        return normalized

    def _convert_metadata_filter_to_gcp(self, filter_dict: Dict) -> Dict[str, Any]:
        """Convert metadata filter dict to GCP filter expression."""
        if not filter_dict:
            return {}

        if "$and" in filter_dict or "$or" in filter_dict:
            return filter_dict

        clauses: List[Dict[str, Any]] = []
        for field, condition in filter_dict.items():
            if isinstance(condition, dict):
                clauses.append({field: condition})
            else:
                clauses.append({field: {"$eq": condition}})

        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @retry_on_error()
    def _upsert_data_objects(self, collection_name: str, data_objects: List[Dict]):
        """Upsert data objects to GCP Vector Search with batching and error handling."""
        collection_path = self._get_collection_path(collection_name)
        batch_size = min(GCP_VECTOR_SEARCH_BATCH_SIZE, len(data_objects))

        for i in range(0, len(data_objects), batch_size):
            batch = data_objects[i:i + batch_size]

            requests = [
                vectorsearch_v1beta.CreateDataObjectRequest(
                    parent=collection_path,
                    data_object_id=obj["id"],
                    data_object=obj["data_object"],
                )
                for obj in batch
            ]
            request = vectorsearch_v1beta.BatchCreateDataObjectsRequest(
                parent=collection_path,
                requests=requests,
            )

            try:
                self.data_object_client.batch_create_data_objects(
                    request=request,
                    timeout=GCP_VECTOR_SEARCH_REQUEST_TIMEOUT,
                )
                logger.debug(f"Upserted batch of {len(batch)} data objects (created)")
            except Exception as batch_error:
                if gcp_exceptions and isinstance(batch_error, gcp_exceptions.AlreadyExists):
                    self._upsert_data_objects_individual(collection_path, batch)
                    continue

                if self._is_retryable_error(batch_error):
                    logger.warning(
                        f"Batch upsert failed (retryable): {self._format_exception_details(batch_error)}"
                    )
                    raise batch_error
                logger.error(
                    f"Batch upsert failed (non-retryable): {self._format_exception_details(batch_error)}"
                )
                raise batch_error

    def _upsert_data_objects_individual(self, collection_path: str, batch: List[Dict]):
        """Fallback upsert path for batches that include existing IDs."""
        for obj in batch:
            try:
                self.data_object_client.create_data_object(
                    request=vectorsearch_v1beta.CreateDataObjectRequest(
                        parent=collection_path,
                        data_object_id=obj["id"],
                        data_object=obj["data_object"],
                    ),
                    timeout=GCP_VECTOR_SEARCH_REQUEST_TIMEOUT,
                )
            except Exception as create_error:
                if gcp_exceptions and isinstance(create_error, gcp_exceptions.AlreadyExists):
                    update_request = vectorsearch_v1beta.UpdateDataObjectRequest(
                        data_object=vectorsearch_v1beta.DataObject(
                            name=f"{collection_path}/dataObjects/{obj['id']}",
                            data=obj["data_object"].data,
                            vectors=obj["data_object"].vectors,
                        ),
                        update_mask=field_mask_pb2.FieldMask(paths=["data", "vectors"]),
                    )
                    self.data_object_client.update_data_object(
                        request=update_request,
                        timeout=GCP_VECTOR_SEARCH_REQUEST_TIMEOUT,
                    )
                else:
                    raise create_error

    def _determine_collection_name(self, data_objects: List[Dict]) -> str:
        """Determine which collection to use based on data object content."""
        if not data_objects:
            return GCP_VECTOR_SEARCH_COLLECTION_TEXT

        first_obj = data_objects[0]
        data_source = first_obj.get('data', {}).get('data_source', '')

        # Route code sources to codes collection
        if data_source in ["github_codes", "gitlab_codes"]:
            return GCP_VECTOR_SEARCH_COLLECTION_CODES

        # Route tool responses to dedicated collection
        if first_obj.get('data', {}).get('type') == 'tool_response':
            return GCP_VECTOR_SEARCH_COLLECTION_TOOL_RESPONSES

        # Route ALL other sources (including leanworks_image) to multimodal text collection
        # The multimodal collection handles both text and image embeddings
        return GCP_VECTOR_SEARCH_COLLECTION_TEXT

    def _determine_collection_name_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Determine collection name based on metadata fields."""
        data_source = metadata.get("data_source", "")

        if data_source in ["github_codes", "gitlab_codes"]:
            return GCP_VECTOR_SEARCH_COLLECTION_CODES

        if metadata.get("type") == "tool_response":
            return GCP_VECTOR_SEARCH_COLLECTION_TOOL_RESPONSES

        return GCP_VECTOR_SEARCH_COLLECTION_TEXT

    def _determine_collection_for_namespace(self, namespace: str) -> str:
        """Determine collection name based on namespace."""
        return GCP_VECTOR_SEARCH_COLLECTION_TEXT

    def _get_vector_dimensions_for_collection(self, collection_name: str) -> int:
        """Resolve vector dimensions for a collection."""
        if "image" in collection_name:
            return DEFAULT_IMAGE_EMBEDDING_DIMENSION
        return DEFAULT_EMBEDDING_DIMENSION

    def _get_vector_search_field(self, collection_name: str) -> str:
        """Resolve vector field name for search."""
        if collection_name == GCP_VECTOR_SEARCH_COLLECTION_TEXT:
            return "text_embedding"
        return "embedding"

    def _get_collection_path(self, collection_name: str) -> str:
        """Get full collection path."""
        return (
            f"projects/{self.project_id}/"
            f"locations/{GCP_VECTOR_SEARCH_LOCATION}/"
            f"collections/{collection_name}"
        )

    def delete_by_filter(
        self,
        filter_expr: Optional[Dict[str, Any]],
        collection_names: Optional[List[str]] = None,
    ) -> int:
        """Delete data objects matching filter from one or more collections."""
        total_deleted = 0
        if collection_names is None:
            collection_names = [
                GCP_VECTOR_SEARCH_COLLECTION_TEXT,
                GCP_VECTOR_SEARCH_COLLECTION_IMAGE,
                GCP_VECTOR_SEARCH_COLLECTION_CODES,
                GCP_VECTOR_SEARCH_COLLECTION_TOOL_RESPONSES,
            ]

        for collection_name in collection_names:
            collection_path = self._get_collection_path(collection_name)
            try:
                self.vector_search_client.get_collection(name=collection_path)
            except Exception as e:
                if gcp_exceptions and isinstance(e, gcp_exceptions.NotFound):
                    continue
                raise

            total_deleted += self._delete_data_objects_by_filter(collection_path, filter_expr)

        return total_deleted

    def _delete_data_objects_by_filter(
        self,
        collection_path: str,
        filter_expr: Optional[Dict[str, Any]],
    ) -> int:
        """Delete data objects matching the filter from a collection."""
        total_deleted = 0
        filter_struct = self._dict_to_struct(filter_expr)

        while True:
            response = self.data_object_search_client.query_data_objects(
                request=vectorsearch_v1beta.QueryDataObjectsRequest(
                    parent=collection_path,
                    filter=filter_struct,
                    page_size=GCP_VECTOR_SEARCH_BATCH_SIZE,
                ),
                timeout=GCP_VECTOR_SEARCH_REQUEST_TIMEOUT,
            )

            data_objects = list(getattr(response, "data_objects", []) or [])
            if not data_objects:
                break

            delete_requests = [
                vectorsearch_v1beta.DeleteDataObjectRequest(name=obj.name)
                for obj in data_objects
                if getattr(obj, "name", None)
            ]
            if delete_requests:
                self.data_object_client.batch_delete_data_objects(
                    request=vectorsearch_v1beta.BatchDeleteDataObjectsRequest(
                        parent=collection_path,
                        requests=delete_requests,
                    ),
                    timeout=GCP_VECTOR_SEARCH_REQUEST_TIMEOUT,
                )
                total_deleted += len(delete_requests)

        return total_deleted
