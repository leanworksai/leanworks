from typing import Dict, Any, Optional, List
import uuid
import json
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from leanworks.rag.vectordb import UPSERT_BATCH_SIZE

logger = logging.getLogger(__name__)


class RAGStorageTool:
    """Store unstructured tool responses in vector database for RAG retrieval"""

    def __init__(self, vectordb_client, embedding_client, org_slug: str, chunk_size: int = 512, chunk_overlap: int = 128,
                 use_large_response_indexes: bool = False, large_response_dense_index: Optional[Any] = None,
                 large_response_sparse_index: Optional[Any] = None):
        """
        Initialize RAG storage tool.

        Args:
            vectordb_client: PineconeHybridIndex instance
            embedding_client: GoogleEmbedding instance
            org_slug: Organization slug for namespace
            chunk_size: Size of text chunks for vector storage
            chunk_overlap: Overlap between chunks
            use_large_response_indexes: Whether to use separate large response indexes
            large_response_dense_index: Separate dense index for large responses
            large_response_sparse_index: Separate sparse index for large responses
        """
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client
        self.org_slug = org_slug
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_large_response_indexes = use_large_response_indexes
        
        # Set namespace based on index type
        if use_large_response_indexes:
            self.namespace = f"{org_slug}_large_responses"
            # Use the provided large response indexes
            self.dense_index = large_response_dense_index
            self.sparse_index = large_response_sparse_index
        else:
            self.namespace = f"{org_slug}_tool_responses"
            # Use the vectordb_client indexes
            self.dense_index = None
            self.sparse_index = None

    def store_tool_response_in_vectorstore(
        self,
        content: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store unstructured tool response in vector database.

        Args:
            content: The text content to store
            tool_name: Name of the tool that generated this
            tool_input: Input parameters to the tool
            metadata: Additional metadata

        Returns:
            document_id: Unique identifier for retrieval
        """
        document_id = f"tool_response_{uuid.uuid4()}"

        # Prepare metadata
        doc_metadata = {
            "document_id": document_id,
            "tool_name": tool_name,
            "tool_input": json.dumps(tool_input) if tool_input else "{}",
            "data_source": f"tool_response_{tool_name}",
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
            "type": "tool_response",
            **(metadata or {})
        }

        # Chunk the content using vectordb_client's chunking method
        chunks = self.vectordb_client._chunk_text(content)

        if not chunks:
            logger.warning(f"No chunks generated for document {document_id}")
            return document_id

        # Create chunk metadata list (must include chunk_id for _create_vectors)
        chunk_metadata_list = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_meta = doc_metadata.copy()
            chunk_meta["chunk_id"] = chunk_id
            chunk_meta["chunk_index"] = i
            chunk_meta["total_chunks"] = len(chunks)
            chunk_meta["chunk_text"] = chunk
            chunk_metadata_list.append(chunk_meta)

        # Create vectors using the vectordb_client's method
        try:
            # _create_vectors expects (all_chunks, chunk_metadata_list)
            dense_vectors, sparse_vectors = self.vectordb_client._create_vectors(
                chunks, chunk_metadata_list
            )

            # Get the appropriate indexes to use
            dense_index = self.dense_index if self.use_large_response_indexes else self.vectordb_client.dense_index
            sparse_index = self.sparse_index if self.use_large_response_indexes else self.vectordb_client.sparse_index
            
            # Batch and upsert vectors in parallel for better performance
            def upsert_dense_batch(batch_vectors):
                """Upsert a batch of dense vectors."""
                if batch_vectors:
                    dense_index.upsert(
                        vectors=batch_vectors,
                        namespace=self.namespace
                    )

            def upsert_sparse_batch(batch_vectors):
                """Upsert a batch of sparse vectors."""
                if batch_vectors:
                    sparse_index.upsert(
                        vectors=batch_vectors,
                        namespace=self.namespace
                    )

            # Split vectors into batches
            dense_batches = [
                dense_vectors[i:i + UPSERT_BATCH_SIZE]
                for i in range(0, len(dense_vectors), UPSERT_BATCH_SIZE)
            ]
            sparse_batches = [
                sparse_vectors[i:i + UPSERT_BATCH_SIZE]
                for i in range(0, len(sparse_vectors), UPSERT_BATCH_SIZE)
            ]

            # Upsert dense and sparse vectors in parallel
            logger.info(f"Upserting {len(dense_vectors)} dense and {len(sparse_vectors)} sparse vectors in batches...")

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []

                # Submit all batch upserts
                for batch in dense_batches:
                    futures.append(executor.submit(upsert_dense_batch, batch))

                for batch in sparse_batches:
                    futures.append(executor.submit(upsert_sparse_batch, batch))

                # Wait for all upserts to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error during batch upsert: {e}")
                        raise

            logger.info(f"Stored tool response in RAG: {document_id} ({len(chunks)} chunks, namespace: {self.namespace})")
        except Exception as e:
            logger.error(f"Failed to store tool response in RAG: {e}")
            raise

        return document_id

    def search_tool_response_in_vectorstore(
        self,
        query: str,
        tool_name: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search stored tool responses using RAG.

        Args:
            query: Search query
            tool_name: Optional filter by tool name
            top_k: Number of results

        Returns:
            List of relevant chunks with metadata
        """
        filters = {"type": {"$eq": "tool_response"}}
        if tool_name:
            filters["tool_name"] = {"$eq": tool_name}

        try:
            results = self.vectordb_client.hybrid_search(
                query=query,
                top_k=top_k,
                namespace=self.namespace,
                filter=filters
            )
            return results
        except Exception as e:
            logger.error(f"Failed to search tool responses in RAG: {e}")
            return []

    def cleanup_session_data(self, session_id: str) -> None:
        """
        Clean up RAG namespace data by deleting the entire namespace.
        Deletes all vectors in the namespace from both leanworks-dense and leanworks-sparse indexes.

        Args:
            session_id: The session ID (for logging purposes, but entire namespace is deleted)
        """
        if not session_id:
            logger.warning("No session_id provided for RAG cleanup")
            return

        try:
            # Delete entire namespace from both dense and sparse indexes
            # This effectively removes the namespace from both indexes
            deleted_count = 0

            if self.vectordb_client.dense_index:
                try:
                    # Delete all vectors in the namespace (this deletes the namespace)
                    result = self.vectordb_client.dense_index.delete(
                        delete_all=True,
                        namespace=self.namespace
                    )
                    # Result might be a dict or just indicate success
                    if isinstance(result, dict) and "deleted" in result:
                        deleted_count += result.get("deleted", 0)
                    logger.info(f"Deleted namespace '{self.namespace}' from leanworks-dense index")
                except Exception as e:
                    # Handle 404 errors (namespace not found) gracefully - this is not an error
                    error_str = str(e)
                    # Check for Pinecone namespace not found errors (code 5 or 404)
                    is_namespace_not_found = (
                        "404" in error_str or
                        "Namespace not found" in error_str or
                        '"code":5' in error_str or
                        "'code': 5" in error_str
                    )
                    if is_namespace_not_found:
                        # Namespace doesn't exist, which is fine - nothing to clean up
                        logger.debug(f"Namespace '{self.namespace}' not found in leanworks-dense index, nothing to delete")
                    else:
                        logger.error(f"Failed to delete namespace '{self.namespace}' from leanworks-dense index: {e}")

            if self.vectordb_client.sparse_index:
                try:
                    # Delete all vectors in the namespace (this deletes the namespace)
                    result = self.vectordb_client.sparse_index.delete(
                        delete_all=True,
                        namespace=self.namespace
                    )
                    if isinstance(result, dict) and "deleted" in result:
                        deleted_count += result.get("deleted", 0)
                    logger.info(f"Deleted namespace '{self.namespace}' from leanworks-sparse index")
                except Exception as e:
                    # Handle 404 errors (namespace not found) gracefully - this is not an error
                    error_str = str(e)
                    # Check for Pinecone namespace not found errors (code 5 or 404)
                    is_namespace_not_found = (
                        "404" in error_str or
                        "Namespace not found" in error_str or
                        '"code":5' in error_str or
                        "'code': 5" in error_str
                    )
                    if is_namespace_not_found:
                        # Namespace doesn't exist, which is fine - nothing to clean up
                        logger.debug(f"Namespace '{self.namespace}' not found in leanworks-sparse index, nothing to delete")
                    else:
                        logger.error(f"Failed to delete namespace '{self.namespace}' from leanworks-sparse index: {e}")

            if deleted_count > 0:
                logger.info(f"RAG cleanup completed: deleted namespace '{self.namespace}' ({deleted_count} vector(s))")
            else:
                logger.info(f"RAG cleanup completed: namespace '{self.namespace}' not found or already empty")

        except Exception as e:
            logger.error(f"Error during RAG cleanup for namespace '{self.namespace}': {e}")


class BackgroundIndexingManager:
    """Manages background RAG indexing jobs using thread pool"""

    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs = {}  # job_id -> Future
        self.results = {}  # job_id -> result

    def submit_indexing_job(
        self,
        content: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        document_id: str,
        rag_storage: RAGStorageTool
    ) -> str:
        """Submit RAG indexing job to background thread pool"""
        job_id = str(uuid.uuid4())
        future = self.executor.submit(
            self._index_content,
            content, tool_name, tool_input, document_id, rag_storage
        )
        self.jobs[job_id] = future
        logger.info(f"Started background RAG indexing job: {job_id} for document: {document_id}")
        return job_id

    def _index_content(
        self,
        content: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        document_id: str,
        rag_storage: RAGStorageTool
    ) -> Dict[str, Any]:
        """Worker function that runs in background thread"""
        try:
            logger.info(f"Background indexing started for document: {document_id}")
            rag_storage.store_tool_response_in_vectorstore(
                content=content,
                tool_name=tool_name,
                tool_input=tool_input,
                metadata={"document_id": document_id}
            )
            logger.info(f"Background indexing completed for document: {document_id}")
            return {"status": "success", "document_id": document_id}
        except Exception as e:
            logger.error(f"Background RAG indexing failed for {document_id}: {e}")
            return {"status": "failed", "error": str(e), "document_id": document_id}

    def get_job_status(self, job_id: str) -> str:
        """Check if indexing job is complete"""
        if job_id not in self.jobs:
            return "unknown"
        future = self.jobs[job_id]
        if future.done():
            try:
                result = future.result()
                self.results[job_id] = result
                return "completed"
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                return "failed"
        return "in_progress"

    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get result of completed job"""
        if job_id in self.results:
            return self.results[job_id]
        if job_id in self.jobs and self.jobs[job_id].done():
            try:
                result = self.jobs[job_id].result()
                self.results[job_id] = result
                return result
            except Exception as e:
                return {"status": "failed", "error": str(e)}
        return None

    def shutdown(self, wait=True):
        """Shutdown the thread pool"""
        self.executor.shutdown(wait=wait)