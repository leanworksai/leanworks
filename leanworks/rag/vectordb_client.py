import os
from typing import Optional

from leanworks.rag.vectordb_gcp import GCPVectorSearchIndex


def use_gcp_vector_search() -> bool:
    """Return True if GCP Vector Search backend is enabled."""
    return os.getenv("USE_GCP_VECTOR_SEARCH", "true").lower() in ("true", "1", "yes")


def create_vectordb_client(
    embedding_model_client,
    pinecone_key: Optional[str] = None,
    gcp_credential_path: Optional[str] = None,
    dense_index_name: Optional[str] = None,
    sparse_index_name: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
):
    """Create a vector DB client based on configured backend."""
    if use_gcp_vector_search():
        return GCPVectorSearchIndex(
            gcp_credential_path=gcp_credential_path,
            embedding_model_client=embedding_model_client,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if not pinecone_key:
        raise ValueError("pinecone_key is required when USE_GCP_VECTOR_SEARCH is false")

    from leanworks.rag.vectordb import PineconeHybridIndex

    vectordb_client = PineconeHybridIndex(
        pinecone_key=pinecone_key,
        embedding_model_client=embedding_model_client,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if dense_index_name and sparse_index_name:
        vectordb_client.load_hybrid_index(
            dense_index_name=dense_index_name,
            sparse_index_name=sparse_index_name,
        )

    return vectordb_client
