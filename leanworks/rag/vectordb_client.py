from typing import Optional

from leanworks.rag.vectordb_gcp import GCPVectorSearchIndex


def use_gcp_vector_search() -> bool:
    """Return True if GCP Vector Search backend is enabled."""
    return True


def create_vectordb_client(
    embedding_model_client,
    gcp_credential_path: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
):
    """Create a GCP Vector Search client."""
    return GCPVectorSearchIndex(
        gcp_credential_path=gcp_credential_path,
        embedding_model_client=embedding_model_client,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
