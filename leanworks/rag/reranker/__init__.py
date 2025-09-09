"""Reranker modules for LeanWorks RAG system."""

from leanworks.rag.reranker.base_reranker import BaseReranker
from leanworks.rag.reranker.llm_reranker import CrossEncoderReranker
from leanworks.rag.reranker.reranker_factory import RerankerFactory

# Import BGEReranker only when needed to avoid onnxruntime dependency
def get_bge_reranker():
    """Get BGEReranker class, importing it only when needed."""
    from leanworks.rag.reranker.bge_reranker import BGEReranker
    return BGEReranker

__all__ = [
    "BaseReranker",
    "CrossEncoderReranker", 
    "BGEReranker",
    "RerankerFactory",
    "get_bge_reranker"
]
